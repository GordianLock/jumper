"""Walk to a spot, stop, perform a trained behaviour, return to HOME -- one continuous
simulation for footage, controllers in sequence as the robot's FSM would run them.

    python studies/footage/walk_then_behave.py --model CF1-tri30m-rb-v4b \\
        --task jumper.diag_roll --actor studies/proto3/policies/shoulder_roll/actor.onnx \\
        --out studies/footage/recordings/shoulder_roll_v4b_walk.npz

1. **Walk**: KingKong's posture example policy (`tasks/jumper/posture/out/example`, the
   tripod gait) in jumper.posture's environment, commanded `--walk` (vx m/s, vy m/s,
   wz rad/s, seconds; segments joined by `;`), then a zero command for `--stop` s (the
   walking policy keeps stepping lightly on the spot, so it is never fully still).
2. **Stand**: joint targets ramp from where the joints are to HOME over `RAMP` s and
   hold until the joints are within `SETTLED` rad of HOME and the body is still: the
   start every behaviour was trained from.
3. **Behaviour**: the state (root pose and velocity, joint positions and velocities)
   is written into the behaviour task's own environment and its policy runs, with
   its finishing termination off, until that termination's predicate holds.
4. **Return**: the stand controller again.
5. **On**: back in the walking environment from that state, `--walk-after` (a few
   steps forward), a zero command, the stand controller, then `--tail` s still.

Both environments are play mode on the nominal robot (no randomisation, no pushes),
with the legs and the carapace colliding with the body, as the roll tasks have it.
Physics is simulated throughout; the hand-over between the two environments is an
exact copy of the state. The recording is in the demonstration format
`render_footage.py` draws.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tasks  # noqa: E402
from mjrl.backend.resolve import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402
from tasks.jumper.common.constants import GAIT_JOINTS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--task", required=True)
p.add_argument("--actor", required=True)
p.add_argument("--out", required=True)
p.add_argument("--walk", default="0.3,0,0,4.0")
p.add_argument("--stop", type=float, default=1.0, help="s of zero walking command before standing")
p.add_argument("--walk-after", default="0.3,0,0,1.5",
               help="after the behaviour and the return to HOME, walk this too (same "
                    "format as --walk; empty for none)")
p.add_argument("--tail", type=float, default=0.8)
a = p.parse_args()

use_backend(resolve("native", "cpu", num_envs=1, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from tasks.jumper.common import body_collision as bc  # noqa: E402
from tasks.jumper.horn_roll import env_cfg as horn  # noqa: E402

RAMP, SETTLED, STILL_W, STILL_V = 0.5, 0.05, 0.1, 0.01
XML = ROOT / "assets" / "jumper" / ("jumper.xml" if a.model == "jumper" else f"jumper_{a.model}.xml")
WALKER = ROOT / "tasks" / "jumper" / "posture" / "out" / "example" / "actor.onnx"


def nominal(cfg):
    cfg.scene.num_envs = 1
    cfg.events.pop("push_robot", None)
    for k in [k for k, e in cfg.events.items() if e.mode == "startup"]:
        cfg.events.pop(k)
    return cfg


rec = {k: [] for k in ("q", "root_pos", "root_quat")}
phases = []


#: The recording's step. jumper.posture's control runs faster than the roll tasks'
#: 20 ms, so its steps are logged every REC_DT only.
REC_DT = 0.02
_count = {}


def log(env, ids):
    n = _count[id(env)] = _count.get(id(env), 0) + 1
    if n % max(1, round(REC_DT / env.step_dt)):
        return
    r = env.scene["robot"]
    rec["q"].append(r.data.joint_pos[0, ids].numpy().copy())
    rec["root_pos"].append(r.data.root_link_pos_w[0].numpy() - env.scene.env_origins[0].numpy())
    rec["root_quat"].append(r.data.root_link_quat_w[0].numpy().copy())


def still(r):
    return (float(r.data.root_link_ang_vel_b[0].norm()) < STILL_W
            and float(r.data.root_link_lin_vel_b[0].norm()) < STILL_V)


def stand(env, ids, max_s=3.0):
    """Ramp the joint targets to HOME and hold until settled there."""
    r = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    scale = term._scale[0].numpy() if torch.is_tensor(term._scale) else float(term._scale)
    home = r.data.default_joint_pos[0, ids].numpy()
    start = r.data.joint_pos[0, ids].numpy().copy()
    n_ramp = int(RAMP / env.step_dt)
    for k in range(int(max_s / env.step_dt)):
        f = min(1.0, (k + 1) / n_ramp)
        env.step(torch.as_tensor((((1 - f) * (start - home)) / scale)[None], dtype=torch.float32))
        log(env, ids)
        err = np.abs(r.data.joint_pos[0, ids].numpy() - home).max()
        if k >= n_ramp and err < SETTLED and still(r):
            return
    print("warning: did not settle at HOME within", max_s, "s")


# ── 1-2: walk, stop, stand ─────────────────────────────────────────────────────
wcfg = nominal(tasks.load_env_cfg("jumper.posture", asset=XML, play=False))
wcfg.curriculum = {}
wcfg.episode_length_s = 120.0
tw = wcfg.commands["twist"]
tw.resampling_time_range, tw.rel_standing_envs, tw.heading_command = (1e9, 1e9), 0.0, False
tw.ranges.heading = None
wcfg.commands["posture"].resampling_time_range = (1e9, 1e9)
robot_cfg = wcfg.scene.entities["robot"]
carapace = a.model != "jumper"
robot_cfg.collisions = tuple(robot_cfg.collisions) + bc.arm_body_collision(
    horn.CARAPACE_GEOMS if carapace else None, horn.CARAPACE_FRICTION if carapace else None
) + (bc.walking_leg_body_collision(),)
wcfg.terminations = {k: v for k, v in wcfg.terminations.items() if k == "time_out"}
# Start at the origin facing +x, set down just above the floor, so the walk's direction
# (and the camera, which is placed relative to it) is the same every time.
wcfg.events["reset_base"].params["pose_range"] = {"z": (0.01, 0.01)}
wenv = ManagerBasedRlEnv(cfg=wcfg, device="cpu")
wr = wenv.scene["robot"]
wterm = wenv.action_manager.get_term("joint_pos")
wids = wterm._target_ids
assert tuple(wr.joint_names[i] for i in wids) == tuple(GAIT_JOINTS)
vc, pc = wenv.command_manager.get_term("twist"), wenv.command_manager.get_term("posture")
walker = ort.InferenceSession(str(WALKER))
wname = walker.get_inputs()[0].name
okey = [k for k in wenv.observation_manager.active_terms if k != "critic"][0]
obs, _ = wenv.reset()


def walk(obs, spec):
    """Run the walking policy through `spec`'s segments, then `--stop` s at zero."""
    segments = [tuple(float(x) for x in seg.split(",")) for seg in spec.split(";")]
    for vx, vy, wz, secs in segments + [(0.0, 0.0, 0.0, None)]:
        k = 0
        while True:
            vc.vel_command_b[:] = torch.tensor([vx, vy, wz])
            pc.posture_command[:, :3] = 0.0
            pc.posture_command[:, 3] = pc.cfg.neutral_height
            act = walker.run(None, {wname: obs[okey].numpy().astype(np.float32)})[0]
            obs, *_ = wenv.step(torch.as_tensor(act))
            log(wenv, wids)
            k += 1
            if secs is not None and k * wenv.step_dt >= secs:
                break
            if secs is None and k * wenv.step_dt >= a.stop:
                break


walk(obs, a.walk)
phases.append(("walk", len(rec["q"])))
stand(wenv, wids)
phases.append(("stand", len(rec["q"])))
root = wr.data.root_state_w.clone() if hasattr(wr.data, "root_state_w") else torch.cat(
    [wr.data.root_link_pos_w, wr.data.root_link_quat_w, wr.data.root_link_lin_vel_w,
     wr.data.root_link_ang_vel_w], dim=1).clone()
root[:, :3] -= wenv.scene.env_origins
jpos, jvel = wr.data.joint_pos.clone(), wr.data.joint_vel.clone()
jnames = list(wr.joint_names)

# ── 3: the behaviour, from exactly that state ──────────────────────────────────
bcfg = nominal(tasks.load_env_cfg(a.task, asset=XML, play=True))
finish = {k: bcfg.terminations.pop(k) for k in ("rolled_and_standing", "standing", "on_back")
          if k in bcfg.terminations}
assert len(finish) == 1, f"{a.task}: no single finishing termination ({list(bcfg.terminations)})"
fin = next(iter(finish.values()))
bcfg.episode_length_s = 60.0
benv = ManagerBasedRlEnv(cfg=bcfg, device="cpu")
br = benv.scene["robot"]
bids = benv.action_manager.get_term("joint_pos")._target_ids
assert list(br.joint_names) == jnames
policy = ort.InferenceSession(str(Path(a.actor)))
pname = policy.get_inputs()[0].name
benv.reset()
root[:, :3] += benv.scene.env_origins
br.write_root_state_to_sim(root)
br.write_joint_state_to_sim(jpos, jvel)
benv.sim.forward()
cmd = benv.command_manager.get_term("roll")
cmd._resample_command(torch.arange(1))                 # the turn counts from here
obs = benv.observation_manager.compute()
for k in range(int(10.0 / benv.step_dt)):
    act = policy.run(None, {pname: obs["actor"].numpy().astype(np.float32)})[0]
    obs, *_ = benv.step(torch.as_tensor(act))
    log(benv, bids)
    if bool(fin.func(benv, **fin.params)[0]):
        break
else:
    print("warning: the behaviour's finish never held")
phases.append((a.task, len(rec["q"])))

# ── 4: back to HOME ────────────────────────────────────────────────────────────
stand(benv, bids)
phases.append(("home", len(rec["q"])))

# ── 5: a few steps on, stop, stand (the walking environment, from this state) ──
if a.walk_after:
    st = torch.cat([br.data.root_link_pos_w - benv.scene.env_origins + wenv.scene.env_origins,
                    br.data.root_link_quat_w, br.data.root_link_lin_vel_w,
                    br.data.root_link_ang_vel_w], dim=1)
    wr.write_root_state_to_sim(st)
    wr.write_joint_state_to_sim(br.data.joint_pos.clone(), br.data.joint_vel.clone())
    wenv.sim.forward()
    walk(wenv.observation_manager.compute(), a.walk_after)
    phases.append(("walk_after", len(rec["q"])))
    stand(wenv, wids)
    env_end, ids_end = wenv, wids
else:
    env_end, ids_end = benv, bids
for _ in range(int(a.tail / env_end.step_dt)):
    env_end.step(torch.zeros(1, env_end.action_manager.total_action_dim))
    log(env_end, ids_end)
phases.append(("end", len(rec["q"])))

dt = REC_DT
assert abs(benv.step_dt - REC_DT) < 1e-9, benv.step_dt
n = len(rec["q"])
print("phases (end, s):", ", ".join(f"{name} {end * dt:.2f}" for name, end in phases))
p0, p1 = np.array(rec["root_pos"][0]), np.array(rec["root_pos"][phases[1][1] - 1])
print(f"walked {np.linalg.norm(p1[:2] - p0[:2]):.2f} m; total {n * dt:.2f} s")
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, dt=np.float32(dt), joint_names=np.array(GAIT_JOINTS), axis=np.array([1, 0, 0], np.float32),
         total=np.zeros(n, np.float32), phases=np.array([e for _, e in phases]),
         phase_names=np.array([nm for nm, _ in phases]),
         **{k: np.array(v, np.float32) for k, v in rec.items()})
print("wrote", out)
