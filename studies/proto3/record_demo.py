"""Record a keyframe plan as jumper.horn_roll's demonstration, in that task's own
environment (its physics, servo curve and carapace collision), and save it in
jumper.momentum_roll's reference format plus the axis it rolled about.

    python studies/proto3/record_demo.py plans/stand_any_impact_s1.npy

`TASK` (default jumper.horn_roll) picks the environment, `KEEP_INCOMPLETE=1` writes a
recording that does not complete (to look at, not to train from), `MODEL` the carapace
(default CF1-tri30m-rb-v3), `OUT` the file written.

The plan is played from standing at HOME with nothing else acting: the feedforward
guide off, play mode (no observation noise, no pushes), one environment. Each control
step's targets are the plan's (`search.controls`, read every 4th physics step).

The axis is the horizontal part of the body-frame rotation accumulated while the body
is between belly-up and upright (as `search.axes_for`); the turn about it is the
tracker's angle, atan2(g . (a x z), -g_z), unwrapped. The recording runs until the
robot has turned a full roll within 0.3 rad, is upright and has stood still for
`HOLD` s, or to the plan's end.

It prints the turn, whether it finished upright, and the hardest hits, and refuses to
write a recording that does not complete the roll in this environment.
"""
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import search as S  # noqa: E402
import tasks  # noqa: E402
from mjrl.backend.resolve import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402
from tasks.jumper.common.constants import GAIT_JOINTS  # noqa: E402

use_backend(resolve("native", "cpu", num_envs=1, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402

PLAN = HERE / sys.argv[1]
TASK = os.environ.get("TASK", "jumper.horn_roll")
#: Where the recording goes: OUT=<path> overrides the task's default demonstration.
OUT = Path(os.environ["OUT"]) if os.environ.get("OUT") else \
    ROOT / "tasks" / "jumper" / "horn_roll" / "ref" / "horn_roll.npz"
MODEL = ROOT / "assets" / "jumper" / \
    f"jumper_{os.environ.get('MODEL', 'CF1-tri30m-rb-v3')}.xml"
HOLD = 0.3
ACTION_SCALE = 0.75

# The environment needs a reference to build; until this one exists, borrow
# jumper.momentum_roll's with a placeholder axis. Nothing of it is used here.
from tasks.jumper.horn_roll import env_cfg as H  # noqa: E402
from tasks.jumper.momentum_roll.mdp.reference import REF_NPZ  # noqa: E402

if TASK == "jumper.horn_roll":
    tmp = Path(tempfile.mkdtemp()) / "bootstrap.npz"
    np.savez(tmp, **dict(np.load(REF_NPZ)), axis=np.array([0.0, 1.0, 0.0]))
    H.REF_NPZ = tmp
cfg = tasks.load_env_cfg(TASK, asset=MODEL, play=True)
cfg.scene.num_envs = 1
cfg.events.pop("push_robot", None)
if os.environ.get("NOMINAL"):
    # The nominal robot: no randomised foot friction, centre of mass or encoder bias.
    for k in [k for k, e in cfg.events.items() if e.mode == "startup"]:
        cfg.events.pop(k)
cfg.terminations.pop("rolled_and_standing", None)
cfg.episode_length_s = 10.0
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
names = [robot.joint_names[i] for i in term._target_ids]
assert tuple(names) == tuple(GAIT_JOINTS), names
home = robot.data.default_joint_pos[0, term._target_ids].numpy()

# The plan's targets, in the search's joint order, mapped to this task's.
m = S.build()
_, snames, shome, lo, hi, drive, _ = S.setup(m)
u = S.controls(np.load(PLAN)[None], shome, lo, hi, drive, (hi - lo) / 2)[0][::S.SUB]
idx = [snames.index(n) for n in names]
targets = u[:, idx]

env.reset()
cmd = env.command_manager.get_term("roll")
origin = env.scene.env_origins[0].numpy()
rec = {k: [] for k in ("q_cmd", "q", "qd", "root_pos", "root_quat", "root_lin", "root_ang")}
w_body, still = [], 0
for k, tgt in enumerate(targets):
    a = torch.as_tensor((tgt - home) / ACTION_SCALE, dtype=torch.float32)[None]
    env.step(a)
    rec["q_cmd"].append(tgt)
    rec["q"].append(robot.data.joint_pos[0, term._target_ids].numpy().copy())
    rec["qd"].append(robot.data.joint_vel[0, term._target_ids].numpy().copy())
    rec["root_pos"].append(robot.data.root_link_pos_w[0].numpy() - origin)
    rec["root_quat"].append(robot.data.root_link_quat_w[0].numpy().copy())
    rec["root_lin"].append(robot.data.root_link_lin_vel_w[0].numpy().copy())
    rec["root_ang"].append(robot.data.root_link_ang_vel_w[0].numpy().copy())
    w_body.append(robot.data.root_link_ang_vel_b[0].numpy().copy())
rec = {k: np.array(v, dtype=np.float32) for k, v in rec.items()}
w_body = np.array(w_body)
dt = env.step_dt

# The axis, and the turn about it.
g_b = quat_apply_inverse(torch.as_tensor(rec["root_quat"]),
                         torch.tensor([[0.0, 0.0, -1.0]]).expand(len(targets), 3)).numpy()
up = -g_b[:, 2]
mid = (up > -0.9) & (up < 0.9)
th = (w_body * mid[:, None]).sum(0) * dt
axis = np.array([th[0], th[1], 0.0]) / max(1e-6, math.hypot(th[0], th[1]))
side = np.cross(axis, [0.0, 0.0, 1.0])
total = np.unwrap(np.arctan2(g_b @ side, -g_b[:, 2]))
total -= total[0]
if total[np.argmax(np.abs(total))] < 0:          # positive the way it rolled
    axis, side, total = -axis, -side, -total

done = (total >= 2 * math.pi - 0.3) & (up > 0.9) & (np.linalg.norm(w_body, axis=1) < 1.0)
end = len(targets)
if done.any():
    first = int(np.argmax(done))
    end = min(len(targets), first + int(HOLD / dt) + 1)
rec = {k: v[:end] for k, v in rec.items()}
total = total[:end]
print(f"axis {math.degrees(math.atan2(axis[1], axis[0])):.0f} deg (0 = centreline, 90 = end "
      f"over end); turn {math.degrees(total[-1]):.0f} deg; upright at the end: "
      f"{bool(up[end - 1] > 0.9)}; recording {end} steps, {end * dt:.2f} s")
print(f"hardest hits over the recording: limbs {float(cmd.peak_limb):.0f} N, stock shell "
      f"{float(cmd.peak_shell):.0f} N, carapace impact {float(cmd.peak_carapace_impact):.0f} N, "
      f"floor impact {float(cmd.peak_ground_impact):.0f} N, longest flight "
      f"{float(cmd.longest_flight) * dt:.2f} s")
if hasattr(cmd, "peak_load"):
    print(f"face {float(cmd.peak_front):.0f} N, worst floor-contact joint torque "
          f"{float(cmd.peak_load):.2f} N*m")
if not done.any() and not os.environ.get("KEEP_INCOMPLETE"):
    sys.exit("the plan does not complete the roll in this environment; nothing written "
             "(KEEP_INCOMPLETE=1 writes it anyway, to look at, never to train from)")
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, dt=np.float32(dt), joint_names=np.array(names), total=total.astype(np.float32),
         axis=axis.astype(np.float32), source=str(PLAN.relative_to(ROOT)), **rec)
print("wrote", OUT)
env.close()
