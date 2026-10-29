"""Keyframe roll search inside jumper.horn_roll's own environment: its physics, the
servo curve and thermal limit, and the carapace collision. Plans found by
`search.py` (ideal servos) reach the carapace and stall on their backs here.

Same plan shape and schedule as `search.py` (five keyframes, 50 Hz, HOME to 3.5 s).
Each round plays one plan per environment from standing, for the whole population at
once. Scored as `search.py`'s "any" mode -- progress and success about the axis each
rollout turned about, upright at the end -- with costs on limb, stock-shell and sensor
force, carapace and floor impact (the task's own tracker) and air time. Joint torque is
not in the score here; the recording and replay measure it.

    python -u studies/proto3/env_search.py ROUNDS POP START_PLAN OUT_PLAN [STD]

Environment variables:

- `TASK` (default `jumper.horn_roll`): the environment searched in. With
  `jumper.tuck_roll` the score also charges the face's floor force (stock shell, base,
  sensors ahead of the base origin) and the worst floor-contact joint torque |J^T f|
  (both from that task's tracker), and the claws collide with the body.
- `MODE` (default `any`): `diagonal` keeps only rolls about either diagonal --
  right shoulder to left rear hip or its mirror, `search.py`'s axes -- scoring the
  finish by how close the measured axis is (`search.py`'s on-axis factor, 15 deg).
"""
import math
import sys
import tempfile
import time
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

ROUNDS, POP = int(sys.argv[1]), int(sys.argv[2])
START, OUT = HERE / sys.argv[3], HERE / sys.argv[4]
STD = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
ELITE = max(4, POP // 8)
use_backend(resolve("native", "cpu", num_envs=POP, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
import os  # noqa: E402

from tasks.jumper.horn_roll import env_cfg as H  # noqa: E402
from tasks.jumper.momentum_roll.mdp.reference import REF_NPZ  # noqa: E402

TASK = os.environ.get("TASK", "jumper.horn_roll")
MODE = os.environ.get("MODE", "any")
TUCK = TASK == "jumper.tuck_roll"
if not TUCK:
    tmp = Path(tempfile.mkdtemp()) / "bootstrap.npz"     # the tracker needs a reference
    np.savez(tmp, **dict(np.load(REF_NPZ)), axis=np.array([0.0, 1.0, 0.0]))
    H.REF_NPZ = tmp
cfg = tasks.load_env_cfg(TASK,
                         asset=ROOT / "assets/jumper/jumper_CF1-tri30m-rb-v3.xml", play=True)
cfg.scene.num_envs = POP
cfg.events.pop("push_robot", None)
cfg.terminations = {}
cfg.episode_length_s = 10.0
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
cmd = env.command_manager.get_term("roll")
names = [robot.joint_names[i] for i in term._target_ids]
home = robot.data.default_joint_pos[0, term._target_ids].numpy()
m = S.build()
_, snames, shome, lo, hi, drive, _ = S.setup(m)
half = (hi - lo) / 2
idx = [snames.index(n) for n in names]
nd = int(drive.sum())
dt = env.step_dt
FREE = {"limbs": 20.0, "sensors": 2.0, "shell": 30.0, "carapace_impact": 50.0,
        "ground_impact": 60.0, "flight": 0.06}
COST = {"limbs": 0.08, "sensors": 0.5, "shell": 0.04, "carapace_impact": 0.03,
        "ground_impact": 0.04, "flight": 40.0}
if TUCK:
    # The claws are the arms the roll goes over: their load has its own band there
    # (60 N); the face and the joint torque are what that task protects.
    FREE.update(limbs=60.0, face=5.0, load=4.0)
    COST.update(face=0.3, load=1.5)


def play(P):
    """Play each plan in its own environment; return per-env score parts."""
    targets = S.controls(P, shome, lo, hi, drive, half)[:, ::S.SUB][:, :, idx]   # [n, T, 20]
    env.reset()
    n, T = targets.shape[0], targets.shape[1]
    w = np.zeros((n, T, 3))
    up = np.zeros((n, T))
    pk = {k: np.zeros(n) for k in FREE}
    run = np.zeros(n)
    for t in range(T):
        a = torch.as_tensor((targets[:, t] - home) / 0.75, dtype=torch.float32)
        env.step(a)
        w[:, t] = robot.data.root_link_ang_vel_b.numpy()
        up[:, t] = -robot.data.projected_gravity_b[:, 2].numpy()
        limbs = cmd.force("limb_ground_contact")
        if TUCK:
            limbs = torch.maximum(limbs, cmd.force("arm_ground_contact"))
        pk["limbs"] = np.maximum(pk["limbs"], limbs.numpy())
        pk["sensors"] = np.maximum(pk["sensors"], cmd.force("sensor_ground_contact").numpy())
        pk["shell"] = np.maximum(pk["shell"], cmd.force("stock_shell_ground_contact").numpy())
        pk["carapace_impact"] = np.maximum(pk["carapace_impact"], cmd.carapace_impact().numpy())
        pk["ground_impact"] = np.maximum(pk["ground_impact"],
                                         cmd.impact("all_ground_contact").numpy())
        cmd._last = {s: v for s, v in cmd._last.items()}
        for s in (*cmd.cfg.region_sensors, cmd.cfg.ground_sensor):
            h = env.scene[s].data.force_history
            cmd._last[s] = h[:, :, 0].norm(dim=-1).sum(dim=1, keepdim=True)
        air = cmd.in_air().numpy()
        run = np.where(air, run + 1, 0)
        pk["flight"] = np.maximum(pk["flight"], run * dt)
    if TUCK:
        pk["face"] = cmd.peak_front.numpy().copy()
        pk["load"] = cmd.peak_load.numpy().copy()
    mid = ((up > -0.9) & (up < 0.9))[..., None]
    th = np.where((((w * mid).sum(1)[:, :2]) ** 2).sum(1, keepdims=True) > 0.09,
                  (w * mid).sum(1), w.sum(1)) * dt
    ax = th.copy()
    ax[:, 2] = 0
    ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-6)
    cum = np.cumsum((w * ax[:, None]).sum(-1), axis=1) * dt
    sign = np.where(cum.max(1) >= -cum.min(1), 1.0, -1.0)
    signed = cum * sign[:, None]
    need = 2 * math.pi
    prog = np.clip(signed.max(1), 0, need)
    tail = slice(-30, None)
    ok = (signed[:, -1] >= need - 0.5) & (up[:, tail].min(1) > 0.9)
    if MODE == "diagonal":
        cosd = np.abs(ax[:, :2] @ S._DIAGS[:, :2].T).max(1)
        err = np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))
        pk["on_axis"] = np.exp(-(err / 15.0) ** 2)
        ok = ok & (err < 20.0)
    return prog, ok, signed[:, -1], up, pk


def score(prog, ok, up, pk, pen):
    on = pk.get("on_axis", np.ones_like(prog))
    sc = 5 * prog + on * (20 * ok + 3 * (up[:, -30:].min(1) > 0.9))
    for k, v in pk.items():
        if k in COST:
            sc = sc - pen * COST[k] * np.maximum(0.0, v - FREE[k])
    return sc


rng = np.random.default_rng(0)
mu = np.load(START).astype(float)
sd = np.full_like(mu, STD)
best = (-1e9, mu.copy())
t0 = time.time()
for it in range(ROUNDS):
    P = np.clip(mu + sd * rng.standard_normal((POP,) + mu.shape), -1.5, 1.5)
    P[0], P[1] = mu, best[1]
    prog, ok, final, up, pk = play(P)
    pen = min(1.0, it / max(1, ROUNDS // 2))
    s = score(prog, ok, up, pk, pen)
    e = np.argsort(-s)[:ELITE]
    mu, sd = P[e].mean(0), np.maximum(P[e].std(0), 0.05)
    if s[e[0]] > best[0]:
        best = (float(s[e[0]]), P[e[0]].copy())
    print(f"round {it}: rolled {ok.sum()}/{POP}, best turn {math.degrees(final[e[0]]):.0f} deg, "
          f"best score {s[e[0]]:.1f} ({time.time() - t0:.0f}s)", flush=True)
# Choose by reliability, not by one lucky run: the environments differ in their
# randomised foot friction, centre of mass and encoder bias, and each keeps its own
# for the whole search. Each candidate -- the final mean, the best-scoring plan and the
# last round's top elites -- is played in every environment; the one that rolls in the
# most is kept.
cands = [("mean", mu), ("best", best[1])] + [(f"elite{i}", P[e[i]]) for i in range(3)]
results = []
for name, plan in cands:
    prog, ok, final, up, pk = play(np.repeat(plan[None], POP, 0))
    results.append((int(ok.sum()), name, plan))
    print(f"candidate {name}: rolls in {ok.sum()}/{POP} environments, median turn "
          f"{math.degrees(float(np.median(final))):.0f} deg", flush=True)
n_ok, name, plan = max(results, key=lambda r: r[0])
np.save(OUT, plan)
print(f"saved {OUT.relative_to(ROOT)}: candidate {name}, rolls in {n_ok}/{POP}")
