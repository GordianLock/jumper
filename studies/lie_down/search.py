"""Keyframe plan search for jumper.lie_down, inside the task's own environment: its
physics, the servo curve and thermal limit, the legs colliding with the body, the
tracker's hardest hits and floor-contact joint torque.

## The plan

`K` keyframes of the 20 driven joints, each an offset from HOME in units of half the
joint's range (clipped to its limits), reached in turn. Unlike `studies/proto3/search.py`
the targets **ramp** linearly from one keyframe to the next over `SEG` s rather than
stepping, since the motion asked for is slow and deliberate; after the last keyframe
they ramp back to HOME over `SEG` s and stay there for `HOME_HOLD` s, so the plan ends
with the legs at HOME. The plan's clock starts when the tracker releases the robot
(each environment settles at HOME first, as in training).

## The score

- progress: how far onto its back it got (`tracker.best_down - spawn_down`, 0-2), x5;
- success, +20: the task's own finish (`tracker.done`: belly-up within 20 deg, still,
  every joint within 0.15 rad of HOME, held 0.3 s) by the end of the plan;
- costs over the task's free levels (`tasks/jumper/lie_down/env_cfg.py`): shell and
  floor impact, floor-contact joint torque, body rotation, limbs, sensors, air time,
  and steps with legs pressing on each other. Phased in over the first half of the
  rounds, as in `studies/proto3/env_search.py`.

The plan kept is chosen by reliability: each candidate (the final mean, the best
plan, the last round's top elites) is played in every environment -- each with its
own randomised foot friction, centre of mass and encoder bias -- and the one that
finishes in the most is saved, ties broken by score.

    python -u studies/lie_down/search.py ROUNDS POP OUT_PLAN [STD] [START_PLAN]

START_PLAN omitted: the search starts from HOME (all offsets 0).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import tasks  # noqa: E402
from mjrl.backend.resolve import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402

ROUNDS, POP = int(sys.argv[1]), int(sys.argv[2])
OUT = Path(sys.argv[3])
STD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
START = Path(sys.argv[5]) if len(sys.argv) > 5 else None
ELITE = max(4, POP // 8)
use_backend(resolve("native", "cpu", num_envs=POP, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from tasks.jumper.common import body_collision as bc  # noqa: E402
from tasks.jumper.lie_down import env_cfg as LD  # noqa: E402

K = 6
SEG = 0.5
HOME_HOLD = 1.0
SETTLE_MAX = LD.MAX_SETTLE_STEPS

cfg = tasks.load_env_cfg("jumper.lie_down", asset=ROOT / "assets/jumper/jumper.xml",
                         play=True)
cfg.scene.num_envs = POP
cfg.events.pop("push_robot", None)
cfg.terminations = {}
cfg.episode_length_s = 30.0
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
cmd = env.command_manager.get_term("roll")
ids = term._target_ids
home = robot.data.default_joint_pos[0, ids].numpy()
lim = robot.data.soft_joint_pos_limits[0, ids].numpy()
lo, hi = lim[:, 0], lim[:, 1]
half = (hi - lo) / 2
scale = term._scale[0].numpy() if torch.is_tensor(term._scale) else float(term._scale)
dt = env.step_dt
pair_names = tuple(s.name for s in bc.leg_pair_sensors(LD.DECIMATION))

seg = int(round(SEG / dt))
T_PLAN = (K + 1) * seg + int(round(HOME_HOLD / dt))

FREE = {"shell_impact": LD.SHELL_IMPACT_BAND[0], "ground_impact": LD.GROUND_IMPACT_BAND[0],
        "load": LD.LOAD_BAND[0], "spin": LD.SPIN_BAND[0], "limbs": LD.LIMB_BAND[0],
        "sensors": LD.SENSOR_BAND[0], "flight": LD.FLIGHT_BAND[0], "pairs": 0.0}
COST = {"shell_impact": 0.08, "ground_impact": 0.05, "load": 2.0, "spin": 1.5,
        "limbs": 0.08, "sensors": 0.5, "flight": 40.0, "pairs": 0.2}


def targets(P: np.ndarray) -> np.ndarray:
    """Joint targets per plan step, [n, T_PLAN, 20]: HOME, ramps through the K
    keyframes, back to HOME, held."""
    keys = np.clip(home + P * half, lo, hi)                          # [n, K, 20]
    pts = np.concatenate([np.broadcast_to(home, (len(P), 1, len(home))), keys,
                          np.broadcast_to(home, (len(P), 1, len(home)))], axis=1)
    out = np.empty((len(P), T_PLAN, len(home)))
    for i in range(K + 1):
        u = (np.arange(1, seg + 1) / seg)[None, :, None]
        out[:, i * seg:(i + 1) * seg] = (1 - u) * pts[:, i:i + 1] + u * pts[:, i + 1:i + 2]
    out[:, (K + 1) * seg:] = home
    return out


def play(P: np.ndarray):
    """Play each plan in its own environment from its release; per-env results."""
    tg = targets(P)
    n = len(P)
    env.reset()
    clock = np.full(n, -1)                       # plan step; -1 until released
    pairs = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    k = 0
    while (clock < T_PLAN).any() and k < SETTLE_MAX + T_PLAN + 5:
        released = ~cmd.settling.numpy()
        clock = np.where(released & (clock < 0), 0, clock)
        c = np.clip(clock, 0, T_PLAN - 1)
        a = np.where((clock >= 0)[:, None], (tg[np.arange(n), c] - home) / scale, 0.0)
        env.step(torch.as_tensor(a, dtype=torch.float32))
        live = (clock >= 0) & (clock < T_PLAN)
        pairs += bc.leg_collision_pairs(env, pair_names).numpy() * live
        done |= cmd.done.numpy() & live
        clock = np.where(clock >= 0, clock + 1, clock)
        k += 1
    pk = {"shell_impact": cmd.peak_shell_impact.numpy().copy(),
          "ground_impact": cmd.peak_ground_impact.numpy().copy(),
          "load": cmd.peak_load.numpy().copy(), "spin": cmd.peak_spin.numpy().copy(),
          "limbs": cmd.peak_limb.numpy().copy(), "sensors": cmd.peak_sensor.numpy().copy(),
          "flight": cmd.longest_flight.numpy() * dt, "pairs": pairs * dt}
    prog = (cmd.best_down - cmd.spawn_down).numpy().copy()
    return prog, done, pk


def score(prog, done, pk, pen):
    s = 5 * prog + 20 * done
    for key, v in pk.items():
        s = s - pen * COST[key] * np.maximum(0.0, v - FREE[key])
    return s


rng = np.random.default_rng(0)
mu = np.load(START).astype(float) if START else np.zeros((K, len(home)))
assert mu.shape == (K, len(home)), mu.shape
sd = np.full_like(mu, STD)
best = (-1e9, mu.copy())
t0 = time.time()
for it in range(ROUNDS):
    P = np.clip(mu + sd * rng.standard_normal((POP,) + mu.shape), -1.5, 1.5)
    P[0], P[1] = mu, best[1]
    prog, done, pk = play(P)
    pen = min(1.0, it / max(1, ROUNDS // 2))
    s = score(prog, done, pk, pen)
    e = np.argsort(-s)[:ELITE]
    mu, sd = P[e].mean(0), np.maximum(P[e].std(0), 0.05)
    if s[e[0]] > best[0]:
        best = (float(s[e[0]]), P[e[0]].copy())
    b = e[0]
    print(f"round {it}: finished {done.sum()}/{POP}; best: down {prog[b]:.2f}, "
          f"shell {pk['shell_impact'][b]:.0f} N, floor {pk['ground_impact'][b]:.0f} N, "
          f"load {pk['load'][b]:.1f} N*m, spin {pk['spin'][b]:.1f} rad/s, "
          f"score {s[b]:.1f} ({time.time() - t0:.0f}s)", flush=True)

cands = [("mean", mu), ("best", best[1])] + [(f"elite{i}", P[e[i]]) for i in range(3)]
results = []
for name, plan in cands:
    prog, done, pk = play(np.repeat(plan[None], POP, 0))
    s = score(prog, done, pk, 1.0)
    results.append((int(done.sum()), float(np.median(s)), name, plan))
    print(f"candidate {name}: finishes in {done.sum()}/{POP}; worst shell "
          f"{pk['shell_impact'].max():.0f} N, floor {pk['ground_impact'].max():.0f} N, "
          f"load {pk['load'].max():.1f} N*m, spin {pk['spin'].max():.1f} rad/s", flush=True)
n_ok, med, name, plan = max(results, key=lambda r: (r[0], r[1]))
OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT, plan)
print(f"saved {OUT}: candidate {name}, finishes in {n_ok}/{POP}")
