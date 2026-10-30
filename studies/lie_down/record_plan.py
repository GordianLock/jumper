"""Record a lie-down plan (`search.py`'s output) on the nominal stock robot, in the
demonstration format (root pose and joint angles per 20 ms control step, plus the
commanded targets), and report its hardest hits.

    python studies/lie_down/record_plan.py <plan.npy> <out.npz>

The plan is played as `search.py` plays it -- from the tracker's release, keyframes
reached by linear ramps, back to HOME and held -- in jumper.lie_down's own environment
(play mode, no randomisation, no pushes), with the finish termination off so the
recording runs to the end of the plan.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tasks  # noqa: E402
from mjrl.backend.resolve import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402
from tasks.jumper.common.constants import GAIT_JOINTS  # noqa: E402

use_backend(resolve("native", "cpu", num_envs=1, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

PLAN, OUT = np.load(sys.argv[1]), Path(sys.argv[2])
K, SEG, HOME_HOLD = 6, 0.5, 1.0                         # as search.py

cfg = tasks.load_env_cfg("jumper.lie_down", asset=ROOT / "assets/jumper/jumper.xml", play=True)
cfg.scene.num_envs = 1
cfg.events.pop("push_robot", None)
for k in [k for k, e in cfg.events.items() if e.mode == "startup"]:
    cfg.events.pop(k)
cfg.terminations = {}
cfg.episode_length_s = 30.0
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
cmd = env.command_manager.get_term("roll")
ids = term._target_ids
assert tuple(robot.joint_names[i] for i in ids) == tuple(GAIT_JOINTS)
home = robot.data.default_joint_pos[0, ids].numpy()
lim = robot.data.soft_joint_pos_limits[0, ids].numpy()
lo, hi = lim[:, 0], lim[:, 1]
half = (hi - lo) / 2
scale = term._scale[0].numpy() if torch.is_tensor(term._scale) else float(term._scale)
dt = env.step_dt
seg = int(round(SEG / dt))
keys = np.clip(home + PLAN * half, lo, hi)
pts = np.concatenate([home[None], keys, home[None]])
tg = [(1 - u) * pts[i] + u * pts[i + 1] for i in range(K + 1)
      for u in np.arange(1, seg + 1) / seg] + [home] * int(round(HOME_HOLD / dt))

origin = env.scene.env_origins[0].numpy()
rec = {k: [] for k in ("q_cmd", "q", "qd", "root_pos", "root_quat", "root_lin", "root_ang")}
env.reset()
clock, finished = -1, None
while clock < len(tg):
    if clock < 0 and not bool(cmd.settling[0]):
        clock = 0
    target = tg[clock] if clock >= 0 else home
    env.step(torch.as_tensor(((target - home) / scale)[None], dtype=torch.float32))
    if clock >= 0:
        clock += 1
    rec["q_cmd"].append(np.asarray(target, np.float32))
    rec["q"].append(robot.data.joint_pos[0, ids].numpy().copy())
    rec["qd"].append(robot.data.joint_vel[0, ids].numpy().copy())
    rec["root_pos"].append(robot.data.root_link_pos_w[0].numpy() - origin)
    rec["root_quat"].append(robot.data.root_link_quat_w[0].numpy().copy())
    rec["root_lin"].append(robot.data.root_link_lin_vel_w[0].numpy().copy())
    rec["root_ang"].append(robot.data.root_link_ang_vel_w[0].numpy().copy())
    if finished is None and bool(cmd.done[0]):
        finished = len(rec["q"]) * dt
rec = {k: np.array(v, dtype=np.float32) for k, v in rec.items()}
n = len(rec["q"])
print(f"recorded {n} steps, {n * dt:.2f} s; finished (task's finish) at "
      f"{finished if finished is None else round(finished, 2)} s")
print(f"hardest hits: shell impact {float(cmd.peak_shell_impact[0]):.0f} N, floor impact "
      f"{float(cmd.peak_ground_impact[0]):.0f} N, joint torque {float(cmd.peak_load[0]):.1f} N*m, "
      f"spin {float(cmd.peak_spin[0]):.1f} rad/s, limbs {float(cmd.peak_limb[0]):.0f} N, sensors "
      f"{float(cmd.peak_sensor[0]):.0f} N, longest flight {float(cmd.longest_flight[0]) * dt:.2f} s")
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, dt=np.float32(dt), joint_names=np.array(GAIT_JOINTS), axis=np.array([1, 0, 0], np.float32),
         total=np.zeros(n, np.float32), source=Path(sys.argv[1]).name, **rec)
print("wrote", OUT)
