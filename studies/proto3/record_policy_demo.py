"""Record a trained jumper.horn_roll policy's roll as a demonstration, in
jumper.momentum_roll's reference format plus the axis it rolled about -- the format
`record_demo.py` writes from a keyframe plan.

    python studies/proto3/record_policy_demo.py <actor.onnx> <out.npz>

The policy is the exported actor (`scripts/export.py`'s `actor.onnx`, observation
normalisation inside it). It is played in jumper.horn_roll's own environment on
prototype 3: the nominal robot (no randomised friction, centre of mass or encoder
bias), play mode (no pushes, no feedforward guide), one environment, from standing,
with the standing termination off so the recording runs past it.

The axis and the turn about it are worked out as `record_demo.py` does. The
recording stops `HOLD` s after the robot has turned a full roll within 0.3 rad, is
upright and has stopped turning; it refuses to write one that never gets there.
"""
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import tasks  # noqa: E402
from mjrl.backend.resolve import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402
from tasks.jumper.common.constants import GAIT_JOINTS  # noqa: E402

use_backend(resolve("native", "cpu", num_envs=1, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402

ACTOR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODEL = ROOT / "assets" / "jumper" / "jumper_CF1-tri30m-rb-v3.xml"
HOLD = 0.3
MAX_S = 4.0

cfg = tasks.load_env_cfg("jumper.horn_roll", asset=MODEL, play=True)
cfg.scene.num_envs = 1
cfg.events.pop("push_robot", None)
for k in [k for k, e in cfg.events.items() if e.mode == "startup"]:
    cfg.events.pop(k)                               # the nominal robot
cfg.terminations.pop("rolled_and_standing", None)
cfg.episode_length_s = MAX_S + 1.0
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
names = [robot.joint_names[i] for i in term._target_ids]
assert tuple(names) == tuple(GAIT_JOINTS), names
home = robot.data.default_joint_pos[0, term._target_ids].numpy()
scale = float(term._scale[0, 0]) if torch.is_tensor(term._scale) else float(term._scale)

sess = ort.InferenceSession(str(ACTOR))
iname = sess.get_inputs()[0].name

obs, _ = env.reset()
cmd = env.command_manager.get_term("roll")
origin = env.scene.env_origins[0].numpy()
rec = {k: [] for k in ("q_cmd", "q", "qd", "root_pos", "root_quat", "root_lin", "root_ang")}
w_body = []
dt = env.step_dt
for k in range(int(MAX_S / dt)):
    a = sess.run(None, {iname: obs["actor"].numpy().astype(np.float32)})[0]
    obs, *_ = env.step(torch.as_tensor(a))
    rec["q_cmd"].append(a[0] * scale + home)
    rec["q"].append(robot.data.joint_pos[0, term._target_ids].numpy().copy())
    rec["qd"].append(robot.data.joint_vel[0, term._target_ids].numpy().copy())
    rec["root_pos"].append(robot.data.root_link_pos_w[0].numpy() - origin)
    rec["root_quat"].append(robot.data.root_link_quat_w[0].numpy().copy())
    rec["root_lin"].append(robot.data.root_link_lin_vel_w[0].numpy().copy())
    rec["root_ang"].append(robot.data.root_link_ang_vel_w[0].numpy().copy())
    w_body.append(robot.data.root_link_ang_vel_b[0].numpy().copy())
rec = {k: np.array(v, dtype=np.float32) for k, v in rec.items()}
w_body = np.array(w_body)
n = len(w_body)

# The axis, and the turn about it (as record_demo.py).
g_b = quat_apply_inverse(torch.as_tensor(rec["root_quat"]),
                         torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3)).numpy()
up = -g_b[:, 2]
mid = (up > -0.9) & (up < 0.9)
th = (w_body * mid[:, None]).sum(0) * dt
axis = np.array([th[0], th[1], 0.0]) / max(1e-6, math.hypot(th[0], th[1]))
side = np.cross(axis, [0.0, 0.0, 1.0])
total = np.unwrap(np.arctan2(g_b @ side, -g_b[:, 2]))
total -= total[0]
if total[np.argmax(np.abs(total))] < 0:
    axis, side, total = -axis, -side, -total

done = (total >= 2 * math.pi - 0.3) & (up > 0.9) & (np.linalg.norm(w_body, axis=1) < 1.0)
if not done.any():
    sys.exit("the policy does not complete the roll on the nominal robot; nothing written")
end = min(n, int(np.argmax(done)) + int(HOLD / dt) + 1)
rec = {k: v[:end] for k, v in rec.items()}
total = total[:end]
print(f"axis {math.degrees(math.atan2(axis[1], axis[0])):.0f} deg (0 = centreline, 90 = end "
      f"over end); turn {math.degrees(total[-1]):.0f} deg; standing after "
      f"{int(np.argmax(done)) * dt:.2f} s; recording {end} steps, {end * dt:.2f} s")
print(f"hardest hits: limbs {float(cmd.peak_limb):.0f} N, stock shell "
      f"{float(cmd.peak_shell):.0f} N, carapace impact {float(cmd.peak_carapace_impact):.0f} N, "
      f"floor impact {float(cmd.peak_ground_impact):.0f} N, longest flight "
      f"{float(cmd.longest_flight) * dt:.2f} s")
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, dt=np.float32(dt), joint_names=np.array(names), total=total.astype(np.float32),
         axis=axis.astype(np.float32), source=str(ACTOR.name), **rec)
print("wrote", OUT)
env.close()
