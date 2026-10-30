"""Record a trained jumper.self_right policy getting up, on the nominal stock robot, in
the demonstration format `studies/proto3/render_recording.py` draws (root pose and
joint angles per 20 ms control step, plus the axis it turned about and how far).

    python studies/self_right/record_self_right.py <actor.onnx> <out.npz> [--handover]

With `--handover`, the policy hands over to a stand controller the moment the task's
own `standing` termination fires (the point training stopped at; nothing after it was
trained): the joint targets ramp linearly from where the joints are to HOME over
`RAMP` s, then stay at HOME -- what the robot's stand mode would do after a
self-right. The recording runs until the joints are within `SETTLED` rad of HOME and
the body is still, then `HOLD` s more. Physics is simulated throughout.

One episode from the task's own on-the-back spawn (play mode, no randomisation, no
pushes), through the 0.5 s settling and the righting, then `HOLD` s standing.
`total` is the tilt from upright, unwrapped, 0 at the start; `axis` is the horizontal
body axis it turned about while between belly-up and upright.
"""
import math
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

use_backend(resolve("native", "cpu", num_envs=1, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402

ACTOR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
HANDOVER = "--handover" in sys.argv[3:]
HOLD = 0.4
RAMP = 0.5
SETTLED = 0.05
cfg = tasks.load_env_cfg("jumper.self_right", asset=ROOT / "assets/jumper/jumper.xml", play=True)
cfg.scene.num_envs = 1
cfg.events.pop("push_robot", None)
for k in [k for k, e in cfg.events.items() if e.mode == "startup"]:
    cfg.events.pop(k)
for k in ("standing", "settled", "settle_timeout"):
    cfg.terminations.pop(k, None)
stand_func = __import__("tasks.jumper.self_right.mdp.rewards", fromlist=["standing"]).standing
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
names = [robot.joint_names[i] for i in term._target_ids]
assert tuple(names) == tuple(GAIT_JOINTS), names
sess = ort.InferenceSession(str(ACTOR))
iname = sess.get_inputs()[0].name
obs, _ = env.reset()
origin = env.scene.env_origins[0].numpy()
rec = {k: [] for k in ("q", "root_pos", "root_quat")}
wb, stood = [], None
dt = env.step_dt
home = robot.data.default_joint_pos[0, term._target_ids].numpy()
scale = term._scale[0].numpy() if torch.is_tensor(term._scale) else float(term._scale)
handed, start, still = None, None, None
for k in range(int(6.0 / dt)):
    if handed is None:
        a = sess.run(None, {iname: obs["actor"].numpy().astype(np.float32)})[0]
    else:
        f = min(1.0, (k - handed) * dt / RAMP)
        a = ((1.0 - f) * (start - home) / scale)[None].astype(np.float32)
    obs, *_ = env.step(torch.as_tensor(a))
    if HANDOVER and handed is None and bool(stand_func(env, command_name="roll")[0]):
        handed, start = k + 1, robot.data.joint_pos[0, term._target_ids].numpy().copy()
        print(f"handover to the stand controller at {handed * dt:.2f} s")
    rec["q"].append(robot.data.joint_pos[0, term._target_ids].numpy().copy())
    rec["root_pos"].append(robot.data.root_link_pos_w[0].numpy() - origin)
    rec["root_quat"].append(robot.data.root_link_quat_w[0].numpy().copy())
    wb.append(robot.data.root_link_ang_vel_b[0].numpy().copy())
    up = float(-robot.data.projected_gravity_b[0, 2])
    w = float(np.linalg.norm(wb[-1]))
    if stood is None and k * dt > 0.5 and up > math.cos(math.radians(20)) and w < 1.0:
        stood = k
    if not HANDOVER:
        if stood is not None and k >= stood + int(HOLD / dt):
            break
        continue
    if handed is not None and still is None and (k - handed) * dt >= RAMP:
        qe = np.abs(robot.data.joint_pos[0, term._target_ids].numpy() - home).max()
        vb = float(np.linalg.norm(robot.data.root_link_lin_vel_w[0].numpy()))
        if qe < SETTLED and w < 0.1 and vb < 0.01:
            still = k
            print(f"at HOME and still at {still * dt:.2f} s (joints within {qe:.3f} rad)")
    if still is not None and k >= still + int(HOLD / dt):
        break
rec = {k: np.array(v, dtype=np.float32) for k, v in rec.items()}
n = len(rec["q"])
g = quat_apply_inverse(torch.as_tensor(rec["root_quat"]),
                       torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3)).numpy()
upv = -g[:, 2]
wb = np.array(wb)
mid = (upv > -0.9) & (upv < 0.9)
th = (wb * mid[:, None]).sum(0) * dt
axis = np.array([th[0], th[1], 0.0]) / max(1e-6, math.hypot(th[0], th[1]))
tilt = np.arccos(np.clip(upv, -1, 1))
total = (tilt[0] - tilt).astype(np.float32)
print(f"stood after {stood * dt:.2f} s (0.5 s settling)" if stood is not None else "did not stand",
      f"| axis {math.degrees(math.atan2(axis[1], axis[0])):.0f} deg | {n} steps")
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, dt=np.float32(dt), joint_names=np.array(names), total=total,
         axis=axis.astype(np.float32), source=ACTOR.name, **rec)
if HANDOVER and still is None:
    print("never settled at HOME within 6 s")
print("wrote", OUT)
