"""Peak floor-contact and joint-stop torques per joint for KingKong's own shipped example
policies, measured exactly as loads.py measures the rolls: rollouts in each task's
environment (native backend), states recorded each control step, re-simulated in plain
MuJoCo at the task's physics step with the PD re-applied per substep.

Run from the repository root:  python studies/carapace/baseline_loads.py
Produced the first session's baselines (8 envs, 250 control steps, pushes and observation
noise off): jump median 4.8 N*m, worst 5.6, mostly on J1; walking about 2.0; dance about
1.2. The figure is each env's peak floor-contact torque on any joint, median over envs;
joint-stop torque is reported separately and not included in it."""
import sys

import mujoco
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, ".")
import tasks
from mjrl.backend.resolve import resolve
from mjrl.backend.select import use_backend

N = 8
use_backend(resolve("native", "cpu", num_envs=N, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

CASES = [
    ("KingKong jump (ref_free_jump example)", "jumper.ref_free_jump", 250, True),
    ("KingKong walking (posture example)", "jumper.posture", 250, False),
    ("KingKong dance (dance example)", "jumper.dance", 250, True),
]
TAU = 1.7464


def run(label, task, steps, play):
    onnx = f"tasks/jumper/{task.split('.')[1]}/out/example/actor.onnx"
    cfg = tasks.load_env_cfg(task, play=play)
    cfg.scene.num_envs = N
    cfg.events.pop("push_robot", None)
    for g in cfg.observations.values():
        g.enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _ = env.reset()
    sess = ort.InferenceSession(onnx)
    name = sess.get_inputs()[0].name
    want = sess.get_inputs()[0].shape[1]
    if obs["actor"].shape[1] != want:
        print(f"{label}: skipped, policy expects {want} inputs, task now gives {obs['actor'].shape[1]}")
        env.close()
        return
    robot = env.scene["robot"]
    qadr = robot.indexing.joint_q_adr.numpy()
    vadr = robot.indexing.joint_v_adr.numpy()
    kp = float(robot.actuators[0].cfg.stiffness) if hasattr(robot, "actuators") else 10.0
    kd = 0.5
    rec = []
    for t in range(steps):
        qpos = env.sim.data.qpos.clone().numpy()
        qvel = env.sim.data.qvel.clone().numpy()
        o = obs["actor"].numpy().astype(np.float32)
        a = torch.as_tensor(np.concatenate([sess.run(None, {name: o[i:i + 1]})[0] for i in range(N)]))
        obs, _, term, to, _ = env.step(a)
        rec.append((qpos, qvel, robot.data.joint_pos_target.clone().numpy(), (term | to).numpy()))
    m = env.sim.mj_model
    dt_sub = float(m.opt.timestep)
    dec = int(cfg.decimation)
    d = mujoco.MjData(m)
    peak_c = np.zeros((N, len(qadr)))
    peak_l = np.zeros((N, len(qadr)))
    contact_types = [mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
                     mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC,
                     mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS]
    for i in range(N):
        for qpos, qvel, tgt, _ in rec:
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]
            for _ in range(dec):
                tau = np.clip(kp * (tgt[i] - d.qpos[qadr]) - kd * d.qvel[vadr], -TAU, TAU)
                d.qfrc_applied[:] = 0
                d.qfrc_applied[vadr] = tau
                mujoco.mj_step(m, d)
                if d.nefc == 0:
                    continue
                J = d.efc_J.reshape(d.nefc, m.nv)
                f = d.efc_force[:d.nefc]
                ty = d.efc_type[:d.nefc]
                c = np.isin(ty, contact_types)
                lim = ty == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
                peak_c[i] = np.maximum(peak_c[i], np.abs((J[c].T @ f[c])[vadr]))
                peak_l[i] = np.maximum(peak_l[i], np.abs((J[lim].T @ f[lim])[vadr]))
    names = list(robot.joint_names)
    jw = np.median(peak_c, axis=0)
    top = np.argsort(-jw)[:4]
    print(f"{label}: kp {kp:g}, physics {dt_sub*1000:g} ms x {dec}, {steps} control steps")
    print(f"   peak floor-contact torque on any joint: median {np.median(peak_c.max(axis=1)):.1f} N*m, "
          f"worst {peak_c.max():.1f}; joints over 1.75: median {np.median((peak_c > TAU).sum(axis=1)):.0f} of {len(names)}")
    print(f"   peak joint-stop torque: median {np.median(peak_l.max(axis=1)):.1f}, worst {peak_l.max():.1f}")
    print("   most loaded:", ", ".join(f"{names[j].replace('_joint','')} {jw[j]:.1f}" for j in top))
    env.close()


for case in CASES:
    try:
        run(*case)
    except Exception as e:  # report and carry on to the next baseline
        print(f"{case[0]}: failed -- {type(e).__name__}: {e}")
