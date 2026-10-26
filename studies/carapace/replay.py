"""Replay a righting plan from the nominal belly-up start: floor contact forces by part
group, the floor's torque on each joint, and an MP4.

usage: replay.py VARIANT PLAN.npy OUT.mp4
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
import righting as R
from tasks.jumper.common import constants as C

variant, plan, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = R.build(variant); d, names, home, lo, hi, drive, qadr = R.setup(m)
st0 = R.settle(m, d, home, qadr); T0 = st0[0]; how = None
u = R.controls(np.load(plan)[None], home, lo, hi, drive, (hi - lo) / 2)[0]
floor = m.geom("terrain").id
FEET = {m.geom("robot/" + g).id for g in C.FOOT_GEOMS}
def group(g):
    n = m.geom(g).name.split("/")[-1]
    if g in FEET: return "feet"
    if n.startswith("horn"): return "horn"
    if n.startswith(("base_link", "upper_shell")): return "shell"
    if n.startswith(("display", "tof", "camera")): return "sensors"
    return "limbs"
leg_dofs = [m.jnt_dofadr[m.joint("robot/" + n).id] for n in names]
mujoco.mj_setState(m, d, st0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
peak = {k: 0.0 for k in ["feet", "limbs", "shell", "sensors", "horn"]}
jt_peak = np.zeros(len(names)); air = 0; up_t = None
renderer = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 0.9, -15.0, 60.0
frames = []
f6 = np.zeros(6); jac = np.zeros((3, m.nv))
for i in range(len(u) + 100):
    d.ctrl[:] = u[min(i, len(u) - 1)]
    mujoco.mj_step(m, d)
    per = {k: 0.0 for k in peak}; tau = np.zeros(m.nv); touching = False
    for c in range(d.ncon):
        con = d.contact[c]
        if floor not in (con.geom1, con.geom2): continue
        g = con.geom2 if con.geom1 == floor else con.geom1
        mujoco.mj_contactForce(m, d, c, f6)
        fw = con.frame.reshape(3, 3).T @ f6[:3]          # force on geom2, world
        if con.geom1 != floor: fw = -fw                   # make it the force on the robot
        per[group(g)] += f6[0]; touching = True
        mujoco.mj_jac(m, d, jac, None, con.pos, m.geom_bodyid[g])
        tau += jac.T @ fw
    for k in peak: peak[k] = max(peak[k], per[k])
    jt_peak = np.maximum(jt_peak, np.abs(tau[leg_dofs]))
    if not touching: air += 1
    up, _ = R.pose_of(d.qpos)
    if up_t is None and up > 0.9: up_t = d.time - T0
    if how is None and up > 0:
        w, x, y, z = d.qpos[3:7]; fz = 2 * (x * z - w * y); sz = 2 * (y * z + w * x)
        how = "end over end (head %s)" % ("first" if fz < 0 else "last") if abs(fz) > abs(sz) else "sideways (%s side down)" % ("left" if sz < 0 else "right")
    if i % 7 == 0:                                        # ~28.6 fps
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.06]
        renderer.update_scene(d, camera=cam); frames.append(renderer.render())
import mediapy
mediapy.write_video(out, frames, fps=1 / (7 * m.opt.timestep))
j = int(np.argmax(jt_peak))
print(f"{variant} {os.path.basename(plan)}: flips {how}; upright at {up_t:.2f}s, final up {R.pose_of(d.qpos)[0]:.2f}; "
      f"peak floor force by part (N): " + ", ".join(f"{k} {v:.0f}" for k, v in peak.items())
      + f"; airborne {air * m.opt.timestep:.2f}s; worst floor torque on a joint "
      f"{jt_peak[j]:.2f} N*m ({names[j]})")
