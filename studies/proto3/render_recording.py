"""Render a recorded roll (a demonstration .npz: root pose and joint angles per 20 ms
control step, as `record_demo.py` / `record_policy_demo.py` write them) as an MP4.

Nothing is re-simulated: each frame poses prototype 3 at the recorded state (the
unactuated J4 joints at 0) and draws it, so what is shown is exactly what the
training environment produced. Two views side by side -- across the roll and the roll's
own side (looking along the roll axis) -- with the recording's clock and turn.

    MUJOCO_GL=egl python studies/proto3/render_recording.py <recording.npz> <out.mp4> [--speed 0.5]
"""
import argparse
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.insert(0, HERE)
import search as S  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("rec")
p.add_argument("out")
p.add_argument("--speed", type=float, default=0.5)
p.add_argument("--fps", type=int, default=30)
p.add_argument("--label", default="turn", help="the word before the angle in the caption")
p.add_argument("--model", default=None,
               help="carapace model name (e.g. CF1-tri30m-rb-v4); default: the search scene's")
a = p.parse_args()
import mediapy  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

h = np.load(a.rec)
names = [str(n) for n in h["joint_names"]]
if a.model:
    spec = mujoco.MjSpec.from_file(os.path.join(HERE, "..", "..", "assets", "jumper",
                                                "jumper.xml" if a.model == "jumper"
                                                else f"jumper_{a.model}.xml"))
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1],
                            rgba=[0.25, 0.35, 0.5, 1])
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    m = spec.compile()
else:
    m = S.build()
W, H = 480, 400
m.vis.global_.offwidth = max(m.vis.global_.offwidth, 2 * W)
m.vis.global_.offheight = max(m.vis.global_.offheight, H)
d = mujoco.MjData(m)
pre = "robot/" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + names[0]) >= 0 else ""
adr = [m.jnt_qposadr[m.joint(pre + n).id] for n in names]
root = m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]]
dt = float(h["dt"])
T = len(h["q"])
r = mujoco.Renderer(m, H, W)
axis = np.asarray(h["axis"], float)
yaw_axis = math.degrees(math.atan2(axis[1], axis[0]))
frames = []
n_out = int(T * dt / a.speed * a.fps)
for i in range(n_out):
    t = i * a.speed / a.fps
    k = min(T - 1, int(t / dt))
    d.qpos[:] = 0
    d.qpos[root:root + 3] = h["root_pos"][k]
    d.qpos[root + 3:root + 7] = h["root_quat"][k]
    d.qpos[adr] = h["q"][k]
    mujoco.mj_forward(m, d)
    views = []
    for az in (yaw_axis + 90.0, yaw_axis + 180.0):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [d.qpos[root], d.qpos[root + 1], 0.08]
        cam.distance, cam.azimuth, cam.elevation = 0.9, az, -18.0
        r.update_scene(d, cam)
        views.append(r.render())
    img = Image.fromarray(np.hstack(views))
    dr = ImageDraw.Draw(img)
    turn = math.degrees(float(h["total"][k]))
    dr.text((10, 8), f"t = {k * dt:4.2f} s   {a.label} {turn:4.0f} deg   ({a.speed:g}x speed)",
            fill=(255, 255, 255))
    dr.text((10, H - 20), "across the roll axis", fill=(255, 255, 255))
    dr.text((W + 10, H - 20), "along the roll axis", fill=(255, 255, 255))
    frames.append(np.asarray(img))
mediapy.write_video(a.out, frames, fps=a.fps)
print(f"wrote {a.out}: {len(frames)} frames, {T * dt:.2f} s recorded")
