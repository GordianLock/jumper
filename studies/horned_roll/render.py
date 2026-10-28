"""Render a guide_search plan (copied from studies/roll_v2/render.py, 2026-10-28) from the nominal standing start, as an MP4.

Same model, physics and controls as the search (`roll_search.py`); a camera that
follows the body from the front-right, 30 fps, played at `--speed` (0.5 = half
speed). A label gives the plan and the roll's clock.

    MUJOCO_GL=egl python studies/horned_roll/render.py plans/plan_horned_s3.npy out.mp4
"""
import argparse
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.insert(0, HERE)
import guide_search as R  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan"); p.add_argument("out")
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)
    a = p.parse_args()
    import mediapy
    from PIL import Image, ImageDraw

    m = R.build(); d, names, home, lo, hi, drive, qadr = R.setup(m)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)
    plan = np.load(a.plan if os.path.isabs(a.plan) else HERE + a.plan)
    u = R.controls(plan[None], home, lo, hi, drive, (hi - lo) / 2)[0]
    st = R.settle_standing(m, d, home, qadr)
    mujoco.mj_setState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    r = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("robot/base_link").id
    cam.distance, cam.azimuth, cam.elevation = 0.9, 135.0, -15.0
    fps = 30
    every = max(1, round(a.speed / (fps * m.opt.timestep)))
    frames, ups, thetas = [], [], []
    name = os.path.splitext(os.path.basename(a.plan))[0]
    for i in range(len(u)):
        d.ctrl[:] = u[i]; mujoco.mj_step(m, d)
        w, x, y, z = d.qpos[3:7]
        ups.append(1 - 2 * (x * x + y * y)); thetas.append(np.arctan2(2 * (y * z + w * x), ups[-1]))
        if i % every == 0:
            r.update_scene(d, cam)
            img = Image.fromarray(r.render())
            turn = np.degrees(np.unwrap(np.array(thetas))[-1] - np.unwrap(np.array(thetas))[0])
            ImageDraw.Draw(img).text((10, 10), f"{name}   t {i * m.opt.timestep:4.2f} s   "
                                     f"turn {turn:4.0f} deg   ({a.speed:g}x speed)", fill=(255, 255, 255))
            frames.append(np.asarray(img))
    mediapy.write_video(a.out, frames, fps=fps)
    print("wrote", a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
