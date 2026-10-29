"""Render a proto3 search plan (adapted from studies/roll_v2/render.py) from the nominal standing start, as an MP4.

Same model, physics and controls as the search (`roll_search.py`); a camera that
follows the body from the front-right, 30 fps, played at `--speed` (0.5 = half
speed). A label gives the plan and the roll's clock.

    MUJOCO_GL=egl python studies/roll_v2/render.py plans/plan_v2_roll_s3.npy out.mp4
"""
import argparse
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.insert(0, HERE)
import search as R  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan"); p.add_argument("out")
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)
    p.add_argument("--start", default="stand")
    a = p.parse_args()
    import mediapy
    from PIL import Image, ImageDraw

    m = R.build(); d, names, home, lo, hi, drive, qadr = R.setup(m)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)
    plan = np.load(a.plan if os.path.isabs(a.plan) else HERE + a.plan)
    u = R.controls(plan[None], home, lo, hi, drive, (hi - lo) / 2)[0]
    st = R.settle(m, d, home, qadr, a.start)
    mujoco.mj_setState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    r = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("robot/base_link").id
    cam.distance, cam.azimuth, cam.elevation = 0.9, 135.0, -15.0
    fps = 30
    every = max(1, round(a.speed / (fps * m.opt.timestep)))
    frames, states, shots = [], [], []
    nst = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    name = os.path.splitext(os.path.basename(a.plan))[0]
    for i in range(len(u)):
        d.ctrl[:] = u[i]; mujoco.mj_step(m, d)
        x = np.zeros(nst); mujoco.mj_getState(m, d, x, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        states.append(x)
        if i % every == 0:
            r.update_scene(d, cam); shots.append((i, r.render()))
    S = np.array(states)[None]
    _, ok, final, _, up, axes = R.judge(S, m, R.AXES["any"], R.NEED[a.start])
    cum = R.turn(S, m, axes)[0]
    cum = cum * (1 if cum.max() >= -cum.min() else -1)
    print(f"axis {R.axis_deg(axes)[0]:.0f} deg, final turn {np.degrees(final[0]):.0f} deg, success {ok[0]}")
    for i, img in shots:
        img = Image.fromarray(img)
        ImageDraw.Draw(img).text((10, 10), f"{name}   t {i * m.opt.timestep:4.2f} s   turn "
                                 f"{np.degrees(cum[i]):4.0f} deg   ({a.speed:g}x speed)", fill=(255, 255, 255))
        frames.append(np.asarray(img))
    mediapy.write_video(a.out, frames, fps=fps)
    print("wrote", a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
