"""Render every search's best plan for one roll type as a labelled grid video.

usage: grid.py ROLL OUT.mp4     ROLL in {flip, side, diag}
Rows are shells, columns are the three searches (seeds 1-3). Each tile is the plan
replayed from the nominal belly-up start, labelled with its 32-start success rate.
"""
import os, re, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, mediapy
from PIL import Image, ImageDraw, ImageFont
import righting as R

HERE = R.HERE
roll, out = sys.argv[1], sys.argv[2]
SHELLS = os.environ.get("SHELLS", "horn20,horn30,tri20s,tri20m").split(",")
NAMES = {"horn20": "20 deg horn", "horn30": "30 deg horn",
         "tri20s": "20 deg + 2.5 cm corner horns", "tri20m": "20 deg + 3.5 cm corner horns",
         "rt20": "front horn + rear tail", "rc20": "front horn + rear corners",
         "rct20": "front + rear corners + tail",
         "tri20s_rc": "3 front horns + rear corners", "tri20s_rt": "3 front horns + tail",
         "tri20s_rct": "3 front + rear corners + tail"}
W, H, FPS_STEP, SECONDS = 320, 240, 10, 3.2

rates = {}
for log in ["righting3.log", "righting_axes2.log", "righting_tri.log", "righting_rear.log",
            "righting_rear2.log", "righting_tri20s_rear.log"]:
    seen = {}
    if not os.path.exists(HERE + "logs/" + log):
        continue
    for line in open(HERE + "logs/" + log):
        mt = re.match(r"\s+(\S+): best plan rights (\d+)/32", line)
        if mt:
            tag = mt.group(1); seen[tag] = seen.get(tag, 0) + 1
            rates[(tag, seen[tag])] = int(mt.group(2))

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
except OSError:
    font = ImageFont.load_default()


def clip(shell, seed):
    tag = shell if roll == "flip" else f"{shell}_{roll}"
    path = HERE + f"plans/plan_{tag}_s{seed}.npy"
    m = R.build(shell); d, names, home, lo, hi, drive, qadr = R.setup(m)
    st0 = R.settle(m, d, home, qadr)
    u = R.controls(np.load(path)[None], home, lo, hi, drive, (hi - lo) / 2)[0]
    mujoco.mj_setState(m, d, st0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ren = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 0.9, -15.0, 60.0
    frames = []
    n = int(SECONDS / m.opt.timestep)
    r = rates.get((tag, seed))
    label = f"{NAMES[shell]}  |  search {seed}: " + (f"{r}/32 starts" if r is not None else "?")
    for i in range(n):
        d.ctrl[:] = u[min(i, len(u) - 1)]
        mujoco.mj_step(m, d)
        if i % FPS_STEP == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.06]
            ren.update_scene(d, camera=cam)
            img = Image.fromarray(ren.render()); dr = ImageDraw.Draw(img)
            ok = r is not None and r >= 30
            dr.rectangle([0, 0, W, 22], fill=(0, 110, 0) if ok else (150, 30, 30))
            dr.text((6, 3), label, fill=(255, 255, 255), font=font)
            frames.append(np.asarray(img))
    ren.close()
    return frames


rows = []
for shell in SHELLS:
    tiles = [clip(shell, s) for s in (1, 2, 3)]
    rows.append([np.concatenate(fr, 1) for fr in zip(*tiles)])
    print(shell, "rendered", flush=True)
video = [np.concatenate(fr, 0) for fr in zip(*rows)]
mediapy.write_video(out, video, fps=1 / (FPS_STEP * 0.005))
print("wrote", out, len(video), "frames")
