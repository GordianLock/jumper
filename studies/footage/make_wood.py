"""Generate the hardwood floor texture for footage (`render_footage.py --floor hardwood`).

A procedural oak plank floor, so no image of anyone else's is used: planks
`PLANK_W` wide running along the image's x, ends staggered, each its own shade, with
grain stretched along the plank and a dark seam between boards. The image covers
`TILE_M` metres and tiles seamlessly both ways (every plank wraps around the image's
width, and each row's ends fall where the row before's did not), so the renderer
repeats it once per `TILE_M` at true scale.

    python studies/footage/make_wood.py   # writes studies/footage/assets/hardwood.png
"""
from pathlib import Path

import numpy as np
from PIL import Image

TILE_M = 1.2            # metres the image covers
PX = 2048               # image size, square
PLANK_W = 0.10          # board width, m
SEAM_PX = 2
OAK = np.array([0.62, 0.43, 0.26])

rng = np.random.default_rng(7)
px_per_m = PX / TILE_M
rows = int(round(TILE_M / PLANK_W))
row_h = PX / rows
img = np.zeros((PX, PX, 3))
x = np.arange(PX)


def periodic_noise(n, octaves, rng):
    """1D noise that wraps around n samples: a sum of sines with whole periods."""
    out = np.zeros(n)
    for k in octaves:
        out += rng.normal() / k * np.sin(2 * np.pi * k * np.arange(n) / n + rng.uniform(0, 2 * np.pi))
    return out


for r in range(rows):
    y0, y1 = int(round(r * row_h)), int(round((r + 1) * row_h))
    h = y1 - y0
    # Board ends along this row: 1-2 boards per tile width, offset from the row above.
    n = rng.integers(1, 3)
    cuts = np.sort((rng.uniform(0, 1) + np.arange(n) / n + rng.uniform(-0.08, 0.08, n)) % 1.0)
    cuts_px = (cuts * PX).astype(int)
    seg = np.searchsorted(cuts_px, x, side="right") % n            # board index per column
    shade = 1.0 + rng.normal(0, 0.07, n)
    tint = rng.normal(0, 0.012, (n, 3))
    yy = np.arange(h)[:, None]
    # Grain: broad figure bands and fine lines along the board, wavering along it.
    waver = periodic_noise(PX, (1, 2, 3, 5, 8), rng) * 14.0
    t = yy + waver[None, :]
    broad = 0.5 + 0.5 * np.sin(2 * np.pi * t / rng.uniform(28, 45) + rng.uniform(0, 6.3))
    lines = (0.5 + 0.5 * np.sin(2 * np.pi * t / rng.uniform(5, 8) + rng.uniform(0, 6.3))) ** 6
    fine = periodic_noise(PX, range(40, 200, 7), rng)[None, :] * 0.03
    v = (shade[seg][None, :] * (0.86 + 0.14 * broad - 0.10 * lines) + fine)[..., None]
    col = (OAK[None, None, :] + tint[seg][None, :, :]) * v
    # Seams: the long edge at the top of the row, and each board end.
    col[:SEAM_PX] *= 0.45
    for c in cuts_px:
        col[:, max(0, c - 1):c + 1] *= 0.5
    img[y0:y1] = col

out = Path(__file__).resolve().parent / "assets" / "hardwood.png"
out.parent.mkdir(exist_ok=True)
Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out)
print("wrote", out)
