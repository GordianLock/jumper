"""Is a capsule horn clear of the legs' floor-reaching sweep?

usage: python clearance.py  x0 y0 z0  x1 y1 z1  radius
Coordinates in metres, base_link frame (+x head, +y left, +z up; shell top z = 0.072).
Prints the share of sampled points on the capsule that fall inside the keep-out
(leg_keepout_side_voxels.npz, 5 mm clearance included). 0.00 means clear.
Mirror a left-side horn (y > 0) to y < 0 for the other side; the sweep is near-symmetric
but check both.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
z = np.load(HERE + "leg_keepout_side_voxels.npz")
OCC, ORIGIN, VOX = z["occ"], z["origin"], float(z["voxel"])


def overlap(base, tip, radius):
    a, b = np.asarray(base, float), np.asarray(tip, float)
    d = (b - a) / np.linalg.norm(b - a)
    u = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    pts = []
    for t in np.linspace(0, 1, 30):
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for rr in (0, radius * 0.5, radius):
                pts.append(a + t * (b - a) + rr * (np.cos(ang) * u + np.sin(ang) * w))
    for end, sgn in ((b, 1), (a, -1)):                      # hemispherical caps
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for el in np.linspace(0, np.pi / 2, 5):
                pts.append(end + radius * (np.cos(el) * (np.cos(ang) * u + np.sin(ang) * w)
                                           + sgn * np.sin(el) * d))
    p = np.array(pts)
    ijk = np.floor((p - ORIGIN) / VOX).astype(int)
    ok = np.all((ijk >= 0) & (ijk < OCC.shape), axis=1)
    return OCC[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]].sum() / len(p)


if __name__ == "__main__":
    v = [float(x) for x in sys.argv[1:8]]
    print(f"inside keep-out: {overlap(v[0:3], v[3:6], v[6]):.2f}")
