"""How far may a carapace overhang the stock body before the legs lose the floor?

Carapace zone for overhang e: everything above the stock shell top (z > TOP - 5 mm)
whose (x, y) lies within the body's top-view outline grown by e. A leg configuration
counts only if its foot is at or past the floor in a pose AND its joint-space path
from HOME keeps every leg collision point out of the zone. For each e: per leg, the
push stroke past the floor in each pose. The keep-out for e = 0 is meshed (STL).
"""
import sys
import numpy as np, mujoco, trimesh
from matplotlib.path import Path
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure
from shapely.geometry import Polygon

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # repo root
from tasks.jumper.common.assets import JUMPER_XML
from tasks.jumper.common import constants as C

LEGS = ["LF", "RF", "LM", "RM", "LR", "RR"]
TOP = 0.072
OVERHANG = [0.0, 0.01, 0.02, 0.03, 0.05]
POSES = {"flat 8cm": (180, 0, 0.08), "flat 10cm": (180, 0, 0.10),
         "tilt 15deg": (165, 0, 0.10), "horn 20deg": (180, 20, 0.10), "horn 30deg": (180, 30, 0.11)}
N_SAMPLES, INTERP, PTS = 30000, 8, 120
VOX, CLEAR = 0.004, 0.005
rng = np.random.default_rng(2)

m = mujoco.MjModel.from_xml_path(str(JUMPER_XML)); d = mujoco.MjData(m)
d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
for k, v in C.HOME.items(): d.qpos[m.jnt_qposadr[m.joint(k).id]] = v
home = d.qpos.copy(); mujoco.mj_forward(m, d); COM = d.subtree_com[1].copy()
leg_of = lambda b: next((L for L in LEGS if m.body(b).name.startswith(L)), None)
body_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] and leg_of(m.geom_bodyid[g]) is None}
BV = np.concatenate([m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
                     @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g in body_geoms])
hull = BV[ConvexHull(BV[:, :2]).vertices, :2]
ZONES = {e: Path(np.array(Polygon(hull).buffer(e).exterior.coords)) for e in OVERHANG}

def rot(roll, pitch):
    cr, sr = np.cos(roll), np.sin(roll); cp, sp = np.cos(-pitch), np.sin(-pitch)
    return np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
ROTS = {k: (rot(np.radians(r), np.radians(p)), h) for k, (r, p, h) in POSES.items()}

def surface(g, n):
    mid = m.geom_dataid[g]; a, k = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    tm = trimesh.Trimesh(m.mesh_vert[a:a+k], m.mesh_face[fa:fa+fn], process=False)
    return trimesh.sample.sample_surface(tm, n, seed=0)[0]

stroke = {e: {L: {p: 0.0 for p in POSES} for L in LEGS} for e in OVERHANG}
grid_lo = np.array([-0.40, -0.45, -0.30]); shape = np.full(3, int(np.ceil(0.9 / VOX)))
occ = np.zeros(shape, bool)
for L in LEGS:
    ids = [m.joint(f"{L}_J{i}_joint").id for i in range(4 if L in ("LF", "RF") else 3)]
    adr = [m.jnt_qposadr[i] for i in ids]; lo, hi = m.jnt_range[ids, 0], m.jnt_range[ids, 1]
    lg = [g for g in range(m.ngeom) if m.geom_contype[g] and leg_of(m.geom_bodyid[g]) == L]
    coarse = {g: surface(g, PTS) for g in lg}; fine = {g: surface(g, 600) for g in lg}
    sid = m.site(L).id; q0 = home[adr]; n_ok = {e: 0 for e in OVERHANG}
    for _ in range(N_SAMPLES):
        q = lo + rng.random(len(ids)) * (hi - lo)
        d.qpos[:] = home; d.qpos[adr] = q; mujoco.mj_kinematics(m, d)
        p = d.site_xpos[sid] - COM
        z = {k: h + (R @ p)[2] for k, (R, h) in ROTS.items()}
        if min(z.values()) > 0.01: continue
        mujoco.mj_collision(m, d)
        if any(((c.geom1 in coarse and c.geom2 in body_geoms) or (c.geom2 in coarse and c.geom1 in body_geoms))
               and c.dist < -0.001 for c in d.contact[:d.ncon]): continue
        pts = []
        for s in np.linspace(0, 1, INTERP):
            d.qpos[:] = home; d.qpos[adr] = q0 + s * (q - q0); mujoco.mj_kinematics(m, d)
            pts += [P @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, P in coarse.items()]
        pts = np.concatenate(pts); hi_pts = pts[pts[:, 2] > TOP - 0.005]
        clear_e = [e for e in OVERHANG if not ZONES[e].contains_points(hi_pts[:, :2]).any()]
        for e in clear_e:
            n_ok[e] += 1
            for k, zz in z.items(): stroke[e][L][k] = max(stroke[e][L][k], -zz)
        if 0.0 in clear_e and min(abs(v) for v in z.values()) < 0.01:
            for s in np.linspace(0, 1, INTERP * 2):
                d.qpos[:] = home; d.qpos[adr] = q0 + s * (q - q0); mujoco.mj_kinematics(m, d)
                for g, P in fine.items():
                    w = P @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                    ijk = np.floor((w - grid_lo) / VOX).astype(int)
                    ijk = ijk[np.all((ijk >= 0) & (ijk < shape), axis=1)]
                    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    print(L, "usable configurations by overhang:", n_ok, flush=True)

print("\npush stroke past the floor, cm (0 = the foot cannot reach the floor)")
for e in OVERHANG:
    print(f"\noverhang {e*100:.0f} cm")
    print("pose         " + " ".join(f"{L:>5}" for L in LEGS))
    for k in POSES:
        print(f"{k:<12} " + " ".join(f"{max(0, stroke[e][L][k])*100:5.1f}" for L in LEGS))
occ = ndimage.binary_closing(occ, iterations=2)
occ = ndimage.binary_dilation(occ, iterations=int(round(CLEAR / VOX)))
v, f, _, _ = measure.marching_cubes(occ.astype(np.float32), 0.5)
trimesh.Trimesh(v * VOX + grid_lo, f).export("leg_keepout_side.stl")
np.savez_compressed("leg_keepout_side_voxels.npz", occ=occ, origin=grid_lo, voxel=VOX, hull=hull)
import json; json.dump({str(e): stroke[e] for e in OVERHANG}, open("stroke_by_overhang.json", "w"))
print("written leg_keepout_side.stl")
