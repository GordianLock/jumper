"""Leg keep-out volume for a carapace: the space the legs sweep through when they
reach the floor from a belly-up pose, in the robot's base_link frame (metres).

For each leg: sample joint configurations, keep those whose foot is at the floor
(within 1 cm) in any of the candidate belly-up poses and that do not cut into the
stock body, then sweep each one from HOME in joint space. Every collision-mesh
surface point of the leg along the way is voxelised (4 mm), closed, dilated by a
clearance margin and meshed.
"""
import sys
import numpy as np, mujoco, trimesh
from scipy import ndimage
from skimage import measure

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # repo root
from tasks.jumper.common.assets import JUMPER_XML
from tasks.jumper.common import constants as C

OUT = sys.argv[1]
LEGS = ["LF", "RF", "LM", "RM", "LR", "RR"]
VOX, CLEAR = 0.004, 0.005
N_SAMPLES, KEEP, INTERP = 30000, 400, 12
# (roll deg, pitch deg, COM height m): flat on the back, slight tilts, horn 20/30
POSES = [(180, 0, 0.08), (180, 0, 0.10), (165, 0, 0.10), (195, 0, 0.10),
         (180, 20, 0.09), (180, 20, 0.11), (180, 30, 0.10), (180, 30, 0.12)]
rng = np.random.default_rng(1)

m = mujoco.MjModel.from_xml_path(str(JUMPER_XML)); d = mujoco.MjData(m)
d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
for k, v in C.HOME.items(): d.qpos[m.jnt_qposadr[m.joint(k).id]] = v
home = d.qpos.copy(); mujoco.mj_forward(m, d); COM = d.subtree_com[1].copy()

def leg_of(b):
    n = m.body(b).name
    return next((L for L in LEGS if n.startswith(L)), None)

body_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] and leg_of(m.geom_bodyid[g]) is None}

def surface(g):
    mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    tm = trimesh.Trimesh(m.mesh_vert[a:a+n], m.mesh_face[fa:fa+fn], process=False)
    pts, _ = trimesh.sample.sample_surface(tm, 600, seed=0)
    return np.vstack([tm.vertices, pts])

def rot(roll, pitch):
    cr, sr = np.cos(roll), np.sin(roll); cp, sp = np.cos(-pitch), np.sin(-pitch)
    return np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

ROTS = [(rot(np.radians(r), np.radians(p)), h) for r, p, h in POSES]
grid_lo = np.array([-0.40, -0.45, -0.30]); shape = np.ceil(0.9 / VOX).astype(int) * np.ones(3, int)
occ = np.zeros(shape, bool)
per_leg = {}
for L in LEGS:
    joints = [f"{L}_J{i}_joint" for i in range(4 if L in ("LF", "RF") else 3)]
    ids = [m.joint(j).id for j in joints]; adr = [m.jnt_qposadr[i] for i in ids]
    lo, hi = m.jnt_range[ids, 0], m.jnt_range[ids, 1]
    lg = [g for g in range(m.ngeom) if m.geom_contype[g] and leg_of(m.geom_bodyid[g]) == L]
    surf = {g: surface(g) for g in lg}
    sid = m.site(L).id
    good = []
    for _ in range(N_SAMPLES):
        q = lo + rng.random(len(ids)) * (hi - lo)
        d.qpos[:] = home; d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[sid] - COM
        if not any(abs(h + (R @ p)[2]) < 0.01 for R, h in ROTS): continue
        mujoco.mj_collision(m, d)
        if any(((c.geom1 in surf and c.geom2 in body_geoms) or (c.geom2 in surf and c.geom1 in body_geoms))
               and c.dist < -0.001 for c in d.contact[:d.ncon]): continue
        good.append(q)
    per_leg[L] = len(good)
    good = np.array(good)[rng.permutation(len(good))[:KEEP]]
    q0 = home[adr]
    for q in good:
        for s in np.linspace(0, 1, INTERP):
            d.qpos[:] = home; d.qpos[adr] = q0 + s * (q - q0)
            mujoco.mj_kinematics(m, d)
            for g, P in surf.items():
                w = P @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                ijk = np.floor((w - grid_lo) / VOX).astype(int)
                ok = np.all((ijk >= 0) & (ijk < shape), axis=1); ijk = ijk[ok]
                occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    print(L, "floor-reaching configurations:", per_leg[L], flush=True)

occ = ndimage.binary_closing(occ, iterations=2)
occ = ndimage.binary_fill_holes(occ)
occ = ndimage.binary_dilation(occ, iterations=int(round(CLEAR / VOX)))
v, f, _, _ = measure.marching_cubes(occ.astype(np.float32), 0.5)
mesh = trimesh.Trimesh(v * VOX + grid_lo, f)
mesh.export(OUT)
np.savez_compressed(OUT.replace(".stl", "_voxels.npz"), occ=occ, origin=grid_lo, voxel=VOX)
print("keep-out mesh", OUT, "faces", len(f), "bounds", np.round(mesh.bounds, 3))
