"""Belly-up pose study: for a held body pose (roll, pitch, COM height), which legs
reach the floor, how far past it they can push (spring stroke), and how much
righting moment they can make within servo torque limits.

The body is held by an imaginary carapace; poses that would put any part of the
stock body below the floor are infeasible (a carapace can only add material).
"""
import itertools, json, sys
from multiprocessing import Pool
import numpy as np, mujoco
from scipy.optimize import linprog

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # repo root
from tasks.jumper.common.assets import JUMPER_XML
from tasks.jumper.common import constants as C

LEGS = ["LF", "RF", "LM", "RM", "LR", "RR"]
TAU = 1.2      # continuous servo torque, N*m
MU = 0.6       # floor friction, conservative
W = 2.543 * 9.81
N_SAMPLES = 40000
N_LP = 30
F_CAP = 50.0   # per foot, twice the robot weight
rng = np.random.default_rng(0)

m = mujoco.MjModel.from_xml_path(str(JUMPER_XML)); d = mujoco.MjData(m)
d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
for k, v in C.HOME.items(): d.qpos[m.jnt_qposadr[m.joint(k).id]] = v
mujoco.mj_forward(m, d)
COM = d.subtree_com[1].copy()
home_q = d.qpos.copy()

def leg_of(body):
    n = m.body(body).name
    return next((L for L in LEGS if n.startswith(L)), None)

body_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and leg_of(m.geom_bodyid[g]) is None]
BODY_V = []
for g in body_geoms:
    mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    BODY_V.append(m.mesh_vert[a:a+n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
BODY_V = np.concatenate(BODY_V)

def sample_leg(L):
    joints = [f"{L}_J{i}_joint" for i in range(4 if L in ("LF", "RF") else 3)]
    ids = [m.joint(j).id for j in joints]
    lo = m.jnt_range[ids, 0]; hi = m.jnt_range[ids, 1]
    dofs = [m.jnt_dofadr[i] for i in ids]
    leg_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] and leg_of(m.geom_bodyid[g]) == L}
    sid = m.site(L).id
    P, J, ok = [], [], []
    jp = np.zeros((3, m.nv))
    for _ in range(N_SAMPLES):
        q = lo + rng.random(len(ids)) * (hi - lo)
        d.qpos[:] = home_q
        for i, v in zip(ids, q): d.qpos[m.jnt_qposadr[i]] = v
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        hit = False
        for c in d.contact[:d.ncon]:
            a, b = c.geom1, c.geom2
            if (a in leg_geoms and b in body_geoms) or (b in leg_geoms and a in body_geoms):
                if c.dist < -0.001: hit = True; break
        mujoco.mj_jacSite(m, d, jp, None, sid)
        P.append(d.site_xpos[sid].copy()); J.append(jp[:, dofs].copy()); ok.append(not hit)
    P, J, ok = np.array(P), np.array(J), np.array(ok)
    return P[ok], J[ok], ok.mean()

def rot(roll, pitch):
    # roll about body x, then tilt about world y; pitch > 0 lifts the head (+x)
    cr, sr = np.cos(roll), np.sin(roll); cp, sp = np.cos(-pitch), np.sin(-pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx

def best(obj, Jw, R_lin=None):
    # max obj.f  s.t. f_z>=0, |f_x|,|f_y|<=MU f_z, |Jw^T f|<=TAU
    A = [[1, 0, -MU], [-1, 0, -MU], [0, 1, -MU], [0, -1, -MU]]
    b = [0, 0, 0, 0]
    for row in Jw.T: A += [list(row), list(-row)]; b += [TAU, TAU]
    r = linprog(-obj, A_ub=A, b_ub=b, bounds=[(-F_CAP, F_CAP)] * 2 + [(0, F_CAP)], method="highs")
    return -r.fun if r.status == 0 else 0.0

def evaluate(args):
    roll_d, pitch_d, h = args
    R = rot(np.radians(roll_d), np.radians(pitch_d))
    c = np.array([0, 0, h])
    bz = (c + (BODY_V - COM) @ R.T)[:, 2]
    out = {"roll": roll_d, "pitch": pitch_d, "h": h, "clear": float(bz.min())}
    if bz.min() < 0: out["feasible"] = False; return out
    out["feasible"] = True
    head = (c + (np.array([0.094, 0, 0.02]) - COM) @ R.T)[2]
    tail = (c + (np.array([-0.112, 0, 0.02]) - COM) @ R.T)[2]
    out["head_up"] = float(head - tail)
    ax_roll = R @ np.array([1, 0, 0]); ax_roll[2] = 0; ax_roll /= np.linalg.norm(ax_roll)
    ax_pitch = np.cross([0, 0, 1], ax_roll)
    piv = np.array([c[0], c[1], 0])
    legs = {}
    for L in LEGS:
        P, J = SAMPLES[L]
        z = h + ((P - COM) @ R.T)[:, 2]
        stroke = float(max(0.0, -z.min()))
        near = np.where(np.abs(z) < 0.004)[0]
        rec = {"stroke": stroke, "reach": bool(z.min() <= 0.002)}
        if len(near):
            pick = rng.choice(near, min(N_LP, len(near)), replace=False)
            vals = {"roll+": 0, "roll-": 0, "pitch+": 0, "pitch-": 0, "lift": 0}
            for i in pick:
                pw = c + R @ (P[i] - COM); rr = pw - piv; Jw = R @ J[i]
                # moment (rr x f).ax = f.(ax x rr)
                for name, ax, s in [("roll+", ax_roll, 1), ("roll-", ax_roll, -1),
                                    ("pitch+", ax_pitch, 1), ("pitch-", ax_pitch, -1)]:
                    vals[name] = max(vals[name], best(s * np.cross(ax, rr), Jw))
                vals["lift"] = max(vals["lift"], best(np.array([0, 0, 1.0]), Jw))
            rec.update(vals)
        legs[L] = rec
    out["legs"] = legs
    return out

def init(s):
    global SAMPLES; SAMPLES = s

if __name__ == "__main__":
    S = {}
    for L in LEGS:
        P, J, frac = sample_leg(L); S[L] = (P, J); print(L, "free of body", round(frac, 2), flush=True)
    grid = list(itertools.product(range(120, 241, 10), range(-40, 41, 10),
                                  [round(0.07 + 0.01 * i, 2) for i in range(10)]))
    with Pool(initializer=init, initargs=(S,)) as p:
        res = p.map(evaluate, grid, chunksize=4)
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"))
    print("done", len(res))
