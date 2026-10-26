"""Loads of the working plans over 32 perturbed belly-up starts (no rendering).

Per start: did it right itself, worst floor-contact torque on any joint, and the peak
floor force on limbs / shell / horns (sum over the group's contacts per step).
Perturbations as in righting.search: tilt +-8 deg, joint jitter +-0.08 rad, seed 7.
"""
import sys
import numpy as np, mujoco
import righting as R
from tasks.jumper.common import constants as C

PLANS = [("horn20", "plan_horn20_s2", "flip"), ("horn20", "plan_horn20_s3", "flip"),
         ("horn20", "plan_horn20_diag_s1", "diag"),
         ("tri20s", "plan_tri20s_s1", "flip"), ("tri20s", "plan_tri20s_diag_s1", "diag"),
         ("tri20s", "plan_tri20s_diag_s2", "diag"),
         ("tri20m", "plan_tri20m_s1", "flip"), ("tri20m", "plan_tri20m_s2", "flip"),
         ("tri20m", "plan_tri20m_s3", "flip")]
N = 32


def run(shell, plan):
    m = R.build(shell); d, names, home, lo, hi, drive, qadr = R.setup(m)
    u = R.controls(np.load(R.HERE + "plans/" + plan + ".npy")[None], home, lo, hi, drive, (hi - lo) / 2)[0]
    floor = m.geom("terrain").id
    feet = {m.geom("robot/" + g).id for g in C.FOOT_GEOMS}
    grp = {}
    for g in range(m.ngeom):
        n = m.geom(g).name.split("/")[-1]
        grp[g] = ("feet" if g in feet else "horn" if n.startswith("horn") else
                  "shell" if n.startswith(("base_link", "upper_shell")) else
                  "sensors" if n.startswith(("display", "tof", "camera")) else "limbs")
    dofs = [m.jnt_dofadr[m.joint("robot/" + n).id] for n in names]
    rng = np.random.default_rng(7)
    rows = []
    f6 = np.zeros(6); jac = np.zeros((3, m.nv))
    for _ in range(N):
        jit = np.where(drive, rng.uniform(-0.08, 0.08, len(home)), 0)
        st = R.settle(m, d, home, qadr, tilt=tuple(rng.uniform(-8, 8, 2)), jitter=jit)
        mujoco.mj_setState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        pk = {"limbs": 0.0, "shell": 0.0, "horn": 0.0, "sensors": 0.0, "feet": 0.0}
        jt = np.zeros(len(dofs)); ups = []
        for i in range(len(u)):
            d.ctrl[:] = u[i]; mujoco.mj_step(m, d)
            per = dict.fromkeys(pk, 0.0); tau = np.zeros(m.nv)
            for c in range(d.ncon):
                con = d.contact[c]
                if floor not in (con.geom1, con.geom2): continue
                g = con.geom2 if con.geom1 == floor else con.geom1
                mujoco.mj_contactForce(m, d, c, f6)
                fw = con.frame.reshape(3, 3).T @ f6[:3]
                if con.geom1 != floor: fw = -fw
                per[grp[g]] += f6[0]
                mujoco.mj_jac(m, d, jac, None, con.pos, m.geom_bodyid[g]); tau += jac.T @ fw
            for k in pk: pk[k] = max(pk[k], per[k])
            jt = np.maximum(jt, np.abs(tau[dofs]))
            ups.append(R.pose_of(d.qpos)[0])
        ok = min(ups[-120:]) > 0.9
        rows.append((ok, jt.max(), names[int(jt.argmax())], pk))
    return rows


if len(sys.argv) > 1:
    PLANS = [tuple(a.split(",")) for a in sys.argv[1:]]
for shell, plan, roll in PLANS:
    rows = run(shell, plan)
    ok = [r for r in rows if r[0]]
    jt = np.array([r[1] for r in ok]) if ok else np.array([np.nan])
    med = lambda k: np.median([r[3][k] for r in ok]) if ok else np.nan
    worst = max(ok, key=lambda r: r[1])[2] if ok else "-"
    print(f"{shell:7s} {roll:5s} {plan:22s} rights {len(ok):2d}/32 | joint torque median {np.median(jt):.1f} "
          f"p90 {np.percentile(jt, 90):.1f} N*m (worst at {worst}) | median peaks N: limbs {med('limbs'):.0f} "
          f"shell {med('shell'):.0f} horn {med('horn'):.0f} feet {med('feet'):.0f}", flush=True)
