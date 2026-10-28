"""A guide for jumper.horned_roll: a keyframe momentum roll on carapace prototype 2,
searched under that task's contact rules.

Adapted from `studies/roll_v2/roll_search.py` (carapace-design, commit fc77cee, copied
2026-10-28 with its `scene.xml` so that study can change without moving this one).
The plan, schedule, physics, progress, roll clock and axis rule are that search's.
What differs is how contact is scored, to match jumper.horned_roll
(`tasks/jumper/horned_roll/env_cfg.py`):

- the stock shell and base on their own, from 30 N;
- the carapace by **region** (`REGIONS` there): each region sums its pieces, and only
  the hardest region counts, from 80 N and at half the stock shell's cost per newton;
- the **total** floor force, every part summed, feet included, from 100 N, so a hard
  landing costs the same whatever touches down first;
- the longest **flight** (nothing on the floor), from 0.06 s;
- limbs from 20 N and sensors from 2 N, as before. Feet have no charge of their own;
  the total carries their landings.

Costs are phased in from round 10 to round 30, the horn study's scheme. Each search's
best plan is then run from 32 perturbed standing starts and measured like
`studies/carapace/gentle.py`: each start's worst floor-contact torque on any joint
(J^T f), and peak forces by kind; medians quoted. The reference is the stock high jump:
5.9 N*m (p90 7.0), limbs 76 N, feet 175 N.

    python -u studies/horned_roll/guide_search.py 1 2 3      # seeds
"""
import json
import os
import sys
import time

import mujoco
import numpy as np
from mujoco import rollout

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
ROOT = os.path.abspath(HERE + "../..") + "/"
sys.path.insert(0, ROOT)
from tasks.jumper.common import constants as C  # noqa: E402

KP, KD, TAU = 10.0, 0.5, 1.7464
DT, SUB = 0.005, 4                      # 5 ms physics, 50 Hz control
#: (first control step, end, keyframe); -1 is HOME.
SCHED = [(0, 10, 0), (10, 20, 1), (20, 30, 2), (30, 42, 3), (42, 55, 4), (55, 125, -1)]
STEPS = SCHED[-1][1]
K = 5
WINDOW = 1.0
START_TURN = 0.3
FULL_TOL = 0.3

from tasks.jumper.horned_roll.env_cfg import REGIONS  # noqa: E402

FEET = set(C.FOOT_GEOMS)
#: Free level per kind (N; flight in s) and cost per unit above it. "carapace" is the
#: hardest region, "ground" every part summed.
FREE = {"limbs": 20.0, "sensors": 2.0, "shell": 30.0, "carapace": 80.0, "ground": 100.0,
        "flight": 0.06}
COST = {"limbs": 0.08, "sensors": 0.5, "shell": 0.04, "carapace": 0.02, "ground": 0.04,
        "flight": 40.0}
_REGION_OF = {i: r for r, ids in REGIONS.items() for i in ids}


def part_group(name):
    """Contact kind of a geom: feet, limbs, sensors, shell (stock), or carapace:<region>."""
    n = name.split("/")[-1]
    if n in FEET: return "feet"
    if n.startswith("upper_shell_carapace"):
        if not n.endswith("_meshcol"):
            return "visual"                     # the display mesh; it never collides
        return "carapace:" + _REGION_OF[int(n.split("_")[3])]
    if n.startswith(("base_link", "upper_shell")): return "shell"
    if n.startswith(("display", "tof", "camera")): return "sensors"
    return "limbs"


def build():
    import re
    xml = open(HERE + "scene.xml").read()
    geoms = sorted(set(re.findall(r'name="(robot/[^"]+_meshcol)"', xml)))
    sens = "".join(f'\n    <contact geom1="{g}" body2="terrain" data="force" reduce="netforce" '
                   f'name="gc/{g}"/>' for g in geoms)
    xml = xml.replace("  </sensor>", sens + "\n  </sensor>", 1)
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = ROOT + "assets/jumper/"
    for a in spec.actuators:
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = KP; a.biasprm[0:3] = [0, -KP, -KD]
        a.forcelimited = True; a.forcerange = [-TAU, TAU]
    m = spec.compile()
    o = m.opt
    o.timestep = DT
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.iterations, o.ls_iterations, o.tolerance, o.ls_tolerance = 10, 20, 1e-8, 0.01
    o.ccd_iterations = 50
    o.impratio = 1.0
    return m


def groups_of(m):
    g = {}
    for i in range(m.nsensor):
        n = m.sensor(i).name
        if n.startswith("gc/"):
            g.setdefault(part_group(n[3:]), []).append(m.sensor_adr[i])
    return g


def peaks(sens, m):
    """Per rollout: the peak over time of each kind's summed force, the hardest
    carapace region's, every part's total, and the longest flight in seconds."""
    per = {}
    for gname, adrs in groups_of(m).items():
        per[gname] = np.stack([np.linalg.norm(sens[..., a:a + 3], axis=-1) for a in adrs],
                              -1).sum(-1)
    zero = np.zeros(next(iter(per.values())).shape)
    total = sum(per.values())
    regions = [v for k, v in per.items() if k.startswith("carapace:")]
    out = {k: per.get(k, zero).max(-1) for k in ("limbs", "sensors", "shell", "feet")}
    out["carapace"] = np.max(np.stack(regions, -1), -1).max(-1) if regions else zero.max(-1)
    out["ground"] = total.max(-1)
    air = total < 0.5
    run = np.zeros(air.shape[:-1]); longest = np.zeros(air.shape[:-1])
    for t in range(air.shape[-1]):
        run = np.where(air[..., t], run + 1, 0); longest = np.maximum(longest, run)
    out["flight"] = longest * DT
    return out


def setup(m):
    d = mujoco.MjData(m)
    names = [m.actuator(i).name.split("/")[-1] for i in range(m.nu)]
    home = np.array([C.HOME[n] for n in names])
    jid = [m.joint("robot/" + n).id for n in names]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    drive = np.array([not n.endswith("J4_joint") for n in names])
    return d, names, home, lo, hi, drive, [m.jnt_qposadr[j] for j in jid]


def settle_standing(m, d, home, qadr, tilt=(0.0, 0.0), jitter=None, seconds=1.0):
    mujoco.mj_resetData(m, d)
    r, p = np.radians(tilt[0]), np.radians(tilt[1])
    qr = np.array([np.cos(r / 2), np.sin(r / 2), 0, 0]); qp = np.array([np.cos(p / 2), 0, np.sin(p / 2), 0])
    q = np.zeros(4); mujoco.mju_mulQuat(q, qp, qr)
    d.qpos[0:3] = [0, 0, C.STAND_Z + 0.01]; d.qpos[3:7] = q
    d.qpos[qadr] = home + (jitter if jitter is not None else 0)
    d.ctrl[:] = home
    for _ in range(int(seconds / m.opt.timestep)): mujoco.mj_step(m, d)
    st = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return st


def controls(params, home, lo, hi, drive, half):
    n = params.shape[0]
    u = np.empty((n, STEPS * SUB, len(home)))
    for a, b, k in SCHED:
        if k < 0:
            tgt = np.broadcast_to(home, (n, len(home)))
        else:
            off = np.zeros((n, len(home))); off[:, drive] = params[:, k]
            tgt = np.clip(home + off * half, lo, hi)
        u[:, a * SUB:b * SUB] = tgt[:, None, :]
    return u


def kinematics(states, m):
    """Per physics step: body-up, the unwrapped roll angle (right side down
    positive), base height, and body-frame angular velocity."""
    q = states[:, :, 1:1 + m.nq]
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    up = 1 - 2 * (x * x + y * y)
    r21 = 2 * (y * z + w * x)
    theta = np.unwrap(np.arctan2(r21, up), axis=1)
    theta = theta - theta[:, :1]
    omega = states[:, :, 1 + m.nq + 3:1 + m.nq + 6]
    return up, theta, q[..., 2], omega


def roll_clock(up, theta, dt):
    """Seconds from a START_TURN turn to upright after a full turn; nan if never."""
    started = theta >= START_TURN
    done = (theta >= 2 * np.pi - FULL_TOL) & (up > 0.9)
    t0 = np.where(started.any(1), started.argmax(1), -1)
    t1 = np.where(done.any(1), done.argmax(1), -1)
    ok = (t0 >= 0) & (t1 >= 0)
    return np.where(ok, (t1 - t0) * dt, np.nan)


def axis_error_deg(up, omega, dt):
    """Angle between the body-frame turn accumulated mid-roll and the long axis."""
    mask = ((up > -0.9) & (up < 0.9))[..., None]
    th = (omega * mask).sum(1) * dt
    a = np.abs(th[:, 0]) / (np.linalg.norm(th, axis=1) + 1e-6)
    return np.degrees(np.arccos(np.clip(a, 0, 1)))


def score(states, m, sens=None, pen=1.0):
    up, theta, z, omega = kinematics(states, m)
    tail = slice(-30 * SUB, None)
    prog = np.clip(theta.max(1), 0, 2 * np.pi)
    full = theta[:, tail].min(1) >= 2 * np.pi - FULL_TOL
    upright = up[:, tail].min(1) > 0.9
    ok = full & upright
    stand = np.exp(-((z[:, tail].mean(1) - C.STAND_Z) / 0.03) ** 2)
    secs = roll_clock(up, theta, m.opt.timestep)
    late = np.where(np.isnan(secs), 0.0, np.maximum(0.0, secs - WINDOW))
    on_axis = np.exp(-(axis_error_deg(up, omega, m.opt.timestep) / 15.0) ** 2)
    sc = 5 * prog + on_axis * (20 * ok + 3 * stand * upright) - 10 * late * ok
    if sens is not None:
        for gname, pk in peaks(sens, m).items():
            if gname in COST:
                sc = sc - pen * COST[gname] * np.maximum(0, pk - FREE[gname])
    return sc, ok, secs


def joint_loads(m, d, start, u, names):
    """gentle.py's measure for one start: worst |J^T f| over floor contacts on any
    actuated joint, and peaks: summed force per kind, hardest carapace region, every
    part's total, and the longest flight (s)."""
    floor = m.geom("terrain").id
    grp = {g: part_group(m.geom(g).name or "") for g in range(m.ngeom)}
    dofs = [m.jnt_dofadr[m.joint("robot/" + n).id] for n in names]
    mujoco.mj_setState(m, d, start, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    keys = ("feet", "limbs", "sensors", "shell", "carapace", "ground", "flight")
    pk = dict.fromkeys(keys, 0.0); jt = np.zeros(len(dofs)); run = 0
    f6 = np.zeros(6); jac = np.zeros((3, m.nv))
    for i in range(len(u)):
        d.ctrl[:] = u[i]; mujoco.mj_step(m, d)
        per = {}; tau = np.zeros(m.nv)
        for c in range(d.ncon):
            con = d.contact[c]
            if floor not in (con.geom1, con.geom2): continue
            g = con.geom2 if con.geom1 == floor else con.geom1
            mujoco.mj_contactForce(m, d, c, f6)
            fw = con.frame.reshape(3, 3).T @ f6[:3]
            if con.geom1 != floor: fw = -fw
            per[grp[g]] = per.get(grp[g], 0.0) + f6[0]
            mujoco.mj_jac(m, d, jac, None, con.pos, m.geom_bodyid[g]); tau += jac.T @ fw
        total = sum(per.values())
        car = max([v for k, v in per.items() if k.startswith("carapace:")], default=0.0)
        for k in ("feet", "limbs", "sensors", "shell"):
            pk[k] = max(pk[k], per.get(k, 0.0))
        pk["carapace"] = max(pk["carapace"], car); pk["ground"] = max(pk["ground"], total)
        run = run + 1 if total < 0.5 else 0
        pk["flight"] = max(pk["flight"], run * m.opt.timestep)
        jt = np.maximum(jt, np.abs(tau[dofs]))
    return jt.max(), names[int(jt.argmax())], pk


#: Joint torque in the search: worst floor-contact torque on any joint above
#: TORQUE_FREE N*m costs TORQUE_COST per N*m, phased in with the other costs. Every
#: plan of a round is re-simulated to measure it: about 0.1 s each, and on seeds 2 and
#: 3 of the first searches it matched joint_loads() exactly (6.05 and 16.24 N*m).
TORQUE_FREE = 5.0
#: File tag: "_torque" for these runs; the first three searches (no torque) are untagged.
TAG = "_torque"
TORQUE_COST = 3.0
TORQUE_TOP = 96
TORQUE_STRIDE = 1


def floor_torque(m, d, start, u, dofs, floor, stride=TORQUE_STRIDE):
    """Worst |J^T f| over floor contacts on any of `dofs` for one plan from `start`:
    the floor contacts' constraint forces mapped to joint space in one call
    (mj_mulJacTVec), the same quantity joint_loads() sums contact by contact."""
    mujoco.mj_setState(m, d, start, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    worst = 0.0
    tau = np.zeros(m.nv)
    for i in range(len(u)):
        d.ctrl[:] = u[i]; mujoco.mj_step(m, d)
        if i % stride or d.ncon == 0:
            continue
        g = d.contact.geom[:d.ncon]
        on = (g[:, 0] == floor) | (g[:, 1] == floor)
        if not on.any():
            continue
        f = np.zeros(d.nefc)
        for adr, dim in zip(d.contact.efc_address[:d.ncon][on], d.contact.dim[:d.ncon][on]):
            if adr < 0:
                continue
            n = 2 * (dim - 1) if m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL else dim
            f[adr:adr + max(n, 1)] = d.efc_force[adr:adr + max(n, 1)]
        mujoco.mj_mulJacTVec(m, d, tau, f)
        worst = max(worst, float(np.abs(tau[dofs]).max()))
    return worst


def search(seed, iters=50, pop=96, elite=12):
    m = build(); d, names, home, lo, hi, drive, qadr = setup(m)
    half = (hi - lo) / 2
    st0 = settle_standing(m, d, home, qadr)
    rng = np.random.default_rng(seed)
    nd = int(drive.sum())
    mu = np.zeros((K, nd)); sd = np.full((K, nd), 0.6)
    datas = [mujoco.MjData(m) for _ in range(2)]
    best = (-1e9, None)
    dt_data = mujoco.MjData(m)
    dofs = [m.jnt_dofadr[m.joint("robot/" + n).id] for n in names]
    floor = m.geom("terrain").id
    saved = HERE + f"plans/plan_horned{TAG}_s{seed}.npy"
    if os.environ.get("REUSE") and os.path.exists(saved):
        best = (float("nan"), np.load(saved)); iters = 0
        print(f"  seed {seed}: reusing {saved}", flush=True)
    t0 = time.time()
    for it in range(iters):
        P = np.clip(mu + sd * rng.standard_normal((pop, K, nd)), -1.5, 1.5)
        P[0] = mu
        if best[1] is not None: P[1] = best[1]
        u_all = controls(P, home, lo, hi, drive, half)
        st, se = rollout.rollout(m, datas, st0[None], u_all)
        pen = min(1.0, max(0.0, (it - 10) / 20))
        s, ok, secs = score(st, m, se, pen)
        if pen > 0:
            top = np.argsort(-s)[:TORQUE_TOP]
            for j in top:
                jt = floor_torque(m, dt_data, st0, u_all[j], dofs, floor)
                s[j] -= pen * TORQUE_COST * max(0.0, jt - TORQUE_FREE)
            s[np.setdiff1d(np.arange(len(s)), top)] = -1e9
        if best[1] is not None and it in (10, 30):
            best = (-1e9, best[1])
        idx = np.argsort(-s)[:elite]
        mu = P[idx].mean(0); sd = np.maximum(P[idx].std(0), 0.05)
        if s[idx[0]] > best[0]: best = (float(s[idx[0]]), P[idx[0]].copy())
        if it % 5 == 0 or it == iters - 1:
            print(f"  seed {seed} it {it:2d} best {s[idx[0]]:.2f} rolled-in-pop {ok.mean():.2f} "
                  f"fastest {np.nanmin(secs) if np.isfinite(secs).any() else float('nan'):.2f}s "
                  f"({time.time() - t0:.0f}s)", flush=True)
    plan = best[1]
    np.save(HERE + f"plans/plan_horned{TAG}_s{seed}.npy", plan)

    # 32 perturbed standing starts
    starts = []
    for _ in range(32):
        jit = np.where(drive, rng.uniform(-0.08, 0.08, len(home)), 0)
        starts.append(settle_standing(m, d, home, qadr, tilt=tuple(rng.uniform(-3, 3, 2)), jitter=jit))
    starts = np.array(starts)
    u = controls(plan[None], home, lo, hi, drive, half)[0]
    st, se = rollout.rollout(m, datas, starts, np.repeat(u[None], 32, 0))
    _, ok, secs = score(st, m)
    up, _, _, omega = kinematics(st, m)
    err = axis_error_deg(up, omega, m.opt.timestep)
    loads = [joint_loads(m, d, starts[i], u, names) for i in range(32)]
    jt = np.array([x[0] for x in loads])
    worst = [x[1] for x in loads]
    pk = {k: np.array([x[2][k] for x in loads]) for k in loads[0][2]}
    sel = ok if ok.any() else np.ones(32, bool)
    info = {
        "seed": seed, "best_score": best[0], "success": int(ok.sum()),
        "roll_seconds_median": float(np.nanmedian(secs[ok])) if ok.any() else None,
        "roll_seconds_max": float(np.nanmax(secs[ok])) if ok.any() else None,
        "axis_error_deg_median": float(np.median(err)),
        "joint_torque_median": float(np.median(jt[sel])), "joint_torque_p90": float(np.percentile(jt[sel], 90)),
        "joint_torque_max": float(jt[sel].max()),
        "worst_joint_most_often": max(set(worst), key=worst.count),
        "peak_median": {k: float(np.median(v[sel])) for k, v in pk.items()},
        "peak_max": {k: float(v[sel].max()) for k, v in pk.items()},
    }
    print(f"seed {seed}: rolls {ok.sum()}/32, roll time median {info['roll_seconds_median']} s "
          f"(max {info['roll_seconds_max']}), axis error {info['axis_error_deg_median']:.0f} deg | "
          f"joint torque median {info['joint_torque_median']:.1f} p90 {info['joint_torque_p90']:.1f} "
          f"max {info['joint_torque_max']:.1f} N*m (most often {info['worst_joint_most_often']}) | "
          "median peaks: " + " ".join(f"{k} {v:.2f}" if k == "flight" else f"{k} {v:.0f}"
                                     for k, v in info["peak_median"].items()),
          flush=True)
    return info


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    out = [search(s) for s in seeds]
    json.dump(out, open(HERE + f"logs/guide_search{TAG}_s" + "_".join(map(str, seeds)) + ".json", "w"),
              indent=1)
