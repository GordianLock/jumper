"""Keyframe roll search on carapace prototype 3 (CF1-tri30m-rb-v3), in the training
physics, scored for a gentle roll with the horned carapace treated as reinforced.

Written from the horn study (`studies/carapace/righting.py`), not from any other
session's search. Two of the horn study's own faults are fixed here:

- **Physics.** The horn study's scene carried no solver options, so it ran at MuJoCo's
  2 ms Euler default, and its 4 physics steps per control step made the plan run at
  125 Hz instead of 50. Here the options come from the training environment
  (`options.json`, written by `dump_scene.py`): 5 ms, implicitfast, Newton; 4 steps per
  control step is 50 Hz.
- **What counts as a roll.** Success needs the body to have *turned* the required angle
  about the target axis (half a turn from belly-up, a full turn from standing) and to
  end upright. Upright alone is not enough: from standing, doing nothing is upright.

## Model

The training scene, plus:

- motors as position servos, kp 10, kd 0.5, clipped at 1.7464 N*m (the plateau; the
  servo's speed curve is left out);
- **the carapace with friction** (`CARAPACE_CONDIM`, `CARAPACE_FRICTION`). The training
  config gives every non-foot collision piece condim 1, frictionless, which leaves a
  horn nothing to pivot on. The horn study's horns had condim 3, friction 0.6; so do
  these. The stock shell and limbs keep the training's settings;
- one contact sensor per collision piece against the floor.

## Plan, start, roll

Five keyframes of the 20 driven joints (offsets from HOME in half-range units), on a
fixed 50 Hz schedule (`SCHED`), HOME from 2.0 s to 3.5 s.

- `belly`: dropped belly-up and settled 2 s, as in the horn study; needs half a turn.
- `stand`: settled standing 1 s; needs a full turn.

Rolls are named by body axis: `side` (the centreline, x), `flip` (end over end, y),
`diag` (right shoulder to left rear hip, as in the horn study), or `any`: whichever
horizontal body axis the roll turns about, measured per rollout (`axes_for`); turning
about the vertical, spinning, is off-axis in every mode. `diagonal` accepts either
diagonal, whichever each rollout's mid-roll rotation is closer to.

## Turn

The turn about the axis is the integral of the body-frame angular velocity projected
on it. Progress is the furthest it got either way; success is that direction's turn at
the end within `TURN_TOL` of the requirement, upright (body z . world z > 0.9) over
the last 0.6 s. The turn is also printed on every rendered frame (`render.py`), so the
measure is checked against the picture before any result is believed.

## Gentle

The stock shell and base from 30 N, limbs from 20 N, sensors from 2 N (peak force);
the hardest carapace **region**'s **impact** from 50 N and the whole floor's impact
from 60 N, impact being the sharpest rise in force within 20 ms (`rise`), so load the
horns carry as pivots is not charged; the longest flight from 0.06 s; and the worst
floor-contact torque on any joint (J^T f, computed for every plan of every round) from
`TORQUE_FREE`. Costs phase in from round 10 to 30 (the horn study's scheme).

    python -u studies/proto3/search.py belly side 1 2 3
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
SUB = 4                                   # 4 x 5 ms = 50 Hz
SCHED = [(0, 15, 0), (15, 30, 1), (30, 50, 2), (50, 75, 3), (75, 100, 4), (100, 175, -1)]
STEPS = SCHED[-1][1]
K = 5
TURN_TOL = 0.5                            # rad short of the requirement still counts
CARAPACE_CONDIM = 3
CARAPACE_FRICTION = [0.6, 0.005, 0.0001]

AXES = {"side": np.array([1.0, 0.0, 0.0]), "flip": np.array([0.0, 1.0, 0.0]),
        "diag": np.array([-0.116, 0.117, 0.0]) / np.hypot(0.116, 0.117),
        #: Any horizontal body axis: each rollout is measured about the axis it turned
        #: about (`axes_for`); only turning about the vertical (spinning) is off-axis.
        "any": None,
        #: Either diagonal: right shoulder to left rear hip, or its mirror. Each rollout
        #: is measured about whichever of the two its mid-roll rotation is closer to.
        "diagonal": "either"}
_DIAGS = np.array([[-0.116, 0.117, 0.0], [-0.116, -0.117, 0.0]]) / np.hypot(0.116, 0.117)
NEED = {"belly": np.pi, "stand": 2 * np.pi}

#: Carapace pieces by region, from assets/jumper/jumper_CF1-tri30m-rb-v3.xml and its
#: params.json (2026-10-28): a piece belongs to the horn whose axis passes within 3 cm
#: of its centroid, or to the bumper within 2 cm of it (top view); the rest is the
#: front field, or the rear field split by side.
REGIONS = {
    "centre": (0, 9, 10, 13, 14, 21, 22),
    "front_left": (5, 6, 7, 16, 17, 18, 41),
    "front_right": (1, 20, 23, 24, 32),
    "bumper_left_front": (11,), "bumper_left_mid": (12,), "bumper_left_rear": (27,),
    "bumper_right_front": (3,), "bumper_right_mid": (4,), "bumper_right_rear": (2,),
    "front_field": (25,),
    "rear_field": (8, 15, 19, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40),
}
_REGION_OF = {i: r for r, ids in REGIONS.items() for i in ids}

FEET = set(C.FOOT_GEOMS)
#: Charged: plain peak force on the limbs, sensors and stock shell (any real force
#: there is unwanted); **impact**, not load, on the carapace and on the floor as a
#: whole. The carapace's horns, supports and bumpers are built to bear load and to be
#: pivoted on, so a load that builds up as the robot leans and pushes is free; what
#: is charged is how sharply force arrives (`IMPACT_WINDOW`). "carapace_impact" is the
#: hardest region's, "ground_impact" every part's summed. The impact levels are first
#: guesses, set by scale: the robot weighs 25.8 N.
FREE = {"limbs": 20.0, "sensors": 2.0, "shell": 30.0, "carapace_impact": 50.0,
        "ground_impact": 60.0, "flight": 0.06}
COST = {"limbs": 0.08, "sensors": 0.5, "shell": 0.04, "carapace_impact": 0.03,
        "ground_impact": 0.04, "flight": 40.0}
#: Impact is the largest rise in force within this many physics steps: 4 x 5 ms, one
#: control step. A slam rises within it; leaning the weight onto a horn does not.
IMPACT_WINDOW = 4
#: Worst floor-contact torque on any joint: free to 4.8 N*m (KingKong's reference-free
#: jump, measured in the first session with baseline_loads.py), 3 per N*m above.
TORQUE_FREE = 4.8
TORQUE_COST = 3.0


def part_group(name):
    n = name.split("/")[-1]
    if n in FEET:
        return "feet"
    if n.startswith("upper_shell_carapace"):
        return "carapace:" + _REGION_OF[int(n.split("_")[3])] if n.endswith("_meshcol") \
            else "visual"
    if n.startswith(("base_link", "upper_shell")):
        return "shell"
    if n.startswith(("display", "tof", "camera")):
        return "sensors"
    return "limbs"


def build():
    import re
    xml = open(HERE + "scene.xml").read()
    pieces = {int(i) for i in re.findall(r'name="robot/upper_shell_carapace_(\d+)_meshcol"', xml)}
    assert pieces == set(_REGION_OF), f"REGIONS do not match the scene's pieces {sorted(pieces)}"
    geoms = sorted(set(re.findall(r'name="(robot/[^"]+_meshcol)"', xml)))
    sens = "".join(f'\n    <contact geom1="{g}" body2="terrain" data="force" reduce="netforce" '
                   f'name="gc/{g}"/>' for g in geoms)
    xml = xml.replace("  </sensor>", sens + "\n  </sensor>", 1)
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = ROOT + "assets/jumper/"
    for a in spec.actuators:
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = KP
        a.biasprm[0:3] = [0, -KP, -KD]
        a.forcelimited = True
        a.forcerange = [-TAU, TAU]
    for g in spec.geoms:
        if g.name.startswith("robot/upper_shell_carapace") and g.name.endswith("_meshcol"):
            g.condim = CARAPACE_CONDIM
            g.friction = CARAPACE_FRICTION
    m = spec.compile()
    for k, v in json.load(open(HERE + "options.json")).items():
        if k == "gravity":
            m.opt.gravity[:] = v
        else:
            setattr(m.opt, k, v)
    return m


def setup(m):
    d = mujoco.MjData(m)
    names = [m.actuator(i).name.split("/")[-1] for i in range(m.nu)]
    home = np.array([C.HOME[n] for n in names])
    jid = [m.joint("robot/" + n).id for n in names]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    drive = np.array([not n.endswith("J4_joint") for n in names])
    return d, names, home, lo, hi, drive, [m.jnt_qposadr[j] for j in jid]


def settle(m, d, home, qadr, start, tilt=(0.0, 0.0), jitter=None):
    """Belly-up: dropped from 0.3 m, settled 2 s. Standing: set at HOME height, 1 s."""
    mujoco.mj_resetData(m, d)
    r = np.radians((180.0 if start == "belly" else 0.0) + tilt[0])
    p = np.radians(tilt[1])
    qr = np.array([np.cos(r / 2), np.sin(r / 2), 0, 0])
    qp = np.array([np.cos(p / 2), 0, np.sin(p / 2), 0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qp, qr)
    d.qpos[0:3] = [0, 0, 0.30 if start == "belly" else C.STAND_Z + 0.01]
    d.qpos[3:7] = q
    d.qpos[qadr] = home + (jitter if jitter is not None else 0)
    d.ctrl[:] = home
    for _ in range(int((2.0 if start == "belly" else 1.0) / m.opt.timestep)):
        mujoco.mj_step(m, d)
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
            off = np.zeros((n, len(home)))
            off[:, drive] = params[:, k]
            tgt = np.clip(home + off * half, lo, hi)
        u[:, a * SUB:b * SUB] = tgt[:, None, :]
    return u


def up_of(q):
    x, y = q[..., 4], q[..., 5]
    return 1 - 2 * (x * x + y * y)


def _omega(states, m):
    return states[:, :, 1 + m.nq + 3:1 + m.nq + 6]        # free joint: body-frame rad/s


def axes_for(states, m, axis, up):
    """The axis each rollout is measured about, (n, 3). A fixed roll uses its own axis.
    "any" uses the horizontal part (body x-y) of the rotation accumulated mid-roll (body
    between belly-up and upright), or over the whole rollout when there is too little of
    that to say: the axis the body actually rolled about, with spin left out."""
    n = states.shape[0]
    if isinstance(axis, str):                          # "either" diagonal
        w = _omega(states, m)
        mid = ((up > -0.9) & (up < 0.9))[..., None]
        th = np.where(((w * mid).sum(1)[:, :2] ** 2).sum(1, keepdims=True) > 0.09,
                      (w * mid).sum(1), w.sum(1)) * m.opt.timestep
        pick = np.abs(th @ _DIAGS.T).argmax(1)
        return _DIAGS[pick]
    if axis is not None:
        return np.broadcast_to(axis, (n, 3))
    w = _omega(states, m)
    mid = ((up > -0.9) & (up < 0.9))[..., None]
    th_mid = (w * mid).sum(1) * m.opt.timestep
    th_all = w.sum(1) * m.opt.timestep
    th = np.where(np.linalg.norm(th_mid[:, :2], axis=1, keepdims=True) > 0.3, th_mid, th_all)
    h = th.copy()
    h[:, 2] = 0.0
    nrm = np.linalg.norm(h, axis=1, keepdims=True)
    return np.where(nrm > 1e-6, h / np.maximum(nrm, 1e-6), np.array([1.0, 0.0, 0.0]))


def turn(states, m, axes):
    """Per physics step, the accumulated turn about each rollout's axis (body frame), rad."""
    return np.cumsum((_omega(states, m) * axes[:, None, :]).sum(-1), axis=1) * m.opt.timestep


def axis_deg(axes):
    """Direction of each axis in the body's x-y plane, degrees: 0 the centreline (side
    roll), +-90 end over end, about +-45 or +-135 a diagonal."""
    return np.degrees(np.arctan2(axes[:, 1], axes[:, 0]))


def axis_error_deg(states, m, axes, up):
    """Angle between the body-frame rotation accumulated mid-roll and each axis."""
    w = _omega(states, m)
    mid = ((up > -0.9) & (up < 0.9))[..., None]
    th = (w * mid).sum(1) * m.opt.timestep
    a = np.abs((th * axes).sum(1)) / (np.linalg.norm(th, axis=1) + 1e-6)
    return np.degrees(np.arccos(np.clip(a, 0, 1)))


def judge(states, m, axis, need):
    """Progress, success, final turn, time to upright, body-up and the axes used."""
    q = states[:, :, 1:1 + m.nq]
    up = up_of(q)
    axes = axes_for(states, m, axis, up)
    cum = turn(states, m, axes)
    sign = np.where(cum.max(1) >= -cum.min(1), 1.0, -1.0)
    signed = cum * sign[:, None]
    prog = np.clip(signed.max(1), 0, need)
    tail = slice(-30 * SUB, None)                    # last 0.6 s
    ok = (signed[:, -1] >= need - TURN_TOL) & (up[:, tail].min(1) > 0.9)
    t_up = np.where(ok, np.argmax((signed >= need - TURN_TOL) & (up > 0.9), axis=1)
                    * m.opt.timestep, np.nan)
    return prog, ok, signed[:, -1], t_up, up, axes


def groups_of(m):
    g = {}
    for i in range(m.nsensor):
        n = m.sensor(i).name
        if n.startswith("gc/"):
            g.setdefault(part_group(n[3:]), []).append(m.sensor_adr[i])
    return g


def rise(f, w=IMPACT_WINDOW):
    """Largest increase of force `f` (time on axis 1) within `w` steps: for each step,
    the force minus the least force of the `w` steps before it; the maximum over time.
    Extra trailing axes (regions) are kept."""
    lo = f.copy()
    for k in range(1, w + 1):
        shifted = np.concatenate([np.repeat(f[:, :1], k, axis=1), f[:, :-k]], axis=1)
        lo = np.minimum(lo, shifted)
    return (f - lo).max(1)


def peaks(sens, m):
    per = {k: np.stack([np.linalg.norm(sens[..., a:a + 3], axis=-1) for a in adrs], -1).sum(-1)
           for k, adrs in groups_of(m).items()}
    zero = np.zeros(next(iter(per.values())).shape)
    total = sum(per.values())
    regions = [v for k, v in per.items() if k.startswith("carapace:")]
    out = {k: per.get(k, zero).max(-1) for k in ("limbs", "sensors", "shell", "feet")}
    car = np.stack(regions, -1)
    out["carapace"] = car.max(-1).max(-1)                      # peak load, reported
    out["carapace_impact"] = rise(car).max(-1)                  # hardest region's impact
    out["ground"] = total.max(-1)
    out["ground_impact"] = rise(total)
    air = total < 0.5
    run = np.zeros(air.shape[:-1])
    longest = np.zeros(air.shape[:-1])
    for t in range(air.shape[-1]):
        run = np.where(air[..., t], run + 1, 0)
        longest = np.maximum(longest, run)
    out["flight"] = longest * m.opt.timestep
    return out


def floor_torque(m, d, start, u, dofs, floor):
    """Worst |J^T f| over floor contacts on any of `dofs`, one plan from `start`."""
    mujoco.mj_setState(m, d, start, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    worst, tau = 0.0, np.zeros(m.nv)
    for i in range(len(u)):
        d.ctrl[:] = u[i]
        mujoco.mj_step(m, d)
        if d.ncon == 0:
            continue
        g = d.contact.geom[:d.ncon]
        on = (g[:, 0] == floor) | (g[:, 1] == floor)
        if not on.any():
            continue
        f = np.zeros(d.nefc)
        for adr, dim in zip(d.contact.efc_address[:d.ncon][on], d.contact.dim[:d.ncon][on]):
            if adr >= 0:
                n = 2 * (dim - 1) if dim > 1 else 1      # pyramidal cone rows
                f[adr:adr + n] = d.efc_force[adr:adr + n]
        mujoco.mj_mulJacTVec(m, d, tau, f)
        worst = max(worst, float(np.abs(tau[dofs]).max()))
    return worst


def score(states, sens, m, axis, need, pen):
    prog, ok, _, _, up, axes = judge(states, m, axis, need)
    q = states[:, :, 1:1 + m.nq]
    tail = slice(-30 * SUB, None)
    stand = np.exp(-((q[:, tail, 2].mean(1) - C.STAND_Z) / 0.03) ** 2)
    on_axis = np.exp(-(axis_error_deg(states, m, axes, up) / 15.0) ** 2)
    sc = 5 * prog + on_axis * (20 * ok + 3 * stand * (up[:, tail].min(1) > 0.9))
    for k, pk in peaks(sens, m).items():
        if k in COST:
            sc = sc - pen * COST[k] * np.maximum(0.0, pk - FREE[k])
    return sc, ok


def search(start, roll, seed, iters=50, pop=96, elite=12):
    axis, need = AXES[roll], NEED[start]
    m = build()
    d, names, home, lo, hi, drive, qadr = setup(m)
    half = (hi - lo) / 2
    st0 = settle(m, d, home, qadr, start)
    rng = np.random.default_rng(seed)
    nd = int(drive.sum())
    mu, sd = np.zeros((K, nd)), np.full((K, nd), 0.6)
    datas = [mujoco.MjData(m) for _ in range(2)]
    td = mujoco.MjData(m)
    dofs = [m.jnt_dofadr[m.joint("robot/" + n).id] for n in names]
    floor = m.geom("terrain").id
    best, t0 = (-1e9, None), time.time()
    tag = f"{start}_{roll}_s{seed}"
    for it in range(iters):
        P = np.clip(mu + sd * rng.standard_normal((pop, K, nd)), -1.5, 1.5)
        P[0] = mu
        if best[1] is not None:
            P[1] = best[1]
        u_all = controls(P, home, lo, hi, drive, half)
        st, se = rollout.rollout(m, datas, st0[None], u_all)
        pen = min(1.0, max(0.0, (it - 10) / 20))
        s, ok = score(st, se, m, axis, need, pen)
        if pen > 0:
            for j in range(pop):
                jt = floor_torque(m, td, st0, u_all[j], dofs, floor)
                s[j] -= pen * TORQUE_COST * max(0.0, jt - TORQUE_FREE)
        if best[1] is not None and it in (10, 30):
            best = (-1e9, best[1])                     # re-rank once the costs have moved
        idx = np.argsort(-s)[:elite]
        mu, sd = P[idx].mean(0), np.maximum(P[idx].std(0), 0.05)
        if s[idx[0]] > best[0]:
            best = (float(s[idx[0]]), P[idx[0]].copy())
        if it % 5 == 0 or it == iters - 1:
            print(f"  {tag} it {it:2d} best {s[idx[0]]:.2f} rolled-in-pop {ok.mean():.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    plan = best[1]
    np.save(HERE + f"plans/{tag}.npy", plan)
    return evaluate(m, d, plan, start, roll, seed)


def evaluate(m, d, plan, start, roll, seed, n=32):
    """The plan from n perturbed starts: success, turn, axis error, time to upright,
    joint torque and peak forces, medians over the starts that succeed."""
    axis, need = AXES[roll], NEED[start]
    _, names, home, lo, hi, drive, qadr = setup(m)
    rng = np.random.default_rng(1000 + seed)
    tilt = 8.0 if start == "belly" else 3.0
    starts = np.array([settle(m, d, home, qadr, start,
                              tilt=tuple(rng.uniform(-tilt, tilt, 2)),
                              jitter=np.where(drive, rng.uniform(-0.08, 0.08, len(home)), 0))
                       for _ in range(n)])
    u = controls(plan[None], home, lo, hi, drive, (hi - lo) / 2)[0]
    st, se = rollout.rollout(m, [mujoco.MjData(m) for _ in range(2)], starts,
                             np.repeat(u[None], n, 0))
    _, ok, final, t_up, up, axes = judge(st, m, axis, need)
    err = axis_error_deg(st, m, axes, up)
    direction = axis_deg(axes)
    pk = peaks(se, m)
    dofs = [m.jnt_dofadr[m.joint("robot/" + nm).id] for nm in names]
    floor = m.geom("terrain").id
    jt = np.array([floor_torque(m, d, starts[i], u, dofs, floor) for i in range(n)])
    sel = ok if ok.any() else np.ones(n, bool)
    info = {"start": start, "roll": roll, "seed": seed, "success": int(ok.sum()),
            "turn_deg_median": float(np.degrees(np.median(final))),
            "axis_error_deg_median": float(np.median(err)),
            "axis_direction_deg": [round(float(a), 1) for a in direction],
            "t_upright_median": float(np.nanmedian(t_up)) if ok.any() else None,
            "joint_torque_median": float(np.median(jt[sel])),
            "joint_torque_p90": float(np.percentile(jt[sel], 90)),
            "peak_median": {k: float(np.median(v[sel])) for k, v in pk.items()}}
    print(f"{start} {roll} seed {seed}: rolls {ok.sum()}/{n}, turn {info['turn_deg_median']:.0f} deg, "
          f"axis error {info['axis_error_deg_median']:.0f} deg, axis direction median "
          f"{np.median(direction[sel]):.0f} deg, upright at "
          f"{info['t_upright_median']} s | joint torque median {info['joint_torque_median']:.1f} "
          f"p90 {info['joint_torque_p90']:.1f} N*m | median peaks: "
          + " ".join(f"{k} {v:.2f}" if k == "flight" else f"{k} {v:.0f}"
                     for k, v in info["peak_median"].items()), flush=True)
    return info


if __name__ == "__main__":
    start, roll, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]] or [1, 2, 3]
    out = [search(start, roll, s) for s in seeds]
    json.dump(out, open(HERE + f"logs/search_{start}_{roll}_s{'_'.join(map(str, seeds))}.json",
                        "w"), indent=1)
