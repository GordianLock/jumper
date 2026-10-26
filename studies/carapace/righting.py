"""Does a self-righting motion exist from belly-up, and how does a stand-in horn change it?

Physics is jumper.roll's training scene (dumped to scene.xml from the env): same floor,
contact settings and 5 ms step. The motors become position servos, kp 10, kd 0.5,
clipped at 1.7464 N*m (the plateau; the servo's speed curve is left out).

Variants add one stand-in geom to base_link:
  stock   -- nothing
  hornNN  -- a capsule ridge rising toward the front of the top shell, sized so the
             robot rests belly-up on it and the shell's rear edge at about NN deg head-up

Each variant: drop belly-up and settle, then a cross-entropy search over five
keyframes of the 20 leg/arm joints (offsets from HOME in half-range units) on a
fixed 50 Hz schedule, ending at HOME. Score: upright at the end (body z up), standing
height, minus peak spin. The best plan is then rerun from 32 perturbed starts.
"""
import json, os, sys, time
import numpy as np, mujoco
from mujoco import rollout

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # repo root
from tasks.jumper.common import constants as C

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
ROOT = os.path.abspath(HERE + "../..") + "/"
KP, KD, TAU = 10.0, 0.5, 1.7464
SUB = 4                                   # physics steps per control step
SCHED = [(0, 15, 0), (15, 30, 1), (30, 50, 2), (50, 75, 3), (75, 100, 4), (100, 175, -1)]
STEPS = SCHED[-1][1]; K = 5
REAR_X, TOP = -0.112, 0.072


FEET = set(C.FOOT_GEOMS)


def part_group(name):
    n = name.split("/")[-1]
    if n in FEET: return "feet"
    if n.startswith("horn"): return "horn"
    if n.startswith(("base_link", "upper_shell")): return "shell"
    if n.startswith(("display", "tof", "camera")): return "sensors"
    return "limbs"


def build(variant):
    import re
    xml = open(HERE + "scene.xml").read()
    # rear horns on the 20 deg front horn, each placed as far out as stays clear of
    # the legs' floor-reaching sweep (zero overlap with leg_keepout_side_voxels.npz):
    #   rc20  -- two rear corner horns, 2.5 cm, (-0.04, +-0.035) -> (-0.075, +-0.065)
    #   rt20  -- one small rear centreline horn ("tail"), 2.5 cm, (-0.06, 0) -> (-0.095, 0)
    #   rct20 -- both
    # The same rear horns can sit on the three-horn front instead: "<front>_<rear>", for
    # example tri20s_rc, tri20s_rt, tri20s_rct (front = tri20s, rear = rc, rt or rct).
    # Each piece passed clearance.py on its own, so their union is clear of the legs too.
    front = None
    for rear in ("rct", "rc", "rt"):
        if variant.endswith("_" + rear):
            front, variant = variant[: -len(rear) - 1], rear + "20"
            break
    if variant in ("rc20", "rt20", "rct20"):
        line = xml[xml.index('<geom name="robot/base_link_meshcol"'):].split("\n")[0]
        extra = ""
        if variant in ("rc20", "rct20"):
            for sgn, nm in ((1, "horn_rl"), (-1, "horn_rr")):
                extra += (f'\n      <geom name="robot/{nm}" type="capsule" size="0.009" '
                          f'fromto="-0.04 {sgn * 0.035} {TOP} -0.075 {sgn * 0.065} {TOP + 0.025}" mass="0.015" '
                          f'contype="1" conaffinity="0" condim="3" friction="0.6 0.005 0.0001" rgba="0.9 0.5 0.1 1"/>')
        if variant in ("rt20", "rct20"):
            extra += (f'\n      <geom name="robot/horn_tail" type="capsule" size="0.010" '
                      f'fromto="-0.06 0 {TOP} -0.095 0 {TOP + 0.025}" mass="0.015" '
                      f'contype="1" conaffinity="0" condim="3" friction="0.6 0.005 0.0001" rgba="0.9 0.5 0.1 1"/>')
        xml = xml.replace(line, line + extra, 1)
        variant = front or "horn20"
    side = {"tri20s": 0.025, "tri20m": 0.035}.get(variant)
    if side is not None:
        # two smaller horns toward the front corners, angled out and forward; the tip is
        # as far toward the corner as stays clear of the legs' floor-reaching sweep
        # (zero overlap with leg_keepout_side_voxels.npz, 5 mm clearance included)
        rs = 0.009
        line = xml[xml.index('<geom name="robot/base_link_meshcol"'):].split("\n")[0]
        extra = ""
        for sgn, nm in ((1, "horn_l"), (-1, "horn_r")):
            extra += (f'\n      <geom name="robot/{nm}" type="capsule" size="{rs}" '
                      f'fromto="0.02 {sgn * 0.035} {TOP} 0.065 {sgn * 0.065} {TOP + side}" mass="0.015" '
                      f'contype="1" conaffinity="0" condim="3" friction="0.6 0.005 0.0001" rgba="0.9 0.5 0.1 1"/>')
        xml = xml.replace(line, line + extra, 1)
        variant = "horn20"
    if variant.startswith("horn"):
        # calibrated: these sizes settle at 20.1 and 30.1 deg head-up (stock: 7.8)
        deg = {"horn20": 15.43, "horn30": 21.99}.get(variant) or float(variant[4:]); r = 0.012
        x0, x1 = -0.03, 0.085
        # rests on the rear edge (REAR_X, TOP) and the capsule's front underside
        h = (x1 - REAR_X) * np.tan(np.radians(deg)) - r
        line = xml[xml.index('<geom name="robot/base_link_meshcol"'):].split("\n")[0]
        horn = (f'<geom name="robot/horn" type="capsule" size="{r}" fromto="{x0} 0 {TOP} {x1} 0 {TOP + h}" '
                f'mass="0.03" contype="1" conaffinity="0" condim="3" friction="0.6 0.005 0.0001" rgba="0.9 0.5 0.1 1"/>')
        xml = xml.replace(line, line + "\n      " + horn, 1)
    geoms = sorted(set(re.findall(r'name="(robot/[^"]+_meshcol[^"]*|robot/horn[^"]*)"', xml)))
    sens = "".join(f'\n    <contact geom1="{g}" body2="terrain" data="force" reduce="netforce" name="gc/{g}"/>'
                   for g in geoms)
    xml = xml.replace("  </sensor>", sens + "\n  </sensor>", 1)
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = ROOT + "assets/jumper/"
    for a in spec.actuators:
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = KP; a.biasprm[0:3] = [0, -KP, -KD]
    m = spec.compile()
    return m


def groups_of(m):
    g = {}
    for i in range(m.nsensor):
        n = m.sensor(i).name
        if n.startswith("gc/"):
            g.setdefault(part_group(n[3:]), []).append(m.sensor_adr[i])
    return g


#: Contact force each part group may take before it costs, N, and the cost per N
#: beyond it. A 100 N limb hit costs 6.4, against 13 for a full righting.
FREE = {"feet": 50.0, "limbs": 20.0, "shell": 40.0, "horn": 40.0, "sensors": 2.0}
COST = {"feet": 0.03, "limbs": 0.08, "shell": 0.04, "horn": 0.04, "sensors": 0.5}


def peaks(sens, m):
    out = {}
    for gname, adrs in groups_of(m).items():
        f = np.stack([np.linalg.norm(sens[..., a:a + 3], axis=-1) for a in adrs], -1)
        out[gname] = f.max(-1).max(-1)
    return out


def setup(m):
    d = mujoco.MjData(m)
    names = [m.actuator(i).name.split("/")[-1] for i in range(m.nu)]
    home = np.array([C.HOME[n] for n in names])
    jid = [m.joint("robot/" + n).id for n in names]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    drive = np.array([not n.endswith("J4_joint") for n in names])
    return d, names, home, lo, hi, drive, [m.jnt_qposadr[j] for j in jid]


def settle(m, d, home, qadr, tilt=(0, 0), jitter=None, seconds=2.0):
    mujoco.mj_resetData(m, d)
    # belly-up: 180 deg about x, plus optional small extra roll/pitch
    r, p = np.radians(180 + tilt[0]), np.radians(tilt[1])
    qr = np.array([np.cos(r / 2), np.sin(r / 2), 0, 0]); qp = np.array([np.cos(p / 2), 0, np.sin(p / 2), 0])
    q = np.zeros(4); mujoco.mju_mulQuat(q, qp, qr)
    d.qpos[0:3] = [0, 0, 0.30]; d.qpos[3:7] = q
    d.qpos[qadr] = home + (jitter if jitter is not None else 0)
    d.ctrl[:] = home
    for _ in range(int(seconds / m.opt.timestep)): mujoco.mj_step(m, d)
    st = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return st


def pose_of(qpos):
    w, x, y, z = qpos[3:7]
    up = 1 - 2 * (x * x + y * y)                     # body z . world z
    fwd_z = 2 * (x * z - w * y)                       # body x . world z (head up > 0)
    return up, np.degrees(np.arcsin(np.clip(fwd_z, -1, 1)))


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


#: Flip axes in the body frame (unit, horizontal). side: about the centreline;
#: diag: from the right shoulder (RF_J0 anchor, 0.051 -0.061) to the left rear hip
#: (LR_J0 anchor, -0.065 0.056), so the right rear swings over the left shoulder.
AXES = {"end": np.array([0.0, 1.0, 0.0]), "side": np.array([1.0, 0.0, 0.0]),
        "diag": np.array([-0.116, 0.117, 0.0]) / np.hypot(0.116, 0.117)}


def flip_axis(states, m, up):
    """Body-frame rotation accumulated while the body is between belly-up and upright."""
    w = states[:, :, 1 + m.nq + 3:1 + m.nq + 6]        # free-joint angular velocity, body frame
    mask = ((up > -0.9) & (up < 0.9))[..., None]
    return (w * mask).sum(1) * m.opt.timestep


def alignment(theta, axis):
    return np.abs(theta @ axis) / (np.linalg.norm(theta, axis=1) + 1e-6)


def score(states, m, sens=None, pen=1.0, axis=None):
    q = states[:, :, 1:1 + m.nq]                       # FULLPHYSICS: time, qpos, ...
    w, x, y = q[..., 3], q[..., 4], q[..., 5]
    up = 1 - 2 * (x * x + y * y)
    z = q[..., 2]
    tail = slice(-30 * SUB, None)                      # last 0.6 s
    up_end = up[:, tail].mean(1)
    stand = np.exp(-((z[:, tail].mean(1) - C.STAND_Z) / 0.03) ** 2)
    ok = (up[:, tail].min(1) > 0.9)
    t_up = np.where(ok, np.argmax(up > 0.9, axis=1) * m.opt.timestep, np.nan)
    # partial credit for getting round, a large bonus for finishing upright, then
    # the hit costs (phased in by `pen`, so the search finds a righting first)
    if axis is None:
        sc = 4 * up.max(1) + 10 * up_end + 3 * stand + 20 * ok
    else:
        # progress: the furthest the body has turned about the target axis, either way
        w = states[:, :, 1 + m.nq + 3:1 + m.nq + 6]
        cum = np.cumsum(w @ axis, axis=1) * m.opt.timestep
        prog = np.minimum(np.maximum(cum.max(1), -cum.min(1)), np.pi)
        err = np.degrees(np.arccos(np.clip(alignment(flip_axis(states, m, up), axis), 0, 1)))
        on_axis = np.exp(-(err / 15.0) ** 2)            # full credit within ~15 deg of the axis
        sc = 5 * prog + on_axis * (10 * up_end + 3 * stand + 20 * ok)
    if sens is not None:
        for gname, pk in peaks(sens, m).items():
            sc = sc - pen * COST[gname] * np.maximum(0, pk - FREE[gname])
    return sc, ok, t_up, up


def search(variant, iters=35, pop=96, elite=12, seed=0):
    variant, _, ax = variant.partition(":")
    axis = AXES[ax] if ax else None
    tag = variant + (f"_{ax}" if ax else "")
    m = build(variant); d, names, home, lo, hi, drive, qadr = setup(m)
    half = (hi - lo) / 2
    st0 = settle(m, d, home, qadr)
    up0, pitch0 = pose_of(st0[1:1 + m.nq])
    info = {"variant": variant, "rest_up": float(up0), "rest_pitch": float(pitch0), "rest_z": float(st0[3])}
    print(variant, "rests at body-up %.2f, head pitch %.1f deg, base height %.3f m" % (up0, pitch0, st0[3]), flush=True)
    rng = np.random.default_rng(seed)
    nd = int(drive.sum())
    mu = np.zeros((K, nd)); sd = np.full((K, nd), 0.6)
    datas = [mujoco.MjData(m) for _ in range(2)]
    best = (-1e9, None)
    t0 = time.time()
    for it in range(iters):
        P = np.clip(mu + sd * rng.standard_normal((pop, K, nd)), -1.5, 1.5)
        P[0] = mu
        if best[1] is not None: P[1] = best[1]
        st, se = rollout.rollout(m, datas, st0[None], controls(P, home, lo, hi, drive, half))
        pen = min(1.0, max(0.0, (it - 10) / 20))
        s, ok, t_up, _ = score(st, m, se, pen, axis)
        if best[1] is not None and it in (10, 30):
            best = (-1e9, best[1])          # re-rank once the penalty has moved
        idx = np.argsort(-s)[:elite]
        mu = P[idx].mean(0); sd = np.maximum(P[idx].std(0), 0.05)
        if s[idx[0]] > best[0]: best = (float(s[idx[0]]), P[idx[0]].copy())
        if it % 5 == 0 or it == iters - 1:
            print(f"  {variant} it {it:2d} best {s[idx[0]]:.2f} upright-in-pop {ok.mean():.2f} "
                  f"t_up {np.nanmin(t_up) if ok.any() else float('nan'):.2f}s ({time.time()-t0:.0f}s)", flush=True)
    info["best_score"] = best[0]
    np.save(HERE + f"plans/plan_{tag}_s{seed}.npy", best[1]); info["seed"] = seed
    # robustness: 32 perturbed starts (tilt +-8 deg, joint jitter +-0.08 rad)
    starts = []
    for _ in range(32):
        jit = np.where(drive, rng.uniform(-0.08, 0.08, len(home)), 0)
        starts.append(settle(m, d, home, qadr, tilt=tuple(rng.uniform(-8, 8, 2)), jitter=jit))
    starts = np.array(starts)
    st, se = rollout.rollout(m, datas, starts, controls(np.repeat(best[1][None], 32, 0), home, lo, hi, drive, half))
    _, ok, t_up, up = score(st, m)
    th = flip_axis(st, m, up)
    ang = np.degrees(np.arctan2(th[:, 1], th[:, 0]))
    yaw = np.abs(th[:, 2]) / (np.linalg.norm(th, axis=1) + 1e-6)
    info["axis"] = ax; info["flip_axis_deg_median"] = float(np.median(ang)); info["yaw_share_median"] = float(np.median(yaw))
    if axis is not None:
        info["alignment_median"] = float(np.median(alignment(th, axis)))
    print(f"  {tag}: flip axis at {np.median(ang):.0f} deg in the body xy plane (0 = centreline, "
          f"+-90 = end over end), yaw share {np.median(yaw):.2f}"
          + (f", alignment with target {np.median(alignment(th, axis)):.2f}" if axis is not None else ""), flush=True)
    pk = peaks(se, m)
    info["robust_peak_median"] = {k: float(np.median(v)) for k, v in pk.items()}
    print("  median peak floor force (N): " + ", ".join(f"{k} {np.median(v):.0f}" for k, v in pk.items()), flush=True)
    info["robust_success"] = float(ok.mean()); info["robust_t_up_median"] = float(np.nanmedian(t_up)) if ok.any() else None
    print(f"  {tag}: best plan rights {ok.sum()}/32 perturbed starts, median time to upright "
          f"{info['robust_t_up_median']}", flush=True)
    return info


if __name__ == "__main__":
    seeds = [1, 2, 3]
    out = [search(v, iters=50, seed=s) for v in sys.argv[1:] for s in seeds]
    json.dump(out, open(HERE + "logs/righting_" + "_".join(a.replace(":", "-") for a in sys.argv[1:]) + ".json", "w"), indent=1)
