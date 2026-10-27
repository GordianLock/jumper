"""Joint loads and fidelity for three rolls from standing: jumper4 (styled_roll policy),
the demonstration (momentum_roll feedforward, zero action) and jumper3 (momentum_roll
policy).

Rollouts run in the training environment (native backend, observation noise on,
pushes off). Each control step's full state and joint targets are recorded, then
re-simulated in plain MuJoCo on the same compiled model, at the 5 ms physics step,
with the PD (kp 10, kd 0.5, clipped at 1.7464 N*m) re-applied each substep. That
is what lets us read the constraint forces the native backend does not expose:
per joint, the torque that floor contact and joint stops put on it.
"""
import math
import sys

import mujoco
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, ".")
import tasks
from mjrl.backend.resolve import resolve
from mjrl.backend.select import use_backend

N = 16
STEPS = 120
use_backend(resolve("native", "cpu", num_envs=N, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from tasks.jumper.momentum_roll.mdp.reference import REF_NPZ, RollReference  # noqa: E402

KP, KD, TAU = 10.0, 0.5, 1.7464


def rollout(task, onnx, feedforward):
    cfg = tasks.load_env_cfg(task, play=onnx is not None)
    cfg.scene.num_envs = N
    cfg.events.pop("push_robot", None)
    ev = cfg.events["reset_momentum"].params
    ev["demo_fraction"] = 0.0
    ev["back_fraction"] = 0.0
    cfg.actions["joint_pos"].enabled = feedforward
    cfg.observations["actor"].enable_corruption = onnx is not None
    cfg.terminations.pop("rolled_and_standing")
    cfg.episode_length_s = 30.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _ = env.reset()
    sess = ort.InferenceSession(onnx) if onnx else None
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    from tasks.jumper.momentum_roll.mdp.terminations import standing_done
    cmd = env.command_manager.get_term("roll")
    rec = {"qpos": [], "qvel": [], "target": [], "q": [], "total": [], "z": []}
    done_at = torch.full((N,), -1.0)
    for t in range(STEPS):
        if sess is None:
            a = torch.zeros(N, env.action_manager.total_action_dim)
        else:
            o = obs["actor"].numpy().astype(np.float32)
            name = sess.get_inputs()[0].name
            a = torch.as_tensor(np.concatenate([sess.run(None, {name: o[i:i + 1]})[0]
                                                for i in range(N)]))
        rec["qpos"].append(env.sim.data.qpos.clone().numpy())
        rec["qvel"].append(env.sim.data.qvel.clone().numpy())
        obs, _, _, _, _ = env.step(a)
        rec["target"].append(robot.data.joint_pos_target.clone().numpy())
        rec["q"].append(robot.data.joint_pos[:, term._target_ids].clone().numpy())
        rec["total"].append(cmd.total.clone().numpy())
        rec["z"].append((robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).numpy())
        st = standing_done(env, min_own_turn=1.0)
        done_at = torch.where((done_at < 0) & st, torch.full_like(done_at, (t + 1) * 0.02), done_at)
    out = {k: np.stack(v, axis=1) for k, v in rec.items()}  # [N, T, ...]
    out["done_at"] = done_at.numpy()
    out["model"] = env.sim.mj_model
    out["jadr"] = (robot.indexing.joint_q_adr.numpy(), robot.indexing.joint_v_adr.numpy())
    out["act_ids"] = term._target_ids.numpy()
    out["names"] = list(robot.joint_names)
    out["env_origins"] = env.scene.env_origins.numpy()
    env.close()
    return out


def joint_loads(r):
    """Peak |contact torque| and |joint-stop torque| per joint over the roll, and the
    peak contact force per robot link, by re-simulating each control step."""
    m = r["model"]
    d = mujoco.MjData(m)
    qadr, vadr = r["jadr"]
    nj = len(qadr)
    names = r["names"]
    peak_contact = np.zeros((N, nj))
    peak_limit = np.zeros((N, nj))
    link_force = {}
    n_world = m.nq // (7 + nj) if m.nq != 7 + nj else 1
    assert n_world == 1, "one robot per model expected"
    for i in range(N):
        end = r["done_at"][i] if r["done_at"][i] > 0 else STEPS * 0.02
        for k in range(int(round(end / 0.02))):
            d.qpos[:] = r["qpos"][i, k]
            d.qvel[:] = r["qvel"][i, k]
            tgt = r["target"][i, k]
            for _ in range(4):
                q = d.qpos[qadr]
                qd = d.qvel[vadr]
                tau = np.clip(KP * (tgt - q) - KD * qd, -TAU, TAU)
                d.qfrc_applied[:] = 0.0
                d.qfrc_applied[vadr] = tau
                mujoco.mj_step(m, d)
                nefc = d.nefc
                if nefc == 0:
                    continue
                J = d.efc_J.reshape(nefc, m.nv)
                f = d.efc_force[:nefc]
                types = d.efc_type[:nefc]
                contact = np.isin(types, [mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
                                          mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC,
                                          mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS])
                limit = types == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
                tc = np.abs((J[contact].T @ f[contact])[vadr])
                tl = np.abs((J[limit].T @ f[limit])[vadr])
                peak_contact[i] = np.maximum(peak_contact[i], tc)
                peak_limit[i] = np.maximum(peak_limit[i], tl)
                for c in range(d.ncon):
                    con = d.contact[c]
                    b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
                    other = b2 if b1 == 0 else b1 if b2 == 0 else None
                    if other is None:
                        continue
                    fc = np.zeros(6)
                    mujoco.mj_contactForce(m, d, c, fc)
                    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other).split("/")[-1]
                    link_force.setdefault(nm, np.zeros(N))
                    link_force[nm][i] = max(link_force[nm][i], abs(fc[0]))
    return peak_contact, peak_limit, link_force, names


ref = RollReference(REF_NPZ, "cpu")
runs = {
    "jumper4 (styled)": ("jumper.styled_roll", sys.argv[1], False),
    "demonstration": ("jumper.momentum_roll", None, True),
    "jumper3 (momentum)": ("jumper.momentum_roll", sys.argv[2], False),
}
summary = {}
for label, (task, onnx, ff) in runs.items():
    r = rollout(task, onnx, ff)
    ok = r["done_at"] > 0
    pc, pl, lf, names = joint_loads(r)
    # Fidelity over the demonstration's span, from standing.
    T = min(ref.frames, STEPS)
    jerr = np.sqrt(((r["q"][:, :T] - ref.q[:T].numpy()[None]) ** 2).mean(axis=(1, 2)))
    terr = np.degrees(np.abs(r["total"][:, :T] - ref.total[:T].numpy()[None]).mean(axis=1))
    zerr = np.abs(r["z"][:, :T] - ref.root_pos[:T, 2].numpy()[None]).mean(axis=1) * 100
    summary[label] = (pc, pl, names)
    leg_links = {k: v for k, v in lf.items() if not k.startswith("base")}
    worst_links = sorted(leg_links.items(), key=lambda kv: -np.median(kv[1]))[:5]
    print(f"\n== {label}: rolled and stood {int(ok.sum())}/{N}, median {np.median(r['done_at'][ok]) if ok.any() else float('nan'):.2f} s")
    print(f"   vs demonstration: joint RMS {np.degrees(np.median(jerr)):.0f} deg, turn {np.median(terr):.0f} deg, height {np.median(zerr):.1f} cm (median over envs)")
    print(f"   peak floor-contact torque on any joint: median {np.median(pc.max(axis=1)):.1f} N*m, worst {pc.max():.1f}  (servo stall 1.75)")
    print(f"   joints over 1.75 N*m from contact: median {np.median((pc > TAU).sum(axis=1)):.0f} of {len(names)}")
    print(f"   peak joint-stop torque: median {np.median(pl.max(axis=1)):.1f} N*m, worst {pl.max():.1f}")
    print("   hardest-hit links (median peak normal force, N):",
          ", ".join(f"{k} {np.median(v):.0f}" for k, v in worst_links))
    jw = np.median(pc, axis=0)
    top = np.argsort(-jw)[:6]
    print("   joints taking most contact torque (median peak, N*m):",
          ", ".join(f"{names[j].replace('_joint', '')} {jw[j]:.1f}" for j in top))
