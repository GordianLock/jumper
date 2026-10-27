"""Measure the stock jump's floor-contact loads the way studies/carapace/gentle.py measures rolls.

Runs scripts/play.py for jumper.jump with the trained policy, on the native backend, with
NativeSimulation.step wrapped so that after every physics step each environment's own
MjData (refreshed by mj_forward from that environment's state) is inspected:
- worst floor-contact torque on any actuated joint, |J^T f| summed over floor contacts
- peak summed normal force per part group (limbs / shell / sensors / feet)
Each environment is one jump attempt. Summary: medians over environments, like gentle.py.

usage: python jump_loads.py <repo root> <num_envs> <control steps> [physics Hz, default 1000]
"""
import atexit, os, runpy, sys
import numpy as np, mujoco

root, n_envs, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
hz = sys.argv[4] if len(sys.argv) > 4 else "1000"
os.chdir(root)
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "scripts"))   # play.py imports its sibling _cli
from mjrl.backend import native_sim as NS
from tasks.jumper.common import constants as C

state = {}


def setup(sim):
    m = sim._mj_model
    floor = [g for g in range(m.ngeom) if not (m.geom(g).name or "").startswith("robot/")]
    print("floor geoms:", [m.geom(g).name for g in floor], flush=True)
    feet = {m.geom(n).id for n in (f"robot/{g}" for g in C.FOOT_GEOMS)
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0}
    grp = {}
    for g in range(m.ngeom):
        n = (m.geom(g).name or "").split("/")[-1]
        grp[g] = ("feet" if g in feet else
                  "shell" if n.startswith(("base_link", "upper_shell")) else
                  "sensors" if n.startswith(("display", "tof", "camera")) else "limbs")
    act = [m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)]
    names = [m.joint(m.actuator_trnid[a, 0]).name.split("/")[-1] for a in range(m.nu)]
    state.update(floor=set(floor), grp=grp, dofs=act, names=names,
                 jt=np.zeros((sim.num_envs, len(act))),
                 pk={k: np.zeros(sim.num_envs) for k in ("limbs", "shell", "sensors", "feet")},
                 f6=np.zeros(6), jac=np.zeros((3, m.nv)), nsteps=0)


def measure(sim):
    m = sim._mj_model; s = state
    for i in range(sim.num_envs):
        d = sim._datas[i]
        per = dict.fromkeys(s["pk"], 0.0); tau = np.zeros(m.nv)
        for c in range(d.ncon):
            con = d.contact[c]
            if con.geom1 in s["floor"]: g, sign = con.geom2, 1.0
            elif con.geom2 in s["floor"]: g, sign = con.geom1, -1.0
            else: continue
            mujoco.mj_contactForce(m, d, c, s["f6"])
            fw = sign * (con.frame.reshape(3, 3).T @ s["f6"][:3])
            per[s["grp"][g]] += s["f6"][0]
            mujoco.mj_jac(m, d, s["jac"], None, con.pos, m.geom_bodyid[g]); tau += s["jac"].T @ fw
        for k in per: s["pk"][k][i] = max(s["pk"][k][i], per[k])
        s["jt"][i] = np.maximum(s["jt"][i], np.abs(tau[s["dofs"]]))
    s["nsteps"] += 1


orig = NS.NativeSimulation.step


def step(self, nstep=1):
    if not state: setup(self)
    for _ in range(nstep):          # same physics as one rollout of nstep, sampled per step
        orig(self, 1)
        measure(self)


NS.NativeSimulation.step = step


@atexit.register
def report():
    s = state
    if not s: print("no steps measured"); return
    worst = s["jt"].max(1); w = s["names"]
    print(f"jump @{hz} Hz physics  {len(worst)} attempts, {s['nsteps']} physics steps | joint torque median "
          f"{np.median(worst):.1f} p90 {np.percentile(worst, 90):.1f} max {worst.max():.1f} N*m "
          f"(worst joint, most often: {max(set(w[j] for j in s['jt'].argmax(1)), key=list(w[j] for j in s['jt'].argmax(1)).count)}) "
          f"| median peaks N: " + " ".join(f"{k} {np.median(v):.0f}" for k, v in s["pk"].items()),
          flush=True)


sys.argv = ["scripts/play.py", "--task", "jumper.jump", "--backend", "native", "--device", "cpu",
            "--checkpoint", "tasks/jumper/jump/out/example/model_3100.pt",
            "--num_envs", str(n_envs), "--headless", "--steps", str(steps), "--physics-hz", hz]
runpy.run_path("scripts/play.py", run_name="__main__")
