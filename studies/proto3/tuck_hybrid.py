"""Build jumper.tuck_roll's hybrid demonstration.

    python studies/proto3/tuck_hybrid.py

Source: `plans/policy_roll_model_2499.npz`, jumper.horn_roll's first trained policy
(Colab, 2500 iterations) rolling the nominal prototype 3, recorded by
`record_policy_demo.py` (0.84 s to standing, floor impact 99 N).

Everything in that recording -- body motion, turn, joint positions, the states
mid-demonstration spawns start from -- is kept as it is: it is a real roll that
completes. Only the **arm joint commands** (`q_cmd`, fed forward into the action
while the guide lasts) are changed: blended to `TUCK` from step `START` to `FULL`,
held to `END`, and blended back over `RAMP` steps.

`TUCK` puts both palms 15 cm ahead of the base origin and 5 cm either side of the
centreline, forearms 9 cm ahead of the face, every arm part at least 3 mm clear of
the shell, sensors and carapace at HOME's leg pose. Found by bounded least squares on
the left arm (palm target, clearance, front-most arm point within 19 cm) and mirrored
to the right (J0, J2, J3 change sign, as in HOME), 2026-10-29.

Why not a recording of the tuck itself: played back open-loop on the nominal robot in
jumper.tuck_roll's environment (2026-10-29), the tuck from 0.00 s never tipped the
robot (15 deg), its arms' push-off gone; from 0.14 s it went over the nose on its
arms but reached only 203 deg and settled on its back after the front carapace landed
at 127 N -- the right arm, pinned under the body, could not tuck, and the overhead
swing that levers this roll through its back half was gone. So the arms are fed the
tuck and training has to find the rest.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "plans" / "policy_roll_model_2499.npz"
OUT = ROOT / "tasks" / "jumper" / "tuck_roll" / "ref" / "tuck_roll.npz"

ARM_JOINTS = ("LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint",
              "RF_J0_joint", "RF_J1_joint", "RF_J2_joint", "RF_J3_joint")
TUCK = np.array([-1.4376, 1.26, 0.5645, -1.89, 1.4376, 1.26, -0.5645, 1.89])
START, FULL, END, RAMP = 7, 10, 17, 3          # control steps (20 ms)

src = dict(np.load(SRC))
names = [str(n) for n in src["joint_names"]]
arm = [names.index(j) for j in ARM_JOINTS]
n = len(src["q_cmd"])
w = np.zeros(n)
for k in range(n):
    if FULL <= k <= END:
        w[k] = 1.0
    elif START <= k < FULL:
        w[k] = (k - START + 1) / (FULL - START + 1)
    elif END < k <= END + RAMP:
        w[k] = 1.0 - (k - END) / (RAMP + 1)
q_cmd = src["q_cmd"].copy()
q_cmd[:, arm] = (1 - w[:, None]) * q_cmd[:, arm] + w[:, None] * TUCK[None]
out = {**src, "q_cmd": q_cmd.astype(np.float32),
       "source": f"{SRC.name}, arm commands tucked over steps {START}-{END + RAMP}"}
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, **out)
print(f"wrote {OUT.relative_to(ROOT)}: {n} steps, arm commands tucked on steps "
      f"{np.nonzero(w)[0].min()}-{np.nonzero(w)[0].max()}")
