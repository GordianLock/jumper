# jumper.tuck_roll, run 3 -- the accepted forward roll on prototype 3

The third jumper.tuck_roll training run on prototype 3 (CF1-tri30m-rb-v3): Colab,
warp backend, 256 environments, 2500 iterations, trained from revision 03a9ece
(`colab-run.json`): run 2's roll as the demonstration, floor-contact joint torque
estimated at every physics step and charged. **Accepted 2026-10-29** as the forward
roll on prototype 3; that roll is set aside here for now.

- `model_2499.pt` -- the rsl_rl checkpoint (`scripts/play.py --checkpoint`).
- `actor.onnx`, `layout.json`, `README.md` -- the deployment export
  (`scripts/export.py`): the actor with observation normalisation inside it, and the
  contract it was exported with.
- `../../plans/tuck_roll_run3_model_2499.npz` -- its roll on the nominal robot,
  recorded in jumper.tuck_roll's environment by `record_policy_demo.py`, in the
  demonstration format.

Replayed on CPU (native backend), 2026-10-29, nominal and four randomised robots,
11 rolls each: 55 of 55 completed in 0.84-0.88 s; no floor force on the face (stock
shell, base, sensors) on any robot; floor impact 108-185 N; carapace impact 97-164 N;
longest flight 0.06 s or less; worst floor-contact joint torque |J^T f| 5.6-7.7 N*m
(nominal 5.58; run 2: 6.7-7.9; KingKong's reference-free jump: 4.8). The torque comes
from the right claw landing as the body comes down onto its back (about 0.40 s,
5.6-7.0 N*m on RF_J0, mostly through the palm pad) and, on 2 of 5 robots, the left
claw catching the body on its hard palm link at 0.12 s (7.4-7.7 N*m on LF_J1). Over
the nose the claws carry the robot alone; the left claw tucks to under 1 cm off the
centreline, the right stays wide.

The .pt and .onnx files are force-added past `.gitignore` on purpose.
