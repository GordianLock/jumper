# jumper.spin_drift, run 1 -- the spin roll on carapace prototype 4

The first jumper.spin_drift training run, on carapace prototype 4 (`CF1-tri30m-rb-v4`):
Colab, warp backend, 256 environments, 2500 iterations, trained from revision 4d6dfc0
(`colab-run.json`). The spin-roll demonstration fed forward and used as a fading prior;
after it fades the roll axis is free to drift. Claws and walking legs collide with the
body and carapace; floor-contact joint torque charged (free to 4 N*m).

- `model_2499.pt` -- the rsl_rl checkpoint (`scripts/play.py --checkpoint`).
- `actor.onnx`, `layout.json`, `README.md` -- the deployment export.
- `../../../proto3/plans/spin_drift_run1_model_2499.npz` -- the nominal robot's roll,
  recorded by `studies/proto3/record_policy_demo.py` (`MODEL=CF1-tri30m-rb-v4`).

Replayed on CPU (native backend), 2026-10-30, nominal and four randomised robots, 7
rolls each: 35 of 35 rolled and stood, 1.26-1.30 s. Worst floor-contact joint torque
|J^T f| 4.2 N*m nominal, 3.6-6.2 randomised, nearly always the left palm landing at
0.84-0.86 s (LF_J2) (KingKong's reference-free jump: 4.8). Carapace impact 118-150 N,
floor impact 123-150 N; stock shell, face and sensors 0 N; arms 53-80 N, walking legs
35-59 N; longest flight 0.04-0.10 s. The axis drifted 20-24 deg from the
demonstration's; on the nominal robot the roll is about 73 deg from the centreline.
Deepest leg inside the body 0.2-0.3 cm (right palm in the upper shell), within the
hulls' fill.

Known: training ends the episode on standing, so nothing after it is trained; in the
recording, which runs 0.3 s past it, the robot hops once (0.12 s in the air).

The .pt and .onnx files are force-added past `.gitignore` on purpose.
