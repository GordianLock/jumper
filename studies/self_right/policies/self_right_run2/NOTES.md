# jumper.self_right, run 2 -- the stock robot getting up from its back

The second jumper.self_right training run on the stock robot (`jumper`): Colab, warp
backend, 256 environments, 2500 iterations, trained from revision ae14627
(`colab-run.json`): the claws collide with the body, and floor-contact joint torque is
estimated at every physics step and charged (free to 4 N*m).

- `model_2499.pt` -- the rsl_rl checkpoint (`scripts/play.py --checkpoint`).
- `actor.onnx`, `layout.json`, `README.md` -- the deployment export.
- `../../plans/self_right_run2_model_2499.npz` -- one episode on the nominal robot,
  recorded by `record_self_right.py` (on its back, 0.5 s settling, up at 0.74 s), in
  the format `studies/proto3/render_recording.py` draws (`--model jumper`).

Replayed on CPU (native backend), 2026-10-29, nominal and four randomised robots: 65 of
65 episodes stood up, about 0.25 s of movement after the 0.5 s settling; floor impact
31-76 N; nothing on the shell or the sensors; longest flight 0.06 s or less. Worst
floor-contact joint torque |J^T f| during the righting move 2.9-3.9 N*m (median per
robot 2.6-3.3; run 1: 6.6-8.4, the right claw pushing off; KingKong's reference-free
jump: 4.8). The highest single reading, 4.1-4.3 N*m at 0.22 s, falls in the settling
window: the middle-right calf meeting the floor as the simulation sets the robot down
on its back, which a robot already lying at rest would not see.

The .pt and .onnx files are force-added past `.gitignore` on purpose.

**Found later (2026-10-29): its walking legs pass through the body.** Trained before
the walking legs collided with it, the policy puts its left-rear calf up to 4.5 cm into
the base and its right-middle foot up to 3.4 cm (Colab replay's joint angles against
the stock collision hulls), so as trained it is not fully valid on the robot. It also
pumps in the settling window (the body turns 180-350 deg/s there). Both are fixed in
jumper.self_right from run 4 on.
