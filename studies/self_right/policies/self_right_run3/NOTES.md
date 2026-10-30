# jumper.self_right, run 3 -- the stock robot getting up from its back, from rest

The third jumper.self_right training run on the stock robot (`jumper`): Colab, warp
backend, 256 environments, 2500 iterations, trained from revision e9c1ec7
(`colab-run.json`). Run 2's set-up plus the settling hold: the action is held at HOME
while the dropped robot settles, and released only once the body is still (at least
25 steps, then 5 steps under 0.17 rad/s and 0.02 m/s, at most 100). The walking legs
do not yet collide with the body in this revision.

- `model_2499.pt` -- the rsl_rl checkpoint (`scripts/play.py --checkpoint`).
- `actor.onnx`, `layout.json`, `README.md` -- the deployment export.
- `../../plans/self_right_run3_model_2499.npz` -- one episode on the nominal robot,
  recorded by `record_self_right.py` (released at 0.50 s, up at 0.92 s), in the format
  `studies/proto3/render_recording.py` draws (`--model jumper`).

Replayed on CPU (native backend), 2026-10-29, nominal and four randomised robots: 49 of
49 episodes stood up; the body at rest by 0.44 s, released at 0.50 s, standing 0.42-0.44
s later. Nothing from the policy during settling. Worst floor-contact joint torque
|J^T f| 4.2 N*m nominal, 4.2-5.1 randomised, always the right claw's RF_J1 pushing off
at 0.82-0.86 s (run 2: 2.9-3.9 but launched while still moving; run 1: 6.6-8.4;
KingKong's reference-free jump: 4.8). Shell floor force after release 102-112 N at
0.62 s: the shell pressed into the floor as the push-off starts (the body lifts at
0.4-1.2 m/s from there), not a drop -- before it the body only settles 4 mm at 0.56 s.
Limbs 26-72 N, sensors 0 N, longest flight 0.06 s.

Walking legs inside the body (`mj_geomDistance` against the stock hulls, every control
step): 0.07-0.13 cm at the deepest, within the hulls' fill of the real parts' hollows
(run 2: 4.5 cm). Accepted as is, without a run with walking-leg collision.

The .pt and .onnx files are force-added past `.gitignore` on purpose.
