# Lay Down -- stock robot (jumper.lie_down, run 2)

Trained as `jumper.lie_down` on the stock robot (`jumper`): Colab, warp backend, 256
environments, 2500 iterations, revision c808f69 (`colab-run.json`), with the finish
loosened to 0.3 rad, 0.8 rad/s, 0.06 m/s, held 10 steps, and the searched plan
(`../../plans/lie_down_s1.npy`, `ref/lie_down.npz`) as the demonstration, fading out
from iteration 300 to 1200. **Kept 2026-10-30, named Lay Down**; no further training
round is planned for this behaviour.

It is not the slow, soft lie-down that was asked for. While the demonstration guided
it no episode finished (all ran to the 6 s time-out); once the guide faded the policy
found a fast flip -- weight thrown back, over the rear legs onto the shell -- and from
iteration 1800 every episode ended on the finish.

- `model_2499.pt` -- the rsl_rl checkpoint (`scripts/play.py --task jumper.lie_down
  --checkpoint`).
- `actor.onnx`, `layout.json`, `README.md` -- the deployment export.

Replayed on CPU (native backend), 2026-10-30, nominal and four randomised stock robots,
9 each: 45 of 45 finished on the back at HOME, 0.52-0.60 s after the release; shell
impact 232-263 N (the floor impact is the same contact); floor-contact joint torque
|J^T f| 2.4-7.3 N*m (7.3 on the nominal robot); body rotation 22.9-24.2 rad/s; limbs
7-73 N; sensors 0 N; airborne 0.10-0.12 s. For comparison, the searched plan: 3.66 s,
shell 83 N, 4.3 N*m, 7.6 rad/s, never airborne. Run 1 (revision 76e61e6, never
finished in training under the strict finish), replayed the same way against today's
finish: 43 of 43, 0.60-0.72 s, shell 235-302 N, 3.0-5.6 N*m, 17.1-18.2 rad/s,
airborne 0.06-0.08 s -- slower over but the hardest shell hits, so run 2 was kept.

The .pt and .onnx files are force-added past `.gitignore` on purpose.
