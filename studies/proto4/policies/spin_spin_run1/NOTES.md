# jumper.spin_spin, run 1 -- the spin roll on carapace prototype 4, free to pivot

The first jumper.spin_spin training run, on carapace prototype 4 (`CF1-tri30m-rb-v4`):
Colab, warp backend, 256 environments, 2500 iterations, trained from revision 847199c
(`colab-run.json`; the episode ends at the finish -- the settle-at-HOME ending had been
reverted). jumper.spin_drift without the charge on pivoting: the spin-roll
demonstration fed forward and used as a fading prior, claws and walking legs colliding
with the body and carapace, floor-contact joint torque charged (free to 4 N*m).

- `model_2499.pt` -- the rsl_rl checkpoint (`scripts/play.py --checkpoint`).
- `actor.onnx`, `layout.json`, `README.md` -- the deployment export.

Replayed on CPU (native backend), 2026-10-30, nominal and four randomised robots, 8
rolls each: 40 of 40 rolled and stood, 1.14-1.18 s. Carapace impact 0 N on every robot
(spin_drift run 1: 118-150), floor impact 70-129 N, stock shell 0-28 N, face 0 N, arms
47-88 N, walking legs 30-61 N, longest flight 0.04-0.08 s. Worst floor-contact joint
torque |J^T f| 5.7 N*m nominal, 5.8-6.6 randomised (spin_drift run 1: 3.6-6.2;
KingKong's reference-free jump: 4.8), always a claw's palm on the floor -- the right at
about 0.3 s tipping over, or the left at about 0.77 s coming down. The axis is 23-27 deg
off the demonstration's; it pivots 17-24 deg about the vertical per roll. The right
claw's finger presses 0.5-0.7 cm into the shell or carapace at its deepest: contact
under load (the claws collide with the body here), not passing through.

The .pt and .onnx files are force-added past `.gitignore` on purpose.
