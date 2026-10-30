# Shoulder roll -- prototype 3 (jumper.diag_roll, run 1)

Trained as `jumper.diag_roll` on prototype 3 (CF1-tri30m-rb-v3): Colab, warp backend,
256 environments, 2500 iterations, revision bf536b3 (`colab-run.json`), from the
diagonal demonstration (axis 47 deg off the centreline). The trained policy drifted
off the diagonal to a roll about an axis 81 deg off the centreline -- nearly end over
end, going over a front shoulder -- once the demonstration's guide faded: the task
measures the turn about the demonstration's axis, and a roll about a nearby axis still
completes it. **Kept as is, 2026-10-29, named the shoulder roll**: a second forward
roll beside the tuck roll (`../tuck_roll_run3/`), slower and gentler.

- `model_2499.pt` -- the rsl_rl checkpoint (`scripts/play.py --task jumper.diag_roll
  --checkpoint`).
- `actor.onnx`, `layout.json`, `README.md` -- the deployment export.
- `../../plans/shoulder_roll_model_2499.npz` -- its roll on the nominal robot,
  recorded in jumper.diag_roll's environment by `record_policy_demo.py` (axis 81 deg,
  376 deg, standing after 1.16 s), in the demonstration format
  (`render_recording.py --model CF1-tri30m-rb-v3` draws it).

Replayed on CPU (native backend), 2026-10-29, nominal and four randomised robots, 8
rolls each: 40 of 40 completed in 1.12-1.16 s; no floor force on the face on any
robot; floor impact 78-145 N; carapace impact 0-125 N; stock shell 20-44 N; longest
flight 0.04 s or less; worst floor-contact joint torque |J^T f| 3.4-4.2 N*m on four
robots (under KingKong's reference-free jump, 4.8), 6.3 on one (the right claw's
finger striking at 0.90 s). The tuck roll, for comparison: 5.6-7.7 N*m, 0.84-0.88 s.

The .pt and .onnx files are force-added past `.gitignore` on purpose.
