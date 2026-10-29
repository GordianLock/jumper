# jumper.tuck_roll, run 2 -- backup of the trained policy

The second jumper.tuck_roll training run on prototype 3 (CF1-tri30m-rb-v3): Colab,
warp backend, 256 environments, 2500 iterations, trained from revision 46d0de9
(`colab-run.json`). Kept here so the roll survives later changes to the task.

- `model_2499.pt` -- the rsl_rl checkpoint (`scripts/play.py --checkpoint`).
- `actor.onnx`, `layout.json`, `README.md` -- the deployment export
  (`scripts/export.py`): the actor with observation normalisation inside it, and the
  contract it was exported with.
- `../../plans/tuck_roll_run2_model_2499.npz` -- its roll on the nominal robot,
  recorded in jumper.tuck_roll's environment by `record_policy_demo.py`, in the
  demonstration format.

Replayed on CPU (native backend), 2026-10-29, nominal and four randomised robots,
11 rolls each: 55 of 55 completed in 0.84-0.88 s; no floor force on the face (stock
shell, base, sensors) on any robot; floor impact 85-137 N; carapace impact 71-121 N;
longest flight 0.04 s or less; worst floor-contact joint torque |J^T f| 6.7-7.9 N*m
(KingKong's reference-free jump: 4.8), from the left palm landing at 0.12-0.14 s or
the right claw's finger striking at 0.42 s. The left claw tucks to 3-4 cm off the
centreline while the nose is down; the right stays 16 cm out.

The .pt and .onnx files are force-added past `.gitignore` on purpose.
