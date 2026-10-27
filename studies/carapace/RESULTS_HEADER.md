# Carapace horn study: results

The **tables below this header are generated** by `compile_results.py` from `logs/`. Do not
edit them by hand; rerun the script after new runs. `results.csv` holds every search, one row
each, with the same numbers.

## What was tested

The robot was dropped belly-up onto a flat floor (jumper.roll's training scene). For each
shell, a cross-entropy search looked for a five-keyframe joint motion (3.5 s, ending at
HOME) that turns it back upright. Three roll types were searched:

- **flip**: head-first, end over end.
- **side**: about the body centreline.
- **diag**: about a shoulder-to-opposite-hip axis.

There were three searches (seeds 1–3) per shell and roll. `HANDOFF.md` has the method in
full; `righting.py` and `gentle.py` are the code.

- **Search reliability** is each search's best plan replayed from 32 perturbed starts
  (tilt ±8°, joint jitter ±0.08 rad). A search "works" at 29/32 or better. A side or
  diagonal plan must also turn about its target axis: alignment 0.9 or better, measured
  from the body's integrated rotation. (`axis_deg` in the CSV is a median of angles and
  can misreport near ±180°; the alignment is the reliable check.)
- **Loads** (`gentle.py`) come from replaying a working plan from another 32 perturbed
  starts, and are taken over the runs that right themselves:
  - **joint torque** is the median, over runs, of each run's worst floor-contact torque on
    any joint (Jᵀf over every floor contact). **p90** is its 90th percentile.
  - **limbs, feet, shell, horn** are median peak floor forces on each part group, in
    newtons. Feet are meant to touch the floor; limbs (shins, arms, claws) and the shell
    are not.

## Horn geometry (stand-in capsules)

All coordinates are in the `base_link` frame, in metres: +x forward toward the head, +y to
the robot's left, +z up out of the shell. The stock shell top is at z = 0.072, and the
stock body spans x −0.112 to +0.094 and y ±0.091. Every horn passed `clearance.py`
against the leg keep-out volume (`leg_keepout_side_voxels.npz`).

| Feature | Radius | From | To |
|---|---|---|---|
| Centre horn, 20° | 0.012 | (−0.030, 0, 0.072) | (0.085, 0, 0.114) |
| Centre horn, 30° | 0.012 | (−0.030, 0, 0.072) | (0.085, 0, 0.140) |
| Front corner, low (`s`) | 0.009 | (0.020, ±0.035, 0.072) | (0.065, ±0.065, 0.097) |
| Front corner, tall (`m`) | 0.009 | (0.020, ±0.035, 0.072) | (0.065, ±0.065, 0.107) |
| Rear corner | 0.009 | (−0.040, ±0.035, 0.072) | (−0.075, ±0.065, 0.097) |
| Tail | 0.010 | (−0.060, 0, 0.072) | (−0.095, 0, 0.097) |

"20°" and "30°" name the angle the robot rests at head-up on that centre horn, not the
horn's own slope. The tip height comes from `righting.build()`:
h = (0.085 + 0.112)·tan(d) − 0.012, with d = 15.43° or 21.99°.

Shell names combine these: `tri<20|30><s|m>` is the centre horn plus front corners; `_rc`,
`_rt` and `_rct` add rear corners, the tail, or both.

## Caveats a designer should know

- **Simulation only.** Servos are modelled as position controllers (kp 10, kd 0.5,
  1.7464 N·m cap) without the speed curve. The numbers compare shells; they are not
  hardware loads.
- **Horn mass is ignored.** `base_link` has an explicit inertia, so the stand-in horns'
  15–30 g change nothing. A real carapace's mass and centre of mass will move the rest
  pose and the rolls, so a printed design must be re-tested with its true inertia.
- **Rest pose matters most.** The pose a shell rests in belly-up decides which rolls are
  easy. The 30° horn plus rear corners has **two** rest poses: 55° sitting on the rear
  corners (most drops) and 16° lying. Its plans work only from the sitting pose.
- **Plans are open-loop keyframes, not policies.** Within one shell, the three searches'
  loads vary widely: gentleness depends on the motion as much as the shape. A trained
  roll policy should be held to the gentle versions.
- **The jumps were measured with a different tool.** `jump_loads.py` measured the high
  jump live in its own environment at its trained 400 Hz physics; `baseline_loads.py`
  measured the reference-free jump. The rolls ran at 5 ms steps. Both jump scripts use
  the same metric as the rolls.
