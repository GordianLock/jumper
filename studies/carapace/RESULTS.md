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

## Which rolls each shell can do

Searches that found a working plan, out of 3, with the gentlest working plan's median joint torque in N·m. A dash means not searched. `*` marks the two-pose shells, where "working" means it works from every start in the 55° sitting pose.

| Shell | Description | Rests head-up / height | Flip | Side | Diagonal |
|---|---|---|---|---|---|
| `stock` | no horns | 8° / 6.1 cm | 1/3 | — | — |
| `horn20` | 20 deg centre horn | 20° / 8.9 cm | 3/3 (4.2) | 0/3 | 1/3 (8.0) |
| `horn30` | 30 deg centre horn | 30° / 9.0 cm | 3/3 (4.7) | 0/3 | 3/3 (5.5) |
| `tri20s` | 20 deg centre + low (2.5 cm) front corners | 20° / 8.9 cm | 1/3 (9.1) | 0/3 | 2/3 (5.1) |
| `tri20m` | 20 deg centre + tall (3.5 cm) front corners | 23° / 8.9 cm | 3/3 (4.3) | 0/3 | 0/3 |
| `rt20` | 20 deg centre + tail | 5° / 10.3 cm | 0/3 | 0/3 | 0/3 |
| `rc20` | 20 deg centre + rear corners | 7° / 11.5 cm | 3/3 (4.3) | 0/3 | 3/3 (5.7) |
| `rct20` | 20 deg centre + rear corners + tail | 6° / 11.6 cm | 2/3 (4.2) | 1/3 (5.6) | 1/3 (6.2) |
| `tri20s_rt` | 20 deg centre + low front corners + tail | 11° / 11.1 cm | 0/3 | 0/3 | 0/3 |
| `tri20s_rc` | 20 deg centre + low front corners + rear corners | 7° / 11.5 cm | 3/3 (4.3) | 2/3 (4.1) | 1/3 (5.1) |
| `tri20s_rct` | 20 deg centre + low front corners + rear corners + tail | 6° / 11.6 cm | 1/3 (5.1) | 1/3 (5.5) | 1/3 (6.5) |
| `tri30s` | 30 deg centre + low (2.5 cm) front corners | 30° / 9.0 cm | 1/3 (5.1) | 3/3 (5.5) | 3/3 (6.6) |
| `tri30m` | 30 deg centre + tall (3.5 cm) front corners | 30° / 9.0 cm | 3/3 (3.5) | 1/3 (6.5) | 2/3 (6.9) |
| `tri30s_rc` | 30 deg centre + low front corners + rear corners | 55° / 12.6 cm (or 16°) | 3/3* (3.7) | 3/3* (6.5) | 3/3* (5.6) |
| `tri30m_rc` | 30 deg centre + tall front corners + rear corners | 55° / 12.6 cm (or 16°) | 3/3* (3.3) | 0/3* | 3/3* (5.3) |

## Against the stock jumps

| Stock motion | Attempts | Joint torque (p90) | Limbs | Feet | Shell |
|---|---|---|---|---|---|
| High jump `jumper.jump` (the shipped jump) | 32 | 5.9 (7.0) | 76 | 175 | 0 |
| Reference-free jump `jumper.ref_free_jump` | 8 | 4.8 (worst 5.6) | — | — | — |

### Working plans that beat the high jump on all four measures

Joint torque, p90, limbs and feet all below the high jump's.

| Roll | Shell | Plan | Starts | Joint torque (p90) | Limbs | Feet | Shell | Horn |
|---|---|---|---|---|---|---|---|---|
| flip | `tri30m_rc` | `plan_tri30m_rc_s3` | 30/32 | 3.3 (3.4) | 50 | 35 | 36 | 42 |
| flip | `tri30m` | `plan_tri30m_s3` | 32/32 | 3.5 (3.6) | 67 | 89 | 37 | 28 |
| flip | `tri30m_rc` | `plan_tri30m_rc_s1` | 30/32 | 3.7 (3.7) | 61 | 44 | 36 | 47 |
| flip | `tri30s_rc` | `plan_tri30s_rc_s1` | 30/32 | 3.7 (3.7) | 61 | 44 | 36 | 47 |
| flip | `horn20` | `plan_horn20_s3` | 32/32 | 4.2 (4.3) | 55 | 64 | 0 | 36 |
| flip | `rct20` | `plan_rct20_s2` | 32/32 | 4.2 (4.2) | 66 | 56 | 34 | 50 |
| flip | `tri30s_rc` | `plan_tri30s_rc_s3` | 30/32 | 4.2 (4.3) | 36 | 32 | 30 | 65 |
| flip | `rc20` | `plan_rc20_s1` | 32/32 | 4.3 (4.7) | 57 | 77 | 0 | 69 |
| flip | `tri20m` | `plan_tri20m_s3` | 32/32 | 4.3 (4.5) | 44 | 49 | 27 | 44 |
| flip | `tri30m_rc` | `plan_tri30m_rc_s2` | 30/32 | 4.5 (4.5) | 57 | 37 | 42 | 42 |
| flip | `horn30` | `plan_horn30_s2` | 31/32 | 4.7 (5.2) | 73 | 88 | 24 | 37 |
| side | `tri20s_rc` | `plan_tri20s_rc_side_s1` | 32/32 | 4.7 (4.7) | 56 | 65 | 0 | 155 |
| flip | `horn20` | `plan_horn20_s2` | 32/32 | 4.8 (5.1) | 47 | 49 | 0 | 35 |
| flip | `tri30m` | `plan_tri30m_s2` | 32/32 | 4.9 (5.1) | 56 | 38 | 29 | 32 |
| flip | `tri20m` | `plan_tri20m_s2` | 32/32 | 5.0 (5.5) | 66 | 82 | 0 | 44 |
| flip | `tri30s_rc` | `plan_tri30s_rc_s2` | 30/32 | 5.0 (5.0) | 46 | 35 | 33 | 33 |
| diag | `tri20s_rc` | `plan_tri20s_rc_diag_s2` | 32/32 | 5.1 (5.1) | 74 | 88 | 0 | 52 |
| diag | `tri30m_rc` | `plan_tri30m_rc_diag_s2` | 30/32 | 5.3 (5.3) | 64 | 67 | 38 | 40 |
| side | `tri30s` | `plan_tri30s_side_s2` | 32/32 | 5.5 (6.0) | 73 | 70 | 7 | 51 |
| side | `rct20` | `plan_rct20_side_s1` | 32/32 | 5.6 (5.8) | 41 | 65 | 95 | 48 |
| flip | `tri20s_rc` | `plan_tri20s_rc_s2` | 32/32 | 5.6 (5.8) | 64 | 93 | 0 | 73 |
| diag | `tri30s_rc` | `plan_tri30s_rc_diag_s3` | 30/32 | 5.6 (5.6) | 67 | 100 | 0 | 52 |

- flip: 16 plan(s); gentlest `plan_tri30m_rc_s3` at 3.3 N·m

- side: 3 plan(s); gentlest `plan_tri20s_rc_side_s1` at 4.7 N·m

- diag: 3 plan(s); gentlest `plan_tri20s_rc_diag_s2` at 5.1 N·m

### Working plans at or below the reference-free jump's joint torque (4.8 N·m)

| Roll | Shell | Plan | Starts | Joint torque (p90) | Limbs | Feet | Shell | Horn |
|---|---|---|---|---|---|---|---|---|
| flip | `tri30m_rc` | `plan_tri30m_rc_s3` | 30/32 | 3.3 (3.4) | 50 | 35 | 36 | 42 |
| flip | `tri30m` | `plan_tri30m_s3` | 32/32 | 3.5 (3.6) | 67 | 89 | 37 | 28 |
| flip | `tri30m` | `plan_tri30m_s1` | 32/32 | 3.6 (3.7) | 85 | 91 | 39 | 41 |
| flip | `tri30m_rc` | `plan_tri30m_rc_s1` | 30/32 | 3.7 (3.7) | 61 | 44 | 36 | 47 |
| flip | `tri30s_rc` | `plan_tri30s_rc_s1` | 30/32 | 3.7 (3.7) | 61 | 44 | 36 | 47 |
| side | `tri20s_rc` | `plan_tri20s_rc_side_s2` | 32/32 | 4.1 (4.4) | 89 | 114 | 0 | 55 |
| flip | `horn20` | `plan_horn20_s3` | 32/32 | 4.2 (4.3) | 55 | 64 | 0 | 36 |
| flip | `rct20` | `plan_rct20_s2` | 32/32 | 4.2 (4.2) | 66 | 56 | 34 | 50 |
| flip | `tri30s_rc` | `plan_tri30s_rc_s3` | 30/32 | 4.2 (4.3) | 36 | 32 | 30 | 65 |
| flip | `rc20` | `plan_rc20_s1` | 32/32 | 4.3 (4.7) | 57 | 77 | 0 | 69 |
| flip | `tri20m` | `plan_tri20m_s3` | 32/32 | 4.3 (4.5) | 44 | 49 | 27 | 44 |
| flip | `tri20s_rc` | `plan_tri20s_rc_s1` | 31/32 | 4.3 (4.3) | 76 | 90 | 0 | 55 |
| flip | `tri30m_rc` | `plan_tri30m_rc_s2` | 30/32 | 4.5 (4.5) | 57 | 37 | 42 | 42 |
| flip | `horn30` | `plan_horn30_s2` | 31/32 | 4.7 (5.2) | 73 | 88 | 24 | 37 |
| side | `tri20s_rc` | `plan_tri20s_rc_side_s1` | 32/32 | 4.7 (4.7) | 56 | 65 | 0 | 155 |
| flip | `horn20` | `plan_horn20_s2` | 32/32 | 4.8 (5.1) | 47 | 49 | 0 | 35 |

## Every load-measured working plan, gentlest first

| Roll | Shell | Plan | Starts | Joint torque (p90) | Limbs | Feet | Shell | Horn |
|---|---|---|---|---|---|---|---|---|
| flip | `tri30m_rc` | `plan_tri30m_rc_s3` | 30/32 | 3.3 (3.4) | 50 | 35 | 36 | 42 |
| flip | `tri30m` | `plan_tri30m_s3` | 32/32 | 3.5 (3.6) | 67 | 89 | 37 | 28 |
| flip | `tri30m` | `plan_tri30m_s1` | 32/32 | 3.6 (3.7) | 85 | 91 | 39 | 41 |
| flip | `tri30m_rc` | `plan_tri30m_rc_s1` | 30/32 | 3.7 (3.7) | 61 | 44 | 36 | 47 |
| flip | `tri30s_rc` | `plan_tri30s_rc_s1` | 30/32 | 3.7 (3.7) | 61 | 44 | 36 | 47 |
| side | `tri20s_rc` | `plan_tri20s_rc_side_s2` | 32/32 | 4.1 (4.4) | 89 | 114 | 0 | 55 |
| flip | `horn20` | `plan_horn20_s3` | 32/32 | 4.2 (4.3) | 55 | 64 | 0 | 36 |
| flip | `rct20` | `plan_rct20_s2` | 32/32 | 4.2 (4.2) | 66 | 56 | 34 | 50 |
| flip | `tri30s_rc` | `plan_tri30s_rc_s3` | 30/32 | 4.2 (4.3) | 36 | 32 | 30 | 65 |
| flip | `rc20` | `plan_rc20_s1` | 32/32 | 4.3 (4.7) | 57 | 77 | 0 | 69 |
| flip | `tri20m` | `plan_tri20m_s3` | 32/32 | 4.3 (4.5) | 44 | 49 | 27 | 44 |
| flip | `tri20s_rc` | `plan_tri20s_rc_s1` | 31/32 | 4.3 (4.3) | 76 | 90 | 0 | 55 |
| flip | `tri30m_rc` | `plan_tri30m_rc_s2` | 30/32 | 4.5 (4.5) | 57 | 37 | 42 | 42 |
| flip | `horn30` | `plan_horn30_s2` | 31/32 | 4.7 (5.2) | 73 | 88 | 24 | 37 |
| side | `tri20s_rc` | `plan_tri20s_rc_side_s1` | 32/32 | 4.7 (4.7) | 56 | 65 | 0 | 155 |
| flip | `horn20` | `plan_horn20_s2` | 32/32 | 4.8 (5.1) | 47 | 49 | 0 | 35 |
| flip | `tri30m` | `plan_tri30m_s2` | 32/32 | 4.9 (5.1) | 56 | 38 | 29 | 32 |
| flip | `tri20m` | `plan_tri20m_s2` | 32/32 | 5.0 (5.5) | 66 | 82 | 0 | 44 |
| flip | `tri30s_rc` | `plan_tri30s_rc_s2` | 30/32 | 5.0 (5.0) | 46 | 35 | 33 | 33 |
| diag | `tri20s` | `plan_tri20s_diag_s1` | 31/32 | 5.1 (5.1) | 101 | 69 | 42 | 123 |
| diag | `tri20s_rc` | `plan_tri20s_rc_diag_s2` | 32/32 | 5.1 (5.1) | 74 | 88 | 0 | 52 |
| flip | `tri20s_rct` | `plan_tri20s_rct_s2` | 32/32 | 5.1 (5.5) | 85 | 107 | 0 | 100 |
| flip | `tri30s` | `plan_tri30s_s3` | 32/32 | 5.1 (5.2) | 90 | 80 | 7 | 35 |
| diag | `tri30m_rc` | `plan_tri30m_rc_diag_s2` | 30/32 | 5.3 (5.3) | 64 | 67 | 38 | 40 |
| diag | `horn30` | `plan_horn30_diag_s2` | 32/32 | 5.5 (6.0) | 129 | 88 | 32 | 35 |
| side | `tri20s_rct` | `plan_tri20s_rct_side_s2` | 32/32 | 5.5 (5.9) | 112 | 108 | 0 | 88 |
| side | `tri30s` | `plan_tri30s_side_s2` | 32/32 | 5.5 (6.0) | 73 | 70 | 7 | 51 |
| side | `rct20` | `plan_rct20_side_s1` | 32/32 | 5.6 (5.8) | 41 | 65 | 95 | 48 |
| flip | `tri20s_rc` | `plan_tri20s_rc_s2` | 32/32 | 5.6 (5.8) | 64 | 93 | 0 | 73 |
| diag | `tri30s_rc` | `plan_tri30s_rc_diag_s3` | 30/32 | 5.6 (5.6) | 67 | 100 | 0 | 52 |
| diag | `rc20` | `plan_rc20_diag_s1` | 32/32 | 5.7 (6.2) | 83 | 73 | 0 | 99 |
| diag | `tri20s` | `plan_tri20s_diag_s2` | 32/32 | 5.7 (6.0) | 78 | 75 | 0 | 46 |
| diag | `horn30` | `plan_horn30_diag_s3` | 32/32 | 5.9 (6.0) | 96 | 45 | 41 | 32 |
| diag | `rct20` | `plan_rct20_diag_s3` | 32/32 | 6.2 (6.4) | 70 | 48 | 95 | 77 |
| diag | `rc20` | `plan_rc20_diag_s3` | 31/32 | 6.3 (6.3) | 116 | 101 | 0 | 85 |
| diag | `tri20s_rct` | `plan_tri20s_rct_diag_s1` | 32/32 | 6.5 (7.2) | 83 | 54 | 0 | 49 |
| side | `tri30m` | `plan_tri30m_side_s2` | 32/32 | 6.5 (6.6) | 67 | 44 | 6 | 77 |
| side | `tri30s_rc` | `plan_tri30s_rc_side_s3` | 30/32 | 6.5 (6.6) | 131 | 54 | 33 | 60 |
| diag | `tri30m_rc` | `plan_tri30m_rc_diag_s3` | 30/32 | 6.6 (6.7) | 68 | 83 | 111 | 44 |
| diag | `tri30s` | `plan_tri30s_diag_s3` | 32/32 | 6.6 (7.0) | 119 | 53 | 40 | 66 |
| flip | `rc20` | `plan_rc20_s3` | 32/32 | 6.7 (6.7) | 96 | 57 | 40 | 97 |
| diag | `tri30s` | `plan_tri30s_diag_s2` | 32/32 | 6.7 (6.9) | 108 | 96 | 28 | 50 |
| flip | `tri20s_rc` | `plan_tri20s_rc_s3` | 32/32 | 6.9 (7.1) | 73 | 86 | 0 | 49 |
| diag | `tri30m` | `plan_tri30m_diag_s2` | 32/32 | 6.9 (7.3) | 85 | 78 | 36 | 37 |
| diag | `tri30s_rc` | `plan_tri30s_rc_diag_s2` | 30/32 | 7.0 (7.0) | 58 | 95 | 34 | 39 |
| side | `tri30s_rc` | `plan_tri30s_rc_side_s2` | 30/32 | 7.0 (7.1) | 124 | 42 | 37 | 37 |
| side | `tri30s_rc` | `plan_tri30s_rc_side_s1` | 30/32 | 7.2 (7.3) | 103 | 52 | 29 | 36 |
| diag | `tri30m_rc` | `plan_tri30m_rc_diag_s1` | 30/32 | 7.4 (7.4) | 53 | 46 | 38 | 37 |
| diag | `tri30m` | `plan_tri30m_diag_s3` | 32/32 | 7.5 (8.2) | 66 | 97 | 36 | 44 |
| diag | `tri30s_rc` | `plan_tri30s_rc_diag_s1` | 30/32 | 7.7 (7.8) | 67 | 52 | 35 | 49 |
| diag | `horn20` | `plan_horn20_diag_s1` | 32/32 | 8.0 (8.4) | 62 | 85 | 0 | 50 |
| diag | `tri30s` | `plan_tri30s_diag_s1` | 32/32 | 8.1 (9.0) | 76 | 52 | 38 | 54 |
| side | `tri30s` | `plan_tri30s_side_s3` | 30/32 | 8.3 (9.3) | 102 | 56 | 12 | 34 |
| flip | `horn30` | `plan_horn30_s3` | 32/32 | 8.4 (8.5) | 70 | 78 | 29 | 36 |
| diag | `rc20` | `plan_rc20_diag_s2` | 32/32 | 8.5 (8.5) | 78 | 50 | 0 | 50 |
| flip | `tri20m` | `plan_tri20m_s1` | 32/32 | 9.0 (9.9) | 85 | 67 | 0 | 37 |
| flip | `horn30` | `plan_horn30_s1` | 32/32 | 9.1 (9.1) | 55 | 46 | 51 | 33 |
| flip | `tri20s` | `plan_tri20s_s1` | 32/32 | 9.1 (9.1) | 71 | 47 | 31 | 45 |
| flip | `rc20` | `plan_rc20_s2` | 32/32 | 9.4 (10.0) | 117 | 71 | 55 | 66 |
| flip | `rct20` | `plan_rct20_s1` | 32/32 | 9.4 (9.5) | 91 | 80 | 11 | 57 |
| diag | `horn30` | `plan_horn30_diag_s1` | 32/32 | 13.2 (14.1) | 101 | 58 | 27 | 64 |
| side | `tri30s` | `plan_tri30s_side_s1` | 32/32 | 15.3 (15.5) | 102 | 75 | 27 | 54 |

Every search, including the ones that failed and the axis checks, is in `results.csv`.
