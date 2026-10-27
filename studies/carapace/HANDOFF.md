# Carapace horn study: handoff

This document lets a fresh session continue the work without the original conversation.
Read it first, then the scripts it points to.

## Who and why

The user is entering KingKong's **Crab Robot Open Motion Challenge** with the Jumper
robot. The movement work is done here, with Claude. The shell or carapace look is being
designed separately, in ChatGPT.

The user's fork is `GordianLock/jumper`. Its branches:

- `roll-guided`: the trained roll tasks.
- `carapace-study`: this branch, the probes. It is not meant to be merged.

The user is new to GitHub and Colab, works on Windows, and keeps run outputs in
`C:\Users\stran\OneDrive\Documents\jumperkrab\`. Explain plainly. Prefer doing the work
over describing it. Keep replies short; the user cares about usage limits.

The question being explored is which belly-up carapace shape, mainly horns, makes
self-righting rolls possible, reliable and gentle on the joints. The user's priorities:

- Visually interesting rolls.
- Gentle forces, especially on non-load-bearing parts such as shins, claws and sensors.
- Three roll types:
  - **head-first flip**: end over end.
  - **side roll**: about the body centreline, right side over left or the reverse.
  - **diagonal (cross-body) roll**: about the axis from one shoulder to the opposite
    rear hip, for example the right rear going over the left shoulder.

## Robot facts

- Jumper hexapod, 22 joints:
  - Two 5-joint front arms with claws (`LF`, `RF`). Joint `J4` is the gripper, held
    still here.
  - Four 3-joint legs (`LM`, `RM`, `LR`, `RR`).
- Mass about 2.54 kg (about 25 N).
- Servo plateau 1.7464 N·m (continuous 1.2), kp 10, kd 0.5. The policy runs at 50 Hz on
  a 5 ms physics step.
- `base_link` frame, in metres:
  - +x points forward toward the head, +y to the robot's left, +z up out of the top shell.
  - The stock body spans x −0.112 to +0.094 and y ±0.091.
  - **The stock shell top is at z = 0.072.**
- On its back, the stock robot rests about 8° head-up with its centre of mass about 8 cm
  above the floor.

## Method

Run everything from `studies/carapace/` after `pip install -e .` at the repo root. Extra
pip packages needed: `shapely scikit-image trimesh mediapy pillow`.

- **`scene.xml`** is jumper.roll's training scene, dumped from the environment by
  `dump_scene.py`. It keeps the same floor, contact settings and sensors.
- **`righting.py`** builds the model and turns the motors into position servos (kp 10,
  kd 0.5, clipped at 1.7464 N·m; the servo speed curve is not modelled). Stand-in horns
  are added in `build(variant)` as capsule geoms on `robot/base_link`. Any geom named
  `robot/horn*` is counted in the "horn" force group automatically.
  - The robot is dropped belly-up and left to settle for 2 s (`settle`).
  - A **cross-entropy search** then looks for a motion. A plan is five keyframes of the
    20 driven joints, written as offsets from HOME in half-range units, on a fixed 50 Hz
    schedule that ends at HOME after 3.5 s. Each round has 96 candidates and keeps the 12
    best. It runs for 50 rounds and always keeps the best plan found so far.
  - The score:
    - Credit for progress.
    - +20 for ending upright (body z·world z > 0.9 over the last 0.6 s).
    - Credit for standing height.
    - Minus floor-hit costs, phased in from round 10 to round 30: limbs free to 20 N,
      shell 40 N, horns 40 N, sensors 2 N, feet 50 N. Costs are per newton above those
      levels: 0.08, 0.04, 0.04, 0.5 and 0.03.
  - With a target axis (`variant:side` or `variant:diag`), the score gives progress as
    the turn about that axis and multiplies the finishing bonus by
    exp(−(axis error / 15°)²). The axis is measured from the integrated body-frame
    angular velocity during the flip.
  - Each search's best plan is retried from **32 perturbed starts** (tilt ±8°, joint
    jitter ±0.08 rad), and reports how many succeed.
  - Usage: `python -u righting.py SHELL SHELL:side SHELL:diag`. That runs three searches
    (seeds 1–3) per argument, which takes about 3–4 minutes per search on two CPU cores.
    Plans go to `plans/plan_<shell>[_<axis>]_s<seed>.npy` and a summary to `logs/`.
- **`gentle.py SHELL,plan_name,roll ...`** replays plans from 32 perturbed starts (seed 7)
  and reports success. Per run it records the **worst floor-contact torque on any joint**
  (J^T f for every floor contact) and peak floor force by part group; the summary prints
  medians. This is the gentleness measure; quote the median joint torque.
  - Baselines for that torque (KingKong's own motions, measured the same way):
    walking about 2.0 N·m, dance 1.2, jump 4.8. The jump figure could not be reproduced.
    `jump_loads.py` runs the trained jumper.jump policy (`out/example/model_3100.pt`)
    for 32 attempts and measures every physics step (`logs/gentle_stock_jump.log`):
    - 200 Hz physics, the rolls' own step: median **6.4 N·m** (p90 7.0), limbs 113 N,
      shell 0, feet 178 N.
    - 400 Hz, the jump's training rate: 5.9 N·m.
    - 1000 Hz, play's default: 5.8 N·m.

    Compare rolls against **6.4**.
- **`grid.py ROLL OUT.mp4`**: labelled grid video. Rows are shells (set with the
  `SHELLS=` environment variable); columns are searches 1–3. A green label means the plan
  worked from 30 or more of the 32 starts.
  - Rendering is software-only and slow, about 25 s per tile at 320×240.
  - Run long jobs in the background with a long timeout.
- **`replay.py SHELL plans/X.npy out.mp4`**: one video plus nominal loads. Its "flips
  sideways / end over end" label is a crude snapshot; trust the axis numbers from the
  search instead.
- **`clearance.py x0 y0 z0 x1 y1 z1 r`**: is a capsule horn clear of the legs' path?
  0.00 means clear. Every horn must pass this.
  - The legs reach the floor by swinging around the outside of the body. A shape over
    the stock footprint is free; outside it, the legs need the space.
  - `leg_keepout_side_voxels.npz` is the swept volume, from `keepout2.py`, with 5 mm of
    clearance added. `carapace_keepout.png` is a picture of it.
- **`study.py` and `analyse.py`**: the earlier still-pose reach study.
- **`keepout.py` and `keepout2.py`**: build the leg keep-out volume and the overhang table.

A practical warning: never run `pkill -f righting`. The pattern matches your own shell
command and kills the session's shell. Use `pkill -f "python -u righting.py"`.

## Results so far

### Still-pose study, no carapace

- Each centimetre the carapace raises the body costs about a centimetre of leg push past
  the floor. Push is 6.7 cm at 8 cm height, 4.7 at 10 cm, 1.7 at 13 cm, and at 15 cm only
  two legs reach the floor. **Keep added thickness over the back to 2–3 cm or less.**
- Overhang past the stock outline costs little up to about 2 cm and hurts by 5 cm, the
  front arms most.
- Past about 30° of side tilt, the high-side legs lose the floor.
- A head-up pose of 20–30° gives the rear legs 8.6–9.8 cm of push, against 6.7 cm flat.

### Shells tested

Each shell below is a 20° front horn plus additions, except where noted.

- **horn20 / horn30**: centreline capsule, radius 1.2 cm, from (−0.030, 0, 0.072) to
  (0.085, 0, top).
  - horn20 top at 0.114: it stands 4.2 cm above the shell at the front and the robot
    rests 20° head-up.
  - horn30 top at 0.140 (6.8 cm): the robot rests 30° head-up.
- **tri20s / tri20m**: horn20 plus two front corner horns, radius 0.9 cm, from
  (0.02, ±0.035, 0.072) to (0.065, ±0.065, 0.072 + 0.025 or 0.035). These are as far out
  as stays clear of the legs, about 3 cm short of the real corners. An earlier version
  that reached the real corners overlapped the leg path, and the user asked for it to
  be kept clear.
- **rc20**: horn20 plus two rear corner horns, 2.5 cm, from (−0.04, ±0.035, 0.072) to
  (−0.075, ±0.065, 0.097).
- **rt20**: horn20 plus a rear centreline "tail", radius 1.0 cm, 2.5 cm tall, from
  (−0.06, 0, 0.072) to (−0.095, 0, 0.097).
- **rct20**: rc20 plus the tail.

### How each shell rests belly-up

| Shell | Head-up | Side lean | Height off floor |
|---|---|---|---|
| stock | 8° | none | 6.1 cm |
| horn20 | 20° | none | 8.9 cm |
| horn30 | 30° | none | 9.0 cm |
| rc20 | 7° | none | 11.5 cm |
| rt20 | 5° | about 29° | 10.3 cm |
| rct20 | 6° | none | 11.6 cm |

Heights here are of the body's origin, not the centre of mass.

The rear corner horns turn the back into a level tripod. The tail makes a centreline
ridge that tips the robot onto one side.

### Searches that found a reliable plan

Each cell shows how many of the three searches found a plan that worked from at least
29–30 of the 32 starts.

| Shell | Head-first flip | Side roll | Diagonal roll |
|---|---|---|---|
| stock | 1/3 | — | — |
| horn20 | 3/3 | 0/3 | 1/3 |
| horn30 | 3/3 | 0/3 (best 15/32) | 3/3 |
| tri20s | 1/3 | 0/3 | 2/3 (one 30/32) |
| tri20m | 3/3 | 0/3 | 0/3 |
| rt20 | 0/3 (about 18/32) | 0/3 | 0/3 |
| rc20 | 3/3 | 0/3 | 3/3 |
| rct20 | 2/3 | **1/3 (29–32/32)**, the only side roll found | 1/3 |

### Gentlest plans

The joint torque is the median over 32 starts of each run's worst floor-contact torque on
any joint. The forces are the matching median peaks.

| Roll | Shell, plan | Joint torque | Limbs | Shell | Horn |
|---|---|---|---|---|---|
| flip | horn20 `plan_horn20_s3` | **4.2 N·m** | 55 N | 0 | 36 N |
| flip | rct20 `plan_rct20_s2` | 4.2 | 66 | 34 | 50 |
| flip | rc20 `plan_rc20_s1` | 4.3 | 57 | 0 | 69 |
| flip | tri20m `plan_tri20m_s3` | 4.3 | 44 | 27 | 44 |
| diag | rc20 `plan_rc20_diag_s1` | **5.7** | 83 | 0 | 99 |
| diag | tri20s `plan_tri20s_diag_s2` | 5.7 | 78 | 0 | 46 |
| diag | horn20 `plan_horn20_diag_s1` | 8.0 | 62 | 0 | 50 |
| side | rct20 `plan_rct20_side_s1` | **5.6** | 41 | **95** | 48 |

Shell is the floor force on the stock shell.

The same shell's three searches can range from 4.3 to 9.4 N·m, so gentleness depends on
the motion as well as the shape. Trained policies should be held to the gentle version.

All of these numbers are in `logs/`. In that folder, `gentle*.log` holds the loads and
`righting*.log` the searches.

### Takeaways the user has heard

- A 20° front horn is the best single feature.
- Front corner horns that stay clear of the legs don't help. At 3.5 cm they stop the
  diagonal roll, but the user found the flip they produce visually interesting.
- Rear corner horns make the diagonal roll dependable and keep the flip gentle.
- Rear corners plus the tail give the only side roll, but it hits the shell with about
  95 N, roughly four times the robot's weight.
- The tail alone hurts every roll.

### Rear horns on the three-horn front (session 2)

The user's goal was clarified: not one shell that does every roll, but a map of which
shell can do which roll. The rear horn sets were tested on the three-horn front
(centre horn20 plus the low front corners of tri20s) instead of the single horn:
`tri20s_rc`, `tri20s_rt`, `tri20s_rct`. `righting.build()` takes these as
`<front>_<rear>`. The taller front corners (tri20m) are not tested yet; the user may want
them next.

Rest poses belly-up:

| Shell | Head-up | Side lean | Height | On the floor |
|---|---|---|---|---|
| tri20s_rc | 7° | none | 11.5 cm | centre horn, both rear corners (same as rc20) |
| tri20s_rt | 11° | about 20° | 11.1 cm | one front corner, tail |
| tri20s_rct | 6° | none | 11.6 cm | centre horn, a rear corner, tail |

Search results. Each cell shows how many of the three searches reached at least 29 of the
32 starts, with each search's best rate in brackets:

| Shell | Flip | Side | Diagonal |
|---|---|---|---|
| tri20s_rc | 3/3 (31, 32, 32) | **2/3 (32, 32, 0)** | 1/3 (0, 32, 0) |
| tri20s_rt | 0/3 (16, 18, 16) | 0/3 (7, 0, 2) | 0/3 (0, 18, 0) |
| tri20s_rct | 1/3 (12, 30, 12) | 1/3 (12, 29, 13) | 1/3 (31, 0, 13) |

The side-roll axes were checked: tri20s_rc's two plans are 11° and 13° off the
centreline, and tri20s_rct's is 14° off it. Its diagonal plans sit on the diagonal
(alignment 1.00).

Loads, measured with `gentle.py` (`logs/gentle_tri20s_rear.log`). The earlier
horn20 flip, rc20 diagonal and rct20 side plans were re-measured in the same run and
reproduced 4.2, 5.7 and 5.6 N·m.

| Roll | Plan | Joint torque | Limbs | Shell | Horn |
|---|---|---|---|---|---|
| flip | `plan_tri20s_rc_s1` | 4.3 | 76 | 0 | 55 |
| flip | `plan_tri20s_rc_s2` | 5.6 | 64 | 0 | 73 |
| flip | `plan_tri20s_rc_s3` | 6.9 | 73 | 0 | 49 |
| flip | `plan_tri20s_rct_s2` | 5.1 | 85 | 0 | 100 |
| side | `plan_tri20s_rc_side_s2` | **4.1** | 89 | 0 | 55 |
| side | `plan_tri20s_rc_side_s1` | 4.7 | 56 | 0 | 155 |
| side | `plan_tri20s_rct_side_s2` | 5.5 | 112 | 0 | 88 |
| diag | `plan_tri20s_rc_diag_s2` | **5.1** | 74 | 0 | 52 |
| diag | `plan_tri20s_rct_diag_s1` | 6.5 | 83 | 0 | 49 |

Takeaways:

- **tri20s_rc can do all three rolls.** Its side roll is the best found so far: 32 of
  32 starts and 4.1 N·m, below the jump's 4.8. Unlike rct20's side roll (95 N), it never
  touches the shell. Its diagonal, at 5.1 N·m, is also the gentlest so far, though only
  one search of three found it.
- With a single front horn, the rear corners never produced a side roll. The front
  corners are what make it possible.
- The tail still hurts. tri20s_rt does nothing reliably. Adding the tail to the corners
  (tri20s_rct) lowers the flip and side reliability and adds no gentleness. Its one gain
  is that the side roll no longer hits the shell (0 N against rct20's 95 N).

## Next steps the user has asked about or hinted at

- Combine the **front corner horns with the rear horns**. tri20s is done (session 2,
  above). tri20m with the same three rear sets is the possible follow-up the user
  mentioned: `tri20m_rc`, `tri20m_rt`, `tri20m_rct` already build.
- Other sizes and angles of these horns.
- **Side rails or channels** low along the shell sides at shell-top height, to rock the
  body onto its edge. This is the most likely fix for the side roll, and it might soften
  rct20's shell hit.
- The handoff notes for ChatGPT (the carapace designer) are in the user's files as
  `CARAPACE_SELF_RIGHTING_NOTES.md`. Results meant for the designer should say: frame
  `base_link`, metres, keep-out STL, thickness of 2–3 cm or less, overhang of 2 cm or less.
- Later, outside this study: train self-righting as a real policy on the chosen shell
  (the roll tasks are on `roll-guided`), then a carapace from ChatGPT's actual meshes.

## Adding a shell

1. Pick the horn endpoints and radius. Check every horn with `clearance.py` (both sides).
2. Add a branch in `righting.build()`, modelled on `rc20`. Insert geom XML after the
   `robot/base_link_meshcol` line; name horns `robot/horn_*`; give each about 15 g, with
   `contype=1 conaffinity=0 condim=3 friction=0.6`. Then set `variant = "horn20"` so the
   front horn is added too.
3. Settle it and print the rest pose (see `settle` and `pose_of`). Tell the user how it
   rests before searching; the rest pose is often the biggest effect.
4. Run `python -u righting.py NEW NEW:side NEW:diag` in the background, about 30 minutes.
5. Run `gentle.py` on the plans that reached 29/32 or better, then render the grids with
   `SHELLS=NEW` (add the name to `NAMES` and the log to the list in `grid.py`).
6. Send the videos with a short table: reliability, joint torque, and limb, shell and
   horn forces.
