# Parametric carapace design: handoff

This document lets a fresh session start the carapace design work without the earlier
conversations. Read it first, then `studies/carapace/RESULTS.md`.

## Who and why

The user is entering KingKong's **Crab Robot Open Motion Challenge** with the Jumper robot.
The entry has two parts:

- **A carapace (shell).** It must be printable in the submitted version.
- **A trained movement: rolling.** This includes self-righting from belly-up, done with
  head-first, side and diagonal rolls, and a momentum roll.

The roll training lives in `tasks/jumper/*_roll` and `tasks/jumper/roll`.

A simulation study on the `carapace-study` branch tested which **horn configurations**
make belly-up self-righting rolls possible, reliable and gentle on the joints. It used
capsule stand-ins for the horns. **This branch, `carapace-design`, turns those findings
into a real, parametric carapace.** Its versions should be printable and also usable as
the robot model for roll training.

### The user

- New to GitHub and Colab. Works on Windows and keeps run outputs in
  `C:\Users\stran\OneDrive\Documents\jumperkrab\`. Explain things plainly.
- Prefers that you do the work rather than describe it. Keeps replies short, and watches
  usage limits.
- Wants to discuss before expensive simulation runs, and brainstorms before committing
  ideas to designs.
- Earlier carapace drafts and notes were made in ChatGPT, including
  `CARAPACE_SELF_RIGHTING_NOTES.md` in the user's own files, not in this repo. Ask for
  them if they would help.

### The user's design intent

- **Theme: pattern biomimicry.** Simple rules produce complex natural patterns, such as
  cellular automata and integer sequences. These are expressed through filament colour
  and topographic surface relief, with relief inspired by crustacean carapaces.
- **Relief is only ever added on top of the base shell, never cut into it.** That protects
  the fit. Relief height is up to about 20 mm for now, possibly 30.
- **A somewhat domed form to help it roll.** Flat-ish channels placed to guide a roll
  toward orientations that complete it or self-right from the back.
- **Adjustable** so it can be tuned to print and fit well, and so versions with different
  horn lengths and angles can be produced.
- The user organises surface-pattern designs as a taxonomy, with IDs of the form
  Family-Pattern-Coupling-Version, for example `CA1-R30-T-v1`. Ask how carapace-form
  versions should be named before inventing a scheme.

## The task

Build a **parametric carapace generator**. From a small set of parameters it should
produce:

1. **Printable geometry**, as STL or 3MF, that mounts on the stock robot.
2. **A matching MuJoCo robot model**: a copy of `assets/jumper/jumper.xml` with the
   carapace added to `base_link` as collision geometry, and the body inertia updated. Roll
   training and the study scripts can then run on that version.

Parameters to expose include, at least:

- Dome height and profile.
- Centre front horn: rest angle (or tip height), length, radius.
- Front corner horns: height, position, angle.
- Rear corner horns.
- Tail, on or off.
- Channels or side rails.
- Wall thickness.

Changing horn lengths and angles must be a parameter change, not a remodel.

Brainstorm the approach with the user first. Possible routes include a Python CAD
library (CadQuery or build123d), trimesh/manifold3d, and OpenSCAD. Check what installs in
the workspace before committing to one.

## Hard constraints

All coordinates are in the robot's **`base_link` frame, in metres**: +x forward toward the
head, +y to the robot's left, +z up out of the shell.

- **Mounting.** The stock body spans x −0.112 to +0.094 and y ±0.091. The shell top is at
  z = 0.072. The stock top shell is `upper_shell_link`, at (−0.0073, 0, 0.0315) in
  `base_link`. Its meshes are `assets/jumper/meshes/upper_shell_link_col.STL` and
  `assets/jumper/urdf/jumper/meshes/visual/upper_shell_link.stl`. The carapace is
  built **outward from the stock shell surface** and never removes material.
- **The legs' path.** The legs reach the floor by swinging around the outside of the body.
  Nothing may enter their swept volume:
  - `studies/carapace/leg_keepout_side_voxels.npz` is the volume with 5 mm of clearance
    added. `carapace_keepout.png` is a picture of it.
  - `studies/carapace/clearance.py` checks a capsule against it. A mesh needs a similar
    voxel-overlap check; write one.
  - Shapes over the stock footprint are free.
- **Thickness and overhang.** Each centimetre added over the back costs about a centimetre
  of leg push past the floor when belly-up, which is what self-righting needs. Keep
  **added thickness over the back to 2–3 cm or less**. Keep **overhang past the stock
  outline to 2 cm or less**; at 5 cm the front arms suffer most. Horns are the exception:
  the study's horns are what make the rolls work.
- **Sensors at the front stay uncovered and keep their view.** The display module is at
  about (0.080, 0, 0.050). The ToF sensor and camera are at about (0.091, 0, 0.017) and
  (0.092, 0.014, 0.017), and face forward.
- **Collision geometry.** MuJoCo collides a mesh geom as its **convex hull**. A concave
  carapace with horns must therefore be split into convex pieces (a convex
  decomposition, such as CoACD or V-HACD) or approximated with primitives. Otherwise the
  simulated shell is a solid blob and the roll results are wrong.
  `assets/jumper/tools/check_hulls.py` may help.
- **Mass and inertia.** `base_link` has an explicit `<inertial>`, so added geoms' masses
  are ignored unless the inertial is recomputed. Estimate the printed carapace's mass and
  centre of mass from its volume and the filament density, and fold them into
  `base_link`'s inertial. The horn study ignored horn mass; a real carapace must not.

## What the horn study found

The full tables are in `studies/carapace/RESULTS.md`; `results.csv` has one row per
search. The horn geometry used is tabulated there too. The short version:

- **The belly-up rest pose decides most of it.** Horns work by setting how the robot
  rests on its back. A new carapace's rest pose should be checked first.
- **A centre front horn is the single most useful feature.** With one, nearly every
  shell can do a head-first flip gently.
- **Gentlest flips:** the 30° centre horn with tall (3.5 cm) front corners, with or
  without rear corners. They reach 3.3–3.5 N·m, against the stock high jump's 5.9.
- **Side roll:** low (2.5 cm) front corners make it possible.
  - 20° centre horn + low front corners + rear corners (`tri20s_rc`): 4.1–4.7 N·m.
  - 30° centre horn + low front corners (`tri30s`): reliable in 3 of 3 searches.
- **Diagonal roll:** the gentlest is `tri20s_rc` at 5.1 N·m. `rc20`, `horn30` and
  `tri30s` are the most dependable.
- **A rear centreline tail hurts every roll.** Leave it out.
- **The 30° horn plus rear corners has two rest poses:** sitting up at 55° on the rear
  corners, or lying at 16°. Its plans work only from the sitting pose. Prefer a design
  with one stable rest pose.
- **Comparison with the stock jumps.** 22 working plans beat the high jump (the jump the
  robot ships with) on joint torque, p90, limb force and foot force: 16 flips, 3 side
  rolls and 3 diagonals. `RESULTS.md` lists them and also compares against the gentler
  reference-free jump (4.8 N·m).

No single shell was best at every roll, and the user did not ask for one. The aim was to
know which configurations can do which rolls. A parametric design lets the user choose.

## Testing a carapace version

The study tools on this branch can evaluate a generated version:

1. **Keep-out:** no overlap with the leg voxels.
2. **Rest pose:** drop it belly-up and settle (`righting.settle`, `righting.pose_of`), with
   several perturbed drops to catch a second rest pose. Report head-up angle, side lean and
   height.
3. **Rolls:** `righting.py` searches. It is about 4 minutes per search on this machine's 2
   cores, and 3 searches × 3 roll types per shell. Then use `gentle.py` for loads and
   `grid.py` for videos.
   - `righting.build()` currently loads `scene.xml` and adds capsule horns. Testing a mesh
     carapace needs a build path that loads the generated model instead.
   - **Ask the user before starting search runs.**
4. **Training:** `scripts/train.py` and `scripts/play.py` accept `--model path/to/model.xml`,
   so a generated model can train the roll tasks without registering it.
   - `jumper.xml` finds its meshes through `<compiler meshdir="./"/>`, relative to the
     XML file. A copy saved elsewhere fails to load: tested, it cannot open
     `urdf/jumper/meshes/visual/base_link.stl`. Either write generated models into
     `assets/jumper/`, or set an absolute `meshdir` and give the carapace meshes absolute
     paths.
   - `--dry-run` resolves the path without loading the model, so also load it once with
     `mujoco.MjModel.from_xml_path` before training.

`studies/carapace/HANDOFF.md` explains the study's method, its scripts and its pitfalls.
One pitfall: never `pkill -f righting`, because it kills your own shell.

## Repository notes

- Read `CLAUDE.md` at the repo root. Everything committed is in English.
  `pip install -e .` is required.
- Keep design code and outputs under `design/carapace/`. Keep large generated meshes out
  of git unless the user wants them committed.
- Commit and push to `carapace-design`. `carapace-study` is the study's branch; it is not
  meant to be merged, and this branch is built on it.
