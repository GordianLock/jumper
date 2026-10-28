# Parametric carapace

Horned carapaces ("crests") grown on the stock Jumper top shell, generated from a small set of
parameters, each version giving a printable replacement top shell and a matching MuJoCo model
for roll training. `HANDOFF.md` is the design brief; this file is how to use what is here.

## Versions

| Version | What it is | Model for training |
|---|---|---|
| `CF1-tri30m-rc-v1` | First roll-training prototype. Horns on the horn study's `tri30m_rc` lines: 30° centre horn, tall front corners, rear corners, each shortened just enough to clear the legs' path (3.5 / 4.5 / 5 mm). No dome. Belly-up rest poses: ≈52° head-up sitting on the rear horns and rear legs, or ≈17° lying back. | `assets/jumper/jumper_CF1-tri30m-rc-v1.xml` |
| `CF1-tri30m-rc-v2` | Second prototype: v1 with the rear corner horns lowered from 28.5° to 5° elevation (roots and tip x/y unchanged; tips 75.6 mm high, 41.9 mm long). One belly-up rest pose, ≈23° head-up: the sitting pose disappears below about 10°. | `assets/jumper/jumper_CF1-tri30m-rc-v2.xml` |

| `CF1-tri30m-rb-v3` | Third prototype: no rear horns, three bumpers per side instead (9 mm-radius spheres two-thirds below the shell, each as close to the rear edge as the legs' path allows, 35.5 mm apart). Centre horn 25°, sunk 5 mm, its root extension diving onto the shell short of the legs' path; front corner horns 28°, sunk 10 mm. One belly-up rest pose, ≈24° head-up. | `assets/jumper/jumper_CF1-tri30m-rb-v3.xml` |

Body-only centre of mass (body, top shell, display, sensors and carapace; no limbs), resting
on its underside / on its back: stock 2.9 / 3.8 cm, `CF1-tri30m-rb-v3` 3.3 / 6.2 cm.

The rest poses are static, not simulated: the stable faces of the convex hull of the training
model's collision geometry (legs at `HOME`), tested against its centre of mass. The same
calculation on the horn study's shapes reproduces its dropped rest poses (55° / 16°) within a
few degrees.

Names are `CF<family>-<study horn layout>-v<n>`; every number a version was built with is in its
`params.json`, so the name does not have to carry them.

Each version folder holds:

- `params.json` — the parameters (the input; edit a copy to make a new version)
- `report.json` — checks from the generator: legs' path overlap, overhang, volume, mass
- `model.json` — what went into the MuJoCo model: collision pieces, mass, inertia
- `printable_top_shell.3mf` — the replacement top shell (stock shell + carapace, fused), in
  the shell's own frame, millimetres. Not test-printed yet.
- `carapace.stl` — the carapace alone, base_link frame, metres; generated, not committed

## Making a version

From the repository root:

```bash
python design/carapace/generate.py design/carapace/versions/<NAME>/params.json   # ~40 s
python design/carapace/build_version.py design/carapace/versions/<NAME>           # ~1 min
```

`generate.py` needs trimesh, scikit-image and scipy; `build_version.py` also needs
manifold3d and coacd (`pip install trimesh scikit-image manifold3d coacd`).

What the parameters do is documented at `DEFAULTS` in `generate.py`. Horns are given by root
`a`, tip `b`, spine radius `r` and tip radius `r_tip` (base_link frame, metres); length and
angle are just those points. The generator reports, and must keep at zero, the overlap with
the legs' swept volume (`legs_path_overlap_mm3`, the study's volume mirrored so both sides
match, 5 mm clearance included).

## Training on a version

```bash
python scripts/train.py --task jumper.roll --model assets/jumper/jumper_CF1-tri30m-rc-v1.xml ...
```

Every roll task takes `--model`. What carries over automatically:

- The carapace's collision pieces collide because `tasks/jumper/common/constants.py`
  adds them for any model that has them (`CARAPACE_COLLISION`); until 2026-10-28 they
  were switched off and passed through the floor. They are named `upper_shell_carapace_NN_meshcol` on
  `upper_shell_link`. The roll tasks select collision geoms by the `_meshcol` ending, so their
  shell-impact sensor (`body_ground_contact`) and the rolled-start reset include the carapace
  without any change. `studies/carapace/gentle.py` groups names starting `upper_shell` as
  "shell".
- `upper_shell_link`'s inertial includes the carapace: +124 g (typical print estimate), robot
  2.543 → 2.667 kg, centre of mass and inertia recomputed.

Known approximations in `CF1-tri30m-rc-v1`:

- MuJoCo collides each mesh as its convex hull, so the carapace is 44 convex pieces (CoACD).
  Their surface stands up to 2.6 mm proud of the real carapace (95th percentile; 4.4 mm at
  most), mostly by filling the concave valleys and fillets slightly. None reaches the legs'
  path. Tighter: lower `threshold` / raise `max_convex_hull` in `build_version.py`, at the cost
  of more pieces to collide.
- The mass is an estimate (1.2 mm walls, 15 % infill, PLA), spread uniformly over the
  carapace's volume. Re-weigh the print and update it.
- Not yet checked in simulation: the belly-up rest pose and the rolls on the real shape and
  mass. The horn study's results used capsule horns without mass.

## Where the shell gets hit (for bumpers)

The plan is to add bumpers where rolls hit the shell, which needs the impact **locations** as
well as forces. Neither is recorded today:

- The roll tasks' `body_ground_contact` sensor asks only for `fields=("force",)` with
  `reduce="netforce"`: one net force per step, all shell contacts summed, positions dropped.
- `studies/carapace/gentle.py` uses each contact's position only to compute joint torques and
  keeps peak forces per part group.

The contact sensor already supports positions (`rl/mjlab/sensor/contact_sensor.py`: per-contact
fields `pos`, `normal`, `force`, ...; `reduce="maxforce"` or `"none"` keeps contacts apart).
Recording them is a change to the roll task or a replay tool, for example:

1. A second sensor (keep `body_ground_contact` as it is, since a reward reads it) with
   `fields=("force", "pos")`, `reduce="maxforce"` or `"none"` and `num_slots` of 8–16, on the
   same `_meshcol` geoms against the terrain.
2. When replaying a trained policy (`scripts/play.py`), log each contact's position
   transformed into the `base_link` frame (subtract the base position, rotate by the inverse
   base orientation), its normal force and which geom it hit.
3. Accumulate into a map over the shell surface (force-weighted hit counts, and peak force),
   from many perturbed rolls. That map is what bumper placement comes from.

Per `CLAUDE.md`, a task's parameters live in its own `env_cfg.py`, and a one-off measuring
probe stays outside `tests/`.
