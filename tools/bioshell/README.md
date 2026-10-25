# Jumper bioshell CAD — v01 prototype handoff

Three **modeled, printable, simulation-ready** derivatives of the stock upper
shell, designed for the `roll-guided` training branch. The physical and visual
dimensions come from `assets/jumper/urdf/jumper/meshes/visual/upper_shell_link.stl`,
**not** from an AI-generated concept drawing. That original reference is
watertight, weighs 102.44 g in the stock MJCF, and has a small screen opening on
the **+X front** short face.

| Variant stem | Physical intent | Suggested first use |
|---|---|---|
| `bioshell-crowned-dome-v01` | Spaced rounded nubs on an organic crown | Contact-point exploration |
| `bioshell-central-keel-v01` | Shallow lengthwise ridge with shoulder bosses | Directional roll testing |
| `bioshell-low-rocker-crest-v01` | Low offset ridge and lateral rolling features | Lowest silhouette and smoother rolling |

All three are genuine mesh geometry, not just images. Their properties and
generated paths are recorded in
`assets/jumper/bioshell/bioshell-v01-manifest.json`.

## Two geometric fidelity classes per design

- `...-visual.stl`: detailed, watertight visual/physics-source shell, native
  robot mesh units **meters**, same `upper_shell_link` frame and all source
  triangles retained; geometry is modeled using smooth parametric deformation
  of the outward-facing top surface only.
- `...-collision.stl`: one watertight low-triangle **convex** contact proxy in
  **meters**, support-map sampled to retain raised extremities. MuJoCo uses a
  convex collision representation of meshes; this intentionally abstracts away
  dents and color/texture, and is **not** a fully faithful collision surface.

Bonus export formats for design and fabrication:

- `...-print-mm.stl`: full detail resized to **millimeters** for slicers.
- `...-colored.3mf`: full detail in **millimeters** with four base-material
  color territories. Verify AMS/material assignments in the intended slicer.
- `...-colored.ply`: full detail in **meters** with vertex color data for 3D
  renderers, inspection or further sculpting.
- `bioshell-v01-contact-sheet.png`: technical oblique/top previews; not
  photographic renderings.

## Reproduce and adjust the CAD

The entire modeling recipe is in `tools/bioshell/generate.py`. Dependencies:
Python 3.10+, NumPy, SciPy, and optionally Matplotlib for the preview sheet.

```bash
python tools/bioshell/generate.py
python tools/bioshell/validate.py
```

The three editable height fields live in `crown_field()`. They affect only the
upper exterior. `exterior_weight()` protects the shell underside, lateral rim,
rear, mounts, and the front display approach. The source faces remain exactly
the same connectivity, maintaining the stock print geometry's manifold status.
The generated artifacts do not include robot legs or move the display module.

## Handoff to Claude / roll-guided

The generator writes a distinct full robot model for each variant in
`assets/jumper/jumper-bioshell-<name>-v01.xml`. These XMLs differ from stock
`assets/jumper/jumper.xml` **only** in:

1. `upper_shell_link` visual asset file;
2. `upper_shell_link_col` convex collision asset file;
3. `upper_shell_link` mass, local COM and full inertia tensor.

The existing `upper_shell_link` rigid-body translation and every actuator,
joint, leg, display, camera and contact geom name remain unchanged. Claude can
use the existing training CLI, with no engine rewrite:

```bash
python scripts/train.py --task jumper.roll --model assets/jumper/jumper-bioshell-central-keel-v01.xml
python scripts/train.py --task jumper.roll --model assets/jumper/jumper-bioshell-crowned-dome-v01.xml
python scripts/train.py --task jumper.roll --model assets/jumper/jumper-bioshell-low-rocker-crest-v01.xml
```

The same `--model` mechanism can be used in other roll task variants, subject
to their own configuration. Start each comparison with equal training seeds,
environment counts and reward weights, then report upright success, episode
duration, peak shell/ground impact force, roll direction, repeated contacts and
motor effort. A model that rolls quickly but rebounds violently may not be the
best candidate for the physical competition.

## Important limits / prototype status

- Shell contact shaping and visual fidelity are **v01 exploration**, not
  experimentally certified self-righting geometry. Right-side-down lateral
  roll behavior requires simulation and physical trials.
- Mass and inertias are derived by solid-mesh volume integration with density
  calibrated to the stock shell mass (102.44 g). **The model assumes solid
  material**. Hollow prints, infill, ribs, paint and hardware will differ;
  once a shell is actually printed, weigh it and ideally measure its COM and
  recalibrate the simulation.
- Front screen module remains on the stock robot and is not a camera. Our new
  shell does not cover the opening; final fit must be verified with the exact
  printer, fasteners, display, moving legs and source CAD tolerances.
- A single collision hull fills concavities (including the screen opening).
  For high-fidelity side impacts, replace it with a handful of convex pieces,
  and update contact filtering accordingly; **do not** pass the detailed mesh
  as a concave MuJoCo collider and assume it will behave as a concave solid.
- This work does **not** claim that new RL training runs have been performed.
  Only geometry, mass/inertia, XML load and static MuJoCo forward dynamics are
  part of the handoff verification.

## Naming / versioning

File stem: `bioshell-<concept>-vNN`. Increment `vNN` whenever silhouette,
mounting protection region or mass model changes. Keep both the visual and
collision assets under that same stem to avoid accidental mismatches.

The generator intentionally lives apart from `tasks/` and training code.
Only generated bioshell files under `assets/jumper/` are introduced.
