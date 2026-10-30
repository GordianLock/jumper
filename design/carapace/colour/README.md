# Carapace colour patterns

Colour patterns for the carapace versions, exported as **one display mesh per filament** so a
simulation render (or a multi-filament print plan) can use them directly. Patterns are designed
in the interactive colour preview; each folder here is one saved setup applied to one version.

| Folder | Version | Setup |
|---|---|---|
| `CF1-tri30m-rb-v4b__30-oct-b/` | `CF1-tri30m-rb-v4b` | "30 Oct B": Fibonacci bands, 2.5 mm unit, grown from the shell surface up, filament = n mod 4, accent on added surface under 0.6 mm |

## What a folder holds

- `setup.json` -- the pattern's parameters and colours (the input).
- `colours.json` -- what was produced: each piece's file, hex and `rgba`, the relief interval
  each filament covers, and an area check (the pieces sum to the printable shell's area).
- `<filament>.stl` -- one surface mesh per filament: `stock`, `accent`, `c0`..`c3`.
  **Metres, in the `upper_shell_link` frame** -- the same frame as
  `assets/jumper/urdf/jumper/meshes/visual/upper_shell_link.stl` and the version's
  `carapace/<version>/visual.stl`, so they attach to `upper_shell_link` with no `pos`/`quat`.

Together the six pieces are the whole printable top shell surface (stock shell, underside
included, plus carapace), with no gaps or overlaps. They are open surfaces, not closed
solids: right for display, not for slicing as separate print bodies.

How the pattern follows relief: each surface point's height above the stock top shell picks
its filament. The untouched stock surface is `stock`; added surface below the accent cut-off
(the thin CAD-blend leftovers that form the striping, and the feathered horn roots) is
`accent`; everything higher is banded by the integer sequence. Triangles are cut along the
band edges, so boundaries are smooth.

## Using it in simulation (display only)

Display-only geoms (`class="visual"`: `contype="0" conaffinity="0" group="2"`, `density="0"`)
change nothing in the physics. Loaded on MuJoCo 3.11.0 with the six pieces added to
`jumper_CF1-tri30m-rb-v4b.xml`: the model compiles and `upper_shell_link`'s mass stays
0.174437 kg.

1. In `<asset>`, add one mesh per piece. Mesh paths resolve against `assets/jumper/`:

   ```xml
   <mesh name="colour_stock"  file="../../design/carapace/colour/CF1-tri30m-rb-v4b__30-oct-b/stock.stl"/>
   <mesh name="colour_accent" file="../../design/carapace/colour/CF1-tri30m-rb-v4b__30-oct-b/accent.stl"/>
   <mesh name="colour_c0"     file="../../design/carapace/colour/CF1-tri30m-rb-v4b__30-oct-b/c0.stl"/>
   <mesh name="colour_c1"     file="../../design/carapace/colour/CF1-tri30m-rb-v4b__30-oct-b/c1.stl"/>
   <mesh name="colour_c2"     file="../../design/carapace/colour/CF1-tri30m-rb-v4b__30-oct-b/c2.stl"/>
   <mesh name="colour_c3"     file="../../design/carapace/colour/CF1-tri30m-rb-v4b__30-oct-b/c3.stl"/>
   ```

2. In `<body name="upper_shell_link">`, add one visual geom per piece with its `rgba` from
   `colours.json`:

   ```xml
   <geom name="upper_shell_colour_stock_visual"  class="visual" type="mesh" density="0" rgba="0.8275 0.2039 0.0863 1" mesh="colour_stock"/>
   <geom name="upper_shell_colour_accent_visual" class="visual" type="mesh" density="0" rgba="0.0471 0.051 0.051 1"  mesh="colour_accent"/>
   <geom name="upper_shell_colour_c0_visual"     class="visual" type="mesh" density="0" rgba="0.1216 0.1216 0.1216 1" mesh="colour_c0"/>
   <geom name="upper_shell_colour_c1_visual"     class="visual" type="mesh" density="0" rgba="0.6118 0.7608 0.1176 1" mesh="colour_c1"/>
   <geom name="upper_shell_colour_c2_visual"     class="visual" type="mesh" density="0" rgba="0.902 0.8549 0.1961 1"  mesh="colour_c2"/>
   <geom name="upper_shell_colour_c3_visual"     class="visual" type="mesh" density="0" rgba="0.3255 0.0392 0.4588 1" mesh="colour_c3"/>
   ```

3. Hide the two visuals the pieces replace, or they z-fight with the pattern: set alpha to 0
   on `upper_shell_link_visual` and `upper_shell_carapace_visual`.

4. **Hide collision geoms (group 1) when rendering.** MuJoCo draws groups 0-2 by default, and
   the carapace's convex collision hulls stand up to a few mm proud of the real surface, so
   they cover the pattern with their own orange. In a renderer, pass an `MjvOption` with
   `geomgroup[1] = 0`; in the viewer, toggle group 1 off.

## Regenerating

```bash
python design/carapace/colour/split_colours.py design/carapace/colour/<folder>/setup.json
```

Needs trimesh and rtree. Only the Fibonacci sequence and `grow_from: "base"` are implemented
in the script so far; the preview offers more, and the script raises rather than guessing.
