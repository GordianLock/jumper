# Carapace colour patterns

Colour patterns for the carapace versions, exported as **one display mesh per filament** so a
simulation render (or a multi-filament print plan) can use them directly. Patterns are designed
in the interactive colour preview; each folder here is one saved setup applied to one version.

| Folder | Version | Setup |
|---|---|---|
| `CF1-tri30m-rb-v4b__30-oct-b/` | `CF1-tri30m-rb-v4b` | "30 Oct B": Fibonacci bands, 2.5 mm unit, grown from the shell surface up, filament = n mod 4, accent on added surface under 0.6 mm |
| `CF1-tri30m-rb-v4__30-oct-kolakoski/` | `CF1-tri30m-rb-v4` | "30 Oct Kolakoski": Kolakoski bands, 0.25 mm unit (bands 0.25 or 0.5 mm), counted down from the highest tip, filament = n mod 3, accent on added surface under 1.1 mm. Three band filaments, so the pieces are `stock`, `accent`, `c0`..`c2`. |
| `CF1-tri30m-rb-v4__30-oct-kolakoski-v2/` | `CF1-tri30m-rb-v4` | "30 Oct Kolakoski v2": the same bands and accent as `30-oct-kolakoski` (those pieces are byte-identical), plus a stepped height fade on the stock surface: 6 level steps from `#0F0F0F` at the top to `#B20606` at the bottom, starting 22 % and ending 88 % of the way down from the top of the stock shell to its lowest point. The stock piece is replaced by `stock_s0` (bottom) .. `stock_s5` (top); `colours.json` lists each step's colour and z range. |
| `CF1-tri30m-rb-v4b__30-oct-c/` | `CF1-tri30m-rb-v4b` | "30 Oct C": Fibonacci bands, 2.5 mm unit, grown from the shell surface up, filament = a(n) mod 4, accent under 0.6 mm, minimum band for a 0.4 mm nozzle; stock fades down the sides in 4 steps from `#B72306` on the whole top to `#1F1F1F` at the lower edge (`stock_s0` bottom .. `stock_s3` top; `colours.json` lists each step's colour and wall-fraction range). 9 pieces. |
| `CF1-tri30m-rb-v4__30-oct-thue-morse/` | `CF1-tri30m-rb-v4` | "30 Oct Thue-Morse": Thue-Morse (+1) bands, 3.55 mm unit (bands 3.55 or 7.1 mm), grown from the shell surface up, filament = n mod 3, accent under 1.5 mm, minimum band for a 0.4 mm nozzle; stock fades by height in 3 steps from `#141414` at the top to `#5F0202` at the bottom, between 27 % and 100 % (`stock_s0` bottom .. `stock_s2` top). 7 pieces. |
| `CF1-tri30m-rb-v4b__30-oct-d/` | `CF1-tri30m-rb-v4b` | "30 Oct D": triangular-number bands, 1.9 mm unit, counted down from the highest tip, filament = n mod 4, accent under 1.4 mm, minimum band for a 0.6 mm nozzle; stock fades down the sides in 4 steps from `#3616D4` on the whole top to `#171616` at the lower edge (`stock_s0` bottom .. `stock_s3` top). 9 pieces. |
| `CF1-tri30m-rb-v4__30-oct-e/` | `CF1-tri30m-rb-v4` | "30 Oct E": Thue-Morse (+1) bands, 0.35 mm unit (bands 0.35 or 0.7 mm), counted down from the highest tip, filament = n mod 4, accent under 1.4 mm, minimum band for a 0.2 mm nozzle; stock fades down the sides in 4 steps from `#427321` on the whole top to `#171616` at the lower edge (`stock_s0` bottom .. `stock_s3` top). 9 pieces. |
| `CF1-tri30m-rb-v3b__30-oct-f/` | `CF1-tri30m-rb-v3b` | "30 Oct F": Fibonacci bands, 0.5 mm unit, grown from the shell surface up, filament = a(n) mod 4, accent under 1.35 mm, minimum band for a 0.2 mm nozzle; stock fades down the sides in 4 steps from `#000000` on the whole top to `#1116A2` at the lower edge (`stock_s0` bottom .. `stock_s3` top). 9 pieces. |
| `CF1-tri30m-rb-v3__30-oct-f/` | `CF1-tri30m-rb-v3` | "30 Oct F" on prototype 3: the same setup as `CF1-tri30m-rb-v3b__30-oct-f` (only `version` differs), so the same bands, accent, fade and colours; prototype 3's taller front corner horns (28 degrees, against 20 on 3b) reach further into the bands. 9 pieces. |
| `CF1-tri30m-rb-v3__30-oct-fa/`, `CF1-tri30m-rb-v3b__30-oct-fa/` | `CF1-tri30m-rb-v3`, `CF1-tri30m-rb-v3b` | "30 Oct Fa": 30 Oct F with the stock fade changed to 5 steps from `#000000` on the whole top to `#3E43C1` at the lower edge (`stock_s0` bottom .. `stock_s4` top); bands, accent and colours as in 30 Oct F. 10 pieces each. |

## What a folder holds

- `setup.json` -- the pattern's parameters and colours (the input).
- `colours.json` -- what was produced: each piece's file, hex and `rgba`, the relief interval
  each filament covers, and an area check (the pieces sum to the printable shell's area).
- `<filament>.stl` -- one surface mesh per filament: `stock` (or `stock_s0`..`stock_s<n-1>` when
  the setup has a stepped stock fade), `accent`, `c0`..`c(k-1)`.
  **Metres, in the `upper_shell_link` frame** -- the same frame as
  `assets/jumper/urdf/jumper/meshes/visual/upper_shell_link.stl` and the version's
  `carapace/<version>/visual.stl`, so they attach to `upper_shell_link` with no `pos`/`quat`.

Together the pieces are the whole printable top shell surface (stock shell, underside
included, plus carapace), with no gaps or overlaps. They are open surfaces, not closed
solids: right for display, not for slicing as separate print bodies.

How the pattern follows relief: each surface point's height above the stock top shell picks
its filament. The untouched stock surface is `stock`; added surface below the accent cut-off
(the thin CAD-blend leftovers that form the striping, and the feathered horn roots) is
`accent`; everything higher is banded by the integer sequence. Triangles are cut along the
band edges, so boundaries are smooth.

## Using it in simulation (display only)

Display-only geoms (`class="visual"`: `contype="0" conaffinity="0" group="2"`, `density="0"`)
change nothing in the physics. Loaded on MuJoCo 3.11.0 with the pieces added, each model
compiles and `upper_shell_link`'s mass is unchanged: 0.174437 kg for `30-oct-b` on
`jumper_CF1-tri30m-rb-v4b.xml`, 0.176437 kg for `30-oct-kolakoski` on
`jumper_CF1-tri30m-rb-v4.xml`, and the same for `30-oct-kolakoski-v2` (10 pieces); 0.174437 kg
for `30-oct-c` (9 pieces) on `jumper_CF1-tri30m-rb-v4b.xml`; 0.176437 kg for `30-oct-thue-morse`
(7 pieces) on `jumper_CF1-tri30m-rb-v4.xml`; 0.174437 kg for `30-oct-d` (9 pieces) on
`jumper_CF1-tri30m-rb-v4b.xml`; 0.176437 kg for `30-oct-e` (9 pieces) on `jumper_CF1-tri30m-rb-v4.xml`; 0.186437 kg for
`30-oct-f` (9 pieces) on `jumper_CF1-tri30m-rb-v3b.xml`, 0.188437 kg on `jumper_CF1-tri30m-rb-v3.xml`, and the same two
masses for `30-oct-fa` (10 pieces).

The snippets below are for `CF1-tri30m-rb-v4b__30-oct-b`. For another folder, add the pattern
to that version's model (`jumper_<version>.xml`), use that folder's name in the paths, list
the pieces its `colours.json` lists, and take each `rgba` from there.

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

Needs trimesh and rtree. The script implements every sequence the preview offers
(`fibonacci`, `padovan`, `triangular`, `primes`, `kolakoski`, `thue_morse_plus_1`) the same
way the preview generates them, both growth directions (`grow_from`: `"base"` or `"tip"`) and
both colour rules (`colour_rule`: `"a_mod_k"` or `"n_mod_k"`), a stepped stock fade in
either of the preview's modes (`stock_fade.mode`: `"height"` or `"wall"`, the preview's "down
the sides"; a smooth fade cannot be split into filaments), and the preview's minimum band
(`min_band_mm`, the nozzle width: unit, accent cut-off and fade steps are checked, slivers are
folded into the neighbouring band as the preview folds them). For `30-oct-c` its relief
intervals match the preview's Copy settings ranges, and its wall fraction agrees with the
preview's to within 0.002 (the preview stores it to 1/255). Its relief intervals for
`30-oct-kolakoski`, merged and rounded to 0.1 mm, match all 107 ranges in the preview's Copy
settings output for that setup.
