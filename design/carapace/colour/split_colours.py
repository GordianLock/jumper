"""Split a carapace version's printable top shell into one surface mesh per filament.

The colour pattern follows relief: every point of the shell's surface gets its height above
the stock top shell (distance to the stock visual mesh), and that height decides the filament.

- `stock`: the untouched stock surface (within `stock_tolerance_mm`; on v4b the stock surface
  sits within 1e-4 mm and the first added surface starts at about 5e-3 mm, so the split is
  clean).
- `accent`: added surface lower than `accent_below_mm` -- the thin CAD-blend leftovers that
  form the striping, plus the feathered edge where horns meet the shell.
- `c0..c(k-1)`: bands from an integer sequence. Band n is a(n) * unit_mm thick, counted up from
  the stock surface; its filament is a(n) mod k or n mod k.

Triangles that straddle a band edge are cut along the iso-line (height is interpolated
linearly across each triangle), so colour boundaries are smooth rather than stepped.

Output, in the setup's folder: `<filament>.stl` per filament (metres, `upper_shell_link`
frame, the frame of urdf/jumper/meshes/visual/upper_shell_link.stl) and `colours.json`.

    python design/carapace/colour/split_colours.py design/carapace/colour/<folder>/setup.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[3]
STOCK_VISUAL = REPO / "assets/jumper/urdf/jumper/meshes/visual/upper_shell_link.stl"


def sequence(kind, count):
    if kind == "fibonacci":
        a = [1, 1]
        while len(a) < count:
            a.append(a[-1] + a[-2])
        return a[:count]
    raise ValueError(f"sequence {kind!r} not implemented here yet")


def intervals(setup, dmax):
    """Relief intervals [lo, hi) in mm, each with its filament name."""
    tol, acc = setup["stock_tolerance_mm"], setup["accent_below_mm"]
    if setup["grow_from"] != "base":
        raise ValueError("only grow_from='base' is implemented")
    out = [(-1.0, tol, "stock"), (tol, acc, "accent")]
    top = 0.0
    for n, a in enumerate(sequence(setup["sequence"], 200), start=1):
        if top >= dmax:
            break
        lo, hi = top, top + a * setup["unit_mm"]
        top = hi
        rule = setup["colour_rule"]
        ci = (a if rule == "a_mod_k" else n) % setup["k"]
        lo = max(lo, acc)
        if hi > lo:
            out.append((lo, hi, f"c{ci}"))
    out[-1] = (out[-1][0], dmax + 1.0, out[-1][2])
    return out


def clip(poly, value, keep_above):
    """Sutherland-Hodgman: keep the part of a polygon of (xyz, height) above/below value."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        p_in = p[1] >= value if keep_above else p[1] < value
        q_in = q[1] >= value if keep_above else q[1] < value
        if p_in:
            out.append(p)
        if p_in != q_in:
            t = (value - p[1]) / (q[1] - p[1])
            out.append((p[0] + t * (q[0] - p[0]), value))
    return out


def split(setup_path):
    setup_path = Path(setup_path).resolve()
    setup = json.loads(setup_path.read_text())
    out_dir = setup_path.parent
    version = setup["version"]
    shell = trimesh.load(
        REPO / f"design/carapace/versions/{version}/printable_top_shell.3mf", force="mesh"
    )
    stock = trimesh.load(STOCK_VISUAL)
    stock.apply_scale(1000.0)  # the printable is in mm
    _, height, _ = trimesh.proximity.closest_point(stock, shell.vertices)
    dmax = float(height.max())

    verts, faces = shell.vertices, shell.faces
    fh = height[faces]
    fmin, fmax = fh.min(1), fh.max(1)
    tris = {name: [] for name in setup["colours"]}
    plan = intervals(setup, dmax)
    for lo, hi, name in plan:
        whole = (fmin >= lo) & (fmax < hi)
        tris[name].append(verts[faces[whole]])
        cut = []
        for f in np.where(~whole & (fmax >= lo) & (fmin < hi))[0]:
            poly = clip([(verts[j], height[j]) for j in faces[f]], lo, True)
            if len(poly) >= 3:
                poly = clip(poly, hi, False)
            for k in range(1, len(poly) - 1):
                cut.append([poly[0][0], poly[k][0], poly[k + 1][0]])
        if cut:
            tris[name].append(np.array(cut))

    pieces, total = [], 0.0
    for name, chunks in tris.items():
        chunks = [c for c in chunks if len(c)]
        if not chunks:
            continue
        t = np.concatenate(chunks) / 1000.0  # metres, as the MuJoCo model expects
        mesh = trimesh.Trimesh(t.reshape(-1, 3), np.arange(len(t) * 3).reshape(-1, 3))
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-12))
        mesh.remove_unreferenced_vertices()
        mesh.export(out_dir / f"{name}.stl")
        hexc = setup["colours"][name]
        rgb = [round(int(hexc[i:i + 2], 16) / 255, 4) for i in (1, 3, 5)]
        area = mesh.area * 1e6
        total += area
        pieces.append(dict(filament=name, file=f"{name}.stl", hex=hexc, rgba=rgb + [1],
                           triangles=len(mesh.faces), area_mm2=round(area, 1)))
    for p in pieces:
        p["area_share"] = round(p["area_mm2"] / total, 4)
    manifest = dict(
        setup=setup["name"], version=version,
        frame="upper_shell_link (as urdf/jumper/meshes/visual/upper_shell_link.stl)",
        units="metres",
        relief_intervals_mm=[dict(lo=round(max(lo, 0.0), 3), hi=round(min(hi, dmax), 3),
                                  filament=n) for lo, hi, n in plan],
        pieces=pieces,
        area_check_mm2=dict(pieces=round(total, 1), printable=round(shell.area, 1)),
    )
    (out_dir / "colours.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    m = split(sys.argv[1])
    for p in m["pieces"]:
        print(f"{p['filament']:>7} {p['hex']} {p['triangles']:>6} tris {p['area_share']:.1%}")
    print("area check (mm2):", m["area_check_mm2"])
