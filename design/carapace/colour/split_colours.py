"""Split a carapace version's printable top shell into one surface mesh per filament.

The colour pattern follows relief: every point of the shell's surface gets its height above
the stock top shell (distance to the stock visual mesh), and that height decides the filament.

- `stock`: the untouched stock surface (within `stock_tolerance_mm`; on v4b the stock surface
  sits within 1e-4 mm and the first added surface starts at about 5e-3 mm, so the split is
  clean).
- `accent`: added surface lower than `accent_below_mm` -- the thin CAD-blend leftovers that
  form the striping, plus the feathered edge where horns meet the shell.
- `c0..c(k-1)`: bands from an integer sequence. Band n is a(n) * unit_mm thick, counted up from
  the stock surface (`grow_from: "base"`) or down from the highest point of the relief
  (`"tip"`); its filament is a(n) mod k (`"a_mod_k"`) or n mod k (`"n_mod_k"`).

Optional `min_band_mm` (the nozzle width) applies the colour preview's printability rule:
every visible colour region is at least MIN_BAND_MARGIN x min_band_mm thick in relief, which
keeps it at least min_band_mm wide on the surface. The preview already raises the unit and the
accent cut-off to fit, so here they are only checked; slivers left where a band is cut off at
the accent or the top are folded into the neighbouring band (the one below, or the one above
for the lowest), as the preview folds them. Stepped fade steps are checked the same way.

Optional `stock_fade` splits the stock surface into stepped colours fading from
`colours.stock` (top) to `stock_fade.low` (bottom), as the preview's stepped fade does, in
either of the preview's modes:

- `mode: "height"`: level cuts, a percentage of the way down from the top of the stock shell
  (0 %) to its lowest point (100 %).
- `mode: "wall"` (the preview's "down the sides"): a percentage of the way down the side, from
  the rim of the top surface (0 %) to the lower edge (100 %), so the whole top keeps the top
  colour however it slopes. Rim and lower-edge heights are found per degree around the shell
  and smoothed exactly as the preview found them (`wall_curves`); the fraction is cut along
  linearly within each triangle, as the preview's shader interpolates it.
 The step colours are interpolated in
linear RGB, as the preview interpolates them. Pieces are `stock_s0` (bottom, the low colour) to
`stock_s<steps-1>` (top, `colours.stock`). A smooth fade (steps 0) cannot be split into
filaments and is refused.

Triangles that straddle a band edge are cut along the iso-line (height is interpolated
linearly across each triangle), so colour boundaries are smooth rather than stepped.

Output, in the setup's folder: `<filament>.stl` per filament (metres, `upper_shell_link`
frame, the frame of urdf/jumper/meshes/visual/upper_shell_link.stl) and `colours.json`.

    python design/carapace/colour/split_colours.py design/carapace/colour/<folder>/setup.json
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[3]
STOCK_VISUAL = REPO / "assets/jumper/urdf/jumper/meshes/visual/upper_shell_link.stl"
# Height-mode fade frame, as the preview defines it: the stock shell's lowest point (z, mm, in
# upper_shell_link) and its height range to the top (the stock visual spans -7.502..40.397).
FADE_Z0_MM, FADE_RANGE_MM = -7.5, 47.9
# Relief is a distance, so in theory a band is never narrower on the surface than it is thick;
# on the meshes the relief climbs up to 1.21 mm per mm along the surface (v4 and v4b, steepest
# triangle; 1.13 on v3, 1.15 on v3b), so bands are kept 1.25 x the nozzle thick. Must match the
# preview's MARGIN.
MIN_BAND_MARGIN = 1.25


def srgb_to_linear(c):
    return np.where(c < 0.04045, c * 0.0773993808, (c * 0.9478672986 + 0.0521327014) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, c ** 0.41666 * 1.055 - 0.055)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def wall_curves(stock):
    """Rim of the top surface and lower edge, z per degree around the stock shell (mm).

    The top surface is the largest connected patch of upward faces (normal z > 0.5); the rim
    height is the mean z of its outermost 10 % of samples in each 1-degree sector, the lower
    edge the lowest z among the outer 30 % of surface samples there; both are smoothed with a
    Gaussian of sigma 8 degrees. Same seeds and steps as the preview, which agrees with this to
    within half of its 8-bit rounding.
    """
    up = np.where(stock.face_normals[:, 2] > 0.5)[0]
    adj = stock.face_adjacency
    both = np.isin(adj[:, 0], up) & np.isin(adj[:, 1], up)
    graph = nx.Graph()
    graph.add_nodes_from(up)
    graph.add_edges_from(adj[both])
    top = list(max(nx.connected_components(graph),
                   key=lambda c: stock.area_faces[list(c)].sum()))
    top_pts, _ = trimesh.sample.sample_surface_even(stock.submesh([top], append=True), 60000,
                                                    seed=1)
    all_pts, _ = trimesh.sample.sample_surface_even(stock, 120000, seed=2)
    nb = 360
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    def binned(pts, pick):
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        rad = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.clip(np.digitize(ang, edges) - 1, 0, nb - 1)
        out = np.full(nb, np.nan)
        for i in range(nb):
            sel = idx == i
            if sel.sum() >= 3:
                out[i] = pick(pts[sel], rad[sel])
        return out

    def smooth(x, sigma=8):
        i = np.arange(nb)
        ok = ~np.isnan(x)
        x = np.interp(i, i[ok], x[ok], period=nb)
        g = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
        g /= g.sum()
        return np.convolve(np.r_[x[-3 * sigma:], x, x[:3 * sigma]], g, "valid")

    rim = binned(top_pts, lambda p, r: p[np.argsort(r)[-max(3, len(r) // 10):], 2].mean())
    low = binned(all_pts, lambda p, r: p[r >= np.percentile(r, 70), 2].min())
    return centres, smooth(rim), smooth(low)


def wall_fraction_fn(stock):
    """Point (mm) -> fraction of the way up the side, 0 at the lower edge, 1 at the rim."""
    centres, rim, low = wall_curves(stock)

    def frac(pts):
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        r = np.interp(ang, centres, rim, period=2 * np.pi)
        lo = np.interp(ang, centres, low, period=2 * np.pi)
        return np.clip((pts[:, 2] - lo) / np.maximum(r - lo, 1e-3), 0, 1)

    return frac, float((rim - low).min())


def fade_steps(fade, top_hex, stock):
    """Stepped fade: (field, [(lo, hi, hex)] per step, bottom step first).

    `field` maps points (mm) to the quantity the steps are cut along: z in mm for the height
    mode, the wall fraction for the wall mode.
    """
    if fade["mode"] not in ("height", "wall"):
        raise ValueError("stock_fade mode must be 'height' or 'wall'")
    n = int(fade["steps"])
    if n < 2:
        raise ValueError("a smooth fade (steps 0) cannot be split into filaments; use 2..8 steps")
    f_top, f_bot = 1 - fade["from_pct"] / 100, 1 - fade["to_pct"] / 100
    nz = fade.get("min_band_mm")
    if fade["mode"] == "wall":
        frac, min_span = wall_fraction_fn(stock)
        if nz and min_span * (f_top - f_bot) / n < nz * MIN_BAND_MARGIN - 1e-9:
            raise ValueError(f"{n} fade steps are thinner than {nz} mm on the shortest wall")
        z = [-np.inf] + [f_bot + k / n * (f_top - f_bot) for k in range(1, n)] + [np.inf]
        field = frac
    else:
        if nz and FADE_RANGE_MM * (f_top - f_bot) / n < nz - 1e-9:
            raise ValueError(f"{n} fade steps are thinner than {nz} mm")
        z = [-np.inf] + [FADE_Z0_MM + FADE_RANGE_MM * (f_bot + k / n * (f_top - f_bot))
                         for k in range(1, n)] + [np.inf]

        def field(pts):
            return pts[:, 2]
    lo_lin, top_lin = srgb_to_linear(hex_rgb(fade["low"])), srgb_to_linear(hex_rgb(top_hex))
    out = []
    for k in range(n):
        rgb = linear_to_srgb(lo_lin + (top_lin - lo_lin) * k / (n - 1))
        out.append((z[k], z[k + 1], "#" + "".join(f"{round(float(c) * 255):02X}" for c in rgb)))
    return field, out


def split_by_field(tris, field, z_lo, z_hi):
    """Triangles (N, 3, 3) clipped to z_lo <= field < z_hi, the field taken at the corners and
    interpolated linearly along the edges."""
    tz = field(tris.reshape(-1, 3)).reshape(-1, 3)
    whole = (tz.min(1) >= z_lo) & (tz.max(1) < z_hi)
    out = [tris[whole]]
    cut = []
    for t, tv in zip(tris[~whole & (tz.max(1) >= z_lo) & (tz.min(1) < z_hi)],
                     tz[~whole & (tz.max(1) >= z_lo) & (tz.min(1) < z_hi)]):
        poly = list(zip(t, tv))
        if np.isfinite(z_lo):
            poly = clip(poly, z_lo, True)
        if len(poly) >= 3 and np.isfinite(z_hi):
            poly = clip(poly, z_hi, False)
        for k in range(1, len(poly) - 1):
            cut.append([poly[0][0], poly[k][0], poly[k + 1][0]])
    if cut:
        out.append(np.array(cut))
    return np.concatenate(out)


def sequence(kind, count):
    """The first `count` terms, generated exactly as the colour preview generates them."""
    a = []
    if kind == "fibonacci":
        a = [1, 1]
        while len(a) < count:
            a.append(a[-1] + a[-2])
    elif kind == "padovan":
        a = [1, 1, 1]
        while len(a) < count:
            a.append(a[-2] + a[-3])
    elif kind == "triangular":
        a = [i * (i + 1) // 2 for i in range(1, count + 1)]
    elif kind == "primes":
        p = 2
        while len(a) < count:
            if all(p % q for q in range(2, int(p ** 0.5) + 1)):
                a.append(p)
            p += 1
    elif kind == "kolakoski":
        a = [1, 2, 2]
        i = 2
        while len(a) < count:
            a.extend([1 if i % 2 == 0 else 2] * a[i])
            i += 1
    elif kind == "thue_morse_plus_1":
        a = [bin(i).count("1") % 2 + 1 for i in range(count)]
    else:
        raise ValueError(f"unknown sequence {kind!r}")
    return a[:count]


def intervals(setup, dmax):
    """Relief intervals [lo, hi) in mm, each with its filament name."""
    tol = setup["stock_tolerance_mm"]
    acc = setup.get("accent_below_mm")  # null: no accent
    rule, k, unit = setup["colour_rule"], setup["k"], setup["unit_mm"]
    nz = setup.get("min_band_mm")
    band_min = nz * MIN_BAND_MARGIN if nz else 0.0
    if nz:
        a_min = min(sequence(setup["sequence"], 60))
        if a_min * unit < band_min - 1e-9:
            raise ValueError(f"unit {unit} mm makes bands thinner than {band_min} mm")
        if acc is not None and acc < band_min - 1e-9:
            raise ValueError(f"accent cut-off {acc} mm is thinner than {band_min} mm")
    if setup["grow_from"] not in ("base", "tip"):
        raise ValueError("grow_from must be 'base' (shell surface up) or 'tip' (highest tip down)")
    bands, done = [], 0.0
    for n, a in enumerate(sequence(setup["sequence"], 2000), start=1):
        if done >= dmax:
            break
        w = a * unit
        if setup["grow_from"] == "base":
            lo, hi = done, min(dmax, done + w)
        else:  # counted down from the highest point of the relief
            lo, hi = max(0.0, dmax - done - w), dmax - done
        done += w
        bands.append((lo, hi, f"c{(a if rule == 'a_mod_k' else n) % k}"))
    bands.sort()
    out = [(-1.0, tol, "stock")] + ([(tol, acc, "accent")] if acc is not None else [])
    floor = acc if acc is not None else tol
    vis = []
    for lo, hi, name in bands:
        lo = max(lo, floor)
        if hi > lo:
            vis.append([lo, hi, name])
    if nz:
        vis = fold_slivers(vis, band_min)
    out += [tuple(v) for v in vis]
    out[-1] = (out[-1][0], dmax + 1.0, out[-1][2])
    return out


def merge_same(regions):
    out = []
    for r in regions:
        if out and out[-1][2] == r[2]:
            out[-1][1] = r[1]
        else:
            out.append(list(r))
    return out


def fold_slivers(regions, band_min):
    """Fold regions thinner than band_min into a neighbour, exactly as the preview does."""
    regions = merge_same(regions)
    folded = True
    while folded and len(regions) > 1:
        folded = False
        for i, (lo, hi, _) in enumerate(regions):
            if hi - lo < band_min - 1e-9:
                if i > 0:
                    regions[i - 1][1] = hi
                else:
                    regions[1][0] = lo
                regions.pop(i)
                regions = merge_same(regions)
                folded = True
                break
    return regions


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

    colours = dict(setup["colours"])
    steps = []
    if setup.get("stock_fade"):
        stock_tris = np.concatenate([c for c in tris.pop("stock") if len(c)])
        top_hex = colours.pop("stock")
        field, steps = fade_steps(
            {**setup["stock_fade"], "min_band_mm": setup.get("min_band_mm")}, top_hex, stock)
        fade_tris = {}
        for k, (z_lo, z_hi, hexc) in enumerate(steps):
            fade_tris[f"stock_s{k}"] = [split_by_field(stock_tris, field, z_lo, z_hi)]
            colours[f"stock_s{k}"] = hexc
        tris = {**fade_tris, **tris}

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
        hexc = colours[name]
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
    if steps and setup["stock_fade"]["mode"] == "wall":  # fraction up the side; open ends null
        manifest["stock_fade_steps_wall_fraction"] = [
            dict(piece=f"stock_s{k}", hex=h, lo=round(a, 4) if np.isfinite(a) else None,
                 hi=round(b, 4) if np.isfinite(b) else None)
            for k, (a, b, h) in enumerate(steps)]
    elif steps:  # z in mm, upper_shell_link frame; open ends are null
        manifest["stock_fade_steps_mm"] = [
            dict(piece=f"stock_s{k}", hex=h, z_lo=round(a, 3) if np.isfinite(a) else None,
                 z_hi=round(b, 3) if np.isfinite(b) else None)
            for k, (a, b, h) in enumerate(steps)]
    (out_dir / "colours.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    m = split(sys.argv[1])
    for p in m["pieces"]:
        print(f"{p['filament']:>7} {p['hex']} {p['triangles']:>6} tris {p['area_share']:.1%}")
    print("area check (mm2):", m["area_check_mm2"])
