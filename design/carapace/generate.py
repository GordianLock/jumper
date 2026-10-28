"""Parametric carapace generator: horns (crests) grown on the stock Jumper top shell.

usage: python design/carapace/generate.py design/carapace/versions/<NAME>/params.json

Writes, next to the params file:
  carapace.stl  -- the material added on top of the stock top shell, base_link frame, metres
  report.json   -- clearance, overhang, thickness and mass checks

Every horn is built from its root a, tip b and radius alone, so changing a horn's length or
angle is a parameter change and the blending holds:

  spine   the horn study's capsule (same axis, same radius), with a ball tip, carried on
          behind the root along the same line until its top meets the shell
  strut   under the tip: a tube of the spine's radius whose centre line leaves the tip along
          the spine and curves down to meet the shell vertically (quadratic Bezier), with a
          web filling up to the spine. Its front is rounded; it fades in behind the ball tip
          so the tip stays a sphere
  walls   under the spine the sides leave it junction_angle_deg above its widest line, along
          its own tangent, and curve out (concave) to land flat on the shell under_flare
          wider. Where a spine runs low they become an arc of base_fillet_radius instead
  flanks  wide concave flanks along the horn, easing to the wall's width at the strut

Neighbouring horns meet in rounded valleys (valley_blend). Material is only ever added on
top of the shell; the legs' swept volume from the horn study (mirror-symmetrised, 5 mm
clearance included) is cut away, except the study's own capsules, which are never trimmed.
The display screen stays uncovered.

The geometry is a signed-distance field on a 1 mm grid, meshed with marching cubes. Run
from the repository root.
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from matplotlib.path import Path as Polygon
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from skimage.measure import marching_cubes

REPO = Path(__file__).resolve().parents[2]
VISUAL = REPO / "assets/jumper/urdf/jumper/meshes/visual"
KEEPOUT = REPO / "studies/carapace/leg_keepout_side_voxels.npz"

#: The stock top shell (upper_shell_link) and display module, placed in base_link
#: (offsets from assets/jumper/jumper.xml).
SHELL_OFF = np.array([-0.0073137, 0, 0.0315])
DISPLAY_OFF = np.array([0.0802718, 1.01591e-05, 0.0495238])

#: Defaults; a version's params.json lists every value it was built with.
DEFAULTS = dict(
    vox=0.001,                      # grid spacing (m)
    top_min_z=0.035,                # the carapace covers the upward-facing shell top:
    top_max_slope=0.6,              # gentle slope only, never the steep sides or front face
    dome_rise=0.0,                  # optional dome over the shell top (0: none)
    dome_xc=-0.012, dome_ax=0.105, dome_ay=0.088, dome_p=0.7, dome_edge_blend=0.018,
    bumps=[],                       # optional low mounds: dict(c=(x, y), h=height, r=radius)
    flank_width=0.032,              # wide flanks: half-width beyond the spine at flank_depth
    flank_depth=0.05,               # below the axis,
    flank_power=2.0,                # growing as depth**power (concave)
    flank_taper=1.0,                # flanks ease to the wall's width over this many reaches
    valley_blend=0.012,             # rounded valleys between horns and onto the shell
    support_fillet=0.012,           # concave fillet where tip supports meet the shell
    support_reach=0.37,             # strut front lands this share of the tip's height
                                    # behind the tip (then steps back to stay on the top)
    junction_angle_deg=10.0,        # walls leave the spine this far above its widest line
    under_flare=0.008,              # walls land on the shell this much wider per side
    base_fillet_radius=0.015,       # where a spine runs low, the wall is an arc this radius
    tip_round_len=1.5,              # junction and flare fade in over this many tip radii
    display_gap=0.002,              # material keeps this far from the display module
    edge_round=0.004,               # rounding where the carapace meets a boundary
    crests=[],                      # dict(name, a=root, b=tip, r=spine radius, r_tip)
)


def smin(a, b, k):
    """Smooth minimum (union with a fillet of about k)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    """Smooth maximum (intersection with a rounded edge of about k)."""
    return -smin(-a, -b, k)


def build(P):
    """Build the carapace for parameters P. Returns (mesh, report)."""
    def load(name, off):
        m = trimesh.load(VISUAL / f"{name}.stl")
        m.apply_translation(off)
        return m

    shell = load("upper_shell_link", SHELL_OFF)
    display = load("display_module_link", DISPLAY_OFF)
    ko = np.load(KEEPOUT)

    # ---- grid and the stock shell's upper surface ----
    v = P["vox"]
    xs = np.arange(-0.125, 0.115 + v, v)
    ys = np.arange(-0.105, 0.105 + v, v)
    zs = np.arange(0.035, 0.165 + v, v)
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    orig = np.c_[X2.ravel(), Y2.ravel(), np.full(X2.size, 0.3)]
    locs, idx, _ = shell.ray.intersects_location(orig, np.tile([0, 0, -1.0], (X2.size, 1)),
                                                 multiple_hits=False)
    Zs = np.full(X2.size, np.nan)
    Zs[idx] = locs[:, 2]
    Zs = Zs.reshape(X2.shape)
    gx, gy = np.gradient(Zs, v)
    gentle = ~np.isnan(Zs) & (np.hypot(gx, gy) < P["top_max_slope"]) & (Zs > P["top_min_z"])
    lab, _ = label(gentle)
    top = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    shell_or_top = np.where(np.isnan(Zs), 0.072, Zs)

    # optional mounds and dome (both off in the current versions)
    bumps_h = np.zeros_like(X2)
    for bp in P["bumps"]:
        d = np.hypot(X2 - bp["c"][0], Y2 - bp["c"][1]) / bp["r"]
        bumps_h += np.where(d < 1, bp["h"] * np.cos(0.5 * np.pi * np.clip(d, 0, 1)) ** 2, 0)
    e_ = np.clip(distance_transform_edt(top) * v / P["dome_edge_blend"], 0, 1)
    blend = e_ * e_ * (3 - 2 * e_)
    r_ = 1 - ((X2 - P["dome_xc"]) / P["dome_ax"]) ** 2 - (Y2 / P["dome_ay"]) ** 2
    dome = P["dome_rise"] * np.clip(r_, 0, None) ** P["dome_p"] * blend
    # the reference surface sits 2 mm inside the shell where nothing is added, so it adds no
    # skin, but fillets still form against the shell surface
    Ztop = np.where(top, Zs - 0.002 + dome + bumps_h * blend, -1.0)
    Zbase = np.where(top, Zs - 0.001, 1.0)
    Zfloor_caps = np.where(top, Zs - 0.001, -1.0)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    FP = P["flank_power"]
    FA = P["flank_width"] / P["flank_depth"] ** FP

    def grid_at(px, py):
        return (int(np.clip(np.searchsorted(xs, px), 0, len(xs) - 1)),
                int(np.clip(np.searchsorted(ys, py), 0, len(ys) - 1)))

    def root_extension(c):
        """How far the spine runs on behind its root, along its own line, before its top
        meets the shell. Crossing another horn, they fuse; it stops early only at the legs'
        path or the edge of the shell."""
        a, b = np.asarray(c["a"]), np.asarray(c["b"])
        d = (b - a) / np.linalg.norm(b - a)
        lift = c["r"] / np.hypot(d[0], d[1])    # capsule top above the axis, vertically
        for e in np.arange(0, 0.2, 0.001):
            p = a - e * d
            hit, _, _ = shell.ray.intersects_location([[p[0], p[1], 0.3]], [[0, 0, -1.0]],
                                                      multiple_hits=False)
            if not len(hit):
                return e, "leaves the shell"
            if p[2] + lift <= hit[0, 2]:
                return e, "top meets the shell"
            ijk = np.floor((p - ko["origin"]) / float(ko["voxel"])).astype(int)
            if ko["occ"][tuple(ijk)]:
                return e, "stops at the legs' path"
        return e, "limit"

    root_ext = {c["name"]: root_extension(c) for c in P["crests"]}

    def horn_frame(c):
        a, b = np.asarray(c["a"], float), np.asarray(c["b"], float)
        hd = (b - a)[:2]
        run = np.linalg.norm(hd)
        hd = hd / run
        slope = (b[2] - a[2]) / run
        th = np.arctan(slope)
        axis3 = (b - a) / np.linalg.norm(b - a)
        ext = root_ext[c["name"]][0]
        a_ext = a - axis3 * ext
        u_end = -(run + ext * np.cos(th))
        # the strut's foot lands on the shell's upper surface, front face included
        z_F = a[2]
        for _ in range(3):
            reach = P["support_reach"] * (b[2] - z_F)
            u_F = -(reach + c["r"])
            while u_F > -run:
                front = grid_at(b[0] + (u_F + c["r"]) * hd[0], b[1] + (u_F + c["r"]) * hd[1])
                if top[front]:
                    break
                u_F -= 0.001
            zf = Zs[grid_at(b[0] + u_F * hd[0], b[1] + u_F * hd[1])]
            z_F = float(zf) if np.isfinite(zf) else 0.072
        reach = -u_F
        z_C = max(b[2] + slope * u_F, z_F + 1e-4)
        return dict(a=a, b=b, hd=hd, slope=slope, a_ext=a_ext, u_end=u_end,
                    u_F=u_F, z_F=z_F, z_C=z_C, reach=reach)

    def junction_depth(c, f, fade=1.0):
        cth = np.cos(np.arctan(f["slope"]))
        return -(c["r"] / cth) * np.sin(np.radians(P["junction_angle_deg"]) * fade)

    def wall_profile(c, f, u, Zq, h_cap=None, fade=1.0):
        """Half-width W(h) of the wall under a spine at depth h below its axis (h < 0 above
        it), and dW/dh. In the vertical section the spine is an ellipse (half-width r,
        half-height r/cos(slope)). The wall leaves it junction_angle above its widest line
        along its tangent and lands flat on the shell under_flare wider (a quadratic Bezier
        bowing into the corner); where the drop is too tall for that, it follows the tangent.
        Where the spine runs low it is an arc of base_fillet_radius instead, meeting the
        spine above the junction line. fade (0..1) scales junction and flare (0 near the
        tip); h_cap is a depth below which the wall stops widening."""
        b, r, slope = f["b"], c["r"], f["slope"]
        cth = np.cos(np.arctan(slope))
        e = r / cth
        phi = np.radians(P["junction_angle_deg"]) * fade
        w_J, h_J, m = r * np.cos(phi), -e * np.sin(phi), np.tan(phi) * cth
        za = b[2] + slope * u
        shell_z = shell_or_top[:, :, None]
        D = np.maximum(za - shell_z - h_J, 0.001)
        wS = r + P["under_flare"] * fade
        wC = w_J + m * D
        filleted = wS > wC
        h = za - Zq
        hu = h if h_cap is None else np.minimum(h, h_cap)
        t = np.clip(hu - h_J, 0, None)
        tt = np.minimum(t, D)
        sB = 1 - np.sqrt(np.clip(1 - tt / D, 0, 1))
        Wb = (1 - sB) ** 2 * w_J + 2 * sB * (1 - sB) * wC + sB ** 2 * wS
        dW_ds = 2 * (1 - sB) * (wC - w_J) + 2 * sB * (wS - wC)
        dt_ds = np.maximum(2 * D * (1 - sB), 1e-6)
        WA = np.where(filleted, np.where(t >= D, wS, Wb), w_J + m * t)
        dWA = np.where(filleted, np.where(t >= D, 0.0, np.minimum(dW_ds / dt_ds, 50.0)), m)
        # the low-spine arc, worked in a section scaled vertically by cos(slope)
        H = za - shell_z
        Rf = P["base_fillet_radius"]
        Hs = H * cth
        ysc = (H - hu) * cth

        def ell(yy):
            return r * np.sqrt(np.clip(1 - ((Hs - yy) / r) ** 2, 0, 1))

        def dell_dh(yy):
            q = np.clip((Hs - yy) / r, -0.999, 0.999)
            return -cth * q / np.sqrt(1 - q * q)

        wc = np.sqrt(np.clip((r + Rf) ** 2 - (Hs - Rf) ** 2, 0, None))
        ysT = Hs + r * (Rf - Hs) / (r + Rf)
        arc_in = np.clip(Rf ** 2 - (ysc - Rf) ** 2, 1e-12, None)
        WB = np.where(ysc < 0, wc, np.where(ysc < ysT, wc - np.sqrt(arc_in), ell(ysc)))
        dWB = np.where(ysc < 0, 0.0,
                       np.where(ysc < ysT, np.minimum(cth * (Rf - ysc) / np.sqrt(arc_in), 50.0),
                                dell_dh(ysc)))
        h_T = H - ysT / cth
        Hs_lo = Rf - (r + Rf) * np.sin(phi)
        Hs_hi = np.maximum(Rf, Hs_lo + 1e-4)
        kb = np.clip((Hs_hi - Hs) / (Hs_hi - Hs_lo), 0, 1)
        beta = kb * kb * (3 - 2 * kb)
        WA = np.where(hu >= h_J, WA, ell((H - hu) * cth))
        dWA = np.where(hu >= h_J, dWA, dell_dh((H - hu) * cth))
        W = (1 - beta) * WA + beta * WB
        dW = (1 - beta) * dWA + beta * dWB
        h_top = np.where(beta > 0, np.minimum(h_J, h_T), h_J)
        return h, h_top, W, dW

    def crest_sdf(c, tip_zone=False):
        """Signed distance to one horn: spine, ball and flanks (tip_zone=False) or the strut
        and web under the tip (tip_zone=True). u: horizontal distance along the horn from the
        tip (negative towards the root); w: lateral distance from its vertical plane."""
        f = horn_frame(c)
        b, hd, r = f["b"], f["hd"], c["r"]
        u = (X - b[0]) * hd[0] + (Y - b[1]) * hd[1]
        w = -(X - b[0]) * hd[1] + (Y - b[1]) * hd[0]
        ball = np.sqrt((X - b[0]) ** 2 + (Y - b[1]) ** 2 + (Z - b[2]) ** 2) - c["r_tip"]
        if not tip_zone:
            ae = f["a_ext"]
            e = b - ae
            t = np.clip(((X - ae[0]) * e[0] + (Y - ae[1]) * e[1] + (Z - ae[2]) * e[2])
                        / e.dot(e), 0, 1)
            cap = np.sqrt((X - ae[0] - t * e[0]) ** 2 + (Y - ae[1] - t * e[1]) ** 2
                          + (Z - ae[2] - t * e[2]) ** 2) - r
            del t
            hs, h_J, W, dW = wall_profile(c, f, u, Z)
            h = np.clip(hs, 0, None)
            k = np.clip((f["u_F"] - u) / (P["flank_taper"] * f["reach"]), 0, 1)
            tau = k * k * (3 - 2 * k)
            wide, dwide = r + tau * FA * h ** FP, tau * FA * FP * h ** (FP - 1)
            use_wide = (hs > 0) & (wide > W)
            width = np.where(use_wide, wide, W)
            dw = np.where(use_wide, dwide, dW)
            fl = (np.abs(w) - width) / np.sqrt(1 + dw * dw)
            fl = np.where(hs >= h_J, fl, np.inf)
            fl = np.maximum(fl, np.maximum(u - f["u_F"], f["u_end"] - u))
            return np.minimum(np.minimum(cap, ball), fl)
        u_F, z_F, z_C = f["u_F"], f["z_F"], f["z_C"]
        qq = np.clip(u / u_F, 0, 1)
        sq = 1 - np.sqrt(1 - qq)
        z_line = (1 - sq) ** 2 * b[2] + 2 * sq * (1 - sq) * z_C + sq ** 2 * z_F
        kf = np.clip(-u / (P["tip_round_len"] * c["r_tip"]), 0, 1)
        fade = kf * kf * (3 - 2 * kf)
        hs, h_J, W, dW = wall_profile(c, f, u, Z, h_cap=b[2] + f["slope"] * u - z_line,
                                      fade=fade)
        del qq, sq, z_line, kf
        res = 0.0005
        ug = np.arange(u_F - r - 0.015, c["r_tip"] + 0.015, res)
        zg = np.arange(z_F - r - 0.015, b[2] + c["r_tip"] + 0.015, res)
        UG, ZG = np.meshgrid(ug, zg, indexing="ij")
        sB = 1 - np.sqrt(1 - np.clip(UG / u_F, 0, 1))
        z_path = (1 - sB) ** 2 * b[2] + 2 * sB * (1 - sB) * z_C + sB ** 2 * z_F
        kg = np.clip(-UG / (P["tip_round_len"] * c["r_tip"]), 0, 1)
        top_line = b[2] + f["slope"] * UG - junction_depth(c, f, kg * kg * (3 - 2 * kg))
        region = (UG >= u_F) & (UG <= 0) & (ZG >= z_path) & (ZG <= top_line)
        d2 = distance_transform_edt(~region) * res
        dR = map_coordinates(d2, [((u - ug[0]) / res).ravel(), ((Z - zg[0]) / res).ravel()],
                             order=1, cval=1.0).reshape(X.shape)
        side = np.maximum(np.abs(w) - (W - r), 0)
        sup = np.maximum(np.sqrt(side * side + dR * dR) - r, h_J - hs)
        return np.minimum(sup, ball)

    def capsule_sdf(c):
        a, b = np.asarray(c["a"]), np.asarray(c["b"])
        ab = b - a
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2])
                    / ab.dot(ab), 0, 1)
        d = np.sqrt((X - a[0] - t * ab[0]) ** 2 + (Y - a[1] - t * ab[1]) ** 2
                    + (Z - a[2] - t * ab[2]) ** 2) - c["r"]
        tip = np.sqrt((X - b[0]) ** 2 + (Y - b[1]) ** 2 + (Z - b[2]) ** 2) - c["r_tip"]
        return np.minimum(d, tip)

    # ---- horns, valleys, shell ----
    sdf = Z - Ztop[:, :, None]
    for c in P["crests"]:
        sdf = smin(sdf, crest_sdf(c), P["valley_blend"])
    ed = (distance_transform_edt(top) - distance_transform_edt(~top)) * v
    sdf = smax(sdf, -ed[:, :, None], P["edge_round"])
    sdf = np.maximum(sdf, Zbase[:, :, None] - Z)
    caps = np.full(X.shape, np.inf)
    for c in P["crests"]:
        caps = np.minimum(caps, capsule_sdf(c))
    caps = np.maximum(caps, Zfloor_caps[:, :, None] - Z)
    sdf = np.minimum(sdf, caps)
    # tip supports: on the shell's upper surface, or above shell-top height beyond it
    Zsup = np.where(top, Zs - 0.001, 0.072)
    shell_face = np.where(top[:, :, None], Z - (Zs - 0.002)[:, :, None], 1.0)
    sup_all = None
    for c in P["crests"]:
        sup = crest_sdf(c, tip_zone=True)
        sup_all = sup if sup_all is None else smin(sup_all, sup, P["valley_blend"])
    if sup_all is not None:
        near = sup_all < 3 * P["support_fillet"]
        sup_all = np.where(near, smin(sup_all, shell_face, P["support_fillet"]), sup_all)
        sup_all = np.maximum(sup_all, Zsup[:, :, None] - Z)
        sdf = np.minimum(sdf, sup_all)

    # ---- the legs' path (mirror-symmetrised) is cut away, but never the study capsules ----
    def sample_ko(XX, YY, ZZ):
        g = np.floor((np.stack([XX, YY, ZZ], -1) - ko["origin"]) / float(ko["voxel"])).astype(int)
        inside = np.all((g >= 0) & (g < ko["occ"].shape), axis=-1)
        out = np.zeros(XX.shape, bool)
        out[inside] = ko["occ"][g[inside][:, 0], g[inside][:, 1], g[inside][:, 2]]
        return out

    occ = sample_ko(X, Y, Z) | sample_ko(X, -Y, Z)
    carved = int(((sdf < 0) & occ).sum())
    caps_in_keepout = int(((caps < 0) & occ).sum())
    kd = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * v
    sdf = smax(sdf, -kd, P["edge_round"])
    sdf = np.minimum(sdf, caps)

    # ---- the display screen stays uncovered ----
    lo_, hi_ = display.bounds[0] - P["display_gap"], display.bounds[1] + P["display_gap"]
    inside_box = np.max(np.stack([lo_[0] - X, lo_[1] - Y, lo_[2] - Z,
                                  X - hi_[0], Y - hi_[1], Z - hi_[2]]), axis=0)
    sdf = smax(sdf, -inside_box, P["edge_round"])

    # ---- mesh ----
    sdf[[0, -1], :, :] = sdf[:, [0, -1], :] = v
    sdf[:, :, [0, -1]] = v
    sdf = np.where(sdf == 0, 1e-9, sdf)
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(v, v, v))
    verts += np.array([xs[0], ys[0], zs[0]])
    car = trimesh.Trimesh(verts, faces, process=True)
    car = trimesh.util.concatenate([p for p in car.split(only_watertight=False)
                                    if abs(p.volume) > 5e-8])
    if not car.is_watertight:
        trimesh.repair.fill_holes(car)
        trimesh.repair.fix_normals(car)
    if car.volume < 0:
        car.invert()

    # ---- checks ----
    solid = sdf < 0
    pts = np.c_[X[solid], Y[solid], Z[solid]]
    hit = (occ & (caps >= 0))[solid]
    outside = ~Polygon(ko["hull"]).contains_points(pts[:, :2])
    over = 0.0
    if outside.any():
        hp = ko["hull"]
        seg_a, seg_b = hp, np.roll(hp, -1, axis=0)
        q = pts[outside][:, :2]
        ab = seg_b - seg_a
        t = np.clip(((q[:, None, :] - seg_a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
        proj = seg_a + t[..., None] * ab
        over = float(np.min(np.linalg.norm(q[:, None, :] - proj, axis=-1), axis=1).max())
    vol = solid.sum() * v ** 3
    report = dict(
        legs_path_overlap_mm3=int(hit.sum()),
        study_capsules_in_legs_path_mm3=caps_in_keepout,
        trimmed_for_legs_mm3=carved,
        overhang_past_stock_outline_mm=round(over * 1000, 1),
        volume_cm3=round(vol * 1e6, 1),
        mass_g_solid_pla=round(vol * 1e6 * 1.24),
        # typical print: 1.2 mm of walls over the whole surface plus 15 % infill inside
        mass_g_typical_print=round(1.24e6 * (car.area * 0.0012
                                             + 0.15 * max(vol - car.area * 0.0012, 0))),
        watertight=bool(car.is_watertight),
        root_extension_mm={k: [round(e * 1000), why] for k, (e, why) in root_ext.items()},
    )
    return car, report


def main():
    params_path = Path(sys.argv[1]).resolve()
    P = dict(DEFAULTS)
    P.update(json.loads(params_path.read_text()))
    car, report = build(P)
    out = params_path.parent
    car.export(out / "carapace.stl")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
