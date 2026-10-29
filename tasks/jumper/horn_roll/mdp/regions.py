"""Carapace regions for a model, worked out from the model itself.

Impact on the carapace is measured per region, so that a hit landing across the seam
between two collision pieces of one horn counts in full, while load shared between
different horns or bumpers counts only its hardest region.

A carapace version's design parameters (`design/carapace/versions/<name>/params.json`,
from the carapace design session) give its horns ("crests": root `a`, tip `b`) and
bumpers ("spheres": top-view centre `c`). Each collision piece
(`upper_shell_carapace_NN_meshcol`) goes to:

- the horn whose axis passes within `HORN_REACH` of the piece's centroid, nearest
  first;
- else the bumper whose centre is within `BUMPER_REACH` of it in top view;
- else the front or rear field (valleys and flats), by the sign of x, the rear split
  left and right.

A model with no params.json (a path given by hand) gets one region per piece.
Checked against prototype 3 (CF1-tri30m-rb-v3, 2026-10-28): every bumper is one piece
of its own, as its params say; horns have 5-7 pieces each.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HORN_REACH = 0.03
BUMPER_REACH = 0.02
_PIECE = re.compile(r"^upper_shell_carapace_(\d+)_meshcol$")


def _params_for(asset: Path) -> dict | None:
    name = asset.stem.removeprefix("jumper_")
    root = asset.resolve().parents[2]                    # assets/jumper/<xml> -> repo
    p = root / "design" / "carapace" / "versions" / name / "params.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _seg_dist(p: np.ndarray, a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def carapace_regions(asset: Path | None) -> dict[str, tuple[int, ...]]:
    """Region name -> carapace piece numbers. Empty for a model without a carapace."""
    if asset is None:
        return {}
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(asset))
    d = mujoco.MjData(m)
    d.qpos[3] = 1.0
    mujoco.mj_kinematics(m, d)
    base = m.body("base_link").id
    origin, rot = d.xpos[base], d.xmat[base].reshape(3, 3)
    centroids = {}
    for g in range(m.ngeom):
        mt = _PIECE.match(m.geom(g).name)
        if not mt:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        centroids[int(mt.group(1))] = ((w - origin) @ rot).mean(axis=0)
    if not centroids:
        return {}
    params = _params_for(Path(asset))
    if params is None:
        return {f"piece_{i:02d}": (i,) for i in sorted(centroids)}

    regions: dict[str, list[int]] = {}
    for i, c in sorted(centroids.items()):
        horns = sorted((_seg_dist(c, h["a"], h["b"]), h["name"]) for h in params.get("crests", []))
        bumps = sorted((float(np.linalg.norm(c[:2] - np.asarray(s["c"][:2]))), k)
                       for k, s in enumerate(params.get("spheres", [])))
        if horns and horns[0][0] <= HORN_REACH:
            name = f"horn_{horns[0][1]}"
        elif bumps and bumps[0][0] <= BUMPER_REACH:
            name = f"bumper_{bumps[0][1]}"
        elif c[0] >= 0:
            name = "front_field"
        else:
            name = "rear_field_left" if c[1] >= 0 else "rear_field_right"
        regions.setdefault(name, []).append(i)
    return {k: tuple(v) for k, v in sorted(regions.items())}


__all__ = ["carapace_regions"]
