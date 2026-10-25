#!/usr/bin/env python3
"""Validate generated fabrication geometry and MuJoCo handoff invariants."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

import generate as g


def assert_close(a, b, tolerance=1e-6):
    if not np.allclose(a, b, atol=tolerance, rtol=0):
        raise AssertionError(f"Unexpected mismatch: {a} vs {b}")


def main():
    manifest_path = g.OUTPUT / "bioshell-v01-manifest.json"
    data = json.loads(manifest_path.read_text())
    stock_v, stock_f = g.load_stl(g.SOURCE)
    source_xml = ET.parse(g.SOURCE_MJCF).getroot()
    expected_source_geom_names = [x.get("name") for x in source_xml.findall(".//geom")]
    optional_mujoco = None
    try:
        import mujoco
        optional_mujoco = mujoco
    except ImportError:
        pass

    results = []
    for stem, spec in data["variants"].items():
        files = {key: g.ROOT / spec[key] for key in
                 ("visual_mesh", "collision_mesh", "print_stl_mm", "colored_preview_ply", "colored_print_3mf", "mjcf")}
        for key, path in files.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"Missing or empty {key}: {path}")
        v, f = g.load_stl(files["visual_mesh"])
        p, pf = g.load_stl(files["print_stl_mm"])
        collider_v, collider_f = g.load_stl(files["collision_mesh"])
        if not g.watertight(f) or not g.watertight(collider_f) or not g.watertight(pf):
            raise AssertionError(f"Nonmanifold mesh in {stem}")
        if len(v) != len(stock_v) or len(f) != len(stock_f):
            raise AssertionError(f"Altered shell topology in {stem}")
        # STLs are triangle soups; equal coordinates are sorted by load_stl.
        # Validate separate millimeter print file using AABBs and volumes.
        assert_close(p.min(0), v.min(0) * 1000, .00005)
        assert_close(p.max(0), v.max(0) * 1000, .00005)
        x, y, z = stock_v.T
        protected = (z <= .010) | (x >= .078) | (x <= -.086) | (np.abs(y) >= .081)
        assert_close(v[protected], stock_v[protected], 2e-7)
        assert_close(v[:, 0:2], stock_v[:, 0:2], 2e-7)
        if np.any(v[:, 2] + 1e-7 < stock_v[:, 2]):
            raise AssertionError("An indentation may compromise original internal clearance")
        props = g.mass_properties(v, f, data["density_kg_m3"])
        assert_close(props["mass_kg"], spec["mass_kg"], .00002)
        # Every original support extreme should be close to the convex proxy.
        rng = np.random.default_rng(2026)
        normals = rng.normal(size=(500, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        gaps = np.max(v @ normals.T, axis=0) - np.max(collider_v @ normals.T, axis=0)
        max_gap = float(np.max(gaps))
        if max_gap > .00075:
            raise AssertionError(f"Collision proxy misses silhouette by {max_gap*1000:.2f} mm")
        # Color model is a real 3MF part in printer-native millimeter units.
        with zipfile.ZipFile(files["colored_print_3mf"]) as zf:
            model = ET.fromstring(zf.read("3D/3dmodel.model"))
            if model.get("unit") != "millimeter":
                raise AssertionError("3MF incorrectly scaled")
            names = {e.tag.split("}")[-1] for e in model.iter()}
            if not {"basematerials", "mesh", "vertices", "triangles"} <= names:
                raise AssertionError("3MF missing core mesh/color elements")
        xml_root = ET.parse(files["mjcf"]).getroot()
        geom_names = [x.get("name") for x in xml_root.findall(".//geom")]
        if geom_names != expected_source_geom_names:
            raise AssertionError("Shell XML changed the robot's named geoms")
        if optional_mujoco is not None:
            model = optional_mujoco.MjModel.from_xml_path(str(files["mjcf"]))
            sim = optional_mujoco.MjData(model)
            optional_mujoco.mj_forward(model, sim)
            if not np.isfinite(sim.qacc).all():
                raise AssertionError("Nonfinite forward dynamics")
            idx = optional_mujoco.mj_name2id(model, optional_mujoco.mjtObj.mjOBJ_BODY, "upper_shell_link")
            assert_close(model.body_mass[idx], spec["mass_kg"], .000001)
        results.append({"design": stem, "watertight": True,
                        "manufacturing_scale_mm": True,
                        "protected_mount_and_front_shift_m": 0,
                        "collision_gap_random_directions_m": max_gap,
                        "mujoco_load": optional_mujoco is not None})
        print(f"OK {stem}: manufacturing scale, topology, protective zones, collider and MJCF checked; "
              f"proxy gap {max_gap*1000:.3f}mm")

    (g.OUTPUT / "bioshell-v01-validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Validation passed for all variants")


if __name__ == "__main__":
    main()
