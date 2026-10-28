"""Turn a generated carapace into the printable part and a matching MuJoCo model.

usage: python design/carapace/build_version.py design/carapace/versions/<NAME>

Reads <NAME>/carapace.stl and report.json (from generate.py) and writes:

  <NAME>/printable_top_shell.3mf
      the replacement top shell: the stock upper_shell_link with the carapace fused on, in
      the shell's own frame, millimetres (what a slicer expects)
  <NAME>/model.json
      what went into the model: piece count, mass, inertia
  assets/jumper/carapace/<NAME>/hull_NN.stl, visual.stl
      the carapace split into convex pieces (MuJoCo collides each mesh as its convex hull,
      so a concave carapace needs this), and a lighter mesh to look at
  assets/jumper/jumper_<NAME>.xml
      jumper.xml with the carapace on upper_shell_link. It sits beside jumper.xml because
      that model finds its meshes relative to its own file (compiler meshdir="./").

The collision pieces are named upper_shell_carapace_NN_meshcol: the roll tasks select
collision geoms by the `_meshcol` ending (their shell-impact sensor and rolled-start
reset), and studies/carapace/gentle.py counts names starting `upper_shell` as shell.

upper_shell_link has an explicit <inertial>, so the carapace's mass is folded into it:
the mass is the typical-print estimate from report.json (1.2 mm walls, 15 % infill, PLA),
spread uniformly over the carapace's volume for the centre of mass and inertia.
Run from the repository root.
"""
import json
import re
import sys
from pathlib import Path

import coacd
import manifold3d
import mujoco
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
JUMPER = REPO / "assets/jumper"
SHELL_OFF = np.array([-0.0073137, 0, 0.0315])     # upper_shell_link in base_link


def to_manifold(m):
    return manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(m.vertices, np.float32),
        tri_verts=np.asarray(m.faces, np.uint32)))


def from_manifold(mf):
    mesh = mf.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts))


def inertia_tensor(diag, quat):
    """Full inertia from MuJoCo's principal moments and orientation (w, x, y, z)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    R = R.reshape(3, 3)
    return R @ np.diag(diag) @ R.T


def main():
    vdir = Path(sys.argv[1]).resolve()
    name = vdir.name
    report = json.loads((vdir / "report.json").read_text())
    car = trimesh.load(vdir / "carapace.stl")
    car.apply_translation(-SHELL_OFF)               # into upper_shell_link's frame
    assert car.is_watertight and car.volume > 0, "carapace.stl must be a closed, outward mesh"

    # ---- printable replacement top shell (mm) ----
    stock = trimesh.load(JUMPER / "urdf/jumper/meshes/visual/upper_shell_link.stl")
    fused = from_manifold(to_manifold(stock) + to_manifold(car))
    fused.apply_scale(1000.0)
    fused.export(vdir / "printable_top_shell.3mf")

    # ---- collision pieces and visual mesh ----
    adir = JUMPER / "carapace" / name
    adir.mkdir(parents=True, exist_ok=True)
    for old in adir.glob("*.stl"):
        old.unlink()
    light = car.simplify_quadric_decimation(face_count=20000)
    parts = coacd.run_coacd(coacd.Mesh(light.vertices, light.faces), threshold=0.03,
                            max_convex_hull=48, preprocess_mode="auto", seed=1)
    hull_files = []
    hull_vol = 0.0
    for i, (vv, ff) in enumerate(parts):
        hm = trimesh.Trimesh(vv, ff).convex_hull
        fn = adir / f"hull_{i:02d}.stl"
        hm.export(fn)
        hull_files.append(fn)
        hull_vol += hm.volume
    car.simplify_quadric_decimation(face_count=30000).export(adir / "visual.stl")

    # ---- mass and inertia, folded into upper_shell_link ----
    xml = (JUMPER / "jumper.xml").read_text()
    m_inert = re.search(r'(<body name="upper_shell_link"[^>]*>\s*)(<inertial ([^/]*)/>)', xml)
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m_inert.group(3)))
    m0 = float(attrs["mass"])
    c0 = np.array(attrs["pos"].split(), float)
    I0 = inertia_tensor(np.array(attrs["diaginertia"].split(), float), attrs["quat"].split())
    m1 = report["mass_g_typical_print"] / 1000.0
    props = car.copy()
    props.density = m1 / car.volume
    c1 = props.center_mass
    I1 = props.moment_inertia                       # about c1
    M = m0 + m1
    com = (m0 * c0 + m1 * c1) / M

    def shift(inertia, m, c):                       # parallel axis, to the combined centre
        d = c - com
        return inertia + m * (d.dot(d) * np.eye(3) - np.outer(d, d))

    Itot = shift(I0, m0, c0) + shift(I1, m1, c1)
    full = " ".join(f"{x:.6g}" for x in (Itot[0, 0], Itot[1, 1], Itot[2, 2],
                                         Itot[0, 1], Itot[0, 2], Itot[1, 2]))
    new_inertial = (f'<inertial pos="{com[0]:.6g} {com[1]:.6g} {com[2]:.6g}" '
                    f'mass="{M:.6g}" fullinertia="{full}"/>')

    # ---- the model ----
    rel = f"carapace/{name}"
    assets = "\n".join(f'    <mesh name="carapace_hull_{i:02d}" file="{rel}/{f.name}"/>'
                       for i, f in enumerate(hull_files))
    assets += f'\n    <mesh name="carapace_visual" file="{rel}/visual.stl"/>'
    geoms = ('\n        <geom name="upper_shell_carapace_visual" class="visual" type="mesh" '
             'density="0" rgba="0.95 0.65 0.25 1" mesh="carapace_visual"/>')
    geoms += "".join(
        f'\n        <geom name="upper_shell_carapace_{i:02d}_meshcol" class="collision" '
        f'type="mesh" rgba="0.95 0.65 0.25 1" mesh="carapace_hull_{i:02d}"/>'
        for i in range(len(hull_files)))
    out = xml.replace(m_inert.group(2), new_inertial, 1)
    out = out.replace("<asset>", "<asset>\n" + assets, 1)
    anchor = re.search(r'\n(\s*)<geom name="upper_shell_link_meshcol"[^\n]*', out)
    out = out[:anchor.end()] + geoms + out[anchor.end():]
    header = (f"<!-- Generated by design/carapace/build_version.py for carapace {name} "
              f"(design/carapace/versions/{name}). Do not edit by hand. -->\n")
    out = re.sub(r"(<\?xml[^>]*\?>\s*)?", lambda mm: (mm.group(0) or "") + header, out, count=1)
    model_xml = JUMPER / f"jumper_{name}.xml"
    model_xml.write_text(out)

    model = mujoco.MjModel.from_xml_path(str(model_xml))   # it must load
    stock_model = mujoco.MjModel.from_xml_path(str(JUMPER / "jumper.xml"))
    info = dict(
        model=str(model_xml.relative_to(REPO)),
        collision_pieces=len(hull_files),
        hull_volume_cm3=round(hull_vol * 1e6, 1),
        carapace_volume_cm3=round(car.volume * 1e6, 1),
        carapace_mass_g=round(m1 * 1000),
        upper_shell_mass_g=[round(m0 * 1000, 1), round(M * 1000, 1)],
        robot_mass_kg=[round(float(stock_model.body_mass.sum()), 4),
                       round(float(model.body_mass.sum()), 4)],
        printable=str((vdir / "printable_top_shell.3mf").relative_to(REPO)),
        printable_watertight=bool(fused.is_watertight),
    )
    (vdir / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
