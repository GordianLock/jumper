#!/usr/bin/env python3
"""Generate print, colored, collision and MuJoCo variants of Jumper's bioshell.

The original watertight upper_shell_link is a manufacturing datum. We smoothly
deform only its exterior crown, preserving front screen clearance, rim and
mounts, then recalculate physical properties. Coordinates are METERS.

Dependencies: numpy, scipy; matplotlib for previews.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/jumper"
SOURCE = ASSETS / "urdf/jumper/meshes/visual/upper_shell_link.stl"
SOURCE_MJCF = ASSETS / "jumper.xml"
OUTPUT = ASSETS / "bioshell"
SOURCE_MASS = 0.102436653
STL_STRUCT = np.dtype([("normal", "<f4", (3,)), ("vertex", "<f4", (3, 3)), ("attribute", "<u2")])

DESIGNS = {
    "crowned-dome": ("Broad dome and spaced blunt nodules", ("#45616B", "#29434A", "#A99C7E", "#B66D43")),
    "central-keel": ("Longitudinal crest and shoulder bosses", ("#4E6062", "#303D42", "#B5A181", "#AB6443")),
    "low-rocker-crest": ("Low asymmetric rocker and lateral contact ribs", ("#525D56", "#373F3A", "#BFA78A", "#8F624A")),
}


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def gaussian(x, y, cx, cy, sx, sy):
    return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def load_stl(path):
    binary = path.read_bytes()
    count = struct.unpack_from("<I", binary, 80)[0]
    if len(binary) != 84 + 50 * count:
        raise ValueError(f"Invalid binary STL: {path}")
    triangles = np.frombuffer(binary, dtype=STL_STRUCT, count=count, offset=84)["vertex"]
    vertices, index = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices.astype(np.float64), index.reshape(-1, 3).astype(np.int32)


def face_normals(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(area2[:, None], 1e-15), area2 / 2


def exterior_weight(vertices, faces):
    normals, area = face_normals(vertices, faces)
    mean = np.zeros_like(vertices)
    total = np.zeros(len(vertices))
    for i in range(3):
        np.add.at(mean, faces[:, i], normals * area[:, None])
        np.add.at(total, faces[:, i], area)
    nz = (mean / np.maximum(total[:, None], 1e-15))[:, 2]
    x, y, z = vertices.T
    # Interior ceiling has negative outward z normals and must not be changed.
    weight = smoothstep(-.16, .60, nz) * smoothstep(.010, .024, z)
    # Protect screen-facing SHORT front (+x), the full rim, rear and screw mounts.
    weight *= smoothstep(-.086, -.069, x) * (1 - smoothstep(.061, .078, x))
    weight *= 1 - smoothstep(.064, .081, np.abs(y))
    return weight


def crown_field(name, x, y):
    if name == "crowned-dome":
        # The stock roof is already domed: add texture and contact bosses,
        # not another solid hemispherical layer of substantial weight.
        field = .0007 * gaussian(x, y, -.010, 0, .062, .067)
        for i in range(12):
            theta = 2 * np.pi * i / 12 + .15
            cx, cy = -.011 + .057 * np.cos(theta), .052 * np.sin(theta)
            height = .0032 + .0010 * (.5 + .5 * math.sin(2.7 * i))
            field += height * gaussian(x, y, cx, cy, .0095, .0105)
        for cx, cy, h in ((-.006, 0, .0042), (-.030, -.025, .0030), (.024, .022, .0027)):
            field += h * gaussian(x, y, cx, cy, .011, .013)
    elif name == "central-keel":
        field = .0050 * gaussian(x, y, -.009, 0, .058, .013)
        field += .0010 * gaussian(x, y, -.022, 0, .018, .008)
        for cx, cy, h in ((-.055, -.051, .0027), (-.014, -.056, .0031), (.036, -.049, .0027),
                          (-.046, .050, .0028), (.005, .054, .0033), (.040, .041, .0024)):
            field += h * gaussian(x, y, cx, cy, .012, .012)
    elif name == "low-rocker-crest":
        field = .0025 * gaussian(x, y, -.010, -.018, .067, .025)
        field += .0015 * gaussian(x, y, -.020, .027, .056, .020)
        for cx, cy, h in ((-.051, -.054, .0027), (-.004, -.059, .0028), (.044, -.050, .0025),
                          (-.047, .056, .0022), (.015, .052, .0027)):
            field += h * gaussian(x, y, cx, cy, .015, .014)
    else:
        raise ValueError(name)
    return field


def deform(original, faces, name):
    x, y, _ = original.T
    added = exterior_weight(original, faces) * crown_field(name, x, y)
    result = original.copy()
    result[:, 2] += added
    return result, added


def watertight(faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def mass_properties(vertices, faces, density):
    triangles = vertices[faces]
    vol_faces = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6
    volume = float(vol_faces.sum())
    if volume <= 0:
        raise ValueError("Surface is not positively oriented")
    com = np.einsum("t,ti->i", vol_faces, triangles.sum(axis=1)) / (4 * volume)
    sums = triangles.sum(axis=1)
    moment2 = (np.einsum("t,ti,tj->ij", vol_faces, sums, sums)
               + np.einsum("t,tki,tkj->ij", vol_faces, triangles, triangles)) / 20
    mass = volume * density
    center_second = density * moment2 - mass * np.outer(com, com)
    inertia = np.trace(center_second) * np.eye(3) - center_second
    if np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("Nonpositive inertia tensor")
    return {"volume_m3": volume, "mass_kg": mass, "center_of_mass_m": com.tolist(),
            "inertia_about_com_kg_m2": inertia.tolist(), "density_kg_m3": density}


def write_stl(path, vertices, faces):
    normal, _ = face_normals(vertices, faces)
    entries = np.zeros(len(faces), dtype=STL_STRUCT)
    entries["normal"] = normal
    entries["vertex"] = vertices[faces].astype(np.float32)
    header = b"Jumper bioshell | unit: meter | original datum retained"
    with path.open("wb") as stream:
        stream.write(header.ljust(80, b" "))
        stream.write(struct.pack("<I", len(faces)))
        stream.write(entries.tobytes())


def hex_rgb(h):
    return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def colors_for(vertices, added):
    x, y, z = vertices.T
    indices = np.zeros(len(vertices), dtype=np.uint8)
    indices[(.5 * np.sin(110*x + 19*y) + .5 * np.cos(88*y)) > .66] = 1
    indices[added > .0018] = 2
    indices[added > .0038] = 3
    indices[z < .012] = 1
    return indices


def write_ply(path, vertices, faces, rgb):
    header = ("ply\nformat ascii 1.0\ncomment units meter\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    with path.open("w", encoding="utf8") as stream:
        stream.write(header)
        for p, c in zip(vertices, rgb):
            stream.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]} {c[1]} {c[2]}\n")
        for a, b, c in faces:
            stream.write(f"3 {a} {b} {c}\n")


def write_3mf(path, vertices, faces, palette, colors):
    """3MF with four painted base-material swatches, sized in millimeters."""
    ns = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    ET.register_namespace("", ns)
    model = ET.Element(f"{{{ns}}}model", {"unit": "millimeter", "xml:lang": "en-US"})
    resources = ET.SubElement(model, f"{{{ns}}}resources")
    mats = ET.SubElement(resources, f"{{{ns}}}basematerials", {"id": "1"})
    for i, color in enumerate(palette):
        ET.SubElement(mats, f"{{{ns}}}base", {"name": ("shell", "shadow", "ridge", "tips")[i], "displaycolor": color.upper()})
    obj = ET.SubElement(resources, f"{{{ns}}}object", {"id": "2", "type": "model", "pid": "1", "pindex": "0"})
    mesh = ET.SubElement(obj, f"{{{ns}}}mesh")
    vert_el = ET.SubElement(mesh, f"{{{ns}}}vertices")
    for x, y, z in vertices * 1000:
        ET.SubElement(vert_el, f"{{{ns}}}vertex", {"x": f"{x:.6f}", "y": f"{y:.6f}", "z": f"{z:.6f}"})
    tri_el = ET.SubElement(mesh, f"{{{ns}}}triangles")
    for a, b, c in faces:
        ET.SubElement(tri_el, f"{{{ns}}}triangle", {"v1": str(a), "v2": str(b), "v3": str(c),
                                               "p1": str(colors[a]), "p2": str(colors[b]), "p3": str(colors[c])})
    build = ET.SubElement(model, f"{{{ns}}}build")
    ET.SubElement(build, f"{{{ns}}}item", {"objectid": "2"})
    types = ('<?xml version="1.0" encoding="UTF-8"?>'
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
             '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            '</Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("3D/3dmodel.model", ET.tostring(model, encoding="utf-8", xml_declaration=True))


def collision_hull(vertices, samples=1600):
    """Conservative proxy: compact convex support hull, same model coordinate frame."""
    angle = math.pi * (3 - math.sqrt(5))
    i = np.arange(samples, dtype=float) + .5
    z = 1 - 2*i/samples
    radial = np.sqrt(1 - z*z)
    dirs = np.column_stack([radial*np.cos(i*angle), radial*np.sin(i*angle), z])
    dirs = np.concatenate([dirs, np.eye(3), -np.eye(3)])
    source_idx = np.unique(np.argmax(vertices @ dirs.T, axis=0))
    points = vertices[source_idx]
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    t = points[faces]
    n = np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0])
    flip = np.einsum("ij,ij->i", n, t.mean(1)-points.mean(0)) < 0
    faces[flip] = faces[flip][:,[0,2,1]]
    gaps = np.max(vertices@dirs.T, axis=0) - np.max(points@dirs.T, axis=0)
    return points, faces, float(gaps.max())


def write_mjcf(path, visual, collision, props):
    tree = ET.parse(SOURCE_MJCF)
    root = tree.getroot()
    for mesh in root.findall("./asset/mesh"):
        if mesh.get("name") == "upper_shell_link":
            mesh.set("file", "bioshell/"+visual)
        elif mesh.get("name") == "upper_shell_link_col":
            mesh.set("file", "bioshell/"+collision)
    inertial = root.find("./worldbody/body/body[@name='upper_shell_link']/inertial")
    if inertial is None:
        raise ValueError("Stock XML lacks shell inertia")
    I = props["inertia_about_com_kg_m2"]
    inertial.attrib.clear()
    inertial.set("pos", " ".join(f"{v:.10g}" for v in props["center_of_mass_m"]))
    inertial.set("mass", f"{props['mass_kg']:.10g}")
    # MuJoCo fullinertia order: Ixx Iyy Izz Ixy Ixz Iyz.
    inertial.set("fullinertia", " ".join(f"{v:.10g}" for v in (I[0][0],I[1][1],I[2][2],I[0][1],I[0][2],I[1][2])))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def contact_sheet(items):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(15,9), facecolor="#f5f1e9")
    for k,(name, v, faces, rgb) in enumerate(items):
        for row,(elev,azim) in enumerate(((26,-60),(86,-90))):
            ax = fig.add_subplot(2,3,k+1+row*3,projection="3d")
            # All faces; sampling creates false white holes on this shell.
            f = faces
            mesh = Poly3DCollection(v[f]*1000,facecolors=rgb[f].mean(1)/255,edgecolor="none")
            ax.add_collection3d(mesh)
            ax.set(xlim=(-108,108),ylim=(-98,98),zlim=(-12,60))
            ax.set_box_aspect((216,196,92),zoom=1.22)
            ax.view_init(elev=elev,azim=azim)
            ax.set_axis_off()
            ax.set_title(("Oblique" if row==0 else "Top")+" · "+name.replace("-"," ").title())
    fig.subplots_adjust(wspace=0,hspace=.01,left=.02,right=.98,bottom=.02,top=.97)
    fig.savefig(OUTPUT/"bioshell-v01-contact-sheet.png",dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-sheet", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    original, faces = load_stl(SOURCE)
    if not watertight(faces):
        raise ValueError("Stock shell is not watertight")
    base_vol = mass_properties(original, faces, 1)["volume_m3"]
    density = SOURCE_MASS / base_vol
    manifest = {"schema":"jumper-bioshell-v01","coordinate_frame":"upper_shell_link",
                "unit":"meter","source_stl":"assets/jumper/urdf/jumper/meshes/visual/upper_shell_link.stl",
                "stock_mass_kg":SOURCE_MASS,"stock_volume_m3":base_vol,
                "density_kg_m3":density,"front_plus_x":True,
                "parent_body_shell_offset_m":[-.0073137,0,.0315],"variants":{}}
    previews=[]
    for name,(description,palette) in DESIGNS.items():
        stem=f"bioshell-{name}-v01"
        visual=f"{stem}-visual.stl"
        collision=f"{stem}-collision.stl"
        print_stl=f"{stem}-print-mm.stl"
        colored=f"{stem}-colored.ply"
        print3mf=f"{stem}-colored.3mf"
        xml=f"jumper-{stem}.xml"
        v, added = deform(original, faces, name)
        props = mass_properties(v, faces, density)
        hull_vertices,hull_faces,gap = collision_hull(v)
        indices = colors_for(v, added)
        rgb=np.array([hex_rgb(c) for c in palette])[indices]
        write_stl(OUTPUT/visual,v,faces)
        write_stl(OUTPUT/collision,hull_vertices,hull_faces)
        write_stl(OUTPUT/print_stl,v*1000,faces)
        write_ply(OUTPUT/colored,v,faces,rgb)
        write_3mf(OUTPUT/print3mf,v,faces,palette,indices)
        write_mjcf(ASSETS/xml,visual,collision,props)
        x,y,z=original.T
        protected=(z<=.010)|(x>=.078)|(x<=-.086)|(np.abs(y)>=.081)
        error=float(np.max(np.linalg.norm(v[protected]-original[protected],axis=1)))
        manifest["variants"][stem]={"description":description,
          "visual_mesh":"assets/jumper/bioshell/"+visual,
          "collision_mesh":"assets/jumper/bioshell/"+collision,
          "print_stl_mm":"assets/jumper/bioshell/"+print_stl,
          "colored_preview_ply":"assets/jumper/bioshell/"+colored,
          "colored_print_3mf":"assets/jumper/bioshell/"+print3mf,
          "mjcf":"assets/jumper/"+xml,
          "visual_triangles":int(len(faces)),"collision_triangles":int(len(hull_faces)),
          "max_added_height_m":float(added.max()),"max_protected_vertex_shift_m":error,
          "collision_support_gap_test_m":gap,**props}
        previews.append((name,v,faces,rgb))
        print(f"{stem}: {len(faces)} visual tris / {len(hull_faces)} collider tris; "
              f"{props['mass_kg']*1000:.1f}g; max rise {added.max()*1000:.1f}mm")
    (OUTPUT/"bioshell-v01-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if not args.no_sheet:
        contact_sheet(previews)


if __name__=="__main__":
    main()
