"""Clean simulation footage of a recorded behaviour, for editing into a video: one
camera angle per file, no captions, at several playback speeds.

Input is any recording in the demonstration format (`record_demo.py`,
`record_policy_demo.py`, `studies/self_right/record_self_right.py`): root pose and
joint angles per 20 ms control step. Nothing is re-simulated -- each frame poses the
model at the recorded state, interpolated between the 20 ms steps (joints linearly,
the root orientation by normalised lerp) so slow motion stays smooth, and draws it. The
unactuated J4 joints are at 0.

The camera is fixed for the whole clip, aimed at the middle of the body's path, and
its angles are relative to the recording's roll axis, so "side" looks across the
motion whichever way the robot rolled. `--hold` repeats the first and last recorded
pose so an editor has handles to cut on (the robot is not moving in either).

    MUJOCO_GL=egl python studies/footage/render_footage.py <recording.npz> <out_dir> \\
        --model jumper --name self_right_run3

Writes `<out_dir>/<name>_<angle>_<speed>x.mp4` for every angle and speed.
"""
import argparse
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "..", "assets", "jumper")

#: name -> (azimuth relative to the roll axis, elevation, distance), degrees and m.
ANGLES = {
    "side": (90.0, -12.0, 0.85),
    "front": (180.0, -12.0, 0.85),
    "three_quarter": (135.0, -22.0, 0.95),
    "low": (60.0, -4.0, 0.75),
    "top": (90.0, -70.0, 1.0),
}

p = argparse.ArgumentParser()
p.add_argument("rec")
p.add_argument("out_dir")
p.add_argument("--model", required=True, help='"jumper" or a carapace name, e.g. CF1-tri30m-rb-v4')
p.add_argument("--name", required=True)
p.add_argument("--speeds", default="1,0.5,0.25")
p.add_argument("--angles", default=",".join(ANGLES))
p.add_argument("--fps", type=int, default=60)
p.add_argument("--width", type=int, default=1920)
p.add_argument("--height", type=int, default=1080)
p.add_argument("--hold", type=float, default=0.5, help="s of still pose added at each end")
p.add_argument("--scene", default=None,
               help="a repository scene id (scripts/play.py --list): its ground, sky and "
                    "lights instead of the footage's own; props are left out")
p.add_argument("--floor", default=None, choices=("hardwood",),
               help="a look for the ground (display only; the physics is untouched): "
                    "hardwood, from studies/footage/make_wood.py")
p.add_argument("--colour", default=None,
               help="a colour pattern folder (design/carapace/colour/<folder>): its display "
                    "meshes replace the top shell's and carapace's visuals")
a = p.parse_args()
import mediapy  # noqa: E402

h = np.load(a.rec)
names = [str(n) for n in h["joint_names"]]
spec = mujoco.MjSpec.from_file(os.path.join(
    ASSETS, "jumper.xml" if a.model == "jumper" else f"jumper_{a.model}.xml"))
if a.colour:
    # The pattern's pieces (design/carapace/colour/README.md): display-only meshes in the
    # upper_shell_link frame, one per filament, replacing the shell's and carapace's visuals.
    folder = os.path.abspath(a.colour)
    cols = json.load(open(os.path.join(folder, "colours.json")))
    shell = spec.body("upper_shell_link")
    for piece in cols["pieces"]:
        mesh = spec.add_mesh(name=f"colour_{piece['filament']}",
                             file=os.path.join(folder, piece["file"]))
        g = shell.add_geom(name=f"upper_shell_colour_{piece['filament']}_visual",
                           type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                           rgba=piece["rgba"], contype=0, conaffinity=0, group=2, density=0)
        del g
    for n in ("upper_shell_link_visual", "upper_shell_carapace_visual"):
        spec.geom(n).rgba[3] = 0.0
if a.scene:
    # The scene compiled onto an empty world the way its exporter does it
    # (scenes/tools/export_web_scene.py `_build_spec`), the robot attached to it.
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    import scenes  # noqa: E402
    from scenes.tools.export_web_scene import _build_spec  # noqa: E402

    world = _build_spec(scenes.load(a.scene))
    world.attach(spec, prefix="robot/", frame=world.worldbody.add_frame())
    spec = world
    if not any(t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in spec.textures):
        # mjlab's default scene has no sky, which renders black.
        spec.add_texture(name="footage_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=512, height=512,
                         rgb1=[0.55, 0.62, 0.70], rgb2=[0.12, 0.14, 0.17])
else:
    spec.add_texture(name="footage_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                     rgb1=[0.30, 0.33, 0.36], rgb2=[0.26, 0.29, 0.32])
    mat = spec.add_material(name="footage_floor", texrepeat=[8, 8], reflectance=0.03)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "footage_grid"
    spec.add_texture(name="footage_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=512, height=512,
                     rgb1=[0.55, 0.62, 0.70], rgb2=[0.12, 0.14, 0.17])
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1],
                            material="footage_floor")
    spec.worldbody.add_light(pos=[0.6, -0.6, 2.0], dir=[-0.3, 0.3, -1],
                             diffuse=[0.75, 0.75, 0.75], castshadow=True)
    spec.worldbody.add_light(pos=[-1, 1, 2.5], dir=[0.4, -0.4, -1], diffuse=[0.3, 0.3, 0.3],
                             castshadow=False)
if a.floor == "hardwood":
    # Visual only: a texture and material on the ground plane. Friction and contact
    # are the geom's and are not touched. The image covers make_wood.TILE_M metres.
    spec.add_texture(name="footage_hardwood", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=os.path.join(HERE, "assets", "hardwood.png"))
    wood = spec.add_material(name="footage_hardwood", texuniform=True,
                             texrepeat=[1 / 1.2, 1 / 1.2], reflectance=0.0, specular=0.25,
                             shininess=0.4)
    wood.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "footage_hardwood"
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.material = "footage_hardwood"
m = spec.compile()
# No mirror in the ground: MuJoCo's reflectance mirrors the whole scene under the floor.
for g in range(m.ngeom):
    if m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD) \
            and m.geom_matid[g] >= 0:
        m.mat_reflectance[m.geom_matid[g]] = 0.0
m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)
m.vis.quality.shadowsize = 4096
m.vis.quality.offsamples = 4
d = mujoco.MjData(m)
pre = "robot/" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + names[0]) >= 0 else ""
adr = [m.jnt_qposadr[m.joint(pre + n).id] for n in names]
root = m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]]
dt = float(h["dt"])
pos, quat, q = h["root_pos"], h["root_quat"], h["q"]
T = len(q)
axis = np.asarray(h["axis"], float)
yaw_axis = math.degrees(math.atan2(axis[1], axis[0]))
look = np.array([(pos[:, 0].min() + pos[:, 0].max()) / 2,
                 (pos[:, 1].min() + pos[:, 1].max()) / 2, 0.07])


def pose(t: float) -> None:
    """The recorded state at time t (s from the first step), interpolated."""
    x = min(max(t / dt, 0.0), T - 1.0)
    k = min(int(x), T - 2)
    u = x - k
    qa, qb = quat[k], quat[k + 1]
    if np.dot(qa, qb) < 0:
        qb = -qb
    qq = (1 - u) * qa + u * qb              # normalised lerp: steps are 20 ms apart
    qq /= np.linalg.norm(qq)
    d.qpos[:] = 0
    d.qpos[root:root + 3] = (1 - u) * pos[k] + u * pos[k + 1]
    d.qpos[root + 3:root + 7] = qq
    d.qpos[adr] = (1 - u) * q[k] + u * q[k + 1]
    mujoco.mj_forward(m, d)


r = mujoco.Renderer(m, a.height, a.width)
#: Collision hulls (group 1) stand proud of the visual surfaces and would cover them.
vopt = mujoco.MjvOption()
vopt.geomgroup[1] = 0
os.makedirs(a.out_dir, exist_ok=True)
length = (T - 1) * dt
for angle in a.angles.split(","):
    az, el, dist = ANGLES[angle]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = look
    cam.azimuth, cam.elevation, cam.distance = yaw_axis + az, el, dist
    for speed in (float(s) for s in a.speeds.split(",")):
        n = int(round((length + 2 * a.hold) / speed * a.fps))
        frames = []
        for i in range(n + 1):
            pose(i * speed / a.fps - a.hold)
            r.update_scene(d, cam, scene_option=vopt)
            frames.append(r.render())
        out = os.path.join(a.out_dir, f"{a.name}_{angle}_{speed:g}x.mp4")
        mediapy.write_video(out, frames, fps=a.fps, crf=18)
        print(f"wrote {out}: {len(frames)} frames, {len(frames) / a.fps:.1f} s")
