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
    "side": (90.0, -25.0, 0.85),
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
p.add_argument("--center", default=None,
               help="aim the camera at the middle of this phase (a walk_then_behave.py "
                    "recording's phase name) instead of the middle of its path")
p.add_argument("--orbit", type=float, default=0.0,
               help="instead of the angles: hold the first pose and circle it once, low, "
                    "in this many seconds (then see --orbit-peak)")
p.add_argument("--orbit-distance", type=float, default=0.55)
p.add_argument("--orbit-elevation", type=float, default=28.0, help="degrees above level")
p.add_argument("--orbit-peak", type=float, default=None,
               help="rise to this many degrees above level (e.g. 85: nearly overhead) at "
                    "--orbit-rise-at and end there")
p.add_argument("--orbit-rise-at", type=float, default=8.0, help="s into the clip the rise starts")
p.add_argument("--orbit-rise", type=float, default=2.5, help="s rising to --orbit-peak")
p.add_argument("--orbit-end-hold", type=float, default=0.5,
               help="s held still overhead, front to the top, before the fade")
p.add_argument("--orbit-hold-rate", type=float, default=12.0,
               help="deg/s the camera turns while held overhead")
p.add_argument("--fade-out", type=float, default=0.0, help="s fading to black at the end")
p.add_argument("--orbit-height", type=float, default=0.09, help="aim point above the floor, m")
p.add_argument("--orbit-start", type=float, default=200.0, help="first azimuth, degrees")
p.add_argument("--track", default=None,
               help="instead of the angles: a close-up following the robot through this "
                    "phase (a walk_then_behave.py recording's phase name), at the first "
                    "of --speeds")
p.add_argument("--track-pad", type=float, default=0.4, help="s before and after the phase")
p.add_argument("--track-pre", type=float, default=None,
               help="s before the phase, if not --track-pad")
p.add_argument("--track-distance", type=float, default=0.6)
p.add_argument("--track-elevation", type=float, default=22.0, help="degrees above level")
p.add_argument("--track-height", type=float, default=0.09)
p.add_argument("--track-start", type=float, default=60.0,
               help="first azimuth, degrees relative to the recording's axis")
p.add_argument("--track-sweep", type=float, default=90.0, help="degrees swung over the clip")
p.add_argument("--zoom", type=float, default=1.0,
               help="camera distance multiplier, for clips that cover more ground")
p.add_argument("--hold", type=float, default=0.5, help="s of still pose added at each end")
p.add_argument("--scene", default=None,
               help="a repository scene id (scripts/play.py --list): its ground, sky and "
                    "lights instead of the footage's own; props are left out")
p.add_argument("--floor", default=None, choices=("hardwood",),
               help="a look for the ground (display only; the physics is untouched): "
                    "hardwood, from studies/footage/make_wood.py")
p.add_argument("--surround", default=None, choices=("grass",),
               help="with --floor: the floor becomes a deck --deck m across around the "
                    "action, and the ground beyond it this (display only)")
p.add_argument("--deck", type=float, default=30.0,
               help="deck size for --surround, m: wide enough that the grass shows only "
                    "towards the horizon")
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
    planes = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if a.surround == "grass":
        # The ground plane keeps its physics and wears the wood, drawn only --deck m
        # across around the action; beyond it a display-only lawn 5 cm lower runs to
        # the horizon. A separate plane rather than a second one coplanar with the
        # floor: two planes millimetres apart fight for the depth buffer at distance.
        # The grass image is the football scene's (two mown bands per repeat).
        spec.add_texture(name="footage_grass", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         file=os.path.join(HERE, "..", "..", "scenes", "assets", "football",
                                           "grass.png"))
        grass = spec.add_material(name="footage_grass", texuniform=True,
                                  texrepeat=[0.7, 0.7], reflectance=0.0)
        grass.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "footage_grass"
        rp = np.load(a.rec)["root_pos"]
        cx, cy = (rp[:, 0].min() + rp[:, 0].max()) / 2, (rp[:, 1].min() + rp[:, 1].max()) / 2
        for g in planes:
            g.material = "footage_hardwood"
            g.size = [a.deck / 2, a.deck / 2, g.size[2]]
            g.pos = [cx, cy, g.pos[2]]
        spec.worldbody.add_geom(name="footage_lawn", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[400.0, 400.0, 0.01], pos=[cx, cy, -0.05],
                                material="footage_grass", contype=0, conaffinity=0, group=2)
        for t in list(spec.textures):
            if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                spec.delete(t)
        spec.add_texture(name="footage_day_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=512, height=512,
                         rgb1=[0.62, 0.77, 0.93], rgb2=[0.30, 0.50, 0.80])
    else:
        for g in planes:
            g.material = "footage_hardwood"
m = spec.compile()
# The camera's near/far planes and the shadow map scale with the model's extent, which
# a ground plane drawn out to the horizon makes hundreds of metres: the near plane then
# cuts into the robot. Set it to the action's scale, and the far plane past the horizon.
m.stat.extent = 2.0
m.vis.map.znear = 0.01
m.vis.map.zfar = 400.0
# No mirror in the ground: MuJoCo's reflectance mirrors the whole scene under the floor.
for g in range(m.ngeom):
    if m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD) \
            and m.geom_matid[g] >= 0:
        m.mat_reflectance[m.geom_matid[g]] = 0.0
m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)
m.vis.quality.shadowsize = 4096
m.vis.quality.offsamples = 4
# The sun's shadow map spans shadowclip x the model's extent, which the ground planes
# make metres wide; at that spread its texels are centimetres and the floor speckles
# (shadow acne). Clip it to SHADOW_REACH m around the scene centre instead.
SHADOW_REACH = 0.4
m.vis.map.shadowclip = min(m.vis.map.shadowclip, SHADOW_REACH / max(m.stat.extent, 1e-6))
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
if a.center:
    # Aim at the middle of a named phase (walk_then_behave.py's recordings carry
    # them): the centre of the ground the robot covers during it, e.g. the behaviour's.
    names = [str(n) for n in h["phase_names"]]
    i = names.index(a.center)
    lo_ = int(h["phases"][i - 1]) if i > 0 else 0
    seg_ = pos[lo_:int(h["phases"][i]), :2]
    look[:2] = (seg_.min(0) + seg_.max(0)) / 2


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
    # The sun's shadow map is centred on the model's centre and spans SHADOW_REACH:
    # keep it on the robot, which may walk a metre or more.
    m.stat.center[:2] = d.qpos[root:root + 2]
    mujoco.mj_forward(m, d)


r = mujoco.Renderer(m, a.height, a.width)
#: Collision hulls (group 1) stand proud of the visual surfaces and would cover them.
vopt = mujoco.MjvOption()
vopt.geomgroup[1] = 0
os.makedirs(a.out_dir, exist_ok=True)
length = (T - 1) * dt
if a.track:
    # A close-up that follows the robot through one phase: the camera aims at the
    # robot's position smoothed over TRACK_SMOOTH s (so it glides rather than
    # jitters with every bounce), and swings --track-sweep degrees round it.
    names = [str(nm) for nm in h["phase_names"]]
    i_ = names.index(a.track)
    pre = a.track_pre if a.track_pre is not None else a.track_pad
    t0 = (int(h["phases"][i_ - 1]) if i_ > 0 else 0) * dt - pre
    t1 = int(h["phases"][i_]) * dt + a.track_pad
    TRACK_SMOOTH = 0.25
    k_ = np.exp(-0.5 * (np.arange(-60, 61) * dt / (TRACK_SMOOTH / 2)) ** 2)
    k_ /= k_.sum()
    padded = np.pad(pos, ((60, 60), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, j], k_, mode="valid") for j in range(3)], 1)
    speed = float(a.speeds.split(",")[0])
    n = int(round((t1 - t0) / speed * a.fps))
    cam = mujoco.MjvCamera()
    cam.elevation, cam.distance = -a.track_elevation, a.track_distance
    frames = []
    for i in range(n + 1):
        t = t0 + i * speed / a.fps
        pose(t)
        x = min(max(t / dt, 0.0), T - 1.0)
        kk, u = min(int(x), T - 2), x - min(int(x), T - 2)
        c = (1 - u) * smooth[kk] + u * smooth[kk + 1]
        cam.lookat[:] = [c[0], c[1], a.track_height]
        cam.azimuth = yaw_axis + a.track_start + a.track_sweep * i / max(n, 1)
        r.update_scene(d, cam, scene_option=vopt)
        frames.append(r.render())
    out = os.path.join(a.out_dir, f"{a.name}_track_{speed:g}x.mp4")
    mediapy.write_video(out, frames, fps=a.fps, crf=18)
    print(f"wrote {out}: {len(frames)} frames, {len(frames) / a.fps:.1f} s")
    sys.exit(0)

if a.orbit:
    # A turntable, the recording's first pose held. The camera circles low at one turn
    # per --orbit s. With --orbit-peak: at --orbit-rise-at s it starts rising, over
    # --orbit-rise s, to that elevation, its turning easing to --orbit-hold-rate deg/s;
    # overhead it keeps turning until the robot's front points to the top of the
    # screen, eases to a stop there, holds --orbit-end-hold s, and fades to black over
    # --fade-out s. Without it, one low turn and the fade.
    pose(0.0)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [d.qpos[root], d.qpos[root + 1], a.orbit_height]
    cam.distance = a.orbit_distance
    w0 = d.qpos[root + 3:root + 7]
    front = math.degrees(math.atan2(2 * (w0[0] * w0[3] + w0[1] * w0[2]),
                                    1 - 2 * (w0[2] ** 2 + w0[3] ** 2)))
    rate = 360.0 / a.orbit
    dt_f = 1.0 / a.fps
    az, el, t = a.orbit_start, a.orbit_elevation, 0.0
    frames = []

    def shot(fade=1.0):
        cam.azimuth, cam.elevation = az, -el
        r.update_scene(d, cam, scene_option=vopt)
        frames.append((r.render() * fade).astype(np.uint8))

    if a.orbit_peak is None:
        while t < a.orbit:
            shot()
            az += rate * dt_f
            t += dt_f
    else:
        while t < a.orbit_rise_at:                          # low
            shot()
            az += rate * dt_f
            t += dt_f
        t_r = 0.0
        while t_r < a.orbit_rise:                           # rising, slowing
            f = t_r / a.orbit_rise
            el = a.orbit_elevation + (a.orbit_peak - a.orbit_elevation) * 0.5 * (1 - math.cos(math.pi * f))
            shot()
            az += (rate + (a.orbit_hold_rate - rate) * f) * dt_f
            t_r += dt_f
        el = a.orbit_peak
        # Overhead: turn on to the next heading with the robot's front at the top of
        # the screen (camera heading = the robot's front), easing to a stop.
        target = az + (front - az) % 360.0
        decel = a.orbit_hold_rate / 1.0                     # stop over about 1 s
        while target - az > 1e-3:
            shot()
            step = min(a.orbit_hold_rate, math.sqrt(2 * decel * (target - az))) * dt_f
            az = min(target, az + max(step, 1e-3))
        az = target
        for _ in range(int(round(a.orbit_end_hold * a.fps))):
            shot()
    n_fade = int(round(a.fade_out * a.fps))
    for k in range(n_fade):
        shot(fade=1.0 - (k + 1) / n_fade)
    out = os.path.join(a.out_dir, f"{a.name}_orbit.mp4")
    mediapy.write_video(out, frames, fps=a.fps, crf=18)
    print(f"wrote {out}: {len(frames)} frames, {len(frames) / a.fps:.1f} s")
    sys.exit(0)

for angle in a.angles.split(","):
    az, el, dist = ANGLES[angle]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = look
    cam.azimuth, cam.elevation, cam.distance = yaw_axis + az, el, dist * a.zoom
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
