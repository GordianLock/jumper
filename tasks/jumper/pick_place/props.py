"""The debris and the receptacle.

Both are built with `five_foot/objects.py`'s contact rules (`_prop_geom`): the
prop priority, condim, friction and solref there were measured for this claw
gripping by friction alone, so the debris here touches the floor and the jaws
exactly as the five-foot task's soap and can do.

## The debris

One box per environment, its size and mass re-drawn per environment at startup
(`env_cfg.py`, `debris_size` and `debris_mass`). A box because a primitive can be
resized per world, where a mesh cannot. The largest size below is shrunk by `SIZE_SCALE`, so
every horizontal side stays within the claw's 73 mm mouth (`claw.py::APERTURE_MM`)
and it can be gripped across either one.

## The receptacle

A square open tray, fixed -- so mjlab wraps it as a mocap body and the reset
places it per episode (`mdp/events.py`). **Its rim height is set by how far it is
sunk into the floor**: the walls are `BIN_WALL_H` tall, and lowering the tray by
`s` leaves a rim of `BIN_WALL_H - s` above the ground. Sunk all the way it is a
square outline on the floor: a pile spot. A body with no degrees of freedom
does not collide with the world, so the sunken part costs nothing.
"""

from __future__ import annotations

from typing import Any

import mujoco

from ..five_foot.objects import _free_body, _prop_geom, _prop_spec

# ── The debris ────────────────────────────────────────────────────────────
#: Half-sizes (x, y, z) in metres **at the largest**: a 56 x 56 x 56 mm block.
#: 56 mm is the widest the claw takes with room to close on it from clear
#: (`mdp/actions.py::PRECLOSE_EXTRA`, measured up to 58).
#: Every environment's is shrunk from this (`mdp/events.py::debris_size`, which
#: says why it can only shrink).
DEBRIS_HALF = (0.028, 0.028, 0.028)
#: Per-axis shrink factors, drawn per environment at startup: every side 22 to
#: 56 mm.
SIZE_SCALE = {0: (0.4, 1.0), 1: (0.4, 1.0), 2: (0.4, 1.0)}
#: Mass as authored, kg. Re-drawn per environment as exp(2 alpha) of this
#: (`DEBRIS_MASS_ALPHA`): 30 to 150 g -- paper, a cup, a toy, a stone.
DEBRIS_MASS = 0.070
DEBRIS_MASS_ALPHA = (-0.42, 0.38)
DEBRIS_RGBA = (0.80, 0.42, 0.18, 1.0)

# ── The receptacle ────────────────────────────────────────────────────────
#: Inner half-width of the tray, m: 80 mm square inside, room for the largest
#: debris with 9 mm to spare each side at its widest.
BIN_INNER_HALF = 0.040
BIN_WALL_T = 0.004
#: Wall height above the tray's floor, m: the tallest rim the robot is asked to
#: clear. The arm's mouth reaches 0.27 m up; this keeps the drop within the
#: region the arm reaches over the tray (see `env_cfg.py`).
BIN_WALL_H = 0.050
BIN_RGBA = (0.40, 0.45, 0.50, 1.0)


def debris_spec() -> Any:
    spec = _prop_spec()
    body = _free_body(spec, "debris")
    # Nonzero so the compiled model keeps its gravity-compensation pass for this
    # body (MuJoCo skips the pass when no body has any); a held spawn turns it up
    # to 1 while the claw closes (`mdp/events.py::HeldSpawn`).
    body.gravcomp = 1e-9
    hx, hy, hz = DEBRIS_HALF
    _prop_geom(
        body, name="debris_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, hz], rgba=list(DEBRIS_RGBA), mass=DEBRIS_MASS,
    )
    return spec


def bin_spec() -> Any:
    """The tray, its floor's underside at the body origin."""
    spec = _prop_spec()
    body = spec.worldbody.add_body(name="bin", pos=[0.0, 0.0, 0.0])
    inner, t, h = BIN_INNER_HALF, BIN_WALL_T, BIN_WALL_H
    rgba = list(BIN_RGBA)
    _prop_geom(body, name="bin_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[inner + t, inner + t, t / 2], pos=[0.0, 0.0, t / 2], rgba=rgba, mass=0.3)
    for name, sx, sy, px, py in (
        ("bin_wall_xp", t / 2, inner + t, inner + t / 2, 0.0),
        ("bin_wall_xm", t / 2, inner + t, -inner - t / 2, 0.0),
        ("bin_wall_yp", inner + t, t / 2, 0.0, inner + t / 2),
        ("bin_wall_ym", inner + t, t / 2, 0.0, -inner - t / 2),
    ):
        _prop_geom(body, name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[sx, sy, h / 2], pos=[px, py, t + h / 2], rgba=rgba, mass=0.1)
    return spec


__all__ = [
    "BIN_INNER_HALF", "BIN_WALL_H", "BIN_WALL_T", "DEBRIS_HALF", "DEBRIS_MASS",
    "DEBRIS_MASS_ALPHA", "SIZE_SCALE", "bin_spec", "debris_spec",
]
