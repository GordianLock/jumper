"""jumper.spin_drift -- the spin roll, standing to standing.

The move (named 2026-10-29): the robot crouches, tips over a front corner, goes up
onto its side until nearly vertical, and comes over and down onto its legs. It does
not roll across its back -- on the demonstration nothing of the carapace or the stock
shell touched the floor -- so it should carry to carapaces of other shapes and to the
stock robot; the task trains on any model in its asset list (prototype 4 first).

Everything is jumper.tuck_roll's (`tuck_roll/env_cfg.py`): the claws colliding with
the body, the face's floor force charged, the horns free to pivot on, floor-contact
joint torque charged at every physics step, the tuck reward, the fading
demonstration with mid-demonstration spawns. Only the demonstration differs.

## The demonstration

`ref/spin_roll.npz`, recorded in jumper.tuck_roll's environment on the nominal
prototype 4 (CF1-tri30m-rb-v4) by `studies/proto3/record_demo.py` from
`studies/proto3/plans/spin_roll_v4.npy`. That plan came from
`studies/proto3/env_search.py` in the same environment on prototype 4, any direction,
seeded from prototype 3's diagonal plan, 12 rounds of 48 (2026-10-29,
`logs/v4_any_from_diag_s1.log`): it rolled on 43 of 48 randomised robots. Recorded:
axis 52 deg off the centreline, 361 deg, standing after 2.3 s; face 0 N, stock shell
0 N, carapace impact 0 N, floor impact 181 N, limbs and claws 146 N, 0.12 s in the
air, worst floor-contact joint torque 11.5 N*m -- the highest of any demonstration so
far, and what training charges down.

Its recorded poses have the claws pressed up to 1.1 cm into the body while they brace
and up to 2.6 cm into each other at 243-253 deg: contact under load, recorded with the
claws colliding in this same environment, so mid-demonstration spawns keep
jumper.momentum_roll's 290 deg.

The turn is counted about the demonstration's axis, and nothing holds the axis once
the guide has faded: jumper.diag_roll's first policy drifted from its diagonal to a
roll nearly end over end that way (kept as the shoulder roll).
"""

from __future__ import annotations

import math
from pathlib import Path

from ..tuck_roll import env_cfg as tuck

REF_NPZ = Path(__file__).resolve().parent / "ref" / "spin_roll.npz"

#: Last turn a mid-demonstration spawn may start at (module docstring).
MAX_SPAWN_TURN = math.radians(290.0)


def env_cfg(asset: Path | None = None, play: bool = False):
    """jumper.tuck_roll with the spin roll's demonstration."""
    cfg = tuck.env_cfg(asset=asset, play=play)
    cfg.commands["roll"].reference = str(REF_NPZ)
    cfg.events["reset_momentum"].params["max_spawn_turn"] = MAX_SPAWN_TURN
    return cfg


__all__ = ["env_cfg"]
