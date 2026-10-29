"""jumper.diag_roll -- a diagonal roll on a horned carapace, standing to standing.

Everything is jumper.tuck_roll's (`tuck_roll/env_cfg.py`): the scene and carapace
collision, the claws colliding with the body, the face's floor force charged, the
horns free to pivot on, floor-contact joint torque charged at every physics step, the
tuck reward, the fading demonstration with mid-demonstration spawns. Only the
demonstration differs: a roll about a diagonal, which the tracker measures the turn
about (it reads the axis from the recording).

## The demonstration

`ref/diag_roll.npz`, recorded in jumper.tuck_roll's environment on the nominal
prototype 3 by `studies/proto3/record_demo.py` from a plan that
`studies/proto3/env_search.py` found in that same environment (`TASK=jumper.tuck_roll
MODE=diagonal`): kept only if it rolls about either diagonal -- right shoulder to left
rear hip or its mirror -- within 20 deg, scored for the face, the claws' and limbs'
load, impacts, air time and joint torque. Which diagonal is the search's choice.

Found 2026-10-29 (`plans/diag_env_s1.npy`, `logs/diag_env_s1.log`): seeded from
`plans/stand_diagonal_s1.npy` (ideal servos, 32/32 there), 12 rounds of 48; the plan
kept rolls diagonally on 32 of 48 randomised robots. Recorded on the nominal robot:
axis 47 deg off the centreline, 381 deg, standing after 1.68 s; face 0 N, stock shell
0 N, carapace impact 0 N, floor impact 154 N, limbs and claws 101 N, 0.10 s in the
air, worst floor-contact joint torque 8.4 N*m -- the last one is what training charges
down.

Its recorded poses have the claws pressed up to 1 cm into the body (0.1-0.6 s, as
they brace against it) and up to 1.9 cm into each other (214-259 deg). Those are
states the recording reached with the claws colliding, contact under load in this same
environment, so mid-demonstration spawns keep jumper.momentum_roll's 290 deg.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..tuck_roll import env_cfg as tuck

REF_NPZ = Path(__file__).resolve().parent / "ref" / "diag_roll.npz"

#: Last turn a mid-demonstration spawn may start at (module docstring).
MAX_SPAWN_TURN = math.radians(290.0)


def env_cfg(asset: Path | None = None, play: bool = False):
    """jumper.tuck_roll about a diagonal."""
    if not REF_NPZ.is_file():
        raise FileNotFoundError(
            f"{REF_NPZ} is not recorded yet: run studies/proto3/env_search.py with "
            "TASK=jumper.tuck_roll MODE=diagonal, then studies/proto3/record_demo.py "
            f"with TASK=jumper.tuck_roll OUT={REF_NPZ}")
    cfg = tuck.env_cfg(asset=asset, play=play)
    cfg.commands["roll"].reference = str(REF_NPZ)
    cfg.events["reset_momentum"].params["max_spawn_turn"] = MAX_SPAWN_TURN
    return cfg


__all__ = ["env_cfg"]
