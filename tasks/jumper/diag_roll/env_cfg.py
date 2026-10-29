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

**Not recorded yet.** Until it is, this task does not build.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..tuck_roll import env_cfg as tuck

REF_NPZ = Path(__file__).resolve().parent / "ref" / "diag_roll.npz"

#: Last turn a mid-demonstration spawn may start at; lowered if the recorded poses
#: put a claw inside the body or the other claw past some turn.
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
