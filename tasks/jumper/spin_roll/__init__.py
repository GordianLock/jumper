"""jumper.spin_roll -- the spin roll: up onto a side, over and down, with
jumper.tuck_roll's protections: face contact charged, horns free, claws colliding
with the body, floor-contact joint torque charged. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..horn_roll import HORN_ROLL_ASSETS

#: Prototype 4 (CF1-tri30m-rb-v4) first: the demonstration was found on it.
SPIN_ROLL_ASSETS = tuple(sorted(HORN_ROLL_ASSETS, key=lambda a: a.name != "CF1-tri30m-rb-v4"))

register(
    id="jumper.spin_roll",
    assets=SPIN_ROLL_ASSETS,
    description=(
        "jumper hexapod spin roll: up onto a side, over and down without rolling across "
        "the back, from prototype 4's demonstration, with jumper.tuck_roll's protections"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior", "carapace"),
)
