"""jumper.tuck_roll -- jumper.horn_roll's forward roll with the claws tucked side by side
under the face, so the arms carry the robot over its nose instead of the front of the
carapace. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..horn_roll import HORN_ROLL_ASSETS

register(
    id="jumper.tuck_roll",
    assets=HORN_ROLL_ASSETS,
    description=(
        "jumper hexapod forward roll on a horned carapace over its tucked arms: claws "
        "side by side under the face, floor contact on the front of the body charged"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior", "carapace"),
)
