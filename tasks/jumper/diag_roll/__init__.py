"""jumper.diag_roll -- a diagonal roll on a horned carapace, standing to standing, with
jumper.tuck_roll's protections: face contact charged, horns free, claws colliding with
the body, floor-contact joint torque charged. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..horn_roll import HORN_ROLL_ASSETS

register(
    id="jumper.diag_roll",
    assets=HORN_ROLL_ASSETS,
    description=(
        "jumper hexapod diagonal roll on a horned carapace, standing to standing: "
        "jumper.tuck_roll's face, claw and joint-torque protections about a diagonal"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior", "carapace"),
)
