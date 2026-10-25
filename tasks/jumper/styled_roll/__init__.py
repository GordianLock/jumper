"""jumper.styled_roll -- the momentum roll's demonstration, performed faithfully:
the fling, the teeter and the hop. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.styled_roll",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod styled roll: jumper.momentum_roll's demonstration tracked "
        "for the whole of training (no recording on the robot)"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior"),
)
