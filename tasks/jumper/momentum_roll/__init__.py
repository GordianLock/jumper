"""jumper.momentum_roll -- a sideways roll driven by a push, learnt from a fading
demonstration, with self-righting from the back. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.momentum_roll",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod momentum roll: crouch, lopsided push, spin over the back, "
        "land and stand; self-rights from the back (demonstration only while training)"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior"),
)
