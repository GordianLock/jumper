"""jumper.pick_place -- pick up debris and put it in a receptacle or on a pile.

Standing on five legs, working with the left claw. No recording. See `env_cfg.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.pick_place",
    assets=JUMPER_ASSETS,
    description=(
        "jumper on five legs: pick a piece of debris off the floor with the left "
        "claw and put it in a receptacle or on a pile (reference-free)"
    ),
    tags=("jumper", "manipulation", "one-shot", "claw"),
)
