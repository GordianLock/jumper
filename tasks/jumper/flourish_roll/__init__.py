"""jumper.flourish_roll -- a slow, expressive roll of the policy's own invention,
charged for contact that is hard on the joints. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.flourish_roll",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod flourish roll: slow and expressive, its own invention, "
        "gentle on limbs, claws and sensors (demonstration only as a brief bootstrap)"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior"),
)
