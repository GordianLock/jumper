"""jumper.showcase_roll -- a slow, gentle, lively roll: jumper.flourish_roll with the
completion bonus scaled by duration and by the hardest limb and sensor hit."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.showcase_roll",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod showcase roll: slow and expressive, the completion bonus "
        "scaled by its duration and its hardest limb and sensor contact"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior"),
)
