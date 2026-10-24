"""jumper.roll -- a sideways barrel roll: a full turn about the body's long axis,
right side first, over the back and onto the feet again.

No recording. The policy commands absolute joint targets around HOME and sees
its body and nothing else; the episode is the roll, and it ends the moment the
robot is standing after a full turn. See `env_cfg.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.roll",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod sideways barrel roll: a full 360-degree turn about the "
        "long axis, ending on its feet (reference-free)"
    ),
    tags=("jumper", "roll", "one-shot"),
)
