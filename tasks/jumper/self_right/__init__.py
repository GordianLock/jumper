"""jumper.self_right -- the stock robot, lying at rest on its back, gets back onto its
feet, any way it likes, gently. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.self_right",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod self-righting: starts at rest on its back, paid for getting "
        "upright by any route, the finish scaled by how gently it went"
    ),
    tags=("jumper", "roll", "one-shot", "self-right"),
)
