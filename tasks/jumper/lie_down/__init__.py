"""jumper.lie_down -- the stock robot, standing, lowers itself onto its back, slowly and
softly, any way it likes, and settles there with its legs at HOME. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.lie_down",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod lying down: from standing onto its back by any route, slowly "
        "and softly, finishing still with its legs at HOME"
    ),
    tags=("jumper", "roll", "one-shot", "lie-down"),
)
