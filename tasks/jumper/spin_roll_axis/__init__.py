"""jumper.spin_roll_axis -- jumper.spin_drift held to its demonstration's roll axis for
the whole of training. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..spin_drift import SPIN_ROLL_ASSETS

register(
    id="jumper.spin_roll_axis",
    assets=SPIN_ROLL_ASSETS,
    description=(
        "jumper hexapod spin roll held to its demonstration's axis: jumper.spin_drift "
        "with off-axis rotation charged and the finishing bonus scaled by the axis"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior", "carapace"),
)
