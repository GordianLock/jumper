"""jumper.spin_spin -- jumper.spin_drift free to pivot on the spot as well. See
`env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..spin_drift import SPIN_ROLL_ASSETS

register(
    id="jumper.spin_spin",
    assets=SPIN_ROLL_ASSETS,
    description=(
        "jumper hexapod spin roll free to pivot on the spot: jumper.spin_drift without "
        "the charge on the roll axis turning in the world"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior", "carapace"),
)
