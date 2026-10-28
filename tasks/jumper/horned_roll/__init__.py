"""jumper.horned_roll -- the momentum roll on carapace prototype 2, gentle, with the
horned carapace treated as the reinforced part it is. See `env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..common.assets import CARAPACE_ASSETS

register(
    id="jumper.horned_roll",
    assets=CARAPACE_ASSETS,
    description=(
        "jumper hexapod momentum roll on carapace prototype 2: gentle on limbs, sensors "
        "and the stock shell, the reinforced carapace allowed more by region, hard "
        "landings and airtime charged whatever touches down first"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior", "carapace"),
)
