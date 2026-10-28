"""jumper.gentle_momentum_roll -- jumper.momentum_roll on a carapace, paid for landing
softly and for a roll of 1 s or less, and indifferent to anything faster. See
`env_cfg.py`."""

from __future__ import annotations

from ...registry import register
from ..common.assets import CARAPACE_ASSETS

register(
    id="jumper.gentle_momentum_roll",
    assets=CARAPACE_ASSETS,
    description=(
        "jumper hexapod momentum roll on carapace prototype 2: the momentum roll's "
        "demonstration as a fading guide, low impact and low torque, done within 1 s"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior", "carapace"),
)
