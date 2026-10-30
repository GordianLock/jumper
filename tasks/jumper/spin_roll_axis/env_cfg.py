"""jumper.spin_roll_axis -- the spin roll, held to its axis.

jumper.spin_drift counts the turn about the demonstration's axis, and once the guide
has faded nothing holds the roll to it: jumper.diag_roll's first policy, trained the
same way, drifted from its 47 deg diagonal to 81 deg, nearly end over end (kept as the
shoulder roll). This task is jumper.spin_drift, identical, plus two terms that last the
whole of training:

- **`off_axis`**, per step: the body's horizontal rotation rate about axes other than
  the demonstration's, charged ((r - `OFF_AXIS_FREE`) / `OFF_AXIS_SCALE`)^2, capped.
  The move is a spin: the demonstration itself turns off its axis at 1.5 rad/s median,
  4.1 at its 90th percentile, 7.8 at most (measured on `spin_drift/ref/spin_roll.npz`),
  so rates up to its 90th percentile are free and the charge is for leaving it.
- **The finishing bonus is scaled by the axis**:
  exp(-(max(0, e - `AXIS_FREE_DEG`) / `AXIS_SIGMA_DEG`)^2), e the angle between the
  axis the episode rolled about (body-frame rotation accumulated between belly-up and
  upright, `tuck_roll/mdp/tracker.py`) and the demonstration's. Within 10 deg the
  bonus is whole, as asked; it keeps 84% at 15 deg, 50% at 20, 20% at 25, 8% at the
  shoulder roll's 29 deg, 6% at 30.

The bonus scaling is what holds the axis; the per-step charge only keeps the roll from
leaving the demonstration's own amount of spin. Everything else, the demonstration
included, is jumper.spin_drift's.

## The episode ends at the finish

Tried and reverted (2026-10-30): keeping the episode going after the finish to settle
at HOME (`common/neutral_finish.py` at revision 7501321, since removed). Colab, 2500 iterations on
prototype 4: once the guide faded the policy rolled (a full turn 97-99% of the time)
and then crouched short of standing for the rest of the 4 s episode, collecting
`finish_upright` per step; the finishing bonus fell to nothing and no episode ever
settled. Finishing had stopped ending the episode, so it now cost that per-step pay,
and in the guide's phase the settle window was unwinnable. Footage gets its neutral
ending from a stand controller handed over to after the finish instead.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.reward_manager import RewardTermCfg

from ..spin_drift import env_cfg as spin
from ..tuck_roll import mdp as tr

AXIS_FREE_DEG = 10.0
AXIS_SIGMA_DEG = 12.0
OFF_AXIS_FREE = 4.5
OFF_AXIS_SCALE = 4.0
OFF_AXIS_WEIGHT = -0.3


def env_cfg(asset: Path | None = None, play: bool = False):
    """jumper.spin_drift held to its demonstration's axis."""
    cfg = spin.env_cfg(asset=asset, play=play)
    cfg.rewards["complete"].params["axis_sigma_deg"] = AXIS_SIGMA_DEG
    cfg.rewards["complete"].params["axis_free_deg"] = AXIS_FREE_DEG
    cfg.rewards["off_axis"] = RewardTermCfg(func=tr.rewards.off_axis, weight=OFF_AXIS_WEIGHT,
                                            params={"free": OFF_AXIS_FREE,
                                                    "scale": OFF_AXIS_SCALE})
    return cfg


__all__ = ["env_cfg"]
