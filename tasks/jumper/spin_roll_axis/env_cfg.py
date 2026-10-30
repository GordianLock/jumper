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

## Settling at HOME after the finish

As jumper.spin_spin (see its docstring for why): the episode goes on after the finish
(`common/neutral_finish.py`) for up to `SETTLE_WINDOW_S` and ends once the robot is
settled -- still standing, every driven joint within `SETTLE_HOME_TOL` of HOME,
turning slower than `SETTLE_ANG_VEL` and moving slower than `SETTLE_LIN_VEL`, for
`SETTLE_HOLD_STEPS` running. The axis-scaled finishing bonus still pays once, on the
finish step; settling adds `SETTLED_BONUS` once, and `settle_home` per second of legs
near HOME while standing.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..common import neutral_finish as nf
from ..common.constants import GAIT_JOINTS
from ..spin_drift import env_cfg as spin
from ..tuck_roll import env_cfg as tuck
from ..tuck_roll import mdp as tr

STEP_DT = tuck.STEP_DT

AXIS_FREE_DEG = 10.0
AXIS_SIGMA_DEG = 12.0
OFF_AXIS_FREE = 4.5
OFF_AXIS_SCALE = 4.0
OFF_AXIS_WEIGHT = -0.3

# ── Settling at HOME after the finish ─────────────────────────────────────────
SETTLE_WINDOW_S = 1.5
SETTLE_HOLD_STEPS = 15
SETTLE_HOME_TOL = 0.15
SETTLE_ANG_VEL = 0.2
SETTLE_LIN_VEL = 0.02
SETTLE_HOME_WEIGHT = 1.0
SETTLE_SIGMA_SQ = 0.10
SETTLED_BONUS = 4.0


def env_cfg(asset: Path | None = None, play: bool = False):
    """jumper.spin_drift held to its demonstration's axis, settling at HOME after."""
    cfg = spin.env_cfg(asset=asset, play=play)
    cfg.rewards["complete"].params["axis_sigma_deg"] = AXIS_SIGMA_DEG
    cfg.rewards["complete"].params["axis_free_deg"] = AXIS_FREE_DEG
    cfg.rewards["off_axis"] = RewardTermCfg(func=tr.rewards.off_axis, weight=OFF_AXIS_WEIGHT,
                                            params={"free": OFF_AXIS_FREE,
                                                    "scale": OFF_AXIS_SCALE})
    # Last: the wrapper copies the finishing bonus's params as they stand.
    nf.settle_after_finish(
        cfg, "rolled_and_standing", SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS)),
        home_tol=SETTLE_HOME_TOL, still_ang_vel=SETTLE_ANG_VEL, still_lin_vel=SETTLE_LIN_VEL,
        hold_steps=SETTLE_HOLD_STEPS, window_steps=round(SETTLE_WINDOW_S / STEP_DT),
        settle_home_weight=SETTLE_HOME_WEIGHT, settle_sigma_sq=SETTLE_SIGMA_SQ,
        settled_weight=SETTLED_BONUS / STEP_DT,
        once_rewards=("complete",), until_rewards=("finish_upright", "finish_home"))
    return cfg


__all__ = ["env_cfg"]
