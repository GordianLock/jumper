"""jumper.spin_spin -- the spin roll, free to pivot on the spot.

jumper.spin_drift (the version that lets the roll drift off its demonstration's axis
once the guide fades) with one term removed: `axis_heading`, jumper.horn_roll's charge
on the roll axis turning in the world, which is what a robot pivoting about the
vertical does. So the roll may wind up into a spin about the vertical as well.

Still in force: the turn is counted about the demonstration's axis and a roll finishes
only when that count reaches a full turn and the robot stands (so the roll itself
stays within roughly 45-60 deg of that axis); `axis_drift`, the charge on sliding
along the roll axis rather than across it; and every gentleness term -- face, claw and
limb load, impacts, air time, floor-contact joint torque.

## Settling at HOME after the finish

jumper.spin_drift's run 1 ended its episode on standing, so nothing after it was
trained: replayed 2026-10-30 past that point, the robot hopped once, 0.12 s in the
air. Here the episode goes on after the finish (`common/neutral_finish.py`) for up to
`SETTLE_WINDOW_S` and ends once the robot is settled: still standing, every driven
joint within `SETTLE_HOME_TOL` of HOME, turning slower than `SETTLE_ANG_VEL` and moving
slower than `SETTLE_LIN_VEL`, for `SETTLE_HOLD_STEPS` running. The finishing bonus
still pays once, on the finish step, and the approach shaping (`finish_upright`,
`finish_home`) up to it, so the roll is learnt as before; settling adds
`SETTLED_BONUS` once, and `settle_home` per second of legs near HOME while standing
(at most `SETTLE_HOME_WEIGHT` x `SETTLE_WINDOW_S`, below the bonus, so lingering does
not pay). Gentleness terms keep charging after the finish, so a hop there costs.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..common import neutral_finish as nf
from ..common.constants import GAIT_JOINTS
from ..spin_drift import env_cfg as spin
from ..tuck_roll import env_cfg as tuck

STEP_DT = tuck.STEP_DT

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
    """jumper.spin_drift without the charge on pivoting, settling at HOME after."""
    cfg = spin.env_cfg(asset=asset, play=play)
    cfg.rewards.pop("axis_heading")
    nf.settle_after_finish(
        cfg, "rolled_and_standing", SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS)),
        home_tol=SETTLE_HOME_TOL, still_ang_vel=SETTLE_ANG_VEL, still_lin_vel=SETTLE_LIN_VEL,
        hold_steps=SETTLE_HOLD_STEPS, window_steps=round(SETTLE_WINDOW_S / STEP_DT),
        settle_home_weight=SETTLE_HOME_WEIGHT, settle_sigma_sq=SETTLE_SIGMA_SQ,
        settled_weight=SETTLED_BONUS / STEP_DT,
        once_rewards=("complete",), until_rewards=("finish_upright", "finish_home"))
    return cfg


__all__ = ["env_cfg"]
