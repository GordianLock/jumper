"""jumper.styled_roll -- the momentum roll's demonstration, performed faithfully.

`ref/momentum_roll.npz` (jumper.momentum_roll's demonstration) is not the fastest
or the gentlest way over: it crouches, flings itself over its back, comes down on
its far edge, teeters, drops onto its feet and hops. jumper.momentum_roll treats
it as scaffolding and trained away from it -- its first policy rolls in 0.78 s
with 0.06 s in the air, where the demonstration takes 1.96 s with 0.26 s. This
task keeps the demonstration's character instead: the same roll, done reliably.

## What differs from jumper.momentum_roll

- **The priors never fade.** Joint angles, the body's turn and -- added here --
  the base's height (`mdp/rewards.py`) are paid against the demonstration at the
  same moment for the whole of training, and they carry most of the reward.
- **The feedforward still fades** (`PRIOR_HOLD_ITERATIONS` to
  `PRIOR_END_ITERATIONS`), so the policy on the robot needs no recording and no
  clock. It keeps the demonstration's timing by tracking its own state, as
  jumper.ref_free_jump keeps its jump's.
- **The costs allow the demonstration's own landing.** Body impact is charged
  only above `IMPACT_THRESHOLD`, set above the demonstration's median 108 N; the
  hop is not charged at all. Both are part of how it looks.
- **No back starts.** Self-righting is a task of its own, later.

## On the numbers below

The prior's kernel widths tighten jumper.momentum_roll's; **the weights and
thresholds are first guesses**, chosen so that tracking the demonstration pays
more per episode than finishing the roll faster another way could.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.tasks.velocity import mdp

from ..common.constants import GAIT_JOINTS
from ..momentum_roll import env_cfg as momentum
from ..momentum_roll import mdp as mr
from ..ref_free_jump.mdp import curriculum as fade
from ..ref_free_jump.mdp.rewards import leg_collision
from ..roll.mdp import rewards as roll_rewards
from . import mdp as styled

STEP_DT = momentum.STEP_DT

# ── Episodes ───────────────────────────────────────────────────────────────────
#: Half the episodes start partway through the demonstration, at its recorded
#: state: every part of it is practised, the teeter and the hop included.
DEMO_FRACTION = 0.5
BACK_FRACTION = 0.0

# ── Tracking the demonstration: permanent ──────────────────────────────────────
PRIOR_POS_WEIGHT = 4.0
#: Mean squared joint error giving exp(-1): 0.10 rad^2, about 18 degrees RMS --
#: tighter than jumper.momentum_roll's 0.20.
PRIOR_POS_SIGMA_SQ = 0.10
PRIOR_TURN_WEIGHT = 2.0
#: 0.45 rad (26 degrees) of turn error gives exp(-1).
PRIOR_TURN_SIGMA_SQ = 0.2
PRIOR_HEIGHT_WEIGHT = 2.0
#: 5 cm of base height error gives exp(-1); the hop is 10 cm.
PRIOR_HEIGHT_SIGMA_SQ = 0.0025

# ── The objective: finish, but not at the cost of the style ────────────────────
PROGRESS_WEIGHT = 1.0
MAX_RATE = 6.0
FINISH_TOLERANCE = 0.5
FINISH_UPRIGHT_WEIGHT = 1.0
FINISH_HOME_WEIGHT = 1.0
FINISH_SIGMA_SQ = 0.10
COMPLETE_BONUS = 10.0

# ── The costs: only what is worse than the demonstration ───────────────────────
#: Body on the floor charged above this many newtons: the demonstration peaks at a
#: median 108 N, so its landing is free and only harder ones pay.
IMPACT_THRESHOLD = 120.0
IMPACT_WEIGHT = -1.0
HEADING_WEIGHT = -0.3
FORWARD_DRIFT_WEIGHT = -0.2
FORWARD_DRIFT_SCALE = 0.20
LEG_COLLISION_WEIGHT = -1.0
JOINT_LIMIT_WEIGHT = -1.0
ACTION_RATE_WEIGHT = -0.01


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """The demonstration's roll, tracked for the whole of training."""
    cfg = momentum.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))

    ev = cfg.events["reset_momentum"].params
    ev["demo_fraction"] = 0.0 if play else DEMO_FRACTION
    ev["back_fraction"] = BACK_FRACTION

    stand = {"command_name": "roll", "min_own_turn": momentum.MIN_OWN_TURN}
    cfg.rewards = {
        "prior_joint_pos": RewardTermCfg(
            func=mr.rewards.prior_joint_pos, weight=PRIOR_POS_WEIGHT,
            params={"asset_cfg": gait_cfg, "sigma_sq": PRIOR_POS_SIGMA_SQ},
        ),
        "prior_turn": RewardTermCfg(
            func=mr.rewards.prior_turn, weight=PRIOR_TURN_WEIGHT,
            params={"sigma_sq": PRIOR_TURN_SIGMA_SQ},
        ),
        "prior_height": RewardTermCfg(
            func=styled.rewards.prior_height, weight=PRIOR_HEIGHT_WEIGHT,
            params={"sigma_sq": PRIOR_HEIGHT_SIGMA_SQ},
        ),
        "progress": RewardTermCfg(
            func=mr.rewards.progress, weight=PROGRESS_WEIGHT, params={"max_rate": MAX_RATE},
        ),
        "finish_upright": RewardTermCfg(
            func=mr.rewards.finish_upright, weight=FINISH_UPRIGHT_WEIGHT,
            params={"tolerance": FINISH_TOLERANCE, "sigma_sq": FINISH_SIGMA_SQ},
        ),
        "finish_home": RewardTermCfg(
            func=mr.rewards.finish_home, weight=FINISH_HOME_WEIGHT,
            params={"asset_cfg": gait_cfg, "tolerance": FINISH_TOLERANCE,
                    "sigma_sq": FINISH_SIGMA_SQ},
        ),
        "complete": RewardTermCfg(
            func=mr.rewards.complete, weight=COMPLETE_BONUS / STEP_DT, params=stand,
        ),
        "body_impact": RewardTermCfg(
            func=roll_rewards.body_impact, weight=IMPACT_WEIGHT,
            params={"sensor_name": "body_ground_contact", "threshold": IMPACT_THRESHOLD},
        ),
        "heading_error": RewardTermCfg(func=roll_rewards.heading_error, weight=HEADING_WEIGHT),
        "forward_drift": RewardTermCfg(
            func=roll_rewards.forward_drift, weight=FORWARD_DRIFT_WEIGHT,
            params={"scale": FORWARD_DRIFT_SCALE},
        ),
        "leg_collision": RewardTermCfg(
            func=leg_collision, weight=LEG_COLLISION_WEIGHT,
            params={"sensor_name": "self_collision", "force_threshold": 1.0},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits, weight=JOINT_LIMIT_WEIGHT
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2, weight=ACTION_RATE_WEIGHT
        ),
    }

    # The feedforward still fades; the priors do not, so only its value is logged.
    cfg.curriculum = {} if play else {
        "feedforward": CurriculumTermCfg(func=fade.report_feedforward),
    }
    return cfg


__all__ = ["env_cfg"]
