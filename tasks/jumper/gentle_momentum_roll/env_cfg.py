"""jumper.gentle_momentum_roll -- jumper.momentum_roll on carapace prototype 2, paid for
landing softly and loading the joints lightly, and for a roll done within a second.

The roll is jumper.momentum_roll's: crouch, lopsided push, over the back, land,
stand, right side first. It starts from the same demonstration
(`momentum_roll/ref/momentum_roll.npz`), fed forward and used as a prior that
fade to nothing, so the policy on the robot needs no recording. What differs is
what the objective pays for.

## The model

The default model is `CF1-tri30m-rc-v2` (`common/assets.py`,
`design/carapace/README.md`): the horned carapace's 49 convex collision pieces
on the top shell, its estimated mass in the shell's inertial. `--model jumper`
runs the same task on the bare robot.

Measured before this task was written (2026-10-28, this workspace, native CPU,
one env, jumper.momentum_roll's replay settings): the demonstration's recorded
targets, sent unchanged on v2, still complete the roll -- 360 degrees, ending
level on its feet -- with a peak of 123 N of shell and carapace on the floor and
0.26 s with nothing touching it. On the stock robot it had measured a median 108 N
over 64 randomised starts (`momentum_roll/mdp/reference.py`); one nominal start is
not that measurement, so read it as about the same.

## Time: within `ROLL_WINDOW`, and close to indifferent to faster

The roll is timed from the step the body has turned `START_TURN` (the crouch is not
counted) to standing. The completion bonus is full for any roll that stands by
`ROLL_WINDOW` seconds and falls to nothing by `ZERO_SECONDS`, and is paid at the
window's end whatever the time of standing, so a quick roll holds its stance
until then. Progress is jumper.momentum_roll's, capped at one full turn per
window (`MAX_RATE`), with gamma 0.998, so a quicker roll gains only about 5% in
discounted progress (`mdp/rewards.py` has the reasoning, and why the first
version's exactly-neutral shaping was dropped). jumper.momentum_roll's per-step
finishing rewards, which paid whoever reached the end first, are folded into the
bonus instead.

The demonstration itself takes about 1.5 s from turning to standing (turning
starts at 0.42 s, standing still comes after its landing teeter and hop, 1.96 s
in all), so the window asks training to tighten it. jumper.momentum_roll's first
policy rolled in 0.78 s.

## Gentle: impact forces and joint load

Two layers, both on floor contact by kind (`jumper.flourish_roll`'s division:
limbs other than the feet, the sensors, the feet, and the shell -- here the stock
shell, base and every carapace piece):

- **Part of the bonus is scaled by the hardest single contact of the episode** of
  each kind but the feet, falling from full at `*_FREE` to nothing at `*_ZERO`
  newtons (jumper.showcase_roll's scheme, the shell added). A slam lasts one
  control step, so a per-step charge alone barely sees it; through the bonus it
  costs the roll. `GENTLE_FLOOR` of the bonus is paid for standing however hard
  the roll landed, so that finishing is always worth more than not trying -- the
  first run, without it, learnt to stand still (`mdp/rewards.py`).
- **Per-step charges** above a threshold, for a gradient between slams; and the
  servos' own: holding over 1.5 N*m (`torque_strain`) and arriving at a joint stop
  at speed (`stop_strike`). The limb charge starts light and reaches its full
  weight as the guide fades out (`LIMB_WEIGHT_START`).

What this task cannot charge directly is the measure the horn study judged
gentleness by: the torque floor contact puts on each joint (J^T f per contact).
It needs a Jacobian per contact per step, too costly in training. Limb contact is
its main cause (jumper.flourish_roll measured the claws and shins as the largest
contributors), which is why limbs are charged hardest. Judge a trained policy by
that measure on replay, against the stock jumps' 5.9 and 4.8 N*m.

## On the numbers below

Taken from jumper.momentum_roll, jumper.flourish_roll and jumper.showcase_roll
where they exist, and said so. The new ones -- the window, the shell's bonus
factor, the finish factor, the gentle floor, the limb ramp -- are **first guesses**, set by reasoning about scale,
not against a trained policy.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from ..common.constants import FOOT_GEOMS, GAIT_JOINTS
from ..flourish_roll import env_cfg as flourish
from ..flourish_roll.mdp import rewards as fr
from ..momentum_roll import env_cfg as momentum
from ..momentum_roll import mdp as mr
from ..momentum_roll.mdp.reference import REF_NPZ
from ..ref_free_jump.mdp import curriculum as fade
from ..ref_free_jump.mdp.rewards import leg_collision
from ..roll.mdp import rewards as roll_rewards
from . import mdp as gm
from .rl_cfg import NUM_STEPS_PER_ENV

STEP_DT = momentum.STEP_DT

# ── Episodes: jumper.momentum_roll's ───────────────────────────────────────────
DEMO_FRACTION = momentum.DEMO_FRACTION
BACK_FRACTION = 0.0
EPISODE_S = momentum.EPISODE_S

# ── The guide: jumper.momentum_roll's schedule ─────────────────────────────────
PRIOR_POS_WEIGHT = momentum.PRIOR_POS_WEIGHT
PRIOR_POS_SIGMA_SQ = momentum.PRIOR_POS_SIGMA_SQ
PRIOR_TURN_WEIGHT = momentum.PRIOR_TURN_WEIGHT
PRIOR_TURN_SIGMA_SQ = momentum.PRIOR_TURN_SIGMA_SQ
PRIOR_HOLD_ITERATIONS = momentum.PRIOR_HOLD_ITERATIONS
PRIOR_END_ITERATIONS = momentum.PRIOR_END_ITERATIONS

# ── Time ───────────────────────────────────────────────────────────────────────
#: Radians of turn at which the roll counts as begun: jumper.showcase_roll's 0.3
#: (17 degrees). The demonstration passes it 0.42 s in, at the end of its crouch.
START_TURN = 0.3
#: Seconds from START_TURN to standing for the full bonus, and the time by which
#: it has fallen to nothing.
ROLL_WINDOW = 1.0
ZERO_SECONDS = 2.0

# ── The objective ──────────────────────────────────────────────────────────────
#: Per radian of new ground: jumper.momentum_roll's 2. The first version's
#: potential-based shaping averaged -0.4 to -0.8 per episode and taught nothing
#: (`mdp/rewards.py`).
PROGRESS_WEIGHT = 2.0
#: Turn rate, rad/s, above which progress pays nothing more: one full turn per
#: window, so a roll faster than ROLL_WINDOW earns no more progress for its speed.
MAX_RATE = 2.0 * math.pi / ROLL_WINDOW
#: jumper.showcase_roll's 20, against progress worth 2 x 2 pi = 12.6 per roll.
COMPLETE_BONUS = 20.0
#: Standing earns this share of the bonus; the rest rises with the legs' closeness
#: to HOME, at jumper.momentum_roll's finish_home width.
HOME_FLOOR = 0.5
HOME_SIGMA_SQ = 0.10

# ── Gentle: the bonus, by the hardest contact of the episode ───────────────────
#: Share of the bonus paid for standing after the roll whatever the hits; the rest
#: is scaled by them. The first run had none: its early rolls kept about 0.05 of the
#: bonus, 0.02-0.06 per episode, against about -0.45 of limb contact alone.
GENTLE_FLOOR = 0.5
#: Limbs and sensors: jumper.showcase_roll's.
LIMB_FREE = 20.0
LIMB_ZERO = 100.0
SENSOR_FREE = 2.0
SENSOR_ZERO = 20.0
#: Shell and carapace: about the robot's 26 N weight is lying on it; by 150 N, near
#: six times the weight, none of the bonus. The demonstration's 123 N on v2 would
#: keep a fifth of it.
SHELL_FREE = 30.0
SHELL_ZERO = 150.0

# ── Gentle: per step ───────────────────────────────────────────────────────────
#: Limbs at jumper.showcase_roll's halved weight; the rest jumper.flourish_roll's,
#: except the shell, charged from jumper.momentum_roll's 30 N body threshold rather
#: than flourish's 60, because here the shell is what was asked to land softly.
LIMB_THRESHOLD = flourish.LIMB_THRESHOLD
LIMB_WEIGHT = -0.5
#: The limb charge's weight while the guide is held, rising to LIMB_WEIGHT as it
#: fades (PRIOR_HOLD_ITERATIONS to PRIOR_END_ITERATIONS). In the first run limb
#: contact was the largest cost while the guide faded (-0.34 to -0.45 per episode,
#: iterations 100-500), so trying a roll was a loss before the policy could land one.
LIMB_WEIGHT_START = -0.1
SENSOR_THRESHOLD = flourish.SENSOR_THRESHOLD
SENSOR_WEIGHT = flourish.SENSOR_WEIGHT
FOOT_THRESHOLD = flourish.FOOT_THRESHOLD
FOOT_WEIGHT = flourish.FOOT_WEIGHT
SHELL_THRESHOLD = 30.0
SHELL_WEIGHT = -0.5
STOP_STRIKE_WEIGHT = flourish.STOP_STRIKE_WEIGHT
TORQUE_STRAIN_WEIGHT = flourish.TORQUE_STRAIN_WEIGHT

# ── The rest: jumper.momentum_roll's ───────────────────────────────────────────
AIRBORNE_WEIGHT = momentum.AIRBORNE_WEIGHT
HEADING_WEIGHT = momentum.HEADING_WEIGHT
FORWARD_DRIFT_WEIGHT = momentum.FORWARD_DRIFT_WEIGHT
FORWARD_DRIFT_SCALE = momentum.FORWARD_DRIFT_SCALE
LEG_COLLISION_WEIGHT = momentum.LEG_COLLISION_WEIGHT
JOINT_LIMIT_WEIGHT = momentum.JOINT_LIMIT_WEIGHT
ACTION_RATE_WEIGHT = momentum.ACTION_RATE_WEIGHT

# ── Contact kinds ──────────────────────────────────────────────────────────────
#: The stock shell, the base and every carapace piece (`upper_shell_carapace_NN`).
SHELL_PATTERN = r"(base_link|upper_shell_link|upper_shell_carapace_\d+)_meshcol"


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """The momentum roll, gentle, within a second, on a carapace."""
    cfg = momentum.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    gait_act = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))

    # ── Sensors: one per contact kind ─────────────────────────────────────────
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        flourish._ground("limb_ground_contact", flourish._LEG,
                         exclude=tuple(FOOT_GEOMS)),
        flourish._ground("sensor_ground_contact", flourish.SENSOR_PARTS),
        flourish._ground("shell_ground_contact", SHELL_PATTERN),
    )

    # ── Command: the roll's clock and its hardest hits ────────────────────────
    cfg.commands = {
        "roll": gm.state.GentleRollTrackerCfg(
            entity_name="robot", reference=str(REF_NPZ),
            limb_sensor="limb_ground_contact", sensor_sensor="sensor_ground_contact",
            shell_sensor="shell_ground_contact", start_turn=START_TURN,
        )
    }

    # ── Episodes ──────────────────────────────────────────────────────────────
    ev = cfg.events["reset_momentum"].params
    ev["demo_fraction"] = 0.0 if play else DEMO_FRACTION
    ev["back_fraction"] = BACK_FRACTION
    cfg.episode_length_s = EPISODE_S

    # ── Rewards ───────────────────────────────────────────────────────────────
    done = {"command_name": "roll", "window": ROLL_WINDOW,
            "min_own_turn": momentum.MIN_OWN_TURN}
    cfg.rewards = {
        # The guide, faded by jumper.momentum_roll's curriculum (kept from it).
        "prior_joint_pos": RewardTermCfg(
            func=mr.rewards.prior_joint_pos, weight=PRIOR_POS_WEIGHT,
            params={"asset_cfg": gait_cfg, "sigma_sq": PRIOR_POS_SIGMA_SQ},
        ),
        "prior_turn": RewardTermCfg(
            func=mr.rewards.prior_turn, weight=PRIOR_TURN_WEIGHT,
            params={"sigma_sq": PRIOR_TURN_SIGMA_SQ},
        ),
        # The objective
        "progress": RewardTermCfg(
            func=mr.rewards.progress, weight=PROGRESS_WEIGHT,
            params={"command_name": "roll", "max_rate": MAX_RATE},
        ),
        "complete": RewardTermCfg(
            func=gm.rewards.gentle_complete, weight=COMPLETE_BONUS / STEP_DT,
            params={**done, "zero_seconds": ZERO_SECONDS, "gentle_floor": GENTLE_FLOOR,
                    "limb_free": LIMB_FREE, "limb_zero": LIMB_ZERO,
                    "sensor_free": SENSOR_FREE, "sensor_zero": SENSOR_ZERO,
                    "shell_free": SHELL_FREE, "shell_zero": SHELL_ZERO,
                    "home_floor": HOME_FLOOR, "home_sigma_sq": HOME_SIGMA_SQ,
                    "asset_cfg": gait_cfg},
        ),
        # Gentle, per step
        "limb_contact": RewardTermCfg(
            func=fr.contact_overload, weight=LIMB_WEIGHT_START,
            params={"sensor_name": "limb_ground_contact", "threshold": LIMB_THRESHOLD},
        ),
        "sensor_contact": RewardTermCfg(
            func=fr.contact_overload, weight=SENSOR_WEIGHT,
            params={"sensor_name": "sensor_ground_contact", "threshold": SENSOR_THRESHOLD},
        ),
        "foot_impact": RewardTermCfg(
            func=fr.contact_overload, weight=FOOT_WEIGHT,
            params={"sensor_name": "feet_ground_contact", "threshold": FOOT_THRESHOLD},
        ),
        "shell_impact": RewardTermCfg(
            func=fr.contact_overload, weight=SHELL_WEIGHT,
            params={"sensor_name": "shell_ground_contact", "threshold": SHELL_THRESHOLD},
        ),
        "stop_strike": RewardTermCfg(
            func=fr.stop_strike, weight=STOP_STRIKE_WEIGHT, params={"asset_cfg": gait_cfg},
        ),
        "torque_strain": RewardTermCfg(
            func=fr.torque_strain, weight=TORQUE_STRAIN_WEIGHT, params={"asset_cfg": gait_act},
        ),
        # jumper.momentum_roll's
        "airborne": RewardTermCfg(func=roll_rewards.airborne, weight=AIRBORNE_WEIGHT),
        "heading_error": RewardTermCfg(func=roll_rewards.heading_error, weight=HEADING_WEIGHT),
        "forward_drift": RewardTermCfg(
            func=roll_rewards.forward_drift, weight=FORWARD_DRIFT_WEIGHT,
            params={"scale": FORWARD_DRIFT_SCALE},
        ),
        "leg_collision": RewardTermCfg(
            func=leg_collision, weight=LEG_COLLISION_WEIGHT,
            params={"sensor_name": "self_collision", "force_threshold": 1.0},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=JOINT_LIMIT_WEIGHT),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=ACTION_RATE_WEIGHT),
    }

    # ── Terminations: standing, once the window has passed ────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "rolled_and_standing": TerminationTermCfg(func=gm.rewards.window_done, params=done),
    }

    # The guide's schedule counts this task's steps per update, not momentum's.
    action = cfg.actions["joint_pos"]
    action.hold_step = NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS
    action.end_step = NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS
    for term in ("prior_joint_pos_weight", "prior_turn_weight"):
        if term in cfg.curriculum:
            p = cfg.curriculum[term].params
            p["start_step"] = NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS
            p["end_step"] = NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS
    # The limb charge rises to its full weight as the guide fades.
    if play:
        cfg.rewards["limb_contact"].weight = LIMB_WEIGHT
    else:
        cfg.curriculum["limb_contact_weight"] = CurriculumTermCfg(
            func=fade.anneal_reward_weight,
            params={"term_name": "limb_contact", "start_weight": LIMB_WEIGHT_START,
                    "end_weight": LIMB_WEIGHT,
                    "start_step": NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
                    "end_step": NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS},
        )
    return cfg


__all__ = ["env_cfg"]
