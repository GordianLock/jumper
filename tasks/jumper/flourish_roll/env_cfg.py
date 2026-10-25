"""jumper.flourish_roll -- a roll of the policy's own invention: slow, strange and
fun to watch, done reliably, without hurting the robot.

jumper.roll and jumper.momentum_roll are paid to get over efficiently;
jumper.styled_roll to reproduce one demonstration. This task is paid to roll
*interestingly*, and it has to come up with the how. What it is paid for:

- **Finishing**, every time: progress round the roll and a bonus on standing.
- **Taking its time**: progress pays up to `MAX_RATE` only, so a full roll takes
  at least `2 pi / MAX_RATE` seconds to earn everything, and the episode is long.
- **Moving like something alive** (`mdp/rewards.py`): limbs in the air that are
  moving (`flourish`), and the body pitching as it rolls (`wobble`).

What it is charged for is what the hardware would not thank it for, kind by kind
(`mdp/rewards.py` has the reasoning; one contact sensor per kind):

- limbs on the floor other than at their tips, above `LIMB_THRESHOLD` per part;
- the display, dToF and camera touching the floor, above `SENSOR_THRESHOLD`;
- feet landing harder than `FOOT_THRESHOLD` each;
- the shell hitting harder than `SHELL_THRESHOLD`;
- joints arriving at their stops at speed, and servos held near the plateau.

## Why the limbs, specifically

Measured on jumper4 (jumper.styled_roll, `model_1999` of the 2026-10-25 Colab
run; 16 rolls, native CPU, states re-simulated in plain MuJoCo at the 5 ms physics
step): floor contact put a median peak of 5.3 N*m on its most-loaded joint, 6.1 at
worst, against a servo plateau of 1.75. KingKong's own shipped jump, measured the
same way, peaks at 4.8 (5.6 worst), walking at 2.0, a dance at 1.2. The rolls'
loads came mostly from the claws and shins meeting the floor side-on -- palms 2.7
N*m median peak (4.5 worst), the rear-left shin 2.5 (4.5) -- and they twisted the
J0 joints, which swing the legs sideways, where the jump loads the J1s.

## A bootstrap, not a template

jumper.momentum_roll's demonstration is fed forward and used as a prior only
briefly (`PRIOR_HOLD_ITERATIONS` to `PRIOR_END_ITERATIONS`), so the policy starts
from *a* roll rather than from nothing, and is then free to drift as far from it
as the style terms pay.

## On the numbers below

The thresholds are set against the measurements above and the robot's 25 N
weight; **every weight is a first guess**, and these ones are the knobs that set
the character of what is learnt (see `STYLE`).
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from ..common.constants import FOOT_GEOMS, GAIT_JOINTS
from ..momentum_roll import env_cfg as momentum
from ..momentum_roll import mdp as mr
from ..ref_free_jump.mdp import curriculum as fade
from ..ref_free_jump.mdp.rewards import leg_collision
from ..roll.mdp import rewards as roll_rewards
from .mdp import rewards as fr
from .rl_cfg import NUM_STEPS_PER_ENV

STEP_DT = momentum.STEP_DT

# ── Episodes ───────────────────────────────────────────────────────────────────
DEMO_FRACTION = 0.3
BACK_FRACTION = 0.0
#: Long enough for a slow roll and a stand.
EPISODE_S = 10.0

# ── The bootstrap: the demonstration, briefly ──────────────────────────────────
PRIOR_HOLD_ITERATIONS = 100
PRIOR_END_ITERATIONS = 600
PRIOR_POS_WEIGHT = 2.0
PRIOR_POS_SIGMA_SQ = 0.20
PRIOR_TURN_WEIGHT = 1.0
PRIOR_TURN_SIGMA_SQ = 0.5

# ── Finishing ──────────────────────────────────────────────────────────────────
PROGRESS_WEIGHT = 3.0
#: Turn rate, rad/s, above which progress pays nothing more: a full roll earns
#: everything only if it takes at least 2 pi / 1.5 = 4.2 s.
MAX_RATE = 1.5
FINISH_TOLERANCE = 0.5
FINISH_UPRIGHT_WEIGHT = 1.0
FINISH_HOME_WEIGHT = 1.0
FINISH_SIGMA_SQ = 0.10
COMPLETE_BONUS = 15.0

# ── Style: the character knobs ─────────────────────────────────────────────────
STYLE = {
    #: Limbs in the air that are moving; joint speeds capped at `FLOURISH_CAP`.
    "flourish": 1.0,
    #: The body pitching mid-roll, up to `WOBBLE_BOUND` (sin of the pitch).
    "wobble": 0.5,
}
FLOURISH_CAP = 4.0
WOBBLE_BOUND = 0.5

# ── Safety ─────────────────────────────────────────────────────────────────────
#: Per part, newtons. Limbs: a fifth of the robot's weight on one shin or claw.
LIMB_THRESHOLD = 5.0
LIMB_WEIGHT = -1.0
#: The display, dToF and camera should not touch the floor at all.
SENSOR_THRESHOLD = 2.0
SENSOR_WEIGHT = -2.0
#: Per foot: about one and a half times the robot's weight on one foot.
FOOT_THRESHOLD = 40.0
FOOT_WEIGHT = -1.0
#: The shell is what the print will be designed to take.
SHELL_THRESHOLD = 60.0
SHELL_WEIGHT = -0.5
#: (rad/s)^2 of joints arriving at a stop.
STOP_STRIKE_WEIGHT = -0.02
#: Share of joints over 1.5 N*m.
TORQUE_STRAIN_WEIGHT = -1.0
AIRBORNE_WEIGHT = -1.0
HEADING_WEIGHT = -0.1
FORWARD_DRIFT_WEIGHT = -0.1
FORWARD_DRIFT_SCALE = 0.30
LEG_COLLISION_WEIGHT = -1.0
JOINT_LIMIT_WEIGHT = -1.0
ACTION_RATE_WEIGHT = -0.005

# ── Contact kinds: every one of the model's 41 collision parts is in one ───────
_LEG = r"(LF|RF|LM|RM|LR|RR)_.*_meshcol.*"
SENSOR_PARTS = ("display_module_link_meshcol", "tof_sensor_link_meshcol",
                "camera_link_meshcol")
SHELL_PARTS = ("base_link_meshcol", "upper_shell_link_meshcol")


def _ground(name: str, pattern, exclude=()) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=pattern, entity="robot", exclude=exclude),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",), reduce="netforce", num_slots=1,
    )


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """A slow, expressive roll that is gentle on the robot."""
    cfg = momentum.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    gait_act = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))

    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        _ground("limb_ground_contact", _LEG, exclude=tuple(FOOT_GEOMS)),
        _ground("sensor_ground_contact", SENSOR_PARTS),
        _ground("shell_ground_contact", SHELL_PARTS),
    )

    ev = cfg.events["reset_momentum"].params
    ev["demo_fraction"] = 0.0 if play else DEMO_FRACTION
    ev["back_fraction"] = BACK_FRACTION
    cfg.episode_length_s = EPISODE_S

    action = cfg.actions["joint_pos"]
    action.hold_step = NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS
    action.end_step = NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS

    stand = {"command_name": "roll", "min_own_turn": momentum.MIN_OWN_TURN}
    cfg.rewards = {
        # Bootstrap
        "prior_joint_pos": RewardTermCfg(
            func=mr.rewards.prior_joint_pos, weight=PRIOR_POS_WEIGHT,
            params={"asset_cfg": gait_cfg, "sigma_sq": PRIOR_POS_SIGMA_SQ},
        ),
        "prior_turn": RewardTermCfg(
            func=mr.rewards.prior_turn, weight=PRIOR_TURN_WEIGHT,
            params={"sigma_sq": PRIOR_TURN_SIGMA_SQ},
        ),
        # Finishing
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
        # Style
        "flourish": RewardTermCfg(
            func=fr.Flourish, weight=STYLE["flourish"],
            params={"sensor_name": "feet_ground_contact", "cap": FLOURISH_CAP},
        ),
        "wobble": RewardTermCfg(
            func=fr.wobble, weight=STYLE["wobble"], params={"bound": WOBBLE_BOUND},
        ),
        # Safety
        "limb_contact": RewardTermCfg(
            func=fr.contact_overload, weight=LIMB_WEIGHT,
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

    def _fade(term: str, start: float) -> CurriculumTermCfg:
        return CurriculumTermCfg(
            func=fade.anneal_reward_weight,
            params={"term_name": term, "start_weight": start, "end_weight": 0.0,
                    "start_step": NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
                    "end_step": NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS},
        )

    cfg.curriculum = {} if play else {
        "prior_joint_pos_weight": _fade("prior_joint_pos", PRIOR_POS_WEIGHT),
        "prior_turn_weight": _fade("prior_turn", PRIOR_TURN_WEIGHT),
        "feedforward": CurriculumTermCfg(func=fade.report_feedforward),
    }
    return cfg


__all__ = ["env_cfg"]
