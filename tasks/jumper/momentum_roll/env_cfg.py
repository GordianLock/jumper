"""jumper.momentum_roll -- a sideways roll driven by a push, guided by a demonstration,
with self-righting as the way out of one that stalls.

Crouch, push hard with one side while the other gives way, spin over the back
without stopping there, land on the far side and stand: a full turn about the
body's long axis, right side first. The push is the jump's mechanism turned into
a twist -- uneven, mostly downward leg forces rather than a sideways shove, which
the feet's grip would not hold.

## How it differs from jumper.roll

jumper.roll learns its roll from the reward alone; its first trained policy rolls
over in 0.88 s and lands with a median 129 N of body on the floor (198 at worst;
16 envs, native CPU, from `model_1499` of the 2026-10-24 Colab run). This task
starts from **a demonstration** -- `ref/momentum_roll.npz`, see
`mdp/reference.py` -- fed forward into the action and used as a prior on the
joints and the turn, both fading to nothing over `PRIOR_HOLD_ITERATIONS` to
`PRIOR_END_ITERATIONS`, which is `jumper.ref_free_jump`'s scheme. By the end the
actor rolls with no recording on the robot and no clock: the roll starts when the
mode does.

## Three kinds of episode (`mdp/events.py`)

- `1 - DEMO_FRACTION - BACK_FRACTION` from standing, the roll the robot is asked for;
- `DEMO_FRACTION` partway through the demonstration, at its recorded state;
- `BACK_FRACTION` on the back, **self-righting**: upright either way round counts.
  On the robot this is the way out of a roll that stalled, and the actor cannot
  tell the two apart -- on its back is on its back.

## The physics is jumper.roll's

The demonstration was recorded in it (50 Hz control, 5 ms physics, kp 10, the
servo curve), and its targets reproduce its roll only there. This file builds on
`roll/env_cfg.py` for the scene, sensors, observations and action space, and
states every parameter of its own: commands, action, rewards, curriculum,
terminations, events and episode.

## On the numbers below

The prior's kernel widths are `jumper.ref_free_jump`'s; **every weight and the
schedule are first guesses**.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from ..common.constants import GAIT_JOINTS
from ..ref_free_jump.mdp import curriculum as fade
from ..ref_free_jump.mdp.rewards import leg_collision
from ..roll import env_cfg as roll_env
from ..roll.mdp import rewards as roll_rewards
from . import mdp as mr
from .rl_cfg import NUM_STEPS_PER_ENV

STEP_DT = roll_env.STEP_DT
ACTION_SCALE = 0.75

# ── Episodes ───────────────────────────────────────────────────────────────────
DEMO_FRACTION = 0.4
BACK_FRACTION = 0.25
#: On the back, give or take 40 degrees, legs near HOME.
BACK_ANGLE_RANGE = (math.radians(140.0), math.radians(220.0))
BACK_JOINT_NOISE = 0.2
#: Mid-roll spawns stop at the demonstration's last frame under this turn, so a
#: 57-degree turn of the policy's own (`MIN_OWN_TURN`) is still ahead of them.
MAX_SPAWN_TURN = math.radians(290.0)
MIN_OWN_TURN = 1.0

EPISODE_S = 4.0

# ── The prior, and how it fades ────────────────────────────────────────────────
PRIOR_POS_WEIGHT = 3.0
PRIOR_POS_SIGMA_SQ = 0.20
PRIOR_TURN_WEIGHT = 1.0
PRIOR_TURN_SIGMA_SQ = 0.5
#: Iterations at full feedforward and prior weight, and the iteration both reach
#: zero. Shorter than jumper.ref_free_jump's 1000 / 3000: a free Colab GPU session
#: is a few hours.
PRIOR_HOLD_ITERATIONS = 300
PRIOR_END_ITERATIONS = 1200

# ── The objective ──────────────────────────────────────────────────────────────
PROGRESS_WEIGHT = 2.0
#: Faster than jumper.roll's 4: the demonstration spins at up to 21 rad/s, and
#: most of its turn happens in a quarter of a second.
MAX_RATE = 6.0
FINISH_TOLERANCE = 0.5
FINISH_UPRIGHT_WEIGHT = 1.0
FINISH_HOME_WEIGHT = 1.0
FINISH_SIGMA_SQ = 0.10
COMPLETE_BONUS = 10.0

# ── The costs ──────────────────────────────────────────────────────────────────
#: Body (not feet) on the floor above this many newtons, charged quadratically.
#: Tighter than jumper.roll's 40 N at weight -1: the demonstration's median peak
#: is 108 N and the roll's trained policy 129 N, four to five times the robot's
#: weight, landing on what will be the shell.
IMPACT_THRESHOLD = 30.0
IMPACT_WEIGHT = -2.0
#: Per second with nothing touching the floor. The demonstration has 0.26 s of it.
AIRBORNE_WEIGHT = -2.0
HEADING_WEIGHT = -0.3
FORWARD_DRIFT_WEIGHT = -0.2
FORWARD_DRIFT_SCALE = 0.20
LEG_COLLISION_WEIGHT = -1.0
JOINT_LIMIT_WEIGHT = -1.0
ACTION_RATE_WEIGHT = -0.01


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Momentum roll from a fading demonstration, with self-righting."""
    cfg = roll_env.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))

    # ── Command: jumper.roll's tracker, with the demonstration ────────────────
    cfg.commands = {
        "roll": mr.commands.MomentumRollTrackerCfg(
            entity_name="robot", reference=str(mr.reference.REF_NPZ)
        )
    }
    cfg.observations["critic"].terms["roll_state"] = ObservationTermCfg(
        func=mr.observations.roll_state, params={"command_name": "roll"}
    )

    # ── Action: targets around HOME, the demonstration fed forward ────────────
    cfg.actions["joint_pos"] = mr.actions.FeedforwardRollActionCfg(
        entity_name="robot",
        actuator_names=list(GAIT_JOINTS),
        scale={j: ACTION_SCALE for j in GAIT_JOINTS},
        use_default_offset=True,
        command_name="roll",
        hold_step=NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
        end_step=NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS,
        enabled=not play,
    )

    # ── Rewards ───────────────────────────────────────────────────────────────
    stand = {"command_name": "roll", "min_own_turn": MIN_OWN_TURN}
    cfg.rewards = {
        "prior_joint_pos": RewardTermCfg(
            func=mr.rewards.prior_joint_pos, weight=PRIOR_POS_WEIGHT,
            params={"asset_cfg": gait_cfg, "sigma_sq": PRIOR_POS_SIGMA_SQ},
        ),
        "prior_turn": RewardTermCfg(
            func=mr.rewards.prior_turn, weight=PRIOR_TURN_WEIGHT,
            params={"sigma_sq": PRIOR_TURN_SIGMA_SQ},
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
        "airborne": RewardTermCfg(func=mr.rewards.airborne, weight=AIRBORNE_WEIGHT),
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

    # ── Curriculum: the prior fades with the feedforward ──────────────────────
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

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "rolled_and_standing": TerminationTermCfg(func=mr.terminations.standing_done,
                                                  params=stand),
    }

    # ── Events: jumper.roll's, with this task's spawns ────────────────────────
    events = {k: v for k, v in cfg.events.items() if k != "reset_rolled"}
    cfg.events = {
        "reset_to_default": events.pop("reset_to_default"),
        "reset_momentum": EventTermCfg(
            func=mr.events.reset_momentum, mode="reset",
            params={"demo_fraction": 0.0 if play else DEMO_FRACTION,
                    "back_fraction": 0.0 if play else BACK_FRACTION,
                    "back_angle_range": BACK_ANGLE_RANGE,
                    "back_joint_noise": BACK_JOINT_NOISE,
                    "max_spawn_turn": MAX_SPAWN_TURN},
        ),
        **events,
    }

    cfg.episode_length_s = EPISODE_S
    return cfg


__all__ = ["env_cfg"]
