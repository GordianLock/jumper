"""jumper.roll -- a sideways barrel roll, reference-free.

One full turn about the body's long axis, right side going down first: onto the
right side, over the back, onto the left side and back onto the feet. The
episode is that roll; it ends, as a true termination, the step the robot stands
after it (`mdp/terminations.py`).

## Why this robot can do it

Measured on this model (`assets/jumper/jumper.xml`, V1.6) rather than assumed:
every foot reaches 0.11-0.13 m above the top of the base -- the front shoulders'
J1 spans 6.15 rad, the leg hips' 3.9 -- so on its back the robot can still put
feet on the ground and push. The base is 0.21 x 0.18 x 0.05 m; most of the
robot's 0.42 m width is legs it can tuck. Standing needs 13% of a servo's
plateau torque, so a slow roll has torque in hand; nothing here asks for speed.

## How it is taught

- **The turn is the reward** (`roll_progress`): radians per second about the
  long axis, the right way, capped at `MAX_ROLL_RATE`, counted to one full roll.
  Turning back costs the same, so rocking pays nothing.
- **Spawns partway round** (`mdp/events.py`): `ROLLED_FRACTION` of episodes
  start turned by an angle in `ROLLED_ANGLE_RANGE` with the legs near HOME, so
  the side, the back and the far side are practised before the policy can reach
  them. The rest start at HOME, where the robot will.
- **Finishing**: near a full turn, level and HOME pay per step, and standing pays
  `COMPLETE_BONUS` once and ends the episode.
- **No "fell over" termination** -- lying on its back is the middle of the roll.

## What is not modelled yet

**The shell.** The model has none, so the body that touches the ground mid-roll
is the bare base. A printed shell changes the contact shape, the mass and the
centre of mass, and the roll should be retrained, or at least checked, against a
collision proxy for it before it goes on a robot. `body_impact` is here for the
shell's sake regardless.

## On the numbers below

The physics and the hardware facts are the walking tasks', from
`common/constants.py`. **Every weight, rate and threshold below is a first
guess**, set by reasoning about scale, not against a trained policy.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..common.constants import (
    ACTUATOR_FORCE_NOISE,
    EFFORT_LIMIT,
    FOOT_GEOMS,
    GAIT_JOINTS,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    PUSH_VELOCITY_RANGE,
    get_jumper_robot_cfg,
)
from ..common.mdp.observations import actuator_force
from ..common.tof import tof_sensor
from ..ref_free_jump.mdp.rewards import leg_collision
from . import mdp as roll_mdp

# ── Physics: the walking tasks' 50 Hz, kp=10 ──────────────────────────────────
#: The skeleton's 5 ms physics step, four per control step. A roll is slow next
#: to a jump's 50 ms push; nothing here needs 200 Hz.
_SIM_TIMESTEP = 0.005
_DECIMATION = 4
STEP_DT = _SIM_TIMESTEP * _DECIMATION

# ── The action ─────────────────────────────────────────────────────────────────
#: Absolute targets around HOME. Getting a leg over the back moves a hip 1.5-2
#: rad from HOME, an output of 2-2.7 at this scale.
ACTION_SCALE = 0.75

# ── The observation's history ──────────────────────────────────────────────────
#: The walking tasks' five frames, 0.1 s at 50 Hz, on the per-joint terms.
OBS_HISTORY = 5
HISTORY_TERMS = frozenset({"joint_pos", "joint_vel", "actions", "actuator_force"})

# ── Spawns ─────────────────────────────────────────────────────────────────────
#: Share of episodes started partway round, and the turn they start at (rad,
#: right side down). Up to 280 degrees: completion needs a turn of the policy's
#: own (`mdp/terminations.py`, `min_own_turn`, 57 degrees), and from 280 that is
#: the whole of the last side. Measured with the legs at HOME, bodies set down
#: below about 310 degrees flop back onto the back rather than onto the feet.
ROLLED_FRACTION = 0.6
ROLLED_ANGLE_RANGE = (math.radians(20.0), math.radians(280.0))
ROLLED_JOINT_NOISE = 0.1

# ── The objective ──────────────────────────────────────────────────────────────
#: Per radian turned, so a full roll pays 2 pi times this.
PROGRESS_WEIGHT = 2.0
#: The tempo: turn rate (rad/s) above which more speed pays nothing. 4 rad/s is a
#: full roll in about 1.6 s at best.
MAX_ROLL_RATE = 4.0

#: Within this of a full turn (rad), the finishing terms start to pay.
FINISH_TOLERANCE = 0.5
FINISH_UPRIGHT_WEIGHT = 1.0
FINISH_HOME_WEIGHT = 1.0
FINISH_SIGMA_SQ = 0.10

#: Paid once, the step it stands after the roll. Rewards are scaled by the step
#: length, so the weight is the bonus over `STEP_DT`.
COMPLETE_BONUS = 10.0

#: What counts as standing after the roll (`mdp/terminations.py`).
STAND_TURN_TOLERANCE = 0.3
STAND_MAX_TILT = math.radians(20.0)
STAND_MIN_FEET = 4
STAND_MAX_ANG_VEL = 2.0

# ── The costs ──────────────────────────────────────────────────────────────────
#: Net non-foot ground force above this (N) is charged quadratically. The robot
#: weighs about 25 N on the model; this charges landing hard, not lying down.
IMPACT_THRESHOLD = 40.0
IMPACT_WEIGHT = -1.0

#: Softened after the first trained run, where these two were the largest costs
#: (-0.30 and -0.44 per second of episode against 0.39 for the turn, iterations
#: 900-999): a body rolling over its side shifts and swings a little, and a push
#: that fails still moves it. They keep the roll sideways; they should not make
#: trying it a loss.
HEADING_WEIGHT = -0.3
FORWARD_DRIFT_WEIGHT = -0.2
FORWARD_DRIFT_SCALE = 0.20

LEG_COLLISION_WEIGHT = -1.0
JOINT_LIMIT_WEIGHT = -1.0
#: On the raw action; a target moved by dq is dq / ACTION_SCALE of it.
ACTION_RATE_WEIGHT = -0.01

#: Long enough to roll slowly and stand, with margin.
EPISODE_S = 6.0

_COLLISION_GEOMS = r".*_meshcol"


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Sideways barrel roll.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- noise and pushes off, every episode from HOME.
    """
    cfg = make_velocity_env_cfg()

    # ── Sim ───────────────────────────────────────────────────────────────────
    cfg.sim.mujoco.timestep = _SIM_TIMESTEP
    cfg.decimation = _DECIMATION
    # More of the body touches the ground than in any walking task.
    cfg.sim.njmax = 768
    cfg.sim.nconmax = 192
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128

    # ── Scene ─────────────────────────────────────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.entities = {"robot": get_jumper_robot_cfg(effort_limit=EFFORT_LIMIT, asset=asset)}

    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan")
    ) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=_DECIMATION,
        ),
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="geom", pattern=_COLLISION_GEOMS, entity="robot",
                                 exclude=FOOT_GEOMS),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
    )

    # ── Observations ──────────────────────────────────────────────────────────
    for group in cfg.observations.values():
        for name in ("command", "height_scan", "foot_height", "foot_air_time",
                     "foot_contact", "foot_contact_forces"):
            group.terms.pop(name, None)
    # No state estimator on the robot: no linear velocity for the actor.
    del cfg.observations["actor"].terms["base_lin_vel"]

    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = dict(term.params or {}) | {"asset_cfg": gait_cfg}
        if "base_ang_vel" in group.terms:
            group.terms["base_ang_vel"].noise = Unoise(
                n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE
            )
        if "projected_gravity" in group.terms:
            group.terms["projected_gravity"].noise = Unoise(
                n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE
            )
        group.terms["actuator_force"] = ObservationTermCfg(
            func=actuator_force,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))},
            noise=Unoise(n_min=-ACTUATOR_FORCE_NOISE, n_max=ACTUATOR_FORCE_NOISE),
        )

    # The critic alone knows how much of the roll is left.
    cfg.observations["critic"].terms["roll_progress"] = ObservationTermCfg(
        func=roll_mdp.observations.roll_progress_obs, params={"command_name": "roll"}
    )

    for group in cfg.observations.values():
        if group.history_length:
            raise ValueError("a group-level history_length would overwrite every term's")
        for name, term in group.terms.items():
            term.history_length = OBS_HISTORY if name in HISTORY_TERMS else 0

    # ── Action ────────────────────────────────────────────────────────────────
    action = cfg.actions["joint_pos"]
    action.actuator_names = list(GAIT_JOINTS)
    action.scale = {j: ACTION_SCALE for j in GAIT_JOINTS}

    # ── Command: the roll's bookkeeping ───────────────────────────────────────
    cfg.commands = {"roll": roll_mdp.commands.RollTrackerCfg(entity_name="robot")}

    # ── Rewards ───────────────────────────────────────────────────────────────
    stand = {
        "command_name": "roll",
        "turn_tolerance": STAND_TURN_TOLERANCE,
        "max_tilt": STAND_MAX_TILT,
        "min_feet": STAND_MIN_FEET,
        "max_ang_vel": STAND_MAX_ANG_VEL,
    }
    cfg.rewards = {
        "roll_progress": RewardTermCfg(
            func=roll_mdp.rewards.roll_progress, weight=PROGRESS_WEIGHT,
            params={"command_name": "roll", "max_rate": MAX_ROLL_RATE},
        ),
        "finish_upright": RewardTermCfg(
            func=roll_mdp.rewards.finish_upright, weight=FINISH_UPRIGHT_WEIGHT,
            params={"command_name": "roll", "tolerance": FINISH_TOLERANCE,
                    "sigma_sq": FINISH_SIGMA_SQ},
        ),
        "finish_home": RewardTermCfg(
            func=roll_mdp.rewards.finish_home, weight=FINISH_HOME_WEIGHT,
            params={"command_name": "roll", "tolerance": FINISH_TOLERANCE,
                    "sigma_sq": FINISH_SIGMA_SQ, "asset_cfg": gait_cfg},
        ),
        "roll_complete": RewardTermCfg(
            func=roll_mdp.rewards.roll_complete, weight=COMPLETE_BONUS / STEP_DT,
            params=stand,
        ),
        "body_impact": RewardTermCfg(
            func=roll_mdp.rewards.body_impact, weight=IMPACT_WEIGHT,
            params={"sensor_name": "body_ground_contact", "threshold": IMPACT_THRESHOLD},
        ),
        "heading_error": RewardTermCfg(
            func=roll_mdp.rewards.heading_error, weight=HEADING_WEIGHT
        ),
        "forward_drift": RewardTermCfg(
            func=roll_mdp.rewards.forward_drift, weight=FORWARD_DRIFT_WEIGHT,
            params={"scale": FORWARD_DRIFT_SCALE},
        ),
        "leg_collision": RewardTermCfg(
            func=leg_collision, weight=LEG_COLLISION_WEIGHT,
            params={"sensor_name": "self_collision", "force_threshold": 1.0},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=JOINT_LIMIT_WEIGHT),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=ACTION_RATE_WEIGHT),
    }

    cfg.curriculum = {}

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "rolled_and_standing": TerminationTermCfg(
            func=roll_mdp.terminations.standing_after_roll, params=stand,
        ),
    }

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=FOOT_GEOMS
    )
    cfg.events["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["push_robot"].params["velocity_range"].update(PUSH_VELOCITY_RANGE)
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events = {
        "reset_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_rolled": EventTermCfg(
            func=roll_mdp.events.reset_rolled,
            mode="reset",
            params={"rolled_fraction": ROLLED_FRACTION, "angle_range": ROLLED_ANGLE_RANGE,
                    "joint_noise": ROLLED_JOINT_NOISE},
        ),
        **cfg.events,
    }

    # ── Episode and viewer ────────────────────────────────────────────────────
    cfg.episode_length_s = EPISODE_S
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.0
    cfg.viewer.elevation = -15.0

    if play:
        for group in cfg.observations.values():
            if group is cfg.observations["actor"]:
                group.enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events["reset_rolled"].params["rolled_fraction"] = 0.0
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg


__all__ = ["env_cfg"]
