"""jumper.pick_place -- pick up a piece of debris and put it in a receptacle or
on a pile.

The robot stands on five legs -- the five-foot task's stance, `RF LM RM LR RR`
-- and works with its left front arm and claw. Each episode a block of debris
lies on the floor in front of it and a tray stands to its left, sunk into the
floor by a random amount so its rim is anywhere from 54 mm to flush (a pile
spot). The policy has to reach down, grip the debris, lift it clear of the rim,
carry it over the tray, put it down inside, and let go. The episode ends there.

**A standing skill, on purpose.** Walking up to debris is what `jumper.five_foot`
already does -- with this claw stowed and a load in it. This task is the part
that skill lacks, the arm, so the two can be chained by the controller: walk
there on five feet, switch to this mode, pick and place, switch back.

## Why the layout is where it is

Measured on `assets/jumper/jumper.xml` (V1.6.1), the arm swept over its whole
range with the legs at HOME, poses where it hits the robot thrown out:

- the claw's mouth reaches **the floor** (15-45 mm up, no part of the claw
  below it) in a band in front of the trunk, x 0.20-0.29 m and y -0.05 to
  +0.15 m, and in an arc to the left; `DEBRIS_ZONE` is inside the band;
- it reaches **70 mm and more up**, the whole claw 30 mm clear of the floor,
  over most of the front-left quarter, x 0.0-0.25 m and y up to 0.30 m;
  `BIN_ZONE` puts the tray's centre where that holds, its walls clear of the
  left-middle foot (at x -0.007, y 0.197) and of the debris zone.

Leaning on the stance legs widens both; nothing here relies on it.

## The grip is friction, so the world runs the five-foot task's prop physics

The claw holds what it squeezes by friction alone, and that only works under the
elliptic friction cone at impratio 50 (`five_foot/objects.py::PROP_CONE`) -- under
the walking tasks' settings a squeezed object creeps out of the jaws. That setting
is for every contact, the feet included, and it costs simulation speed. The jaws
are convex-decomposed (`five_foot/jaws.py`) so the mouth is open to the solver.

## What is not modelled yet

- **Perception.** The policy is told where the debris and the tray are
  (`mdp/observations.py`), as a camera and an object detector would tell it on
  the robot. The deployed controller has no such observation, so this trains and
  replays in simulation and cannot be exported until one is added.
- **The shell.** As for every task: none on the model.

## On the numbers below

The physics and the hardware facts are measured or taken from the five-foot
task. **The reward weights, the zones' edges and the spawn shares are first
guesses**, set by reasoning about scale, not against a trained policy.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
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
    HOME,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    get_jumper_robot_cfg,
)
from ..common.mdp.observations import actuator_force
from ..common.tof import tof_sensor
from ..five_foot.claw import (
    ARM_JOINTS,
    FINGER_JOINT,
    FIVE_FOOT_LEGS,
    LF_GRASP,
    OBSERVED_JOINTS,
    foot_geoms_for,
)
from ..five_foot.jaws import GRIP_FRICTION, JAW_COLLISION, decompose_jaws
from ..five_foot.objects import PROP_CONE, PROP_IMPRATIO
from ..ref_free_jump.mdp.rewards import leg_collision
from . import mdp as pp
from .props import DEBRIS_MASS_ALPHA, SIZE_SCALE, bin_spec, debris_spec

# ── Physics ────────────────────────────────────────────────────────────────────
#: The family's 50 Hz control at a 5 ms physics step: what the five-foot task
#: grips and carries its props under.
_SIM_TIMESTEP = 0.005
_DECIMATION = 4
STEP_DT = _SIM_TIMESTEP * _DECIMATION

#: Contact and constraint capacity. The five-foot task measured 404 contacts with
#: seven props on the GPU backend; two props, and margin.
NCONMAX = 512
NJMAX = 2048

# ── The action: all 21 joints the robot stands and works with ──────────────────
#: The stance legs hold the body and may lean it; the arm travels far from its
#: stow to the floor and the tray; the finger spans its whole travel.
LEG_ACTION_SCALE = 0.25
ARM_ACTION_SCALE = 0.75
FINGER_ACTION_SCALE = 0.5

OBS_HISTORY = 5
HISTORY_TERMS = frozenset({"joint_pos", "joint_vel", "actions", "actuator_force"})

# ── The layout, in the robot's frame at spawn (m) ──────────────────────────────
DEBRIS_ZONE = {"x": (0.21, 0.27), "y": (-0.03, 0.13)}
BIN_ZONE = {"x": (0.09, 0.14), "y": (0.26, 0.29)}
#: Share of episodes whose receptacle is a pile spot, flush with the floor.
PILE_FRACTION = 0.25
#: Share of episodes that start with the debris already in the claw.
HELD_FRACTION = 0.3
HELD_PLACE_STEPS = 10
HELD_STEPS = 30

# ── The ladder ─────────────────────────────────────────────────────────────────
#: Two kernels on the mouth-to-debris distance. The fine one alone gave no
#: gradient from the stow, 0.08-0.15 m away (tanh saturated; a 300-iteration CPU
#: check paid it 0.02 per episode), so a coarse one pulls from there.
REACH_WEIGHT = 1.0
REACH_STD = 0.04
REACH_COARSE_WEIGHT = 1.0
REACH_COARSE_STD = 0.20
GRIP_WEIGHT = 2.0
LIFT_WEIGHT = 4.0
#: The lift is paid up to this above the rim.
LIFT_CLEARANCE = 0.03
CARRY_WEIGHT = 4.0
CARRY_STD = 0.08
IN_BIN_WEIGHT = 6.0
#: Paid once, as the job completes; rewards are scaled by the step, so the
#: weight is the bonus over `STEP_DT`.
PLACED_BONUS = 20.0

# ── The costs ──────────────────────────────────────────────────────────────────
#: Trunk, stance legs' upper links: anything but a foot or the working arm on
#: the floor.
BODY_CONTACT_WEIGHT = -2.0
#: The arm and claw pressing on the floor harder than this, N. It has to touch
#: the floor to pick something off it; it should not lean on it.
ARM_GROUND_THRESHOLD = 15.0
ARM_GROUND_WEIGHT = -0.5
STANCE_FEET_OFF_WEIGHT = -0.5
BASE_DRIFT_WEIGHT = -1.0
TILT_WEIGHT = -2.0
LEG_COLLISION_WEIGHT = -1.0
JOINT_LIMIT_WEIGHT = -1.0
ACTION_RATE_WEIGHT = -0.01
FINGER_EFFORT_WEIGHT = -0.05

#: Paid once, on the step the debris is lost or the robot falls. Without it the
#: early-ending terminations were an escape from the per-step costs: in a
#: 300-iteration CPU check knocking the debris away ended 24% of episodes at
#: first and 41% by the end.
FAILURE_PENALTY = -10.0

EPISODE_S = 10.0

# ── Contact sets ───────────────────────────────────────────────────────────────
GROUND_FOOT_GEOMS: tuple[str, ...] = foot_geoms_for(FIVE_FOOT_LEGS)
_ARM_GEOMS = r"LF_.*_meshcol.*"
_FIXED_JAW = (r"LF_palm_link_meshcol.*", r"LF_palm_pad_f_link_meshcol",
              r"LF_palm_grip_insert_link_meshcol")
_MOVING_JAW = (r"LF_finger_link_meshcol.*", r"LF_finger_tip_link_meshcol",
               r"LF_finger_grip_insert_link_meshcol")

#: The rubber grip inserts on both jaws -- the faces that actually meet a held
#: object (measured: every held debris touched both). `five_foot/jaws.py`'s grip
#: set predates them, so under `JAW_COLLISION` they are condim 1 at priority 0 and a
#: prop meeting one gets the prop's own floor friction, 0.35, rather than the
#: jaws' 1.0. Here they join the grip set.
GRIP_INSERTS = ("LF_palm_grip_insert_link_meshcol", "LF_finger_grip_insert_link_meshcol")


def _with_grip_inserts(base):
    """`JAW_COLLISION` with the grip inserts gripping like the jaw faces."""
    import dataclasses

    grip = "|".join(f"^{n}$" for n in GRIP_INSERTS)

    def first(table, value):
        return {grip: value} | dict(table)

    return dataclasses.replace(
        base,
        condim=first(base.condim, 3),
        priority=first(base.priority, 1),
        friction=first(base.friction, GRIP_FRICTION),
    )


def _scale(j: str) -> float:
    if j == FINGER_JOINT:
        return FINGER_ACTION_SCALE
    if j in ARM_JOINTS:
        return ARM_ACTION_SCALE
    return LEG_ACTION_SCALE


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Pick up the debris and put it in the receptacle or on the pile.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- noise off, every episode from an empty claw.
    """
    cfg = make_velocity_env_cfg()

    # ── Sim ───────────────────────────────────────────────────────────────────
    cfg.sim.mujoco.timestep = _SIM_TIMESTEP
    cfg.decimation = _DECIMATION
    cfg.sim.nconmax = NCONMAX
    cfg.sim.njmax = NJMAX
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.mujoco.cone = PROP_CONE
    cfg.sim.mujoco.impratio = PROP_IMPRATIO

    # ── Scene: the robot with its working claw, the debris, the tray ──────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    robot = get_jumper_robot_cfg(effort_limit=EFFORT_LIMIT, asset=asset)
    robot.init_state.joint_pos = dict(HOME) | dict(LF_GRASP)
    base_spec_fn = robot.spec_fn
    robot.spec_fn = lambda: decompose_jaws(base_spec_fn())
    robot.collisions = (_with_grip_inserts(JAW_COLLISION),)
    cfg.scene.entities = {
        "robot": robot,
        "debris": EntityCfg(spec_fn=debris_spec),
        "bin": EntityCfg(spec_fn=bin_spec),
    }

    debris_body = ContactMatch(mode="body", pattern="debris", entity="debris")
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan", "feet_ground_contact")
    ) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=GROUND_FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="geom", pattern=r".*_meshcol.*", entity="robot",
                                 exclude=GROUND_FOOT_GEOMS + (_ARM_GEOMS,)),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",), reduce="netforce", num_slots=1,
        ),
        ContactSensorCfg(
            name="arm_ground_contact",
            primary=ContactMatch(mode="geom", pattern=_ARM_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",), reduce="netforce", num_slots=1,
        ),
        ContactSensorCfg(
            name="palm_debris",
            primary=ContactMatch(mode="geom", pattern=_FIXED_JAW, entity="robot"),
            secondary=debris_body,
            fields=("force",), reduce="netforce", num_slots=1,
        ),
        ContactSensorCfg(
            name="finger_debris",
            primary=ContactMatch(mode="geom", pattern=_MOVING_JAW, entity="robot"),
            secondary=debris_body,
            fields=("force",), reduce="netforce", num_slots=1,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
            history_length=_DECIMATION,
        ),
    )

    # ── Observations ──────────────────────────────────────────────────────────
    for group in cfg.observations.values():
        for name in ("command", "height_scan", "foot_height", "foot_air_time",
                     "foot_contact", "foot_contact_forces"):
            group.terms.pop(name, None)
    del cfg.observations["actor"].terms["base_lin_vel"]

    observed = SceneEntityCfg("robot", joint_names=list(OBSERVED_JOINTS))
    observed_act = SceneEntityCfg("robot", actuator_names=list(OBSERVED_JOINTS))
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            if name in group.terms:
                group.terms[name].params = dict(group.terms[name].params or {}) | {
                    "asset_cfg": observed
                }
        if "base_ang_vel" in group.terms:
            group.terms["base_ang_vel"].noise = Unoise(
                n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE
            )
        if "projected_gravity" in group.terms:
            group.terms["projected_gravity"].noise = Unoise(
                n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE
            )
        group.terms["actuator_force"] = ObservationTermCfg(
            func=actuator_force, params={"asset_cfg": observed_act},
            noise=Unoise(n_min=-ACTUATOR_FORCE_NOISE, n_max=ACTUATOR_FORCE_NOISE),
        )
        # What a camera would report: a few millimetres of error.
        group.terms["debris"] = ObservationTermCfg(
            func=pp.observations.debris_obs, noise=Unoise(n_min=-0.005, n_max=0.005)
        )
        group.terms["receptacle"] = ObservationTermCfg(
            func=pp.observations.receptacle_obs, noise=Unoise(n_min=-0.005, n_max=0.005)
        )
    cfg.observations["critic"].terms["task_state"] = ObservationTermCfg(
        func=pp.observations.task_state_obs
    )
    for group in cfg.observations.values():
        if group.history_length:
            raise ValueError("a group-level history_length would overwrite every term's")
        for name, term in group.terms.items():
            term.history_length = OBS_HISTORY if name in HISTORY_TERMS else 0

    # ── Action ────────────────────────────────────────────────────────────────
    cfg.actions["joint_pos"] = pp.actions.HeldSpawnJointPositionActionCfg(
        entity_name="robot",
        actuator_names=list(OBSERVED_JOINTS),
        scale={j: _scale(j) for j in OBSERVED_JOINTS},
        use_default_offset=True,
        place_steps=HELD_PLACE_STEPS,
    )

    # ── The task's bookkeeping ────────────────────────────────────────────────
    cfg.commands = {"task": pp.state.PickPlaceStateCfg(held_steps=HELD_STEPS)}

    # ── Rewards ───────────────────────────────────────────────────────────────
    finger = SceneEntityCfg("robot", actuator_names=[FINGER_JOINT])
    cfg.rewards = {
        "reach": RewardTermCfg(func=pp.rewards.reach, weight=REACH_WEIGHT,
                               params={"std": REACH_STD}),
        "reach_coarse": RewardTermCfg(func=pp.rewards.reach, weight=REACH_COARSE_WEIGHT,
                                      params={"std": REACH_COARSE_STD}),
        "grip": RewardTermCfg(func=pp.rewards.grip, weight=GRIP_WEIGHT),
        "lift": RewardTermCfg(func=pp.rewards.lift, weight=LIFT_WEIGHT,
                              params={"clearance": LIFT_CLEARANCE}),
        "carry": RewardTermCfg(func=pp.rewards.carry, weight=CARRY_WEIGHT,
                               params={"std": CARRY_STD}),
        "in_bin": RewardTermCfg(func=pp.rewards.in_bin, weight=IN_BIN_WEIGHT),
        "placed": RewardTermCfg(func=pp.rewards.placed, weight=PLACED_BONUS / STEP_DT),
        "body_contact": RewardTermCfg(
            func=pp.rewards.any_contact, weight=BODY_CONTACT_WEIGHT,
            params={"sensor_name": "body_ground_contact"},
        ),
        "arm_ground_force": RewardTermCfg(
            func=pp.rewards.ground_force, weight=ARM_GROUND_WEIGHT,
            params={"sensor_name": "arm_ground_contact", "threshold": ARM_GROUND_THRESHOLD},
        ),
        "stance_feet_off": RewardTermCfg(
            func=pp.rewards.stance_feet_off, weight=STANCE_FEET_OFF_WEIGHT
        ),
        "base_drift": RewardTermCfg(func=pp.rewards.base_drift, weight=BASE_DRIFT_WEIGHT),
        "tilt": RewardTermCfg(func=pp.rewards.tilt, weight=TILT_WEIGHT),
        "leg_collision": RewardTermCfg(
            func=leg_collision, weight=LEG_COLLISION_WEIGHT,
            params={"sensor_name": "self_collision", "force_threshold": 1.0},
        ),
        "finger_effort": RewardTermCfg(
            func=pp.rewards.finger_effort, weight=FINGER_EFFORT_WEIGHT,
            params={"asset_cfg": finger},
        ),
        "failure": RewardTermCfg(
            func=pp.rewards.failed, weight=FAILURE_PENALTY / STEP_DT,
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=JOINT_LIMIT_WEIGHT),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=ACTION_RATE_WEIGHT),
    }
    cfg.curriculum = {}

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "placed": TerminationTermCfg(func=pp.terminations.placed),
        "debris_lost": TerminationTermCfg(func=pp.terminations.debris_lost),
        "fell": TerminationTermCfg(func=pp.terminations.fell),
    }

    # ── Events ────────────────────────────────────────────────────────────────
    keep = {k: cfg.events[k] for k in ("foot_friction", "base_com", "encoder_bias")
            if k in cfg.events}
    keep["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=GROUND_FOOT_GEOMS
    )
    keep["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    keep["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events = {
        # Per environment, once: how big and how heavy its debris is.
        "debris_size": EventTermCfg(
            func=pp.events.debris_size, mode="startup", params={"scale": SIZE_SCALE},
        ),
        "debris_mass": EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={"asset_cfg": SceneEntityCfg("debris", body_names=("debris",)),
                    "alpha_range": DEBRIS_MASS_ALPHA},
        ),
        "reset_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_layout": EventTermCfg(
            func=pp.events.reset_layout, mode="reset",
            params={"debris_zone": DEBRIS_ZONE, "bin_zone": BIN_ZONE,
                    "pile_fraction": PILE_FRACTION, "held_fraction": HELD_FRACTION},
        ),
        "held_spawn": EventTermCfg(
            func=pp.events.HeldSpawn, mode="step",
            params={"place_steps": HELD_PLACE_STEPS},
        ),
        **keep,
    }

    # ── Episode and viewer ────────────────────────────────────────────────────
    cfg.episode_length_s = EPISODE_S
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 0.9
    cfg.viewer.elevation = -30.0

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events["reset_layout"].params["held_fraction"] = 0.0
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg


__all__ = ["env_cfg"]
