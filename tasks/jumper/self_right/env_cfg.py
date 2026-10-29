"""jumper.self_right -- the stock robot, lying at rest on its back, gets back onto its
feet, any way it likes, gently.

Built on jumper.roll (scene, sensors, observations, actions, the on-the-back spawn)
and nothing else; no demonstration. The horn study found that the stock body can
right itself from flat on its back (`studies/carapace/RESULTS.md`), and jumper.roll's
first policy already got up from its rolled spawns, so this task asks for it with no
guide.

## Episodes

Every episode, training and replay alike, starts on the robot's back: turned
`SPAWN_ANGLE_RANGE` about its long axis (flat on its back +-15 deg), legs at HOME
with `SPAWN_JOINT_NOISE`, set down on the floor (jumper.roll's `reset_rolled`). The
first `SETTLE_STEPS` control steps (0.5 s) it settles; nothing is paid or charged
in them, and how upright it is at their end is the baseline it must rise from.

## Objective

- **Rise** (`mdp/rewards.rise`): new ground in how upright the body is, so any route
  counts -- over either side, end over end, diagonally. 2 units from flat on its back
  to standing, `RISE_WEIGHT` per unit.
- **Standing** ends the episode: within 20 deg of level, four feet loaded, turning
  slower than 2 rad/s, having risen at least 1 unit itself.
- **Finishing bonus** `COMPLETE_BONUS`: half for standing however it went, half
  scaled by the worst of limb force, sensor force, shell impact, floor impact and air
  time. `finish_home` pays legs near HOME once upright.

## Gentle

| Kind | Bonus band (free - zero) | Per step, over |
|---|---|---|
| limbs (feet excepted) | 20 - 100 N | 20 N |
| display / dToF / camera | 2 - 20 N | 2 N |
| stock shell **impact** | 50 - 200 N | 50 N |
| floor impact, every part | 60 - 250 N | 60 N |
| longest time in the air | 0.06 - 0.20 s | per second airborne |

The shell is what it lies on, so its load is free and only its impact -- the sharpest
rise in force within a control step -- is charged. A landing costs the same whatever
part takes it, so hopping onto the feet gains nothing. The bands are
jumper.horn_roll's, which are judgement, not hardware limits.

## The claws meet the body

`common/body_collision.py`: the claws collide with the shell, base and sensors, which
the repository's masks let every leg pass through. The first policy trained without
it (Colab, 2500 iterations, 2026-10-29) put its left claw up to 1.6 cm into the base,
its right 0.7 cm, for about 0.1 s of every righting (checked on the replay's joint
angles). A claw resting on the body is free; leg against leg is charged pair by pair
(`body_collision.leg_collision_pairs`), as before.

## Joint torque from the floor

The first policy, replayed on CPU (2026-10-29; nominal and four randomised robots, 71
of 71 episodes stood up, claws colliding with the body): floor-contact torque |J^T f|
on its busiest joint 5.9-7.1 N*m median per robot, 6.6-8.4 worst, nearly always RF_J1
at about 0.08 s into the move, the right claw's palm and pad pushing off. KingKong's
reference-free jump: 4.8. So it is charged (`common/floor_load.py`, sampled at every
physics step): per step over `LOAD_BAND[0]`, and in the finishing bonus, whose gentle
half is gone at `LOAD_BAND[1]` -- jumper.tuck_roll's band.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from ..common import body_collision as bc
from ..common import floor_load as fl
from ..common.constants import FOOT_GEOMS, GAIT_JOINTS
from ..flourish_roll import env_cfg as flourish
from ..flourish_roll.mdp import rewards as fr
from ..roll import env_cfg as roll_env
from . import mdp as sr

STEP_DT = roll_env.STEP_DT
DECIMATION = 4

# ── Episodes ───────────────────────────────────────────────────────────────────
SPAWN_ANGLE_RANGE = (math.radians(165.0), math.radians(195.0))
SPAWN_JOINT_NOISE = 0.1
SETTLE_STEPS = 25
EPISODE_S = 5.0

# ── Objective ──────────────────────────────────────────────────────────────────
RISE_WEIGHT = 5.0
COMPLETE_BONUS = 20.0
FINISH_HOME_WEIGHT = 1.0
GENTLE_FLOOR = 0.5

# ── Gentle ─────────────────────────────────────────────────────────────────────
LIMB_BAND = (20.0, 100.0)
SENSOR_BAND = (2.0, 20.0)
SHELL_IMPACT_BAND = (50.0, 200.0)
GROUND_IMPACT_BAND = (60.0, 250.0)
FLIGHT_BAND = (0.06, 0.20)
LIMB_WEIGHT = -0.5
SENSOR_WEIGHT = flourish.SENSOR_WEIGHT
SHELL_IMPACT_WEIGHT = -0.3
GROUND_IMPACT_WEIGHT = -0.5
AIRBORNE_WEIGHT = -2.0
STOP_STRIKE_WEIGHT = flourish.STOP_STRIKE_WEIGHT
TORQUE_STRAIN_WEIGHT = flourish.TORQUE_STRAIN_WEIGHT
LEG_COLLISION_WEIGHT = -1.0
#: Floor-contact joint torque |J^T f|, N*m: free up to 4 (under KingKong's 4.8), the
#: gentle half of the bonus gone at 8 (the first policy's worst was 8.4).
LOAD_BAND = (4.0, 8.0)
LOAD_WEIGHT = -2.0
JOINT_LIMIT_WEIGHT = -1.0
ACTION_RATE_WEIGHT = -0.01

SHELL_PARTS = ("base_link_meshcol", "upper_shell_link_meshcol")


def _ground(name: str, pattern, exclude=(), history: int = 0):
    s = flourish._ground(name, pattern, exclude=exclude)
    s.history_length = history
    return s


def up_obs(env, command_name: str = "roll"):
    """How upright the body is, and how far it has risen, [N, 2] (critic only)."""
    import torch

    cmd = env.command_manager.get_term(command_name)
    return torch.stack([cmd.up_now(), cmd.best - cmd.spawn_up], dim=1)


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Self-righting from at rest on its back."""
    cfg = roll_env.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    gait_act = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))

    # ── The claws meet the body ───────────────────────────────────────────────
    robot = cfg.scene.entities["robot"]
    robot.collisions = tuple(robot.collisions) + bc.arm_body_collision()
    pairs = bc.leg_pair_sensors(DECIMATION)

    # ── Sensors ───────────────────────────────────────────────────────────────
    cfg.scene.sensors = tuple(cfg.scene.sensors) + pairs + (
        fl.leg_floor_sensor(DECIMATION),
        _ground("limb_ground_contact", flourish._LEG, exclude=tuple(FOOT_GEOMS)),
        _ground("sensor_ground_contact", flourish.SENSOR_PARTS),
        _ground("shell_ground_contact", SHELL_PARTS, history=DECIMATION),
        _ground("all_ground_contact", r".*_meshcol", history=DECIMATION),
    )

    # ── Command: how upright, and the hardest hits ────────────────────────────
    cfg.commands = {"roll": sr.tracker.SelfRightTrackerCfg(settle_steps=SETTLE_STEPS,
                                                           load_joints=GAIT_JOINTS)}
    critic = cfg.observations["critic"].terms
    critic.pop("roll_progress", None)
    critic["self_right"] = ObservationTermCfg(func=up_obs, params={"command_name": "roll"})

    # ── Every episode starts on its back, replay included ─────────────────────
    ev = cfg.events["reset_rolled"].params
    ev["rolled_fraction"] = 1.0
    ev["angle_range"] = SPAWN_ANGLE_RANGE
    ev["joint_noise"] = SPAWN_JOINT_NOISE
    cfg.episode_length_s = EPISODE_S

    # ── Rewards ───────────────────────────────────────────────────────────────
    stand = {"command_name": "roll"}
    cfg.rewards = {
        "rise": RewardTermCfg(func=sr.rewards.rise, weight=RISE_WEIGHT),
        "complete": RewardTermCfg(
            func=sr.rewards.gentle_complete, weight=COMPLETE_BONUS / STEP_DT,
            params={**stand, "gentle_floor": GENTLE_FLOOR, "limb": LIMB_BAND,
                    "sensor": SENSOR_BAND, "shell_impact": SHELL_IMPACT_BAND,
                    "ground_impact": GROUND_IMPACT_BAND, "flight": FLIGHT_BAND,
                    "load": LOAD_BAND},
        ),
        "joint_load": RewardTermCfg(func=fl.joint_load, weight=LOAD_WEIGHT,
                                    params={"threshold": LOAD_BAND[0]}),
        "finish_home": RewardTermCfg(func=sr.rewards.finish_home, weight=FINISH_HOME_WEIGHT,
                                     params={"asset_cfg": gait_cfg}),
        "limb_contact": RewardTermCfg(
            func=fr.contact_overload, weight=LIMB_WEIGHT,
            params={"sensor_name": "limb_ground_contact", "threshold": LIMB_BAND[0]},
        ),
        "sensor_contact": RewardTermCfg(
            func=fr.contact_overload, weight=SENSOR_WEIGHT,
            params={"sensor_name": "sensor_ground_contact", "threshold": SENSOR_BAND[0]},
        ),
        "shell_impact": RewardTermCfg(func=sr.rewards.shell_impact, weight=SHELL_IMPACT_WEIGHT,
                                      params={"threshold": SHELL_IMPACT_BAND[0]}),
        "ground_impact": RewardTermCfg(func=sr.rewards.ground_impact,
                                       weight=GROUND_IMPACT_WEIGHT,
                                       params={"threshold": GROUND_IMPACT_BAND[0]}),
        "airborne": RewardTermCfg(func=sr.rewards.airborne, weight=AIRBORNE_WEIGHT),
        "stop_strike": RewardTermCfg(func=fr.stop_strike, weight=STOP_STRIKE_WEIGHT,
                                     params={"asset_cfg": gait_cfg}),
        "torque_strain": RewardTermCfg(func=fr.torque_strain, weight=TORQUE_STRAIN_WEIGHT,
                                       params={"asset_cfg": gait_act}),
        "leg_collision": RewardTermCfg(
            func=bc.leg_collision_pairs, weight=LEG_COLLISION_WEIGHT,
            params={"sensor_names": tuple(s.name for s in pairs), "force_threshold": 1.0},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=JOINT_LIMIT_WEIGHT),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=ACTION_RATE_WEIGHT),
    }

    # ── The joint-load estimate, sampled at every physics step ────────────────
    # Its own logged value is a per-step mean; read Metrics/roll/peak_joint_load_Nm.
    cfg.metrics = {**(cfg.metrics or {}),
                   "floor_load_sampler": MetricsTermCfg(func=fl.floor_load_sampler,
                                                        per_substep=True, reduce="max")}

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "standing": TerminationTermCfg(func=sr.rewards.standing, params=stand),
    }
    return cfg


__all__ = ["env_cfg"]
