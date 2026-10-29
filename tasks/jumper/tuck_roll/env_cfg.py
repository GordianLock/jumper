"""jumper.tuck_roll -- jumper.horn_roll's forward roll, carried over the nose on the
arms: claws tucked side by side under the face rather than splayed, so the front of the
body stays off the floor.

Built on jumper.horn_roll (scene, carapace collision and regions, tracker, gentle
terms). What changes is below.

## Why

jumper.horn_roll's first policy (Colab, 2500 iterations, `model_2499`) rolls nose
first in 0.84 s. Replayed on the nominal prototype 3 (2026-10-29, CPU), the floor
pushed on the splayed claws (palms 22 cm either side) from 0.12 to 0.32 s and then on
the front of the carapace, up to 60 N, from 0.26 to 0.40 s, as the body went over its
nose. The display, ToF sensor and camera never touched. Asked for: the claws side by
side under the face, so the forearms carry it over instead.

## The demonstration

`ref/tuck_roll.npz`: the first run's own roll (see below), recorded on the nominal
robot in this environment by `studies/proto3/record_policy_demo.py`
(`plans/tuck_roll_run1_model_2499.npz`): over the nose on its claws, the face off the
floor, nothing inside anything -- a roll that completes and fits together, arms
included. So the joint prior covers all 20 driven joints again.

**Mid-demonstration spawns stop at `MAX_SPAWN_TURN` (270 deg).** At 274-305 deg
(frames 30-33) the claws press into each other up to 1.8 cm (measured on the recorded
states); a spawn there would start with them overlapping.

The first run trained on a hybrid (`studies/proto3/tuck_hybrid.py`): jumper.horn_roll's
policy's roll with only the fed-forward arm commands changed to the tuck, spawns
stopped at 210 deg because that policy's right arm swings through the body.

## The first run, and what changed after it

Colab, 2500 iterations, revision 5c0145d (2026-10-29). Replayed on CPU (nominal and
four randomised robots, 55 of 55 rolls completed, 0.86-0.90 s): over the nose the
floor carried **only the claws** (0.12-0.34 s, 20-30 N each) -- the face no longer
touched. The left palm came to 8-10 cm off the centreline, the right stayed 12-16 cm
out. The front horns took up to 77 N at 0.36-0.42 s, the robot already belly-up.
Floor-contact joint torque rose to 6.4-7.6 N*m on the left claw (horn_roll's policy:
5.5-5.9; KingKong's reference-free jump: 4.8). Decided with the user:

- **The face, not the horns.** `front_force` now covers the stock shell, base and
  sensors only; the carapace, horns included, is as jumper.horn_roll has it.
- **One claw is enough.** `tuck` pays the better-placed claw (`either`).
- **Joint torque is charged** (`tracker.floor_joint_load`, `rewards.joint_load`): per
  step over `LOAD_BAND[0]`, and in the finishing bonus to nothing at `LOAD_BAND[1]`.

## Arms

- **`tuck`**: while the nose points down (`rewards.nose_down`), a palm at least 12 cm
  ahead of the base origin and within 8 cm of the centreline -- the better of the two.
- **The arms collide with the body** (`common/body_collision.py`): the repository leaves
  every leg passing through the shell, base and carapace, and a tuck learnt that way
  could put a forearm where the face is. Arm against body is free -- bracing against
  the front carapace is expected -- and so is arm against arm. Every other pair of
  legs is still charged (`body_collision.leg_collision_pairs`).
- **Their load is the point**: the arms get their own band (60 - 150 N), wider than the
  walking legs' (20 - 100 N), which jumper.horn_roll charged them under.

## The face

Floor force at contacts more than 4 cm ahead of the base origin on the stock shell,
base or sensors (`tracker.front_force`) is charged per step over 5 N and scales the
finishing bonus to nothing at 40 N. The carapace is not part of it: its horns are free
to pivot on.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..common import body_collision as bc
from ..common.constants import FOOT_GEOMS, GAIT_JOINTS
from ..horn_roll import env_cfg as horn
from ..momentum_roll import env_cfg as momentum
from . import mdp as tr

REF_NPZ = Path(__file__).resolve().parent / "ref" / "tuck_roll.npz"
STEP_DT = horn.STEP_DT
DECIMATION = horn.DECIMATION

ARMS = bc.ARMS
ARM_JOINTS = tuple(j for j in GAIT_JOINTS if j.startswith(ARMS))
PALM_BODIES = ("LF_palm_link", "RF_palm_link")

#: Last turn a mid-demonstration spawn may start at (module docstring).
MAX_SPAWN_TURN = math.radians(270.0)

# ── Tuck ───────────────────────────────────────────────────────────────────────
TUCK_WEIGHT = 3.0
TUCK_MIN_X = 0.12
TUCK_MAX_Y = 0.08
TUCK_SIGMA_SQ = 0.01
NOSE_DOWN = 0.3

# ── Contact: bonus bands (free, zero) and per-step charges ────────────────────
FRONT_X = 0.04
FRONT_BAND = (5.0, 40.0)
FRONT_WEIGHT = -0.5
ARM_BAND = (60.0, 150.0)
ARM_WEIGHT = -0.5
LEG_COLLISION_WEIGHT = -1.0
#: Floor-contact joint torque |J^T f|, N*m. Free up to 4 (under KingKong's 4.8),
#: nothing of the bonus's gentle half left at 8 (the first run peaked at 7.6).
LOAD_BAND = (4.0, 8.0)
LOAD_WEIGHT = -2.0

#: Leg pairs free to touch: the claws, tucked side by side.
FREE_PAIRS = (frozenset(bc.ARMS),)


def env_cfg(asset: Path | None = None, play: bool = False):
    """The forward roll over tucked arms."""
    cfg = horn.env_cfg(asset=asset, play=play)

    # ── The arms meet the body ─────────────────────────────────────────────────
    robot = cfg.scene.entities["robot"]
    robot.collisions = tuple(robot.collisions) + bc.arm_body_collision(
        horn.CARAPACE_GEOMS, horn.CARAPACE_FRICTION)

    # ── Sensors: arms apart from the walking legs, the front of the body, leg pairs
    walking = r"(LM|RM|LR|RR)_.*_meshcol.*"
    arms = r"(LF|RF)_.*_meshcol.*"
    sensors = [s for s in cfg.scene.sensors if s.name != "limb_ground_contact"]
    front_parts = tuple(p.strip("^$") for p in bc.BODY_GEOMS)
    cfg.scene.sensors = tuple(sensors) + (
        horn._ground("limb_ground_contact", walking, exclude=tuple(FOOT_GEOMS)),
        horn._ground("arm_ground_contact", arms, exclude=tuple(FOOT_GEOMS)),
        ContactSensorCfg(
            name="front_ground_contact",
            primary=ContactMatch(mode="geom", pattern=front_parts, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force", "pos"), reduce="maxforce", num_slots=2,
        ),
        ContactSensorCfg(
            name="leg_floor_contact",
            primary=ContactMatch(mode="geom", pattern=r"(LF|RF|LM|RM|LR|RR)_.*_meshcol",
                                 entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal", "tangent"), global_frame=True,
            reduce="maxforce", num_slots=2, history_length=DECIMATION,
        ),
    ) + bc.leg_pair_sensors(DECIMATION, FREE_PAIRS)

    # ── Command: jumper.horn_roll's tracker, this demonstration, two more worsts ─
    old = cfg.commands["roll"]
    cfg.commands = {
        "roll": tr.tracker.TuckRollTrackerCfg(
            entity_name="robot", reference=str(REF_NPZ),
            region_sensors=old.region_sensors, front_x=FRONT_X,
            load_sensor="leg_floor_contact", load_joints=GAIT_JOINTS,
        )
    }

    # ── Rewards ───────────────────────────────────────────────────────────────
    r = dict(cfg.rewards)
    stand = {"command_name": "roll", "min_own_turn": momentum.MIN_OWN_TURN}
    r["complete"] = RewardTermCfg(
        func=tr.rewards.gentle_complete, weight=momentum.COMPLETE_BONUS / STEP_DT,
        params={**stand, "gentle_floor": horn.GENTLE_FLOOR, "limb": horn.LIMB_BAND,
                "arm": ARM_BAND, "front": FRONT_BAND, "sensor": horn.SENSOR_BAND,
                "shell": horn.SHELL_BAND, "carapace_impact": horn.CARAPACE_IMPACT_BAND,
                "ground_impact": horn.GROUND_IMPACT_BAND, "flight": horn.FLIGHT_BAND,
                "load": LOAD_BAND},
    )
    r["joint_load"] = RewardTermCfg(func=tr.rewards.joint_load, weight=LOAD_WEIGHT,
                                    params={"threshold": LOAD_BAND[0]})
    r["arm_contact"] = RewardTermCfg(func=tr.rewards.arm_contact, weight=ARM_WEIGHT,
                                     params={"threshold": ARM_BAND[0]})
    r["front_contact"] = RewardTermCfg(func=tr.rewards.front_contact, weight=FRONT_WEIGHT,
                                       params={"threshold": FRONT_BAND[0]})
    r["tuck"] = RewardTermCfg(
        func=tr.rewards.tuck, weight=TUCK_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=list(PALM_BODIES),
                                            preserve_order=True),
                "min_x": TUCK_MIN_X, "max_y": TUCK_MAX_Y, "sigma_sq": TUCK_SIGMA_SQ,
                "min_down": NOSE_DOWN, "either": True},
    )
    r["leg_collision"] = RewardTermCfg(
        func=bc.leg_collision_pairs, weight=LEG_COLLISION_WEIGHT,
        params={"sensor_names": tuple(s.name for s in bc.leg_pair_sensors(DECIMATION, FREE_PAIRS)),
                "force_threshold": 1.0},
    )
    cfg.rewards = r

    # ── Spawns: none from the recorded states with the claws overlapping ──────
    cfg.events["reset_momentum"].params["max_spawn_turn"] = MAX_SPAWN_TURN

    # The prior's fading curriculum names the term; it is still "prior_joint_pos".
    return cfg


__all__ = ["env_cfg"]
