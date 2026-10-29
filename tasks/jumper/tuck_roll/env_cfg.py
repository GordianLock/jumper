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

`ref/tuck_roll.npz`: that policy's own roll, recorded on the nominal robot by
`studies/proto3/record_policy_demo.py` -- smoother and far gentler than
jumper.horn_roll's keyframe demonstration. It splays the arms, so the joint prior
covers the 12 walking-leg joints only (`prior_joint_pos_legs`); the arms are fed the
demonstration forward but free to leave it, and `tuck` pays them to.

## Arms

- **`tuck`**: while the nose points down (`rewards.nose_down`), both palms at least
  12 cm ahead of the base origin and within 8 cm of the centreline. Side by side, not
  crossed.
- **The arms collide with the body** (`_arm_body_collision`): the repository leaves
  every leg passing through the shell, base and carapace, and a tuck learnt that way
  could put a forearm where the face is. Arm against body is free -- bracing against
  the front carapace is expected -- and so is arm against arm. Every other pair of
  legs is still charged (`leg_collision_pairs`).
- **Their load is the point**: the arms get their own band (60 - 150 N), wider than the
  walking legs' (20 - 100 N), which jumper.horn_roll charged them under.

## The front of the body

Floor force at contacts more than 4 cm ahead of the base origin, on the shell, base,
sensors or carapace (`tracker.front_force`), is charged per step over 5 N and scales
the finishing bonus to nothing at 40 N. This is **load**, where jumper.horn_roll
charges only carapace impact: the front corner horns lose the pivot they were free to
be there. Everywhere else the carapace is as jumper.horn_roll has it.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from ..common import constants as K
from ..common.constants import FOOT_GEOMS, GAIT_JOINTS, LEGS
from ..horn_roll import env_cfg as horn
from ..momentum_roll import env_cfg as momentum
from . import mdp as tr

REF_NPZ = Path(__file__).resolve().parent / "ref" / "tuck_roll.npz"
STEP_DT = horn.STEP_DT
DECIMATION = horn.DECIMATION

ARMS = ("LF", "RF")
ARM_JOINTS = tuple(j for j in GAIT_JOINTS if j.startswith(ARMS))
LEG_JOINTS = tuple(j for j in GAIT_JOINTS if not j.startswith(ARMS))
PALM_BODIES = ("LF_palm_link", "RF_palm_link")

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

# ── Collision: the arms against the body ──────────────────────────────────────
#: A bit no other geom uses: legs hold 2-64, the floor 1.
_BODY_BIT = 1 << 7
BODY_GEOMS = (r"^base_link_meshcol$", r"^upper_shell_link_meshcol$",
              r"^display_module_link_meshcol$", r"^tof_sensor_link_meshcol$",
              r"^camera_link_meshcol$")
_LEG_ROOT = {"LF": "LF_shoulder_link", "RF": "RF_shoulder_link", "LM": "LM_hip_link",
             "RM": "RM_hip_link", "LR": "LR_hip_link", "RR": "RR_hip_link"}


def _arm_body_collision() -> tuple[CollisionCfg, ...]:
    """The arms against the stock body and the carapace; everything else as before.

    The body geoms gain `_BODY_BIT` in contype and the arms gain it in conaffinity, so
    only that pair is new: the walking legs' conaffinity lacks the bit, the body's
    conaffinity stays 0. The floor still meets everything through bit 1.
    """
    feet = "|".join(f"^{n}$" for n in FOOT_GEOMS)
    arms = CollisionCfg(
        geom_names_expr=tuple(f"^{a}_.*_meshcol$" for a in ARMS),
        contype={f"^{a}_": K._GROUND_BIT | K._LEG_BIT[a] for a in ARMS},
        conaffinity={f"^{a}_": (K._ALL_LEG_BITS & ~K._LEG_BIT[a]) | _BODY_BIT for a in ARMS},
        condim={feet: 3, ".*": 1},
        priority={feet: 1, ".*": 0},
        disable_other_geoms=False,
    )
    body = CollisionCfg(
        geom_names_expr=BODY_GEOMS,
        contype=K._GROUND_BIT | _BODY_BIT, conaffinity=0, condim=1, priority=0,
        disable_other_geoms=False,
    )
    carapace = CollisionCfg(
        geom_names_expr=(horn.CARAPACE_GEOMS,),
        contype=K._GROUND_BIT | _BODY_BIT, conaffinity=0, condim=3, priority=0,
        friction=horn.CARAPACE_FRICTION, solref=K._CONTACT_SOLREF, solimp=K._CONTACT_SOLIMP,
        disable_other_geoms=False,
    )
    return arms, body, carapace


def _pair_sensors() -> tuple[ContactSensorCfg, ...]:
    """One sensor per leg `j`, against every leg before it in `LEGS`, the arm pair left
    out: 14 of the 15 pairs."""
    out = []
    for j, leg in enumerate(LEGS):
        others = [o for o in LEGS[:j] if {o, leg} != set(ARMS)]
        if not others:
            continue
        out.append(ContactSensorCfg(
            name=f"leg_pair_{leg}",
            primary=ContactMatch(mode="subtree",
                                 pattern=tuple(f"^{_LEG_ROOT[o]}$" for o in others),
                                 entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern=_LEG_ROOT[leg], entity="robot"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            history_length=DECIMATION,
        ))
    return tuple(out)


def env_cfg(asset: Path | None = None, play: bool = False):
    """The forward roll over tucked arms."""
    cfg = horn.env_cfg(asset=asset, play=play)

    # ── The arms meet the body ─────────────────────────────────────────────────
    robot = cfg.scene.entities["robot"]
    robot.collisions = tuple(robot.collisions) + _arm_body_collision()

    # ── Sensors: arms apart from the walking legs, the front of the body, leg pairs
    walking = r"(LM|RM|LR|RR)_.*_meshcol.*"
    arms = r"(LF|RF)_.*_meshcol.*"
    sensors = [s for s in cfg.scene.sensors if s.name != "limb_ground_contact"]
    front_parts = tuple(p.strip("^$") for p in BODY_GEOMS) + (horn.CARAPACE_GEOMS,)
    cfg.scene.sensors = tuple(sensors) + (
        horn._ground("limb_ground_contact", walking, exclude=tuple(FOOT_GEOMS)),
        horn._ground("arm_ground_contact", arms, exclude=tuple(FOOT_GEOMS)),
        ContactSensorCfg(
            name="front_ground_contact",
            primary=ContactMatch(mode="geom", pattern=front_parts, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force", "pos"), reduce="maxforce", num_slots=2,
        ),
    ) + _pair_sensors()

    # ── Command: jumper.horn_roll's tracker, this demonstration, two more worsts ─
    old = cfg.commands["roll"]
    cfg.commands = {
        "roll": tr.tracker.TuckRollTrackerCfg(
            entity_name="robot", reference=str(REF_NPZ),
            region_sensors=old.region_sensors, front_x=FRONT_X,
        )
    }

    # ── Rewards ───────────────────────────────────────────────────────────────
    r = dict(cfg.rewards)
    legs_cfg = SceneEntityCfg("robot", joint_names=list(LEG_JOINTS), preserve_order=True)
    cols = tuple(GAIT_JOINTS.index(j) for j in LEG_JOINTS)
    r["prior_joint_pos"] = RewardTermCfg(
        func=tr.rewards.prior_joint_pos_legs, weight=momentum.PRIOR_POS_WEIGHT,
        params={"asset_cfg": legs_cfg, "columns": cols,
                "sigma_sq": momentum.PRIOR_POS_SIGMA_SQ},
    )
    stand = {"command_name": "roll", "min_own_turn": momentum.MIN_OWN_TURN}
    r["complete"] = RewardTermCfg(
        func=tr.rewards.gentle_complete, weight=momentum.COMPLETE_BONUS / STEP_DT,
        params={**stand, "gentle_floor": horn.GENTLE_FLOOR, "limb": horn.LIMB_BAND,
                "arm": ARM_BAND, "front": FRONT_BAND, "sensor": horn.SENSOR_BAND,
                "shell": horn.SHELL_BAND, "carapace_impact": horn.CARAPACE_IMPACT_BAND,
                "ground_impact": horn.GROUND_IMPACT_BAND, "flight": horn.FLIGHT_BAND},
    )
    r["arm_contact"] = RewardTermCfg(func=tr.rewards.arm_contact, weight=ARM_WEIGHT,
                                     params={"threshold": ARM_BAND[0]})
    r["front_contact"] = RewardTermCfg(func=tr.rewards.front_contact, weight=FRONT_WEIGHT,
                                       params={"threshold": FRONT_BAND[0]})
    r["tuck"] = RewardTermCfg(
        func=tr.rewards.tuck, weight=TUCK_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=list(PALM_BODIES),
                                            preserve_order=True),
                "min_x": TUCK_MIN_X, "max_y": TUCK_MAX_Y, "sigma_sq": TUCK_SIGMA_SQ,
                "min_down": NOSE_DOWN},
    )
    r["leg_collision"] = RewardTermCfg(
        func=tr.rewards.leg_collision_pairs, weight=LEG_COLLISION_WEIGHT,
        params={"sensor_names": tuple(s.name for s in _pair_sensors()),
                "force_threshold": 1.0},
    )
    cfg.rewards = r

    # The prior's fading curriculum names the term; it is still "prior_joint_pos".
    return cfg


__all__ = ["env_cfg"]
