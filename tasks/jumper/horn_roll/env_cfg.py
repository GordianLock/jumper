"""jumper.horn_roll -- a gentle roll on a horned carapace, from a demonstration found on
the carapace itself, with the horns free to be pivoted on.

Built on jumper.momentum_roll (its fading demonstration, mid-demonstration spawns,
progress and finish terms) and jumper.roll (scene, sensors, observations). Nothing in
it comes from jumper.gentle_momentum_roll, jumper.horned_roll or studies/roll_v2.

## The demonstration

`ref/horn_roll.npz`, recorded in this task's own environment by
`studies/proto3/record_demo.py` from the keyframe plan
`studies/proto3/plans/stand_any_impact_s1.npy`: a search on carapace prototype 3 in
the training physics (`studies/proto3/search.py`, "any direction" from standing,
carapace impact rather than load charged). In that search it rolled end over end in
29 of 32 perturbed starts, about the body's y axis (87 deg), landing on the carapace,
with a median worst floor-contact torque on any joint of 5.6 N*m. The recording
stores the axis it rolled about (`axis`), and this task measures its turn about that
axis (`mdp/tracker.py`), so the same code trains a roll about any horizontal axis.

The guide is jumper.momentum_roll's: fed forward into the action and used as a prior
on joints and turn, both holding to `PRIOR_HOLD_ITERATIONS` and fading to nothing by
`PRIOR_END_ITERATIONS`; `DEMO_FRACTION` of spawns start partway through it.

## The carapace

- **It collides, with friction.** The repository's collision config switches off any
  geom it does not list, so this task adds the carapace pieces itself
  (`_carapace_collision`): condim 3, friction 0.6 (the horn study's horns), so a horn
  can grip the floor as a pivot. Feet, limbs and stock shell keep the repository's
  settings.
- **Contact with it is free.** Load on the horns, their supports, the bumpers and the
  valleys costs nothing. What is charged is **impact**: the sharpest rise in force
  within a control step, per region (`mdp/regions.py`), lightly per step and through
  the finishing bonus (`mdp/rewards.py`).

## Gentle, and no reward for leaving the floor

| Kind | Bonus band (free - zero) | Per step, over |
|---|---|---|
| limbs (feet excepted) | 20 - 100 N | 20 N |
| display / dToF / camera | 2 - 20 N | 2 N |
| stock shell and base | 30 - 150 N | 30 N |
| carapace impact, hardest region | 50 - 200 N | 50 N, light |
| floor impact, every part summed | 60 - 250 N | 60 N |
| longest time in the air | 0.06 - 0.20 s | jumper.roll's `airborne` |

A landing costs the same whatever touches down first, so dodging the carapace by
jumping onto the feet gains nothing, and the air time itself is charged twice over.
The bands are the ones `studies/proto3/search.py` scored with, which found the
demonstration; they are judgement, not measurements of hardware limits.

## Not trained on: joint torque

The torque floor contact puts on each joint (J^T f) needs a Jacobian per contact per
step. Judge a trained policy by it on replay (`studies/proto3/search.py`'s
`floor_torque`), against KingKong's reference-free jump, 4.8 N*m, and the
demonstration's 5.6.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.spec_config import CollisionCfg

from ..common import constants as K
from ..common.constants import FOOT_GEOMS, GAIT_JOINTS
from ..flourish_roll import env_cfg as flourish
from ..flourish_roll.mdp import rewards as fr
from ..momentum_roll import env_cfg as momentum
from . import mdp as hr

REF_NPZ = Path(__file__).resolve().parent / "ref" / "horn_roll.npz"
STEP_DT = momentum.STEP_DT
DECIMATION = 4                         # 4 x 5 ms physics steps per 20 ms control step

# ── Contact: bonus bands (free, zero) and per-step thresholds ─────────────────
GENTLE_FLOOR = 0.5
LIMB_BAND = (20.0, 100.0)
SENSOR_BAND = (2.0, 20.0)
SHELL_BAND = (30.0, 150.0)
CARAPACE_IMPACT_BAND = (50.0, 200.0)
GROUND_IMPACT_BAND = (60.0, 250.0)
FLIGHT_BAND = (0.06, 0.20)

LIMB_THRESHOLD = LIMB_BAND[0]
LIMB_WEIGHT = -0.5
SENSOR_THRESHOLD = SENSOR_BAND[0]
SENSOR_WEIGHT = flourish.SENSOR_WEIGHT
SHELL_THRESHOLD = SHELL_BAND[0]
SHELL_WEIGHT = -0.5
#: Light: a nudge to put weight on the horns softly, not a reason to avoid them.
CARAPACE_IMPACT_THRESHOLD = CARAPACE_IMPACT_BAND[0]
CARAPACE_IMPACT_WEIGHT = -0.1
GROUND_IMPACT_THRESHOLD = GROUND_IMPACT_BAND[0]
GROUND_IMPACT_WEIGHT = -0.5
STOP_STRIKE_WEIGHT = flourish.STOP_STRIKE_WEIGHT
TORQUE_STRAIN_WEIGHT = flourish.TORQUE_STRAIN_WEIGHT
AXIS_HEADING_WEIGHT = momentum.HEADING_WEIGHT
AXIS_DRIFT_WEIGHT = momentum.FORWARD_DRIFT_WEIGHT
AXIS_DRIFT_SCALE = momentum.FORWARD_DRIFT_SCALE

# ── The carapace's own collision ───────────────────────────────────────────────
CARAPACE_GEOMS = r"^upper_shell_carapace_\d+_meshcol$"
CARAPACE_FRICTION = (0.6, 0.005, 0.0001)
STOCK_SHELL_PARTS = ("base_link_meshcol", "upper_shell_link_meshcol")


def _carapace_collision() -> CollisionCfg:
    """The carapace pieces against the floor, with friction; every other geom as the
    repository's collision config leaves it."""
    return CollisionCfg(
        geom_names_expr=(CARAPACE_GEOMS,),
        contype=K._GROUND_BIT, conaffinity=0, condim=3, priority=0,
        friction=CARAPACE_FRICTION, solref=K._CONTACT_SOLREF, solimp=K._CONTACT_SOLIMP,
        disable_other_geoms=False,
    )


def _ground(name: str, pattern, exclude=(), history: int = 0):
    s = flourish._ground(name, pattern, exclude=exclude)
    s.history_length = history
    return s


def env_cfg(asset: Path | None = None, play: bool = False):
    """The roll on a horned carapace, from its demonstration."""
    cfg = momentum.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    gait_act = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))

    # ── The carapace collides, with friction ──────────────────────────────────
    regions = hr.regions.carapace_regions(asset)
    if regions:
        robot = cfg.scene.entities["robot"]
        robot.collisions = tuple(robot.collisions) + (_carapace_collision(),)

    # ── Sensors ───────────────────────────────────────────────────────────────
    region_sensors = tuple(
        _ground(f"carapace_{name}_contact",
                "|".join(f"upper_shell_carapace_{i:02d}_meshcol" for i in ids),
                history=DECIMATION)
        for name, ids in regions.items()
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        _ground("limb_ground_contact", flourish._LEG, exclude=tuple(FOOT_GEOMS)),
        _ground("sensor_ground_contact", flourish.SENSOR_PARTS),
        _ground("stock_shell_ground_contact", STOCK_SHELL_PARTS),
        _ground("all_ground_contact", r".*_meshcol", history=DECIMATION),
    ) + region_sensors

    # ── Command: the roll about the demonstration's axis ──────────────────────
    cfg.commands = {
        "roll": hr.tracker.HornRollTrackerCfg(
            entity_name="robot", reference=str(REF_NPZ),
            region_sensors=tuple(s.name for s in region_sensors),
        )
    }

    # ── Rewards: jumper.momentum_roll's guide and objective, gentle on top ────
    stand = {"command_name": "roll", "min_own_turn": momentum.MIN_OWN_TURN}
    r = dict(cfg.rewards)
    for k in ("body_impact", "heading_error", "forward_drift"):
        r.pop(k)                            # replaced by the axis- and part-aware terms
    r["complete"] = RewardTermCfg(
        func=hr.rewards.gentle_complete, weight=momentum.COMPLETE_BONUS / STEP_DT,
        params={**stand, "gentle_floor": GENTLE_FLOOR, "limb": LIMB_BAND,
                "sensor": SENSOR_BAND, "shell": SHELL_BAND,
                "carapace_impact": CARAPACE_IMPACT_BAND,
                "ground_impact": GROUND_IMPACT_BAND, "flight": FLIGHT_BAND},
    )
    r["limb_contact"] = RewardTermCfg(
        func=fr.contact_overload, weight=LIMB_WEIGHT,
        params={"sensor_name": "limb_ground_contact", "threshold": LIMB_THRESHOLD},
    )
    r["sensor_contact"] = RewardTermCfg(
        func=fr.contact_overload, weight=SENSOR_WEIGHT,
        params={"sensor_name": "sensor_ground_contact", "threshold": SENSOR_THRESHOLD},
    )
    r["stock_shell_contact"] = RewardTermCfg(
        func=hr.rewards.stock_shell_contact, weight=SHELL_WEIGHT,
        params={"threshold": SHELL_THRESHOLD},
    )
    if region_sensors:
        r["carapace_impact"] = RewardTermCfg(
            func=hr.rewards.carapace_impact, weight=CARAPACE_IMPACT_WEIGHT,
            params={"threshold": CARAPACE_IMPACT_THRESHOLD},
        )
    r["ground_impact"] = RewardTermCfg(
        func=hr.rewards.ground_impact, weight=GROUND_IMPACT_WEIGHT,
        params={"threshold": GROUND_IMPACT_THRESHOLD},
    )
    r["stop_strike"] = RewardTermCfg(
        func=fr.stop_strike, weight=STOP_STRIKE_WEIGHT, params={"asset_cfg": gait_cfg},
    )
    r["torque_strain"] = RewardTermCfg(
        func=fr.torque_strain, weight=TORQUE_STRAIN_WEIGHT, params={"asset_cfg": gait_act},
    )
    r["axis_heading"] = RewardTermCfg(func=hr.rewards.axis_heading_error,
                                      weight=AXIS_HEADING_WEIGHT)
    r["axis_drift"] = RewardTermCfg(func=hr.rewards.axis_drift, weight=AXIS_DRIFT_WEIGHT,
                                    params={"scale": AXIS_DRIFT_SCALE})
    cfg.rewards = r
    return cfg


__all__ = ["env_cfg"]
