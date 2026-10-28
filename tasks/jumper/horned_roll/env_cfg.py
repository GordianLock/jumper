"""jumper.horned_roll -- jumper.momentum_roll on carapace prototype 2, gentle, with the
horned carapace treated as the reinforced part it is.

A sibling of jumper.gentle_momentum_roll, not a change to it: the same roll, guide,
window, progress and bonus floor, rebuilt here from jumper.momentum_roll so that
either task can change without moving the other. What differs is how contact and
airtime are charged; `mdp/rewards.py` has why, from that task's code and its keyframe
search on v2.

## The brief (2026-10-28)

- A gentle, working roll: forces and joint torque at or below the stock robot's. The
  reference is the stock high jump, measured like the rolls
  (`studies/carapace/RESULTS.md`, "Against the stock jumps"): median worst joint torque
  5.9 N*m (p90 7.0), limbs 76 N, feet 175 N.
- One exception: **the horns and their supports are reinforced** and may take more. The
  roll will probably land on them a lot; that is not to be discouraged, only kept
  somewhat gentle.
- **No incentive to leave the floor** to avoid the carapace: a landing is a harder hit
  than the contact it avoided.

## Contact kinds

| Kind | Parts | Bonus band, N | Per step over, N |
|---|---|---|---|
| limbs | every leg and arm part but the feet | 20 - 100 | 5 |
| sensors | display, dToF, camera | 2 - 20 | 2 |
| stock shell | `base_link`, `upper_shell_link` | 30 - 150 | 30 |
| carapace | hardest **region** (`REGIONS`) | 80 - 250 | 80, light |
| ground | every part summed, feet included | 100 - 250 | 100 |
| flight | longest airborne stretch, s | 0.06 - 0.20 | jumper.roll's `airborne` |

The first three are jumper.gentle_momentum_roll's values (its shell band now covers the
stock shell alone). The rest are **first guesses**, set by reasoning about scale, not
against a trained policy:

- carapace 80 N free: about three times the 26.2 N robot, and the level the stock rolls
  took on the stock shell (jumper.roll's first run, 2026-10-25: about 80-130 N body
  impacts). Nothing of the bonus by 250 N, nearly ten times the weight.
- ground 100 N free: four times the weight. 250 N zero: above the stock high jump's
  175 N on the feet, so a roll landing like that jump keeps half this factor.
- flight 0.06 s free: three control steps, a lift the roll can hardly avoid. Zero by
  0.20 s; the demonstration on v2 spent 0.24 s with nothing touching (the other task's
  measurement), so it would keep none of this factor.

## Regions

Each region's sensor sums its pieces (`netforce`), so a hit straddling the seam between
two collision pieces of one horn reads at full strength; the bonus reads the hardest
region, so load spread across several regions reads low. Pieces were assigned from
`assets/jumper/jumper_CF1-tri30m-rc-v2.xml` and the crests in
`design/carapace/versions/CF1-tri30m-rc-v2/params.json` (2026-10-28): a piece belongs to
the horn whose axis (root `a` to tip `b`) passes nearest its centroid, if that is within
3 cm; the rest are the front or rear field (valleys and flats) by the sign of x.
**Another carapace version needs its own map**: `env_cfg` refuses a model whose pieces
do not match it.

## Not charged in training

The torque floor contact puts on the joints (Jᵀf per contact), which the stock reference
is measured in. Judge a trained policy by it on replay (`studies/carapace/gentle.py`'s
method), against 5.9 N*m.
"""

from __future__ import annotations

import math
import re
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
from . import mdp as hr
from .rl_cfg import NUM_STEPS_PER_ENV

STEP_DT = momentum.STEP_DT

# ── Episodes and the guide: jumper.momentum_roll's ─────────────────────────────
DEMO_FRACTION = momentum.DEMO_FRACTION
BACK_FRACTION = 0.0
EPISODE_S = momentum.EPISODE_S
PRIOR_POS_WEIGHT = momentum.PRIOR_POS_WEIGHT
PRIOR_POS_SIGMA_SQ = momentum.PRIOR_POS_SIGMA_SQ
PRIOR_TURN_WEIGHT = momentum.PRIOR_TURN_WEIGHT
PRIOR_TURN_SIGMA_SQ = momentum.PRIOR_TURN_SIGMA_SQ
PRIOR_HOLD_ITERATIONS = momentum.PRIOR_HOLD_ITERATIONS
PRIOR_END_ITERATIONS = momentum.PRIOR_END_ITERATIONS

# ── Time and the objective: jumper.gentle_momentum_roll's (2026-10-28) ─────────
START_TURN = 0.3
ROLL_WINDOW = 1.0
ZERO_SECONDS = 2.0
PROGRESS_WEIGHT = 2.0
MAX_RATE = 2.0 * math.pi / ROLL_WINDOW
COMPLETE_BONUS = 20.0
HOME_FLOOR = 0.5
HOME_SIGMA_SQ = 0.10
GENTLE_FLOOR = 0.5

# ── Gentle: the bonus bands, N (flight in s); see the table above ─────────────
LIMB_FREE, LIMB_ZERO = 20.0, 100.0
SENSOR_FREE, SENSOR_ZERO = 2.0, 20.0
SHELL_FREE, SHELL_ZERO = 30.0, 150.0
CARAPACE_FREE, CARAPACE_ZERO = 80.0, 250.0
GROUND_FREE, GROUND_ZERO = 100.0, 250.0
FLIGHT_FREE, FLIGHT_ZERO = 0.06, 0.20

# ── Gentle: per step ───────────────────────────────────────────────────────────
LIMB_THRESHOLD = flourish.LIMB_THRESHOLD
LIMB_WEIGHT = -0.5
LIMB_WEIGHT_START = -0.1
SENSOR_THRESHOLD = flourish.SENSOR_THRESHOLD
SENSOR_WEIGHT = flourish.SENSOR_WEIGHT
FOOT_THRESHOLD = flourish.FOOT_THRESHOLD
FOOT_WEIGHT = flourish.FOOT_WEIGHT
SHELL_THRESHOLD = 30.0
SHELL_WEIGHT = -0.5
#: The carapace's charge starts where its bonus band does, and is light: a nudge to
#: land on it softly, not a reason to avoid it.
CARAPACE_THRESHOLD = CARAPACE_FREE
CARAPACE_WEIGHT = -0.1
GROUND_THRESHOLD = GROUND_FREE
GROUND_WEIGHT = -0.5
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
STOCK_SHELL_PARTS = ("base_link_meshcol", "upper_shell_link_meshcol")

#: Carapace pieces (`upper_shell_carapace_NN_meshcol`) by region, for CF1-tri30m-rc-v2.
REGIONS: dict[str, tuple[int, ...]] = {
    "centre": (2, 3, 24, 36, 37, 38, 41),
    "front_left": (0, 5, 6, 12, 13, 21, 23),
    "front_right": (4, 9, 10, 19, 28, 32, 39),
    "rear_left": (1, 15, 16, 25, 26, 27, 29, 30, 31),
    "rear_right": (14, 20, 22, 33, 40, 42, 45, 46, 47),
    "front_field": (34, 35),
    "rear_field": (7, 8, 11, 17, 18, 43, 44),
}
_PIECE = re.compile(r'name="upper_shell_carapace_(\d+)_meshcol"')


def _carapace_pieces(asset: Path | None) -> set[int]:
    if asset is None:
        return set()
    return {int(n) for n in _PIECE.findall(Path(asset).read_text())}


def _region_sensors(asset: Path | None) -> tuple:
    """One `netforce` sensor per region, or none for a model without a carapace."""
    pieces = _carapace_pieces(asset)
    if not pieces:
        return ()
    mapped = {i for ids in REGIONS.values() for i in ids}
    if pieces != mapped:
        raise ValueError(
            f"jumper.horned_roll's REGIONS map CF1-tri30m-rc-v2's 48 carapace pieces; "
            f"{asset} has pieces {sorted(pieces)}. Map this version's pieces to regions "
            f"first (env_cfg.py, 'Regions').")
    return tuple(
        flourish._ground(f"carapace_{name}_contact",
                         "|".join(f"upper_shell_carapace_{i:02d}_meshcol" for i in ids))
        for name, ids in REGIONS.items()
    )


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """The momentum roll on a horned carapace, gentle where it matters."""
    cfg = momentum.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    gait_act = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))

    # ── Sensors ───────────────────────────────────────────────────────────────
    regions = _region_sensors(asset)
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        flourish._ground("limb_ground_contact", flourish._LEG, exclude=tuple(FOOT_GEOMS)),
        flourish._ground("sensor_ground_contact", flourish.SENSOR_PARTS),
        flourish._ground("stock_shell_ground_contact", STOCK_SHELL_PARTS),
    ) + regions

    # ── Command: the roll's clock, hardest hits, landing and flight ───────────
    cfg.commands = {
        "roll": hr.state.HornedRollTrackerCfg(
            entity_name="robot", reference=str(REF_NPZ),
            limb_sensor="limb_ground_contact", sensor_sensor="sensor_ground_contact",
            shell_sensor="stock_shell_ground_contact",
            region_sensors=tuple(s.name for s in regions), start_turn=START_TURN,
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
        # The guide, faded by jumper.momentum_roll's curriculum.
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
            func=hr.rewards.horned_complete, weight=COMPLETE_BONUS / STEP_DT,
            params={**done, "zero_seconds": ZERO_SECONDS, "gentle_floor": GENTLE_FLOOR,
                    "limb_free": LIMB_FREE, "limb_zero": LIMB_ZERO,
                    "sensor_free": SENSOR_FREE, "sensor_zero": SENSOR_ZERO,
                    "shell_free": SHELL_FREE, "shell_zero": SHELL_ZERO,
                    "carapace_free": CARAPACE_FREE, "carapace_zero": CARAPACE_ZERO,
                    "ground_free": GROUND_FREE, "ground_zero": GROUND_ZERO,
                    "flight_free": FLIGHT_FREE, "flight_zero": FLIGHT_ZERO,
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
            params={"sensor_name": "stock_shell_ground_contact",
                    "threshold": SHELL_THRESHOLD},
        ),
        "ground_impact": RewardTermCfg(
            func=hr.rewards.ground_overload, weight=GROUND_WEIGHT,
            params={"threshold": GROUND_THRESHOLD},
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
    if regions:
        cfg.rewards["carapace_impact"] = RewardTermCfg(
            func=hr.rewards.carapace_overload, weight=CARAPACE_WEIGHT,
            params={"threshold": CARAPACE_THRESHOLD},
        )

    # ── Terminations: standing, once the window has passed ────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "rolled_and_standing": TerminationTermCfg(func=hr.rewards.window_done, params=done),
    }

    # The guide's schedule counts this task's steps per update.
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
