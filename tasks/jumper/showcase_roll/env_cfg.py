"""jumper.showcase_roll -- jumper.flourish_roll with the completion bonus made to carry
what that task only asked for per step: slow, gentle, lively.

jumper.flourish_roll's first run (jumper5, `model_2499` of the 2026-10-25 Colab
run) completed 16 of 16 rolls, in 0.76 s, with a median 5.4 N*m peak joint load
(6.9 worst) and its middle-left thigh striking the floor every roll (4.6 N*m median
peak from that contact alone; 16 rolls, native CPU, re-simulated at the 5 ms
physics step). `mdp/rewards.py` has why per-step terms could not hold it to the
brief; the changes:

- **Completion is scaled** by the roll's duration (full at `FULL_SECONDS`, counted
  from the first `start_turn` of turn) and by the hardest single limb and sensor
  contact of the episode (`LIMB_FREE` to `LIMB_ZERO`, `SENSOR_FREE` to
  `SENSOR_ZERO`). A 1 s roll earns a fifth of it; one hit of four times the
  robot's weight on a shin or claw earns none of it.
- **Stalling costs**: mid-roll, `STALL_AFTER` seconds without new ground.
- **Flourish weighs three times as much**, and per-step limb contact half as much
  (`LIMB_STEP_WEIGHT`).
- **gamma 0.998** (`rl_cfg.py`): a 10 s horizon, so the end of a slow roll is
  still worth having from its start.

Everything else -- scene, sensors, bootstrap, style and safety terms -- is
jumper.flourish_roll's, read from its config.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..flourish_roll import env_cfg as flourish
from ..momentum_roll import env_cfg as momentum
from ..momentum_roll.mdp.reference import REF_NPZ
from . import mdp as sc

STEP_DT = flourish.STEP_DT

COMPLETE_BONUS = 20.0
FULL_SECONDS = 5.0
START_TURN = 0.3
#: Hardest single limb part on the floor: no reduction up to 20 N (most of the
#: robot's 25 N weight on one shin is ordinary rolling), none of the bonus by
#: 100 N. jumper5's middle-left thigh, the hit that loaded its joint 4.6 N*m,
#: is the kind this is for.
LIMB_FREE = 20.0
LIMB_ZERO = 100.0
SENSOR_FREE = 2.0
SENSOR_ZERO = 20.0

#: Per-step limb contact at half jumper.flourish_roll's weight: hard hits are now
#: charged through the bonus. At the full weight the demonstration, which the
#: bootstrap starts the policy on, scored -1.2 per episode under these rewards
#: (16 envs, native CPU) against 0 for standing still.
LIMB_STEP_WEIGHT = -0.5

STALL_AFTER = 1.0
STALL_WEIGHT = -1.0
FLOURISH_WEIGHT = 3.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """A slow, gentle, lively roll, the bonus scaled by all three."""
    cfg = flourish.env_cfg(asset=asset, play=play)
    cfg.commands = {
        "roll": sc.state.ShowcaseTrackerCfg(
            entity_name="robot", reference=str(REF_NPZ),
            limb_sensor="limb_ground_contact", sensor_sensor="sensor_ground_contact",
            start_turn=START_TURN,
        )
    }
    rewards = dict(cfg.rewards)
    rewards["complete"] = RewardTermCfg(
        func=sc.rewards.scaled_complete, weight=COMPLETE_BONUS / STEP_DT,
        params={"full_seconds": FULL_SECONDS, "limb_free": LIMB_FREE,
                "limb_zero": LIMB_ZERO, "sensor_free": SENSOR_FREE,
                "sensor_zero": SENSOR_ZERO, "min_own_turn": momentum.MIN_OWN_TURN},
    )
    rewards["stall"] = RewardTermCfg(
        func=sc.rewards.stall, weight=STALL_WEIGHT, params={"after": STALL_AFTER},
    )
    rewards["flourish"].weight = FLOURISH_WEIGHT
    rewards["limb_contact"].weight = LIMB_STEP_WEIGHT
    cfg.rewards = rewards
    return cfg


__all__ = ["env_cfg"]
