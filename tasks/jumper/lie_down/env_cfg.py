"""jumper.lie_down -- the stock robot, standing, lowers itself onto its back: slowly,
deliberately and softly, by any route, and settles there with its legs at HOME.

The other half of jumper.self_right: with the two, the robot lies down and gets up
again. Built on jumper.self_right (scene, sensors, the legs colliding with the body,
the settle window, the hardest-hit tracker, floor-contact joint torque); what differs
is below.

## Episodes

Every episode starts standing at HOME (jumper.roll's default spawn; no turned spawns).
As in jumper.self_right, the joints are held at HOME until the body is still (at least
`SETTLE_STEPS`, at most `MAX_SETTLE_STEPS`); only then does the policy act, and
nothing before is paid or charged.

## Objective

- **Lower** (`mdp/rewards.lower`): new ground in how far onto its back the body is,
  2 units from standing to flat on its back, `LOWER_WEIGHT` per unit. Paid by the
  distance, not the speed.
- **The finish** ends the episode (`tracker.done`): within 20 deg of flat on its back,
  turning slower than 0.2 rad/s and moving slower than 0.02 m/s, every driven joint
  within `HOME_TOL` of HOME, all of it held `HOLD_STEPS` running. jumper.self_right's
  run 3 finished on four feet with a leg still in the air; the legs at HOME are part
  of finishing here, and `back_home` pays them while belly-up.
- **Finishing bonus** `COMPLETE_BONUS`: half however it went, half scaled by the
  worst of the gentle factors below.

## Slow and soft

Asked for: controlled, slow, deliberate and very soft, any direction or method. So
the bands are tighter than jumper.self_right's, and the body's rotation rate is
charged:

| Kind | Bonus band (free - zero) | Per step, over |
|---|---|---|
| limbs (feet excepted) | 20 - 100 N | 20 N |
| display / dToF / camera | 2 - 20 N | 2 N |
| stock shell **impact** | `SHELL_IMPACT_BAND` | its free level |
| floor impact, every part | `GROUND_IMPACT_BAND` | its free level |
| longest time in the air | 0 - 0.06 s | per second airborne |
| floor-contact joint torque | 4 - 8 N*m | 4 N*m |
| body rotation | `SPIN_BAND` rad/s | `SPIN_BAND[0]` |

The shell impact band starts at the rest load: lying still on its back at HOME the
shell carries the whole robot, 25 N (2.54 kg; measured 2026-10-30, native CPU, nominal
stock robot, legs clear of the floor), so an impact of that order is the robot set
down, and `SHELL_IMPACT_BAND[1]` three times it. These are judgement, to be revisited
with the plan search's numbers.

## The demonstration

None yet: a keyframe plan search in this environment (`studies/lie_down/search.py`)
is to find one, which will then be fed forward and used as a fading prior as in
jumper.momentum_roll.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..common.constants import GAIT_JOINTS
from ..self_right import env_cfg as sr_env
from . import mdp as ld

STEP_DT = sr_env.STEP_DT
DECIMATION = sr_env.DECIMATION

# ── Episodes ───────────────────────────────────────────────────────────────────
SETTLE_STEPS = 25
MAX_SETTLE_STEPS = 100
EPISODE_S = 6.0

# ── Objective ──────────────────────────────────────────────────────────────────
LOWER_WEIGHT = 5.0
COMPLETE_BONUS = 20.0
BACK_HOME_WEIGHT = 1.0
GENTLE_FLOOR = 0.5
HOME_TOL = 0.15
HOLD_STEPS = 15

# ── Slow and soft ──────────────────────────────────────────────────────────────
LIMB_BAND = (20.0, 100.0)
SENSOR_BAND = (2.0, 20.0)
SHELL_IMPACT_BAND = (25.0, 75.0)
GROUND_IMPACT_BAND = (40.0, 150.0)
FLIGHT_BAND = (0.0, 0.06)
LOAD_BAND = (4.0, 8.0)
SPIN_BAND = (2.0, 6.0)
LIMB_WEIGHT = -0.5
SENSOR_WEIGHT = sr_env.SENSOR_WEIGHT
SHELL_IMPACT_WEIGHT = -0.5
GROUND_IMPACT_WEIGHT = -0.5
AIRBORNE_WEIGHT = -2.0
LOAD_WEIGHT = -2.0
SPIN_WEIGHT = -0.5
STOP_STRIKE_WEIGHT = sr_env.STOP_STRIKE_WEIGHT
TORQUE_STRAIN_WEIGHT = sr_env.TORQUE_STRAIN_WEIGHT
LEG_COLLISION_WEIGHT = -1.0
JOINT_LIMIT_WEIGHT = -1.0
ACTION_RATE_WEIGHT = -0.01


def down_obs(env, command_name: str = "roll"):
    """How far onto its back the body is, and how far it has lowered, [N, 2] (critic
    only)."""
    import torch

    cmd = env.command_manager.get_term(command_name)
    return torch.stack([cmd.down_now(), cmd.best_down - cmd.spawn_down], dim=1)


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Lying down onto its back from standing."""
    cfg = sr_env.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    old = cfg.rewards

    # ── Command: how far down, the finish, and the hardest hits ───────────────
    cfg.commands = {"roll": ld.tracker.LieDownTrackerCfg(
        settle_steps=SETTLE_STEPS, max_settle_steps=MAX_SETTLE_STEPS,
        load_joints=GAIT_JOINTS, home_tol=HOME_TOL, hold_steps=HOLD_STEPS)}
    critic = cfg.observations["critic"].terms
    critic.pop("self_right", None)
    critic["lie_down"] = ObservationTermCfg(func=down_obs, params={"command_name": "roll"})

    # ── Every episode starts standing ─────────────────────────────────────────
    cfg.events["reset_rolled"].params["rolled_fraction"] = 0.0
    cfg.episode_length_s = EPISODE_S

    # ── Rewards ───────────────────────────────────────────────────────────────
    cfg.rewards = {
        "lower": RewardTermCfg(func=ld.rewards.lower, weight=LOWER_WEIGHT),
        "complete": RewardTermCfg(
            func=ld.rewards.gentle_complete, weight=COMPLETE_BONUS / STEP_DT,
            params={"gentle_floor": GENTLE_FLOOR, "limb": LIMB_BAND, "sensor": SENSOR_BAND,
                    "shell_impact": SHELL_IMPACT_BAND, "ground_impact": GROUND_IMPACT_BAND,
                    "flight": FLIGHT_BAND, "load": LOAD_BAND, "spin": SPIN_BAND},
        ),
        "back_home": RewardTermCfg(func=ld.rewards.back_home, weight=BACK_HOME_WEIGHT,
                                   params={"asset_cfg": gait_cfg}),
        "spin": RewardTermCfg(func=ld.rewards.spin, weight=SPIN_WEIGHT,
                              params={"free": SPIN_BAND[0]}),
        "joint_load": RewardTermCfg(func=old["joint_load"].func, weight=LOAD_WEIGHT,
                                    params={"threshold": LOAD_BAND[0]}),
        "limb_contact": RewardTermCfg(func=old["limb_contact"].func, weight=LIMB_WEIGHT,
                                      params={**old["limb_contact"].params,
                                              "threshold": LIMB_BAND[0]}),
        "sensor_contact": RewardTermCfg(func=old["sensor_contact"].func, weight=SENSOR_WEIGHT,
                                        params={**old["sensor_contact"].params,
                                                "threshold": SENSOR_BAND[0]}),
        "shell_impact": RewardTermCfg(func=old["shell_impact"].func, weight=SHELL_IMPACT_WEIGHT,
                                      params={"threshold": SHELL_IMPACT_BAND[0]}),
        "ground_impact": RewardTermCfg(func=old["ground_impact"].func,
                                       weight=GROUND_IMPACT_WEIGHT,
                                       params={"threshold": GROUND_IMPACT_BAND[0]}),
        "airborne": RewardTermCfg(func=old["airborne"].func, weight=AIRBORNE_WEIGHT),
        "stop_strike": RewardTermCfg(func=old["stop_strike"].func, weight=STOP_STRIKE_WEIGHT,
                                     params=old["stop_strike"].params),
        "torque_strain": RewardTermCfg(func=old["torque_strain"].func,
                                       weight=TORQUE_STRAIN_WEIGHT,
                                       params=old["torque_strain"].params),
        "leg_collision": RewardTermCfg(func=old["leg_collision"].func,
                                       weight=LEG_COLLISION_WEIGHT,
                                       params=old["leg_collision"].params),
        "dof_pos_limits": RewardTermCfg(func=old["dof_pos_limits"].func,
                                        weight=JOINT_LIMIT_WEIGHT),
        "action_rate_l2": RewardTermCfg(func=old["action_rate_l2"].func,
                                        weight=ACTION_RATE_WEIGHT),
    }

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "on_back": TerminationTermCfg(func=ld.rewards.on_back,
                                      params={"command_name": "roll"}),
    }
    return cfg


__all__ = ["env_cfg"]
