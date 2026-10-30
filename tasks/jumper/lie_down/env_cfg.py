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
  turning slower than `FINISH_ANG_VEL` and moving slower than `FINISH_LIN_VEL`, every
  driven joint within `HOME_TOL` of HOME, all of it held `HOLD_STEPS` running (see
  "The finish has to be reachable while exploring" below). jumper.self_right's
  run 3 finished on four feet with a leg still in the air; the legs at HOME are part
  of finishing here, and `back_home` pays them while belly-up.
- **Finishing bonus** `COMPLETE_BONUS`: half however it went, half scaled by the
  worst of the gentle factors below.

## The finish has to be reachable while exploring

The first run (Colab, 2500 iterations, revision 76e61e6) had the finish at 0.15 rad,
0.2 rad/s, 0.02 m/s, held 15 steps, and no episode in training ever finished, so the
finishing bonus -- and every gentleness factor in it -- never paid; the policy learnt
the fastest way over (13-17 rad/s, shell impact about 230 N). Replayed on the nominal
robot (2026-10-30, native and warp, CPU) the same policy finishes within 0.5 s of
landing: what stopped it in training is training's own action noise. With that noise
added (std 0.225, the run's final `Policy/mean_std`), belly-up the body rocked at a
median 0.37 rad/s and the joints sat a median 0.16 rad off HOME; the longest run
meeting the old finish was 2 steps. Meeting 0.3 rad, 0.8 rad/s and 0.06 m/s: 17. Still
far from the flip itself.

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

`ref/lie_down.npz`: the plan from `studies/lie_down/search.py` in this environment
(2026-10-30, 35 rounds of 32 randomised robots, from standing at HOME,
`studies/lie_down/logs/lie_down_s1.log`), chosen for finishing on 31 of the 32, and
recorded on the nominal robot from its release by `studies/lie_down/record_plan.py`:
six keyframes ramped 0.5 s apart, back to HOME and held; the
finish 3.66 s after the release. Shell impact 83 N, floor impact 115 N, joint torque 4.3 N*m, body
rotation 7.6 rad/s, limbs 110 N, sensors 0 N, never airborne (run 1: 13-17 rad/s,
about 230 N on the shell).

Used as jumper.momentum_roll uses its demonstration: fed forward into the action
(`mdp/actions.py`) and a prior on the joints (`momentum_roll.mdp.rewards.
prior_joint_pos`), both from the release, both fading from `PRIOR_HOLD_ITERATIONS` to
nothing at `PRIOR_END_ITERATIONS` -- jumper.momentum_roll's schedule and widths.
Every episode starts at the demonstration's start (standing); there are no mid-
demonstration spawns.

Checked 2026-10-30 (native CPU, nominal robot, training mode, zero policy action,
feedforward whole): the robot follows the recording exactly (joint prior at its full
3.0 throughout), and the episode ends on the finish 3.68 s after the release. The
finishing bonus there is its floor half only: the demonstration's shell impact (83 N)
and limb force (110 N) are past those bands' zero points (75 N, 100 N), which is what
training is to bring down.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
import dataclasses

from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..common.constants import GAIT_JOINTS
from ..momentum_roll.mdp import rewards as mr_rewards
from ..ref_free_jump.mdp import curriculum as fade
from ..self_right import env_cfg as sr_env
from . import mdp as ld

REF_NPZ = Path(__file__).resolve().parent / "ref" / "lie_down.npz"

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
HOME_TOL = 0.3
FINISH_ANG_VEL = 0.8
FINISH_LIN_VEL = 0.06
HOLD_STEPS = 10

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

# ── The demonstration, and how it fades ───────────────────────────────────────
PRIOR_POS_WEIGHT = 3.0
PRIOR_POS_SIGMA_SQ = 0.20
#: Iterations at full feedforward and prior weight, and the iteration both reach 0.
PRIOR_HOLD_ITERATIONS = 300
PRIOR_END_ITERATIONS = 1200
#: rsl_rl's rollout length (`rl_cfg.py`, jumper.roll's): the fade is in env steps.
NUM_STEPS_PER_ENV = 24


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
        load_joints=GAIT_JOINTS, home_tol=HOME_TOL, hold_steps=HOLD_STEPS,
        reference=str(REF_NPZ),
        finish_ang_vel=FINISH_ANG_VEL, finish_lin_vel=FINISH_LIN_VEL)}
    critic = cfg.observations["critic"].terms
    critic.pop("self_right", None)
    critic["lie_down"] = ObservationTermCfg(func=down_obs, params={"command_name": "roll"})

    # ── Every episode starts standing ─────────────────────────────────────────
    cfg.events["reset_rolled"].params["rolled_fraction"] = 0.0
    cfg.episode_length_s = EPISODE_S

    # ── Action: held at HOME while settling, the demonstration fed forward ────
    a = cfg.actions["joint_pos"]
    keep = {f.name: getattr(a, f.name) for f in dataclasses.fields(a)
            if f.init and f.name in {g.name for g in dataclasses.fields(ld.actions.LieDownActionCfg)}}
    keep.update(command_name="roll", hold_step=NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
                end_step=NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS, enabled=not play)
    cfg.actions["joint_pos"] = ld.actions.LieDownActionCfg(**keep)

    # ── Rewards ───────────────────────────────────────────────────────────────
    cfg.rewards = {
        "lower": RewardTermCfg(func=ld.rewards.lower, weight=LOWER_WEIGHT),
        "prior_joint_pos": RewardTermCfg(
            func=mr_rewards.prior_joint_pos, weight=PRIOR_POS_WEIGHT,
            params={"asset_cfg": gait_cfg, "sigma_sq": PRIOR_POS_SIGMA_SQ}),
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

    # ── Curriculum: the prior fades with the feedforward ──────────────────────
    cfg.curriculum = {} if play else {
        "prior_joint_pos_weight": CurriculumTermCfg(
            func=fade.anneal_reward_weight,
            params={"term_name": "prior_joint_pos", "start_weight": PRIOR_POS_WEIGHT,
                    "end_weight": 0.0,
                    "start_step": NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
                    "end_step": NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS}),
        "feedforward": CurriculumTermCfg(func=fade.report_feedforward),
    }

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "on_back": TerminationTermCfg(func=ld.rewards.on_back,
                                      params={"command_name": "roll"}),
    }
    return cfg


__all__ = ["env_cfg"]
