"""The rewards of jumper.gentle_momentum_roll that are its own: a completion bonus that
is full for a roll over in `window` seconds, worth having however hard it landed,
and worth more the softer.

## What the first run taught (2026-10-28, Colab, 256 envs, 2500 iterations)

The first version learnt to stand still. With the demonstration fed in fully
(iteration 100), 40% of episodes ended rolled and standing; as the guide faded,
full rolls fell to 0% by the end, and the replay of `model_2499` stands in place,
tipping onto one side and back. Two reasons, both in the rewards:

- **The bonus was multiplied by one gentleness factor per contact kind**, each 0
  at its `*_zero` force. The early rolls hit limbs at 75-90 N and the shell at
  85-110 N, factors of about 0.1-0.3 each, so their product left almost nothing:
  `complete` earned 0.02-0.06 per episode, against about -0.45 for limb contact
  alone. Standing still paid none of the costs. **Now the bonus has a floor**
  (`gentle_floor`): standing after the roll earns that share whatever the hits,
  and gentleness scales the rest.
- **Progress was potential-based shaping** (gamma * phi(next) - phi(now)), meant to
  pay for turning without paying for when. It also charges (1 - gamma) * phi per
  step for every step spent partway round, so in practice turning further never
  paid: the term averaged -0.4 to -0.8 per episode. **Now it is
  jumper.momentum_roll's progress** (`momentum_roll/mdp/rewards.py`), new ground
  per second, capped at one full turn per `window` (`env_cfg.MAX_RATE`).

## Time: close to neutral inside the window, not exactly

- **The bonus** is paid at the window's end, not at standing: a roll that stands
  early holds its stance until then (`window_done`), so every roll inside the
  window is paid on the same step, and the bonus does not favour a quicker one.
- **Progress** is paid as the roll is made, so the discount favours an earlier
  roll a little. The rate cap means spinning faster than one turn per window pays
  nothing more, and gamma 0.998 (`rl_cfg.GAMMA`) keeps what is left small: a roll
  done in 0.5 s rather than 1 s gets its progress about 5% earlier in discounted
  value. That is the price of a progress reward that teaches the roll.
- **Per-step pay near the end** (jumper.momentum_roll's `finish_upright` and
  `finish_home`) pays whoever gets there sooner, so it is not used; the bonus
  carries the finish instead (`home_floor`).

## Past the window

The bonus falls linearly from full at `window` to none at `zero_seconds`.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...momentum_roll.mdp.terminations import standing_done
from ...showcase_roll.mdp.rewards import _factor


def _cmd(env, name: str):
    return env.command_manager.get_term(name)


def window_done(env, command_name: str = "roll", window: float = 1.0,
                min_own_turn: float = 1.0) -> torch.Tensor:
    """Standing after the roll, and at least `window` seconds since it began.

    Half a control step of slack, so a roll clock that lands on the window between
    steps still counts on that step."""
    cmd = _cmd(env, command_name)
    standing = standing_done(env, command_name=command_name, min_own_turn=min_own_turn)
    late_enough = cmd.roll_seconds() >= window - 0.5 * env.step_dt
    return standing & late_enough & cmd.begun


def gentle_complete(env, command_name: str = "roll", window: float = 1.0,
                    zero_seconds: float = 2.0, gentle_floor: float = 0.5,
                    limb_free: float = 20.0, limb_zero: float = 100.0,
                    sensor_free: float = 2.0, sensor_zero: float = 20.0,
                    shell_free: float = 30.0, shell_zero: float = 150.0,
                    home_floor: float = 0.5, home_sigma_sq: float = 0.10,
                    asset_cfg: SceneEntityCfg | None = None,
                    min_own_turn: float = 1.0) -> torch.Tensor:
    """On the step the episode ends successfully (`window_done`):

        duration x finish x (gentle_floor + (1 - gentle_floor) x gentleness)

    - duration: 1 up to `window` s, 0 by `zero_seconds`;
    - finish: `home_floor` for standing at all, rising to 1 with the legs at HOME;
    - gentleness: one factor per contact kind, each 1 at `*_free` and 0 at `*_zero`
      newtons of the episode's hardest single contact of that kind, multiplied.

    0 on every other step."""
    cmd = _cmd(env, command_name)
    done = window_done(env, command_name=command_name, window=window,
                       min_own_turn=min_own_turn).float()
    over = (cmd.roll_seconds() - window).clamp(min=0.0)
    duration = (1.0 - over / (zero_seconds - window)).clamp(0.0, 1.0)
    limb = torch.maximum(cmd.peak_limb, cmd.part_force(cmd.cfg.limb_sensor))
    sensor = torch.maximum(cmd.peak_sensor, cmd.part_force(cmd.cfg.sensor_sensor))
    shell = torch.maximum(cmd.peak_shell, cmd.part_force(cmd.cfg.shell_sensor))
    gentleness = (_factor(limb, limb_free, limb_zero)
                  * _factor(sensor, sensor_free, sensor_zero)
                  * _factor(shell, shell_free, shell_zero))
    asset_cfg = asset_cfg or SceneEntityCfg("robot")
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    k = torch.exp(-((q - home) ** 2).mean(dim=1) / home_sigma_sq)
    finish = home_floor + (1.0 - home_floor) * k
    return done * duration * finish * (gentle_floor + (1.0 - gentle_floor) * gentleness)


__all__ = ["gentle_complete", "window_done"]
