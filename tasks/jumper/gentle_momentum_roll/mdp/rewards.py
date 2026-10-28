"""The rewards of jumper.gentle_momentum_roll that are its own: a roll that is over in
`window` seconds, paid the same whether it took the whole window or less.

## Neutral about time inside the window, and why that takes care

Nothing below pays a fast roll more than a slow one, as long as both finish inside
`window` seconds, counted from the roll's start (`state.py`, `t0`) to
standing. Three things would pay for speed without saying so, and each is
removed:

- **The discount.** A bonus paid the step the robot stands is worth gamma^steps:
  at 0.99 and 50 Hz a roll that stands at 0.5 s would get 1.28 times what one
  standing at 1.0 s gets. So the bonus is paid at the window's end, not at
  standing: a roll that stands early holds still until then
  (`terminations.done_after_window`), and every roll inside the window is paid on
  the same step.
- **Progress paid as it is made** has the same problem, radian by radian. Here it
  is *potential-based shaping* (Ng, Harada and Russell, 1999): each step pays
  gamma * phi(next) - phi(now), with phi = `weight` x the roll's furthest point, and
  phi = 0 on the step the episode ends successfully. Over any episode those
  payments add up, discounted, to -phi(start) -- the same for every roll -- so they
  guide the policy toward turning without paying for when it turns. The price is
  a large negative payment on the successful step, which the critic learns to
  expect; the bonus on that step is what success is worth.
- **Per-step pay near the end** (jumper.momentum_roll's `finish_upright` and
  `finish_home`) pays more to whoever gets there sooner. Not used; the bonus
  carries the finish instead (`home_floor`).

What still differs with time is real: a faster roll may hit harder, and is charged
for that like any other.

## Past the window

The bonus falls linearly from full at `window` to none at `zero_seconds`.

## Gentle

The bonus is also multiplied by one factor per contact kind, each falling linearly
from 1 at `*_free` newtons to 0 at `*_zero` newtons of the hardest single contact
of that kind in the episode -- jumper.showcase_roll's scheme, with the shell
added. jumper.showcase_roll's reason holds: a hit lasts one control step, so a
per-step charge alone barely notices one slam. The per-step charges are in
`env_cfg.py` as well, for a gradient between slams.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...momentum_roll.mdp.terminations import standing_done
from ...roll.mdp.commands import TWO_PI
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


def shaped_progress(env, command_name: str = "roll", gamma: float = 0.99,
                    window: float = 1.0, min_own_turn: float = 1.0) -> torch.Tensor:
    """gamma * phi(next) - phi(now) per step, as a rate (the manager multiplies by
    the step length), with phi the furthest point of the roll in radians and 0 on
    the step the episode ends successfully. The env_cfg weight scales phi."""
    cmd = _cmd(env, command_name)
    now = cmd.best
    nxt = torch.maximum(now, cmd.total_now().clamp(max=TWO_PI))
    ends = window_done(env, command_name=command_name, window=window,
                       min_own_turn=min_own_turn)
    nxt = torch.where(ends, torch.zeros_like(nxt), nxt)
    return (gamma * nxt - now) / env.step_dt * (~cmd.settling).float()


def gentle_complete(env, command_name: str = "roll", window: float = 1.0,
                    zero_seconds: float = 2.0,
                    limb_free: float = 20.0, limb_zero: float = 100.0,
                    sensor_free: float = 2.0, sensor_zero: float = 20.0,
                    shell_free: float = 30.0, shell_zero: float = 150.0,
                    home_floor: float = 0.5, home_sigma_sq: float = 0.10,
                    asset_cfg: SceneEntityCfg | None = None,
                    min_own_turn: float = 1.0) -> torch.Tensor:
    """On the step the episode ends successfully (`window_done`): a duration factor
    (1 up to `window` s, 0 by `zero_seconds`) x one factor per contact kind x a
    finish factor (`home_floor` for standing at all, rising to 1 with the legs at
    HOME). 0 on every other step."""
    cmd = _cmd(env, command_name)
    done = window_done(env, command_name=command_name, window=window,
                       min_own_turn=min_own_turn).float()
    over = (cmd.roll_seconds() - window).clamp(min=0.0)
    duration = (1.0 - over / (zero_seconds - window)).clamp(0.0, 1.0)
    limb = torch.maximum(cmd.peak_limb, cmd.part_force(cmd.cfg.limb_sensor))
    sensor = torch.maximum(cmd.peak_sensor, cmd.part_force(cmd.cfg.sensor_sensor))
    shell = torch.maximum(cmd.peak_shell, cmd.part_force(cmd.cfg.shell_sensor))
    asset_cfg = asset_cfg or SceneEntityCfg("robot")
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    k = torch.exp(-((q - home) ** 2).mean(dim=1) / home_sigma_sq)
    finish = home_floor + (1.0 - home_floor) * k
    return (done * duration * finish
            * _factor(limb, limb_free, limb_zero)
            * _factor(sensor, sensor_free, sensor_zero)
            * _factor(shell, shell_free, shell_zero))


__all__ = ["gentle_complete", "shaped_progress", "window_done"]
