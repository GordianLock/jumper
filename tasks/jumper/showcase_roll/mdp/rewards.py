"""The two terms jumper.showcase_roll adds to jumper.flourish_roll's.

## Why the completion bonus is scaled

jumper.flourish_roll (jumper5) paid everything per step, and learnt a fast, plain
roll: 0.76 s, 16 of 16 completed, its moving-limb reward falling to a quarter of
its early value. Two reasons, both structural:

- **Discounting favours finishing early.** At gamma 0.995 and 50 Hz a bonus 4.2 s
  away is worth 0.35 of one 0.8 s away, which undid what the progress cap paid a
  slow roll.
- **A hit lasts one control step.** A 100 N slam of a thigh cost at most 10 x 0.02
  = 0.2 under the capped per-step charge, against a 15 bonus. jumper5's middle-left
  thigh still put a median 4.6 N*m peak on a joint (5.2 worst).

So the bonus itself now carries both: it is multiplied by the fraction of
`full_seconds` the roll took (from `roll_start`), and by a safety factor that
falls linearly from 1 at `free` newtons to 0 at `zero` newtons of the hardest
single limb contact of the episode -- and the same for the sensors.

`stall` keeps "slow" from meaning "lies on its back and then finishes": mid-roll,
once the roll has made no new ground for `after` seconds, it costs 1 per second.
"""

from __future__ import annotations

import torch

from ...momentum_roll.mdp.terminations import standing_done


def _factor(peak: torch.Tensor, free: float, zero: float) -> torch.Tensor:
    return (1.0 - (peak - free) / (zero - free)).clamp(0.0, 1.0)


def scaled_complete(env, command_name: str = "roll", full_seconds: float = 5.0,
                    limb_free: float = 20.0, limb_zero: float = 100.0,
                    sensor_free: float = 2.0, sensor_zero: float = 20.0,
                    min_own_turn: float = 1.0) -> torch.Tensor:
    """On the step the robot stands after the roll: duration factor x limb factor x
    sensor factor, each 0-1. 0 on every other step."""
    cmd = env.command_manager.get_term(command_name)
    done = standing_done(env, command_name=command_name, min_own_turn=min_own_turn).float()
    duration = (cmd.roll_seconds() / full_seconds).clamp(0.0, 1.0)
    limb = torch.maximum(cmd.peak_limb, cmd.part_force(cmd.cfg.limb_sensor))
    sensor = torch.maximum(cmd.peak_sensor, cmd.part_force(cmd.cfg.sensor_sensor))
    return done * duration * _factor(limb, limb_free, limb_zero) * _factor(
        sensor, sensor_free, sensor_zero)


def stall(env, command_name: str = "roll", after: float = 1.0, margin: float = 0.3) -> torch.Tensor:
    """1 while mid-roll and the roll has gained no new ground for `after` seconds."""
    cmd = env.command_manager.get_term(command_name)
    since = (env.episode_length_buf.float() - cmd.last_gain) * env.step_dt
    total = cmd.total_now()
    mid = (total > margin) & (total < 2 * torch.pi - margin) & (cmd.roll_start >= 0)
    return (mid & (since > after)).float() * (~cmd.settling).float()


__all__ = ["scaled_complete", "stall"]
