"""The rewards of jumper.horned_roll that are its own.

## What changes from jumper.gentle_momentum_roll, and why

Read from that task as it stood on 2026-10-28 (`carapace-design`, commit fc77cee):

- **The carapace was charged as if it were the stock shell.** One sensor summed the
  stock shell, the base and all 48 carapace pieces; over 30 N it was charged per step,
  and the episode's hardest such contact cut the bonus from 30 N to nothing at 150 N.
  The horns and their supports are built to take hits the stock shell is not, and the
  roll is expected to land on them, so contact with them was among the most expensive
  things a policy could do. The other task's own keyframe search on v2
  (`studies/roll_v2/logs`) shows the trade: its most reliable plan (30/32) never touched
  the carapace and had the highest joint torque of the three, 14.5 N*m against the stock
  high jump's 5.9.
- **Leaving the floor was cheap, and landing on the feet cheaper.** Airtime cost about
  0.04 per step, and foot impacts were charged per step only, never through the bonus.
  Dodging a 130 N carapace hit could save 5-10 of the bonus. So the cheapest way to
  look gentle was to hop over the carapace and land on the feet, which is a harder hit
  than rolling over it.

## The bonus here

    duration x finish x (gentle_floor + (1 - gentle_floor) x gentleness)

duration, finish and gentle_floor are jumper.gentle_momentum_roll's. **gentleness is the
lowest** of six factors, each 1 at or below its `*_free` and 0 at its `*_zero`:

- limbs (other than the feet), sensors: as before;
- the stock shell and base, on their own;
- the hardest carapace region, with its own, much wider band;
- the hardest total floor force, every part summed, feet included: a hard landing
  costs the same whichever part touches down first, so dodging the carapace by landing
  on the feet gains nothing;
- the longest flight, in seconds: a lift of a step or two is free, a real hop is not.

The lowest, not the product: the other task's first run multiplied its factors and
left the policy almost nothing (its `mdp/rewards.py`); with six factors a product would
be worse. The worst offence decides; the per-step charges give the rest a gradient.
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
    """Standing after the roll, and at least `window` seconds since it began
    (jumper.gentle_momentum_roll's, copied). Half a control step of slack."""
    cmd = _cmd(env, command_name)
    standing = standing_done(env, command_name=command_name, min_own_turn=min_own_turn)
    late_enough = cmd.roll_seconds() >= window - 0.5 * env.step_dt
    return standing & late_enough & cmd.begun


def horned_complete(env, command_name: str = "roll", window: float = 1.0,
                    zero_seconds: float = 2.0, gentle_floor: float = 0.5,
                    limb_free: float = 20.0, limb_zero: float = 100.0,
                    sensor_free: float = 2.0, sensor_zero: float = 20.0,
                    shell_free: float = 30.0, shell_zero: float = 150.0,
                    carapace_free: float = 80.0, carapace_zero: float = 250.0,
                    ground_free: float = 100.0, ground_zero: float = 250.0,
                    flight_free: float = 0.06, flight_zero: float = 0.20,
                    home_floor: float = 0.5, home_sigma_sq: float = 0.10,
                    asset_cfg: SceneEntityCfg | None = None,
                    min_own_turn: float = 1.0) -> torch.Tensor:
    """On the step the episode ends successfully (`window_done`), the bonus in the
    module docstring; 0 on every other step. Peaks include the current step."""
    cmd = _cmd(env, command_name)
    done = window_done(env, command_name=command_name, window=window,
                       min_own_turn=min_own_turn).float()
    over = (cmd.roll_seconds() - window).clamp(min=0.0)
    duration = (1.0 - over / (zero_seconds - window)).clamp(0.0, 1.0)

    limb = torch.maximum(cmd.peak_limb, cmd.part_force(cmd.cfg.limb_sensor))
    sensor = torch.maximum(cmd.peak_sensor, cmd.part_force(cmd.cfg.sensor_sensor))
    shell = torch.maximum(cmd.peak_shell, cmd._force(cmd.cfg.shell_sensor))
    carapace = torch.maximum(cmd.peak_carapace, cmd.carapace_force())
    ground = torch.maximum(cmd.peak_ground, cmd.ground_force())
    flight = cmd.longest_flight * env.step_dt
    gentleness = torch.stack([
        _factor(limb, limb_free, limb_zero),
        _factor(sensor, sensor_free, sensor_zero),
        _factor(shell, shell_free, shell_zero),
        _factor(carapace, carapace_free, carapace_zero),
        _factor(ground, ground_free, ground_zero),
        _factor(flight, flight_free, flight_zero),
    ], dim=1).min(dim=1).values

    asset_cfg = asset_cfg or SceneEntityCfg("robot")
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    k = torch.exp(-((q - home) ** 2).mean(dim=1) / home_sigma_sq)
    finish = home_floor + (1.0 - home_floor) * k
    return done * duration * finish * (gentle_floor + (1.0 - gentle_floor) * gentleness)


def _overload(force: torch.Tensor, threshold: float, cap: float) -> torch.Tensor:
    """jumper.flourish_roll's contact charge: ((F - threshold) / threshold)^2, capped."""
    return (((force - threshold).clamp(min=0.0) / threshold) ** 2).clamp(max=cap)


def carapace_overload(env, threshold: float, cap: float = 10.0,
                      command_name: str = "roll") -> torch.Tensor:
    """Per step, the charge on each carapace region over `threshold` N, summed over the
    regions; capped at `cap`. Zero in a spawn's settle window."""
    cmd = _cmd(env, command_name)
    per = torch.stack([_overload(cmd._force(s), threshold, cap)
                       for s in cmd.cfg.region_sensors], dim=1).sum(dim=1)
    return per.clamp(max=cap) * (~cmd.settling).float()


def ground_overload(env, threshold: float, cap: float = 10.0,
                    command_name: str = "roll") -> torch.Tensor:
    """Per step, the charge on the total floor force over `threshold` N, every part
    summed, feet included. Zero in a spawn's settle window."""
    cmd = _cmd(env, command_name)
    return _overload(cmd.ground_force(), threshold, cap) * (~cmd.settling).float()


__all__ = ["carapace_overload", "ground_overload", "horned_complete", "window_done"]
