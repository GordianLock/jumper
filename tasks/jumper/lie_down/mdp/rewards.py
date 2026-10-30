"""Rewards and termination for jumper.lie_down.

- `lower`: new ground in how far onto its back the body is (`tracker.down_now`), per
  second, above the episode's high-water mark: standing to flat on its back is 2 units.
  Paid by the distance, not the speed, so going faster earns nothing more.
- `on_back`, the termination: `tracker.done` -- belly-up, still and legs at HOME,
  held. `gentle_complete` pays on that step, `gentle_floor` of it however it went and
  the rest scaled by the worst factor, as jumper.self_right's.
- `back_home`: while belly-up, legs near HOME (self_right's `finish_home` mirrored).
- `spin`: per step, body rotation rate over a free level.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...self_right.mdp.rewards import _cmd, _factor


def lower(env, command_name: str = "roll") -> torch.Tensor:
    """New ground in `down` this step, per second (the reward is scaled by the step)."""
    cmd = _cmd(env, command_name)
    gain = (cmd.down_now() - cmd.best_down).clamp(min=0.0)
    return gain / env.step_dt * (~cmd.settling).float()


def on_back(env, command_name: str = "roll") -> torch.Tensor:
    return _cmd(env, command_name).done


def gentle_complete(env, command_name: str = "roll", gentle_floor: float = 0.5,
                    limb: tuple[float, float] = (20.0, 100.0),
                    sensor: tuple[float, float] = (2.0, 20.0),
                    shell_impact: tuple[float, float] = (25.0, 75.0),
                    ground_impact: tuple[float, float] = (40.0, 150.0),
                    flight: tuple[float, float] = (0.0, 0.06),
                    load: tuple[float, float] = (4.0, 8.0),
                    spin: tuple[float, float] | None = None) -> torch.Tensor:
    """On the finishing step: `gentle_floor` + (1 - `gentle_floor`) x the lowest factor,
    each 1 at or below its band's free level and 0 at its zero level. With `spin`, the
    episode's fastest body rotation (rad/s) is a factor too."""
    cmd = _cmd(env, command_name)
    done = on_back(env, command_name).float()
    worst = {
        "limb": torch.maximum(cmd.peak_limb, cmd.force(cmd.cfg.limb_sensor)),
        "sensor": torch.maximum(cmd.peak_sensor, cmd.force(cmd.cfg.sensor_sensor)),
        "shell_impact": torch.maximum(cmd.peak_shell_impact, cmd.impact(cmd.cfg.shell_sensor)),
        "ground_impact": torch.maximum(cmd.peak_ground_impact,
                                       cmd.impact(cmd.cfg.ground_sensor)),
        "flight": cmd.longest_flight * env.step_dt,
        "load": torch.maximum(cmd.peak_load, cmd.floor_load.worst()),
    }
    bands = {"limb": limb, "sensor": sensor, "shell_impact": shell_impact,
             "ground_impact": ground_impact, "flight": flight, "load": load}
    if spin is not None:
        worst["spin"] = cmd.peak_spin
        bands["spin"] = spin
    g = torch.stack([_factor(worst[k], *bands[k]) for k in bands], dim=1).min(dim=1).values
    return done * (gentle_floor + (1.0 - gentle_floor) * g)


def back_home(env, asset_cfg: SceneEntityCfg, command_name: str = "roll",
              min_down: float = 0.9, sigma_sq: float = 0.10) -> torch.Tensor:
    """exp(-e / sigma_sq), e the driven joints' mean squared distance from HOME, while
    belly-up (`down` > `min_down`); 0 otherwise."""
    cmd = _cmd(env, command_name)
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    k = torch.exp(-((q - home) ** 2).mean(dim=1) / sigma_sq)
    return k * ((cmd.down_now() > min_down) & ~cmd.settling).float()


def spin(env, free: float, scale: float = 2.0, cap: float = 4.0,
         command_name: str = "roll") -> torch.Tensor:
    """Per step: ((w - free) / scale)^2, capped, w the body's rotation rate, rad/s."""
    cmd = _cmd(env, command_name)
    w = env.scene["robot"].data.root_link_ang_vel_b.norm(dim=1)
    return (((w - free).clamp(min=0.0) / scale) ** 2).clamp(max=cap) * (~cmd.settling).float()


__all__ = ["back_home", "gentle_complete", "lower", "on_back", "spin"]
