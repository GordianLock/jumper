"""Rewards and termination for jumper.self_right.

- `rise`: new ground in how upright the body is (`tracker.up_now`), per second, above
  the episode's high-water mark: from flat on its back to standing is 2 units. Paid
  for whatever route the robot takes; rocking earns nothing twice.
- `standing`, the termination: upright within `max_tilt`, at least `min_feet` feet
  loaded, turning slower than `max_ang_vel`, and risen at least `min_rise` from where
  it settled -- so a spawn that tips onto its feet by itself has not righted itself.
- `gentle_complete`: on that step, a bonus of which `gentle_floor` is paid for
  standing however hard it went, and the rest scaled by the lowest of five factors
  (limb and sensor force, shell and floor impact, time in the air), each 1 at or
  below its free level and 0 at its zero level. A bonus a hard landing could take to
  nothing teaches a policy to stay down; the lowest, not the product, so one bad
  number decides and six good ones are not multiplied away.
- `finish_home`: once upright, legs near HOME.
- `shell_impact`, `ground_impact`: per step, impact over a threshold.
"""

from __future__ import annotations

import math

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _cmd(env, name: str = "roll"):
    return env.command_manager.get_term(name)


def rise(env, command_name: str = "roll") -> torch.Tensor:
    """New ground in `up` this step, per second (the reward is scaled by the step)."""
    cmd = _cmd(env, command_name)
    gain = (cmd.up_now() - cmd.best).clamp(min=0.0)
    return gain / env.step_dt * (~cmd.settling).float()


def standing(env, command_name: str = "roll", sensor_name: str = "feet_ground_contact",
             max_tilt: float = math.radians(20.0), min_feet: int = 4,
             max_ang_vel: float = 2.0, foot_force: float = 1.0,
             min_rise: float = 1.0) -> torch.Tensor:
    """Standing, after rising `min_rise` itself, outside the settle window."""
    cmd = _cmd(env, command_name)
    up = cmd.up_now()
    f = env.scene[sensor_name].data.force
    assert f is not None
    feet = (f.norm(dim=-1) > foot_force).sum(dim=1)
    w = env.scene["robot"].data.root_link_ang_vel_b.norm(dim=1)
    risen = torch.maximum(cmd.best, up) - cmd.spawn_up >= min_rise
    return ((up > math.cos(max_tilt)) & (feet >= min_feet) & (w < max_ang_vel)
            & risen & ~cmd.settling)


def _factor(worst: torch.Tensor, free: float, zero: float) -> torch.Tensor:
    return (1.0 - (worst - free) / (zero - free)).clamp(0.0, 1.0)


def gentle_complete(env, command_name: str = "roll", gentle_floor: float = 0.5,
                    limb: tuple[float, float] = (20.0, 100.0),
                    sensor: tuple[float, float] = (2.0, 20.0),
                    shell_impact: tuple[float, float] = (50.0, 200.0),
                    ground_impact: tuple[float, float] = (60.0, 250.0),
                    flight: tuple[float, float] = (0.06, 0.20),
                    **stand) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    done = standing(env, command_name=command_name, **stand).float()
    worst = {
        "limb": torch.maximum(cmd.peak_limb, cmd.force(cmd.cfg.limb_sensor)),
        "sensor": torch.maximum(cmd.peak_sensor, cmd.force(cmd.cfg.sensor_sensor)),
        "shell_impact": torch.maximum(cmd.peak_shell_impact, cmd.impact(cmd.cfg.shell_sensor)),
        "ground_impact": torch.maximum(cmd.peak_ground_impact,
                                       cmd.impact(cmd.cfg.ground_sensor)),
        "flight": cmd.longest_flight * env.step_dt,
    }
    bands = {"limb": limb, "sensor": sensor, "shell_impact": shell_impact,
             "ground_impact": ground_impact, "flight": flight}
    g = torch.stack([_factor(worst[k], *bands[k]) for k in bands], dim=1).min(dim=1).values
    return done * (gentle_floor + (1.0 - gentle_floor) * g)


def finish_home(env, asset_cfg: SceneEntityCfg, command_name: str = "roll",
                min_up: float = 0.9, sigma_sq: float = 0.10) -> torch.Tensor:
    """exp(-e / sigma_sq), e the driven joints' mean squared distance from HOME, while
    the body is upright (`up` > `min_up`) after rising; 0 otherwise."""
    cmd = _cmd(env, command_name)
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    k = torch.exp(-((q - home) ** 2).mean(dim=1) / sigma_sq)
    up = (cmd.up_now() > min_up) & (cmd.best - cmd.spawn_up >= 1.0) & ~cmd.settling
    return k * up.float()


def _overload(force: torch.Tensor, threshold: float, cap: float) -> torch.Tensor:
    return (((force - threshold).clamp(min=0.0) / threshold) ** 2).clamp(max=cap)


def shell_impact(env, threshold: float, cap: float = 10.0,
                 command_name: str = "roll") -> torch.Tensor:
    cmd = _cmd(env, command_name)
    return _overload(cmd.impact(cmd.cfg.shell_sensor), threshold, cap) * \
        (~cmd.settling).float()


def ground_impact(env, threshold: float, cap: float = 10.0,
                  command_name: str = "roll") -> torch.Tensor:
    cmd = _cmd(env, command_name)
    return _overload(cmd.impact(cmd.cfg.ground_sensor), threshold, cap) * \
        (~cmd.settling).float()


def airborne(env, command_name: str = "roll") -> torch.Tensor:
    """1 while nothing of the robot touches the floor, outside the settle window."""
    cmd = _cmd(env, command_name)
    return (cmd.in_air() & ~cmd.settling).float()


__all__ = ["airborne", "finish_home", "gentle_complete", "ground_impact", "rise",
           "shell_impact", "standing"]
