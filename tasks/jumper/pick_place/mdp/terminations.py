"""How an episode of jumper.pick_place ends early.

- **placed**: the job is done -- a true termination, with the bonus, so nothing
  is asked of the policy afterwards. On the robot the controller leaves the mode.
- **debris_lost**: the debris has been knocked out of the claw's reach, or off
  the floor's plane (thrown). A failure.
- **fell**: tilted past the controller's own 50-degree cut-off, or the trunk
  down on the floor. A failure.
"""

from __future__ import annotations

import math

import torch


def placed(env, state_name: str = "task") -> torch.Tensor:
    s = env.command_manager.get_term(state_name)
    return s.placed() & ~s.settling


def debris_lost(env, state_name: str = "task", max_dist: float = 0.45,
                min_z: float = -0.05) -> torch.Tensor:
    s = env.command_manager.get_term(state_name)
    robot = env.scene["robot"]
    d = (s.debris_pos()[:, :2] - robot.data.root_link_pos_w[:, :2]).norm(dim=1)
    z = s.debris_pos()[:, 2] - env.scene.env_origins[:, 2]
    return (d > max_dist) | (z < min_z)


def fell(env, limit_angle: float = math.radians(50.0), min_height: float = 0.05) -> torch.Tensor:
    robot = env.scene["robot"]
    g = robot.data.projected_gravity_b
    tilted = torch.acos((-g[:, 2]).clamp(-1.0, 1.0)) > limit_angle
    low = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]) < min_height
    return tilted | low


__all__ = ["debris_lost", "fell", "placed"]
