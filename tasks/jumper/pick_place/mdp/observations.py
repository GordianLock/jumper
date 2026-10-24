"""What the policy is told about the debris and the receptacle.

**These are not encoder readings.** On the robot they would come from its camera
and depth sensor -- an object detector giving the debris's position, footprint
and heading, and the receptacle's position and rim. The deployed controller
builds no such terms today (`scripts/export.py::_DEPLOY_TERMS`), so a policy from
this task trains and replays in simulation but cannot be exported to the robot
until a perception term is added to it. Everything is in the robot's base frame,
which is how a body-mounted sensor would report it.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse


def _to_base(env, world: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_link_quat_w, world - robot.data.root_link_pos_w)


def debris_obs(env, state_name: str = "task") -> torch.Tensor:
    """[N, 8]: centre in the base frame (3), heading as (cos 2psi, sin 2psi) (2),
    half-sizes (3)."""
    s = env.command_manager.get_term(state_name)
    return torch.cat([_to_base(env, s.debris_pos()), s.grip_axis_b(), s.half()], dim=1)


def receptacle_obs(env, state_name: str = "task") -> torch.Tensor:
    """[N, 4]: the tray's centre on the floor in the base frame (3), its rim
    height above the floor (1) -- 0 for a pile spot."""
    s = env.command_manager.get_term(state_name)
    return torch.cat([_to_base(env, s.bin_pos), s.rim.unsqueeze(1)], dim=1)


def task_state_obs(env, state_name: str = "task") -> torch.Tensor:
    """Critic only, [N, 9]: mouth-to-debris in the base frame (3), the debris's
    velocity in the base frame (3), gripped, lifted this episode, in the tray."""
    s = env.command_manager.get_term(state_name)
    robot = env.scene["robot"]
    to_debris = quat_apply_inverse(robot.data.root_link_quat_w, s.debris_pos() - s.mouth())
    vel = quat_apply_inverse(robot.data.root_link_quat_w, s.debris.data.root_link_lin_vel_w)
    flags = torch.stack(
        [s.grasped().float(), s.was_lifted.float(), s.in_bin().float()], dim=1
    )
    return torch.cat([to_debris, vel, flags], dim=1)


__all__ = ["debris_obs", "receptacle_obs", "task_state_obs"]
