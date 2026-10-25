"""How an episode of jumper.momentum_roll ends early: standing, done.

A roll episode ends as jumper.roll's does (`standing_after_roll`: a full turn,
level, four feet loaded, still). A self-righting episode ends the same way except
that the turn may be either way round: within `turn_tolerance` of upright, after
at least `min_own_turn` radians of the policy's own.
"""

from __future__ import annotations

import math

import torch

from ...roll.mdp.terminations import standing_after_roll


def standing_done(
    env,
    command_name: str = "roll",
    sensor_name: str = "feet_ground_contact",
    turn_tolerance: float = 0.3,
    max_tilt: float = math.radians(20.0),
    min_feet: int = 4,
    max_ang_vel: float = 2.0,
    foot_force: float = 1.0,
    min_own_turn: float = 1.0,
) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    roll = standing_after_roll(
        env, command_name=command_name, sensor_name=sensor_name,
        turn_tolerance=turn_tolerance, max_tilt=max_tilt, min_feet=min_feet,
        max_ang_vel=max_ang_vel, foot_force=foot_force, min_own_turn=min_own_turn,
    )
    robot = env.scene["robot"]
    g = robot.data.projected_gravity_b
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    f = env.scene[sensor_name].data.force
    assert f is not None
    feet = (f.norm(dim=-1) > foot_force).sum(dim=1)
    w = robot.data.root_link_ang_vel_b.norm(dim=1)
    own = (cmd.total_now() - cmd.spawn).abs() >= min_own_turn
    right = ((cmd.upright_gap() < turn_tolerance) & (tilt < max_tilt) & (feet >= min_feet)
             & (w < max_ang_vel) & own & ~cmd.settling)
    return torch.where(cmd.self_right, right, roll)


__all__ = ["standing_done"]
