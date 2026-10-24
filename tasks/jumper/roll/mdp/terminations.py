"""How an episode of jumper.roll ends early: standing after a full roll.

**A true termination, not a time-out.** The actor cannot tell the start of the
roll from its end -- both are standing at HOME, and it has no clock -- so it is
never asked to do anything after the roll: the episode ends there, with the
completion bonus, and nothing is bootstrapped past it. On the robot the
controller has to leave the roll mode once it is done, as it leaves the jump's.
"""

from __future__ import annotations

import math

import torch

from .commands import TWO_PI


def standing_after_roll(
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
    """True when the body has turned within `turn_tolerance` of a full roll and
    is standing: tilt under `max_tilt`, at least `min_feet` feet loaded over
    `foot_force` N, and turning slower than `max_ang_vel` rad/s."""
    cmd = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    g = robot.data.projected_gravity_b
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    f = env.scene[sensor_name].data.force
    assert f is not None
    feet = (f.norm(dim=-1) > foot_force).sum(dim=1)
    w = robot.data.root_link_ang_vel_b.norm(dim=1)
    return (
        (cmd.total_now() >= TWO_PI - turn_tolerance)
        & (tilt < max_tilt)
        & (feet >= min_feet)
        & (w < max_ang_vel)
        # A spawn that falls onto its feet by itself has not rolled: the settle
        # window has to be over, and the policy has to have turned the body
        # `min_own_turn` radians itself. With the window cut to 5 steps a spawn
        # near 300 degrees can tip onto its feet just after it.
        & ~cmd.settling
        & (cmd.total_now() - cmd.spawn >= min_own_turn)
    )


__all__ = ["standing_after_roll"]
