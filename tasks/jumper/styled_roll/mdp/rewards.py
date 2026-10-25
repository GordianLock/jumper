"""The one reward jumper.styled_roll adds to jumper.momentum_roll's: the base's height
along the demonstration.

The joint and turn priors say what the legs do and where the body is in the roll;
neither sees the demonstration's hop -- the base rising to 0.21 m as it comes over
onto its feet, from 0.11 standing. That hop is a large part of how the roll looks.
"""

from __future__ import annotations

import torch


def prior_height(env, command_name: str = "roll", sigma_sq: float = 0.0025) -> torch.Tensor:
    """exp(-(height error)^2 / sigma_sq) for the base origin above the floor, against
    the demonstration at this moment, inside it; 0 outside."""
    cmd = env.command_manager.get_term(command_name)
    ref = cmd.reference
    robot = env.scene["robot"]
    want = ref.root_pos[ref.index(cmd.demo_t.clamp(min=0.0)), 2]
    have = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    e = (have - want) ** 2
    return torch.exp(-e / sigma_sq) * cmd.in_demo.float() * (~cmd.settling).float()


__all__ = ["prior_height"]
