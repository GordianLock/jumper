"""Critic-only observations for jumper.momentum_roll.

The actor sees its body and nothing else: no clock and no demonstration, so on
the robot the roll starts when the mode does. The critic is told what the
rewards are computed from.
"""

from __future__ import annotations

import torch

from ...roll.mdp.commands import TWO_PI


def roll_state(env, command_name: str = "roll") -> torch.Tensor:
    """[N, 4]: net turn as a fraction of a full roll; the demonstration's phase
    (0-1, or -1 outside it); self-righting episode; distance to upright / pi."""
    cmd = env.command_manager.get_term(command_name)
    phase = torch.where(cmd.in_demo, cmd.demo_t / cmd.reference.span,
                        torch.full_like(cmd.demo_t, -1.0))
    return torch.stack([cmd.total / TWO_PI, phase, cmd.self_right.float(),
                        cmd.upright_gap() / torch.pi], dim=1)


__all__ = ["roll_state"]
