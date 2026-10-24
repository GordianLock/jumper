"""Critic-only observations for jumper.roll.

The value of a state depends on how much of the roll is left, which the actor
cannot see (standing at the start and standing at the end look the same). The
critic is given it.
"""

from __future__ import annotations

import torch

from .commands import TWO_PI


def roll_progress_obs(env, command_name: str = "roll") -> torch.Tensor:
    """The episode's net turn as a fraction of a full roll, [N, 1]."""
    cmd = env.command_manager.get_term(command_name)
    return (cmd.total / TWO_PI).unsqueeze(1)


__all__ = ["roll_progress_obs"]
