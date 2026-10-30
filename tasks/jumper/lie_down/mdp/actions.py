"""The joint-position action for jumper.lie_down: held at HOME while settling, and
the demonstration fed forward after.

jumper.momentum_roll's `FeedforwardRollAction` (the recording's target for this
moment added to the policy's, times a factor fading from 1 to 0 over training, off
in replay) with jumper.self_right's settle hold: while the tracker's `settling` is
True the policy's action is replaced by zero, so the robot stands still at HOME
until released, and the demonstration's clock starts then.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...momentum_roll.mdp.actions import FeedforwardRollAction, FeedforwardRollActionCfg


@dataclass(kw_only=True)
class LieDownActionCfg(FeedforwardRollActionCfg):
    def build(self, env) -> LieDownAction:
        return LieDownAction(self, env)


class LieDownAction(FeedforwardRollAction):
    cfg: LieDownActionCfg

    def process_actions(self, actions: torch.Tensor):
        cmd = self._env.command_manager.get_term(self.cfg.command_name)
        held = cmd.settling.unsqueeze(1)
        super().process_actions(torch.where(held, torch.zeros_like(actions), actions))


__all__ = ["LieDownAction", "LieDownActionCfg"]
