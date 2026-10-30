"""The joint-position action, held at HOME while the robot settles.

jumper.self_right's first two policies used the settling window -- where nothing is
paid or charged -- to rock and pump on their backs, and launched from that swing
before the window ended: on run 2's nominal replay the body was turning 180-350 deg/s
throughout it and 1150 deg/s by its end (2026-10-29). Asked for: getting up from at
rest. So while the command term's `settling` is True the policy's action is replaced
by zero -- every driven joint's target is its HOME angle -- and the robot lies still on
its back until the tracker releases it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class HoldWhileSettlingActionCfg(JointPositionActionCfg):
    command_name: str = "roll"

    def build(self, env) -> HoldWhileSettlingAction:
        return HoldWhileSettlingAction(self, env)


class HoldWhileSettlingAction(JointPositionAction):
    cfg: HoldWhileSettlingActionCfg

    def process_actions(self, actions: torch.Tensor):
        cmd = self._env.command_manager.get_term(self.cfg.command_name)
        held = cmd.settling.unsqueeze(1)
        super().process_actions(torch.where(held, torch.zeros_like(actions), actions))


__all__ = ["HoldWhileSettlingAction", "HoldWhileSettlingActionCfg"]
