"""Absolute joint targets around HOME, with the demonstration fed forward.

While an episode is inside the demonstration (`MomentumRollTracker.in_demo`), the
recording's target for this moment is added to the policy's, times a factor that
is 1 until `hold_step` and 0 from `end_step`. At the start a zero action replays
the demonstration; by the end the policy makes the roll alone. This is
`ref_free_jump/mdp/actions.py`'s scheme, read by time rather than by a go clock.

**Off in replay** (`enabled=False`): replay starts the step counter at 0, which
would turn the feedforward fully on and show the demonstration rather than the
policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class FeedforwardRollActionCfg(JointPositionActionCfg):
    command_name: str = "roll"
    hold_step: int
    end_step: int
    enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.use_default_offset:
            raise ValueError("the feedforward is relative to HOME: use_default_offset must be True")
        if not 0 <= self.hold_step < self.end_step:
            raise ValueError(f"need 0 <= hold_step < end_step, got {self.hold_step}, {self.end_step}")

    def build(self, env) -> FeedforwardRollAction:
        return FeedforwardRollAction(self, env)


class FeedforwardRollAction(JointPositionAction):
    cfg: FeedforwardRollActionCfg

    @property
    def feedforward(self) -> float:
        if not self.cfg.enabled:
            return 0.0
        s = self._env.common_step_counter
        frac = (s - self.cfg.hold_step) / (self.cfg.end_step - self.cfg.hold_step)
        return 1.0 - min(max(frac, 0.0), 1.0)

    def apply_actions(self) -> None:
        target = self._processed_actions
        ff = self.feedforward
        if ff > 0.0:
            cmd = self._env.command_manager.get_term(self.cfg.command_name)
            ref = cmd.reference
            q_cmd = ref.q_cmd[ref.index(cmd.demo_t.clamp(min=0.0))]
            target = target + ff * (q_cmd - self._offset) * cmd.in_demo.unsqueeze(1).float()
        bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target - bias, joint_ids=self._target_ids)


__all__ = ["FeedforwardRollAction", "FeedforwardRollActionCfg"]
