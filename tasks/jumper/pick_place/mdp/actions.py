"""The policy's joint targets, with the claw taken over during a held spawn.

A held spawn (`events.HeldSpawn`) starts the episode with the debris already in
the claw, the way the five-foot task's `play --hold` does: put in the mouth with
the jaws apart, then squeezed shut on it once it is let go.

**Pre-closed, not wide open.** Held 50 mm up at the stow, debris let go of in a
wide-open claw falls out before the finger has crossed the 73 mm mouth -- measured,
a 24 mm block was on the floor 4 control steps after release. So while it is
put in place the jaws are brought to just clear of it (`PRECLOSE_EXTRA`), never
touching it (closing on a body whose pose is being written throws it: the
five-foot task's `PLACE_STEPS`), and the squeeze after release has millimetres
to travel. The action term
writes every target on every physics substep, so a target written by an event
would be overwritten before it acted -- the claw has to be taken over *here*.
Everything but the finger stays the policy's throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from ...five_foot.claw import APERTURE_MM, FINGER_JOINT, GRIPPER_CLOSED
from ...five_foot.mdp.gripper import squeeze_limited

#: Opening added to the debris's width while it is put in the claw, m, read off
#: `APERTURE_MM` at the anvil. The mouth is a wedge, so at the seat's depth it is
#: narrower than the table says: measured in plain MuJoCo with the debris seated
#: by `events.HeldSpawn`, a 2 mm gap to the moving jaw needed the table's aperture
#: to be the width plus 10.7 to 16.0 mm, over widths 22 to 58 mm. 17 keeps at
#: least 2 mm everywhere.
PRECLOSE_EXTRA = 0.017


def finger_for_aperture(mm: torch.Tensor) -> torch.Tensor:
    """The finger angle that leaves `mm` of mouth at the anvil: `APERTURE_MM`
    inverted, linear between its points and clamped to its ends."""
    table = torch.tensor(APERTURE_MM, device=mm.device, dtype=mm.dtype)
    angle, gap = table[:, 0].flip(0), table[:, 1].flip(0)  # gap ascending
    x = mm.clamp(min=float(gap[0]), max=float(gap[-1]))
    i = torch.searchsorted(gap, x).clamp(1, len(gap) - 1)
    g0, g1, a0, a1 = gap[i - 1], gap[i], angle[i - 1], angle[i]
    return a0 + (a1 - a0) * (x - g0) / (g1 - g0)


@dataclass(kw_only=True)
class HeldSpawnJointPositionActionCfg(JointPositionActionCfg):
    state_name: str = "task"
    #: Control steps the debris is placed for, claw open; then it is squeezed
    #: until `PickPlaceStateCfg.held_steps`.
    place_steps: int = 10

    def build(self, env) -> HeldSpawnJointPositionAction:
        return HeldSpawnJointPositionAction(self, env)


class HeldSpawnJointPositionAction(JointPositionAction):
    cfg: HeldSpawnJointPositionActionCfg

    def __init__(self, cfg: HeldSpawnJointPositionActionCfg, env):
        super().__init__(cfg, env)
        names = list(self._target_names)
        self._finger_col = names.index(FINGER_JOINT)
        self._finger_id = int(self._target_ids[self._finger_col])

    def apply_actions(self) -> None:
        target = self._processed_actions.clone()
        state = self._env.command_manager.get_term(self.cfg.state_name)
        settling = state.settling
        if bool(settling.any()):
            steps = self._env.episode_length_buf
            placing = settling & (steps <= self.cfg.place_steps)
            angle = self._entity.data.joint_pos[:, [self._finger_id]]
            speed = self._entity.data.joint_vel[:, [self._finger_id]]
            closed = torch.full_like(angle, GRIPPER_CLOSED)
            # Shut at full speed until a jaw is on the debris, then the operator's
            # squeeze. The squeeze alone is too slow here: its 0.6 N*m against the
            # servo's damping closes at about 1.2 rad/s, and debris let go of 50 mm
            # up was on the floor before the finger had crossed the last 15 mm.
            squeeze = torch.where(
                state.touching().unsqueeze(1),
                squeeze_limited(closed, angle, speed=speed),
                closed,
            ).squeeze(1)
            width = 2.0 * state.half()[:, 0] + PRECLOSE_EXTRA
            preclosed = finger_for_aperture(width * 1000.0)
            finger = torch.where(placing, preclosed, squeeze)
            target[:, self._finger_col] = torch.where(
                settling, finger, target[:, self._finger_col]
            )
        bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target - bias, joint_ids=self._target_ids)


__all__ = ["HeldSpawnJointPositionAction", "HeldSpawnJointPositionActionCfg"]
