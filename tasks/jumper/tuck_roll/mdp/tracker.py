"""jumper.horn_roll's tracker, with two more of the episode's worsts.

- `peak_arm`: floor force on the arms (both claws, shoulder to finger, the palm pads
  excepted), which this task asks to carry the robot over its nose.
- `peak_front`: floor force on the front of the body -- stock shell, base, sensors and
  carapace -- at contact points more than `front_x` ahead of the base origin in the
  body frame. The display, ToF sensor and camera sit 8-9 cm ahead of it, the carapace's
  front edge just behind them.

`front_force` reads the sensor's contact positions, so one geom (the base, the upper
shell) counts only where it actually touches: its back on the floor is not its front.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from ...horn_roll.mdp.tracker import HornRollTracker, HornRollTrackerCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TuckRollTrackerCfg(HornRollTrackerCfg):
    arm_sensor: str = "arm_ground_contact"
    front_sensor: str = "front_ground_contact"
    #: Contacts this far ahead of the base origin (body x, m) count as the front.
    front_x: float = 0.04

    def build(self, env: ManagerBasedRlEnv) -> TuckRollTracker:
        return TuckRollTracker(self, env)


class TuckRollTracker(HornRollTracker):
    cfg: TuckRollTrackerCfg

    def __init__(self, cfg: TuckRollTrackerCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        self.peak_arm = torch.zeros(n, device=self.device)
        self.peak_front = torch.zeros(n, device=self.device)
        self.metrics["peak_arm_N"] = torch.zeros(n, device=self.device)
        self.metrics["peak_front_N"] = torch.zeros(n, device=self.device)

    def front_force(self) -> torch.Tensor:
        """Floor force on the front of the body now, N (module docstring)."""
        data = self._env.scene[self.cfg.front_sensor].data
        f, p = data.force, data.pos                            # [B, N, 3], world pos
        assert f is not None and p is not None
        root = self.robot.data.root_link_pos_w.unsqueeze(1)
        q = self.robot.data.root_link_quat_w.unsqueeze(1).expand(-1, p.shape[1], -1)
        x = quat_apply_inverse(q.reshape(-1, 4), (p - root).reshape(-1, 3)) \
            .reshape(p.shape)[..., 0]
        mag = f.norm(dim=-1)
        return (mag * (x > self.cfg.front_x).float()).sum(dim=1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self.peak_arm[env_ids] = 0.0
        self.peak_front[env_ids] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["peak_arm_N"] = self.peak_arm.clone()
        self.metrics["peak_front_N"] = self.peak_front.clone()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        lf = (~self.settling).float()
        self.peak_arm.copy_(torch.maximum(self.peak_arm, self.force(self.cfg.arm_sensor) * lf))
        self.peak_front.copy_(torch.maximum(self.peak_front, self.front_force() * lf))


__all__ = ["TuckRollTracker", "TuckRollTrackerCfg"]
