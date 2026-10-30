"""jumper.horn_roll's tracker, with two more of the episode's worsts.

- `peak_arm`: floor force on the arms (both claws, shoulder to finger, the palm pads
  excepted), which this task asks to carry the robot over its nose.
- `peak_front`: floor force on the front of the body -- stock shell, base, sensors and
  carapace -- at contact points more than `front_x` ahead of the base origin in the
  body frame. The display, ToF sensor and camera sit 8-9 cm ahead of it, the carapace's
  front edge just behind them.

`front_force` reads the sensor's contact positions, so one geom (the base, the upper
shell) counts only where it actually touches: its back on the floor is not its front.

- `rot_sum`, `axis_error_deg`, `off_axis_rate`: which way the roll is going, for a
  task that holds it to the demonstration's axis (jumper.spin_roll_axis).
- `peak_load`: the episode's worst floor-contact joint torque |J^T f|, N*m, from
  `common/floor_load.py` (`floor_load`), sampled at every physics step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from ...common.floor_load import FloorJointLoad
from ...horn_roll.mdp.tracker import HornRollTracker, HornRollTrackerCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TuckRollTrackerCfg(HornRollTrackerCfg):
    arm_sensor: str = "arm_ground_contact"
    front_sensor: str = "front_ground_contact"
    #: Contacts this far ahead of the base origin (body x, m) count as the front.
    front_x: float = 0.04
    load_sensor: str = "leg_floor_contact"
    #: The driven joints `floor_joint_load` reports, in this order.
    load_joints: tuple[str, ...] = ()

    def build(self, env: ManagerBasedRlEnv) -> TuckRollTracker:
        return TuckRollTracker(self, env)


class TuckRollTracker(HornRollTracker):
    cfg: TuckRollTrackerCfg

    def __init__(self, cfg: TuckRollTrackerCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        self.peak_arm = torch.zeros(n, device=self.device)
        self.peak_front = torch.zeros(n, device=self.device)
        self.peak_load = torch.zeros(n, device=self.device)
        self.metrics["peak_arm_N"] = torch.zeros(n, device=self.device)
        self.metrics["peak_front_N"] = torch.zeros(n, device=self.device)
        self.metrics["peak_joint_load_Nm"] = torch.zeros(n, device=self.device)
        self.floor_load = FloorJointLoad(env, cfg.load_sensor, cfg.load_joints,
                                         cfg.entity_name)
        #: The body-frame rotation accumulated while between belly-up and upright, rad:
        #: its horizontal direction is the axis this episode rolled about.
        self.rot_sum = torch.zeros(n, 3, device=self.device)

    def substep_joint_load(self) -> torch.Tensor:
        """This physics step's floor load per load joint, N*m, [B, J]."""
        return self.floor_load.substep()

    def floor_joint_load(self) -> torch.Tensor:
        """The worst floor load per load joint over this control step, N*m, [B, J]."""
        return self.floor_load.step()

    def off_axis_rate(self) -> torch.Tensor:
        """The body's horizontal rotation rate about axes other than the
        demonstration's, rad/s: |w_xy - (w_xy . a) a|, a the demonstration's axis."""
        w = self.robot.data.root_link_ang_vel_b[:, :2]
        a = self.axis[:2] / self.axis[:2].norm()
        return (w - (w @ a).unsqueeze(1) * a).norm(dim=1)

    def axis_error_deg(self) -> torch.Tensor:
        """The angle between the axis rolled about so far this episode (`rot_sum`) and
        the demonstration's, either way round, degrees; 0 before any rolling."""
        r = self.rot_sum[:, :2]
        a = self.axis[:2] / self.axis[:2].norm()
        n = r.norm(dim=1)
        cos = ((r @ a).abs() / n.clamp(min=1e-6)).clamp(max=1.0)
        return torch.where(n > 0.3, torch.rad2deg(torch.acos(cos)), torch.zeros_like(n))

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
        self.peak_load[env_ids] = 0.0
        self.floor_load.reset(env_ids)
        self.rot_sum[env_ids] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["peak_arm_N"] = self.peak_arm.clone()
        self.metrics["peak_front_N"] = self.peak_front.clone()
        self.metrics["peak_joint_load_Nm"] = self.peak_load.clone()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        lf = (~self.settling).float()
        self.peak_arm.copy_(torch.maximum(self.peak_arm, self.force(self.cfg.arm_sensor) * lf))
        self.peak_front.copy_(torch.maximum(self.peak_front, self.front_force() * lf))
        self.peak_load.copy_(torch.maximum(self.peak_load,
                                           self.floor_joint_load().max(dim=1).values * lf))
        self.floor_load.end_step()
        up = -self.robot.data.projected_gravity_b[:, 2]
        mid = ((up > -0.9) & (up < 0.9) & ~self.settling).float().unsqueeze(1)
        self.rot_sum.add_(self.robot.data.root_link_ang_vel_b * mid * self._env.step_dt)


__all__ = ["TuckRollTracker", "TuckRollTrackerCfg"]
