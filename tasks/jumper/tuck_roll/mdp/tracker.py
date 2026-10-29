"""jumper.horn_roll's tracker, with two more of the episode's worsts.

- `peak_arm`: floor force on the arms (both claws, shoulder to finger, the palm pads
  excepted), which this task asks to carry the robot over its nose.
- `peak_front`: floor force on the front of the body -- stock shell, base, sensors and
  carapace -- at contact points more than `front_x` ahead of the base origin in the
  body frame. The display, ToF sensor and camera sit 8-9 cm ahead of it, the carapace's
  front edge just behind them.

`front_force` reads the sensor's contact positions, so one geom (the base, the upper
shell) counts only where it actually touches: its back on the floor is not its front.

- `peak_load`: the largest torque floor contact puts on any driven joint, |J^T f|, N*m.
  For a hinge, a contact force F at p loads each joint up the chain by
  axis . ((p - anchor) x F), so `floor_joint_load` sums that over the legs' floor
  contacts (`load_sensor`: every leg geom against the floor, two contacts per geom,
  world-frame forces and positions) for every driven joint above the touching link.
  Read over the control step's substeps with this step's contact positions, or the
  last step's for a contact that ended within this one; a contact that began and ended
  within one step is missed. Checked against MuJoCo's own
  `mj_mulJacTVec` on a replay (`env_cfg.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

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
        self._chain: tuple | None = None
        self._last_pos: torch.Tensor | None = None
        self._last_found: torch.Tensor | None = None

    def _build_chain(self) -> tuple:
        """Per load-sensor contact slot, which joints it loads; the joints' bodies,
        anchors and axes in their body frames."""
        m = self._env.sim.mj_model
        sensor = self._env.scene[self.cfg.load_sensor]
        slots = len(sensor.data.found[0]) // len(sensor.primary_names)

        def mj(kind, name):
            for n in (name, f"{self.cfg.entity_name}/{name}"):
                try:
                    return getattr(m, kind)(n)
                except KeyError:
                    continue
            raise KeyError(name)

        joints = [mj("joint", j) for j in self.cfg.load_joints]
        jbody = [int(j.bodyid[0]) for j in joints]
        mask = torch.zeros(len(sensor.primary_names) * slots, len(joints))
        for i, name in enumerate(sensor.primary_names):
            b, chain = int(mj("geom", name.split("/")[-1]).bodyid[0]), set()
            while b > 0:
                chain.add(b)
                b = int(m.body_parentid[b])
            for k, jb in enumerate(jbody):
                if jb in chain:
                    mask[i * slots:(i + 1) * slots, k] = 1.0
        dev = self.device
        return (torch.tensor(jbody, device=dev), mask.to(dev),
                torch.tensor(np.array([j.pos for j in joints]), dtype=torch.float32, device=dev),
                torch.tensor(np.array([j.axis for j in joints]), dtype=torch.float32, device=dev))

    def floor_joint_load(self) -> torch.Tensor:
        """|J^T f| from floor contact per load joint, worst over this step's substeps,
        N*m, [B, J] (module docstring)."""
        if self._chain is None:
            self._chain = self._build_chain()
        jbody, mask, jpos, jaxis = self._chain
        data = self._env.scene[self.cfg.load_sensor].data
        sim = self._env.sim.data
        bq = sim.xquat[:, jbody]                                       # [B, J, 4]
        flat = bq.reshape(-1, 4)
        anchor = sim.xpos[:, jbody] + quat_apply(
            flat, jpos.expand(bq.shape[0], -1, -1).reshape(-1, 3)).reshape(bq.shape[0], -1, 3)
        axis = quat_apply(flat, jaxis.expand(bq.shape[0], -1, -1).reshape(-1, 3)) \
            .reshape(bq.shape[0], -1, 3)
        # A contact that ended within the step keeps the position it had at the end
        # of the step before; one that began and ended within the step has none.
        now = data.found > 0
        if self._last_pos is None:
            self._last_pos, self._last_found = data.pos.clone(), now.clone()
        pos = torch.where(now.unsqueeze(-1), data.pos, self._last_pos)
        live = (now | self._last_found).float().unsqueeze(-1)          # [B, N, 1]
        r = pos.unsqueeze(2) - anchor.unsqueeze(1)                     # [B, N, J, 3]
        f = data.force_history * live.unsqueeze(-1)                    # [B, N, H, 3]
        # tau[b, h, j] = sum_n mask[n, j] * axis_j . (r_nj x f_nh)
        cross = torch.linalg.cross(r.unsqueeze(2).expand(-1, -1, f.shape[2], -1, -1),
                                   f.unsqueeze(3).expand(-1, -1, -1, r.shape[2], -1), dim=-1)
        t = (cross * axis[:, None, None]).sum(-1) * mask[None, :, None, :]   # [B, N, H, J]
        return t.sum(1).abs().max(dim=1).values

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
        if self._last_found is not None:
            self._last_found[env_ids] = False

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
        data = self._env.scene[self.cfg.load_sensor].data
        self._last_pos, self._last_found = data.pos.clone(), data.found > 0


__all__ = ["TuckRollTracker", "TuckRollTrackerCfg"]
