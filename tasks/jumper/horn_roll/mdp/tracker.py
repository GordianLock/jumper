"""The roll tracker for jumper.horn_roll: jumper.momentum_roll's, measured about the
demonstration's own axis, plus what the gentleness terms read.

## The angle, about any horizontal body axis

jumper.roll measures the turn about the body's long axis, theta = atan2(-g_y, -g_z),
g the gravity direction in the base frame. For a turn theta about a horizontal body axis
`a`, gravity in the body frame is (sin theta) (a x z) - (cos theta) z, so

    theta = atan2(g . (a x z), -g_z)

which is jumper.roll's angle for a = x (a x z = -y) and the end-over-end angle
atan2(g_x, -g_z) for a = y. `a` is the demonstration's axis, stored with it (`axis`
in the npz, measured from the recording); the turn is positive the way it rolled.
The rest of the bookkeeping -- unwrapped total, high-water mark, settle window,
demonstration clock -- is jumper.momentum_roll's, unchanged.

## Contact, per episode

- `peak_limb`, `peak_sensor`, `peak_shell`: the hardest floor force on the limbs (feet
  excepted), the display / dToF / camera, and the **stock** shell and base.
- `peak_carapace_impact`: the hardest **impact** on any carapace region, and
  `peak_ground_impact` on the floor as a whole (every part summed). Impact is the
  largest rise in force within one control step (20 ms): each sensor keeps the step's
  four 5 ms substeps (`history_length`), and the step before's last is kept here, so
  a rise across the step boundary is seen too. A load built up over several steps -- a
  horn carrying weight as the robot leans and pushes on it -- is not an impact.
- `flight`, `longest_flight`: control steps with nothing on the floor, now and at most.

All of them skip a spawn's settle window, as jumper.momentum_roll's terms do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...momentum_roll.mdp.commands import MomentumRollTracker, MomentumRollTrackerCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HornRollTrackerCfg(MomentumRollTrackerCfg):
    limb_sensor: str = "limb_ground_contact"
    sensor_sensor: str = "sensor_ground_contact"
    shell_sensor: str = "stock_shell_ground_contact"
    region_sensors: tuple[str, ...] = field(default_factory=tuple)
    ground_sensor: str = "all_ground_contact"

    def build(self, env: ManagerBasedRlEnv) -> HornRollTracker:
        return HornRollTracker(self, env)


class HornRollTracker(MomentumRollTracker):
    cfg: HornRollTrackerCfg

    def __init__(self, cfg: HornRollTrackerCfg, env: ManagerBasedRlEnv):
        data = np.load(cfg.reference)
        if "axis" not in data:
            raise ValueError(f"{cfg.reference}: no 'axis'; record it with "
                             "studies/proto3/record_demo.py")
        a = torch.as_tensor(data["axis"], dtype=torch.float32, device=env.device)
        self.axis = a / a.norm()
        self._side = torch.linalg.cross(self.axis, torch.tensor([0.0, 0.0, 1.0],
                                                                device=env.device))
        super().__init__(cfg, env)
        n = self.num_envs
        self.peak_limb = torch.zeros(n, device=self.device)
        self.peak_sensor = torch.zeros(n, device=self.device)
        self.peak_shell = torch.zeros(n, device=self.device)
        self.peak_carapace_impact = torch.zeros(n, device=self.device)
        self.peak_ground_impact = torch.zeros(n, device=self.device)
        self.flight = torch.zeros(n, device=self.device)
        self.longest_flight = torch.zeros(n, device=self.device)
        self._last: dict[str, torch.Tensor] = {}
        for k in ("peak_limb_N", "peak_shell_N", "peak_carapace_impact_N",
                  "peak_ground_impact_N", "longest_flight_s"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    # ── The angle ─────────────────────────────────────────────────────────────
    def angle_now(self) -> torch.Tensor:
        g = self.robot.data.projected_gravity_b
        return torch.atan2(g @ self._side, -g[:, 2])

    # ── Forces ────────────────────────────────────────────────────────────────
    def force(self, sensor: str) -> torch.Tensor:
        """The sensor's floor force at the end of this control step, N, its parts summed."""
        f = self._env.scene[sensor].data.force
        assert f is not None
        return f.norm(dim=-1).sum(dim=1)

    def impact(self, sensor: str) -> torch.Tensor:
        """The largest rise in the sensor's summed force within this control step's
        substeps and the last substep of the step before, N."""
        h = self._env.scene[sensor].data.force_history          # [B, N, H, 3], 0 newest
        assert h is not None
        f = h.norm(dim=-1).sum(dim=1).flip(dims=[1])            # [B, H], oldest first
        prev = self._last.get(sensor, f[:, :1])
        seq = torch.cat([prev, f], dim=1)
        low = torch.cummin(seq, dim=1).values
        return (seq[:, 1:] - low[:, :-1]).clamp(min=0.0).max(dim=1).values

    def carapace_impact(self) -> torch.Tensor:
        if not self.cfg.region_sensors:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.stack([self.impact(s) for s in self.cfg.region_sensors], dim=1) \
            .max(dim=1).values

    def in_air(self) -> torch.Tensor:
        return self.force(self.cfg.ground_sensor) < 0.5

    # ── Episode bookkeeping ───────────────────────────────────────────────────
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        for buf in (self.peak_limb, self.peak_sensor, self.peak_shell,
                    self.peak_carapace_impact, self.peak_ground_impact,
                    self.flight, self.longest_flight):
            buf[env_ids] = 0.0
        # A new episode has no step before it: no rise is measured from the old one.
        for last in self._last.values():
            last[env_ids] = float("inf")

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["peak_limb_N"] = self.peak_limb.clone()
        self.metrics["peak_shell_N"] = self.peak_shell.clone()
        self.metrics["peak_carapace_impact_N"] = self.peak_carapace_impact.clone()
        self.metrics["peak_ground_impact_N"] = self.peak_ground_impact.clone()
        self.metrics["longest_flight_s"] = self.longest_flight * self._env.step_dt

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        live = ~self.settling
        lf = live.float()
        self.peak_limb.copy_(torch.maximum(self.peak_limb, self.force(self.cfg.limb_sensor) * lf))
        self.peak_sensor.copy_(torch.maximum(self.peak_sensor,
                                             self.force(self.cfg.sensor_sensor) * lf))
        self.peak_shell.copy_(torch.maximum(self.peak_shell,
                                            self.force(self.cfg.shell_sensor) * lf))
        self.peak_carapace_impact.copy_(torch.maximum(self.peak_carapace_impact,
                                                      self.carapace_impact() * lf))
        self.peak_ground_impact.copy_(torch.maximum(self.peak_ground_impact,
                                                    self.impact(self.cfg.ground_sensor) * lf))
        air = self.in_air() & live
        self.flight.copy_(torch.where(air, self.flight + 1.0, torch.zeros_like(self.flight)))
        self.longest_flight.copy_(torch.maximum(self.longest_flight, self.flight))
        # Each impact sensor's newest substep, for the next step's boundary.
        for s in (*self.cfg.region_sensors, self.cfg.ground_sensor):
            h = self._env.scene[s].data.force_history
            self._last[s] = h[:, :, 0].norm(dim=-1).sum(dim=1, keepdim=True)


__all__ = ["HornRollTracker", "HornRollTrackerCfg"]
