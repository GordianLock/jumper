"""jumper.momentum_roll's tracker, plus three things about the episode so far that the
completion bonus is scaled by:

- `roll_start`: the control step the body first turned `start_turn` rad -- the
  roll's duration is counted from here, so standing still before it buys nothing;
- `peak_limb`, `peak_sensor`: the hardest single-part floor contact so far, in
  newtons, of the limbs (other than their tips) and of the display / dToF / camera;
- `last_gain`: the control step the roll last got further round than ever before,
  for the stall cost.

Contact peaks are read in `_update_command`, after the step's physics, so they
include the step just simulated by the time the next reward reads them; the bonus
also folds in the current step (`rewards.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...momentum_roll.mdp.commands import MomentumRollTracker, MomentumRollTrackerCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ShowcaseTrackerCfg(MomentumRollTrackerCfg):
    limb_sensor: str = "limb_ground_contact"
    sensor_sensor: str = "sensor_ground_contact"
    #: Radians of turn at which the roll counts as begun.
    start_turn: float = 0.3

    def build(self, env: ManagerBasedRlEnv) -> ShowcaseTracker:
        return ShowcaseTracker(self, env)


class ShowcaseTracker(MomentumRollTracker):
    cfg: ShowcaseTrackerCfg

    def __init__(self, cfg: ShowcaseTrackerCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        self.roll_start = torch.full((n,), -1.0, device=self.device)
        self.peak_limb = torch.zeros(n, device=self.device)
        self.peak_sensor = torch.zeros(n, device=self.device)
        self.last_gain = torch.zeros(n, device=self.device)
        for k in ("roll_seconds", "peak_limb_N"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    def part_force(self, sensor: str) -> torch.Tensor:
        f = self._env.scene[sensor].data.force
        assert f is not None
        return f.norm(dim=-1).max(dim=1).values

    def roll_seconds(self) -> torch.Tensor:
        steps = self._env.episode_length_buf.float()
        return torch.where(self.roll_start >= 0, steps - self.roll_start,
                           torch.zeros_like(steps)) * self._env.step_dt

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # A spawn already partway round has begun its roll.
        begun = self.total[env_ids] >= self.cfg.start_turn
        self.roll_start[env_ids] = torch.where(begun, 0.0, -1.0)
        self.peak_limb[env_ids] = 0.0
        self.peak_sensor[env_ids] = 0.0
        self.last_gain[env_ids] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["roll_seconds"] = self.roll_seconds()
        self.metrics["peak_limb_N"] = self.peak_limb.clone()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        before = self.best.clone()
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        steps = self._env.episode_length_buf.float()
        start = (self.roll_start < 0) & (self.total >= self.cfg.start_turn)
        self.roll_start.copy_(torch.where(start, steps, self.roll_start))
        self.last_gain.copy_(torch.where(self.best > before + 1e-4, steps, self.last_gain))
        live = (~self.settling).float()
        self.peak_limb.copy_(torch.maximum(self.peak_limb,
                                           self.part_force(self.cfg.limb_sensor) * live))
        self.peak_sensor.copy_(torch.maximum(self.peak_sensor,
                                             self.part_force(self.cfg.sensor_sensor) * live))


__all__ = ["ShowcaseTracker", "ShowcaseTrackerCfg"]
