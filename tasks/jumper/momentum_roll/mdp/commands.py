"""The momentum roll's bookkeeping: jumper.roll's tracker, plus where along the
demonstration each episode is, and which kind of episode it is.

Three kinds of episode (`events.reset_momentum` picks one per reset):

- **from standing** -- the roll the robot will be asked for; the demonstration
  starts with the episode (`demo_t` 0);
- **mid-roll** -- spawned at a frame of the demonstration, so the push, the spin
  and the landing are practised before the policy can reach them (`demo_t` the
  frame's time);
- **on its back** -- self-righting, the way out of a roll that stalled. No
  demonstration (`demo_t` -1), and upright either way round counts: progress is
  the distance to the nearest upright, `upright_gap`, and its high-water mark
  `best_gap`.

`demo_t` is the time since the roll began as the *next* action will see it: 0 on
a standing reset, so the first step sends the recording's first target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...roll.mdp.commands import RollTracker, RollTrackerCfg, _wrap
from .reference import RollReference

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MomentumRollTrackerCfg(RollTrackerCfg):
    reference: str

    def build(self, env: ManagerBasedRlEnv) -> MomentumRollTracker:
        return MomentumRollTracker(self, env)


class MomentumRollTracker(RollTracker):
    cfg: MomentumRollTrackerCfg

    def __init__(self, cfg: MomentumRollTrackerCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        self.reference = RollReference(cfg.reference, self.device)
        #: Seconds into the demonstration; -1 for an episode without one.
        self.demo_t = torch.full((n,), -1.0, device=self.device)
        self.self_right = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: The smallest distance to upright reached this episode (self-righting).
        self.best_gap = torch.full((n,), math.pi, device=self.device)
        self._sr = torch.zeros(n, device=self.device)
        self.metrics["self_right_episode"] = self._sr

    def upright_gap(self) -> torch.Tensor:
        """Radians to the nearest upright, either way round: 0 standing, pi on the back."""
        return _wrap(self.angle_now()).abs()

    @property
    def in_demo(self) -> torch.Tensor:
        return (self.demo_t >= 0.0) & (self.demo_t < self.reference.span)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        spawn = getattr(self._env, "_momentum_spawn", None)
        if spawn is None:
            self.demo_t[env_ids] = 0.0
            self.self_right[env_ids] = False
        else:
            self.demo_t[env_ids] = spawn["demo_t"][env_ids]
            self.self_right[env_ids] = spawn["self_right"][env_ids]
        self.best_gap[env_ids] = _wrap(self.angle[env_ids]).abs()

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self._sr[:] = self.self_right.float()
        self.metrics["self_right_episode"] = self._sr

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        self.demo_t.copy_(torch.where(self.demo_t >= 0.0, self.demo_t + self._env.step_dt,
                                      self.demo_t))
        gap = self.upright_gap()
        self.best_gap.copy_(torch.where(self.settling, gap, torch.minimum(self.best_gap, gap)))


__all__ = ["MomentumRollTracker", "MomentumRollTrackerCfg"]
