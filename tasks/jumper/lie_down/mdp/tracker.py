"""How far onto its back the body is, and whether it has finished, for jumper.lie_down.

jumper.self_right's tracker (`self_right/mdp/tracker.py`) read the other way up: the
same settle window (the robot spawns standing and is held at HOME until still), the
same per-episode hardest hits and floor-contact joint torque. What it adds:

- `down_now()` = -up: -1 standing level, 0 on its side or end, 1 flat on its back;
  `best_down` the episode's high-water mark, so lowering pays new ground only.
- `peak_spin`: the episode's fastest body rotation after the settle window, rad/s.
- `reference`, `demo_t`, `in_demo`: the demonstration (`ref/lie_down.npz`, a
  `momentum_roll.mdp.reference.RollReference`) and the time into it, which starts at 0
  on the release and runs on; -1 while settling. What jumper.momentum_roll's
  feedforward action and joint prior read.
- `done`: the finish -- belly-up within `max_tilt`, the body still, every driven joint
  within `home_tol` of HOME -- held for `hold_steps` control steps running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...momentum_roll.mdp.reference import RollReference
from ...self_right.mdp.tracker import SelfRightTracker, SelfRightTrackerCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class LieDownTrackerCfg(SelfRightTrackerCfg):
    #: The finish (`LieDownTracker.done`).
    max_tilt: float = math.radians(20.0)
    finish_ang_vel: float = 0.2
    finish_lin_vel: float = 0.02
    home_tol: float = 0.15
    hold_steps: int = 15
    #: The demonstration, recorded from the release (`studies/lie_down/record_plan.py`).
    reference: str = ""

    def build(self, env: ManagerBasedRlEnv) -> LieDownTracker:
        return LieDownTracker(self, env)


class LieDownTracker(SelfRightTracker):
    cfg: LieDownTrackerCfg

    def __init__(self, cfg: LieDownTrackerCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        self.spawn_down = torch.zeros(n, device=self.device)
        self.best_down = torch.zeros(n, device=self.device)
        self.held = torch.zeros(n, dtype=torch.long, device=self.device)
        self.peak_spin = torch.zeros(n, device=self.device)
        self.reference = RollReference(cfg.reference, self.device) if cfg.reference else None
        self.demo_t = torch.full((n,), -1.0, device=self.device)
        ids = self.robot.find_joints(list(cfg.load_joints), preserve_order=True)[0]
        self._joint_ids = torch.tensor(ids, device=self.device)
        for k in ("down", "lowered", "held_steps", "peak_spin_rad_s"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    def down_now(self) -> torch.Tensor:
        return -self.up_now()

    def home_error(self) -> torch.Tensor:
        """The driven joints' largest distance from HOME, rad."""
        q = self.robot.data.joint_pos[:, self._joint_ids]
        home = self.robot.data.default_joint_pos[:, self._joint_ids]
        return (q - home).abs().max(dim=1).values

    def finished_pose(self) -> torch.Tensor:
        """This step: belly-up, still, legs at HOME, after the settle window."""
        w = self.robot.data.root_link_ang_vel_b.norm(dim=1)
        v = self.robot.data.root_link_lin_vel_b.norm(dim=1)
        return ((self.down_now() > math.cos(self.cfg.max_tilt))
                & (w < self.cfg.finish_ang_vel) & (v < self.cfg.finish_lin_vel)
                & (self.home_error() < self.cfg.home_tol) & ~self.settling)

    @property
    def in_demo(self) -> torch.Tensor:
        if self.reference is None:
            return torch.zeros_like(self.demo_t, dtype=torch.bool)
        return (self.demo_t >= 0.0) & (self.demo_t < self.reference.span)

    @property
    def done(self) -> torch.Tensor:
        return self.held >= self.cfg.hold_steps

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        d = self.down_now()[env_ids]
        self.spawn_down[env_ids] = d
        self.best_down[env_ids] = d
        self.held[env_ids] = 0
        self.peak_spin[env_ids] = 0.0
        self.demo_t[env_ids] = -1.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["down"] = self.down_now()
        self.metrics["lowered"] = self.best_down - self.spawn_down
        self.metrics["held_steps"] = self.held.float()
        self.metrics["peak_spin_rad_s"] = self.peak_spin.clone()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            return
        d = self.down_now()
        settling = self.settling
        self.spawn_down.copy_(torch.where(settling, d, self.spawn_down))
        self.best_down.copy_(torch.where(settling, d, torch.maximum(self.best_down, d)))
        w = self.robot.data.root_link_ang_vel_b.norm(dim=1)
        self.peak_spin.copy_(torch.maximum(self.peak_spin, w * (~settling).float()))
        fin = self.finished_pose()
        self.held.copy_(torch.where(fin, self.held + 1, torch.zeros_like(self.held)))
        was = self.settling.clone()
        # The parent's bookkeeping ends with the release update.
        super()._update_command(env_ids)
        # The demonstration's clock, as the next action sees it: 0 on the step the
        # robot is released, then one step more each step.
        released = was & ~self.settling
        self.demo_t.copy_(torch.where(released, torch.zeros_like(self.demo_t),
                                      torch.where(self.demo_t >= 0.0,
                                                  self.demo_t + self._env.step_dt,
                                                  self.demo_t)))


__all__ = ["LieDownTracker", "LieDownTrackerCfg"]
