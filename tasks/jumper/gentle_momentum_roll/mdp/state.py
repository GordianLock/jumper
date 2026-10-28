"""jumper.showcase_roll's tracker, plus the hardest shell hit, and a roll clock that
mid-roll spawns inherit from the demonstration.

From jumper.showcase_roll (`showcase_roll/mdp/state.py`): `peak_limb` and
`peak_sensor`, the hardest floor contact of the episode by kind. Added here:

- `peak_shell`: the same for the shell, which on a carapace model is the stock top
  shell, the base and every carapace piece (`env_cfg.SHELL_PATTERN`);
- **its own roll clock**, `begun` and `t0`. The roll counts as begun the step the
  body has turned `start_turn` rad, so the crouch before it is not timed.
  jumper.showcase_roll's `roll_start` starts a mid-roll spawn at a roll time of 0
  and uses negative values to mean "not begun". Here a spawn partway through the
  demonstration starts at the demonstration's own time since it passed
  `start_turn` -- negative `t0` -- so a spawn late in the roll has already used
  that much of it, as a policy that got there itself would have.
  `roll_seconds()` is overridden to read this clock; showcase's `roll_start` is
  still kept up to date by its own code, and nothing here reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...showcase_roll.mdp.state import ShowcaseTracker, ShowcaseTrackerCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class GentleRollTrackerCfg(ShowcaseTrackerCfg):
    shell_sensor: str = "shell_ground_contact"

    def build(self, env: ManagerBasedRlEnv) -> GentleRollTracker:
        return GentleRollTracker(self, env)


class GentleRollTracker(ShowcaseTracker):
    cfg: GentleRollTrackerCfg

    def __init__(self, cfg: GentleRollTrackerCfg, env: ManagerBasedRlEnv):
        n = env.num_envs
        # Before super().__init__: the parent's constructor may already call
        # roll_seconds() through its metrics.
        self.begun = torch.zeros(n, dtype=torch.bool, device=env.device)
        #: The episode step the roll began; negative for a spawn already partway.
        self.t0 = torch.zeros(n, device=env.device)
        super().__init__(cfg, env)
        self.peak_shell = torch.zeros(n, device=self.device)
        self.metrics["peak_shell_N"] = torch.zeros(n, device=self.device)
        # The demonstration's time at which it passed `start_turn`: `total[k]` is
        # the turn after step k, and the target sent during step k is read at
        # demo time k * dt, so the state after step k is at (k + 1) * dt.
        ref = self.reference
        passed = (ref.total >= cfg.start_turn).nonzero()
        k = int(passed[0]) if len(passed) else ref.frames - 1
        self.demo_start_t = (k + 1) * ref.dt

    def roll_seconds(self) -> torch.Tensor:
        steps = self._env.episode_length_buf.float()
        return torch.where(self.begun, steps - self.t0, torch.zeros_like(steps)) * \
            self._env.step_dt

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        begun = self.total[env_ids] >= self.cfg.start_turn
        demo = self.demo_t[env_ids]
        elapsed = torch.where(demo >= 0.0, (demo - self.demo_start_t).clamp(min=0.0),
                              torch.zeros_like(demo))
        self.begun[env_ids] = begun
        # A reset episode is at step 0, so t0 = -elapsed / dt reads `elapsed`.
        self.t0[env_ids] = torch.where(begun, -elapsed / self._env.step_dt,
                                       torch.zeros_like(elapsed))
        self.peak_shell[env_ids] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["peak_shell_N"] = self.peak_shell.clone()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        steps = self._env.episode_length_buf.float()
        start = ~self.begun & (self.total >= self.cfg.start_turn)
        self.t0.copy_(torch.where(start, steps, self.t0))
        self.begun.copy_(self.begun | start)
        live = (~self.settling).float()
        self.peak_shell.copy_(torch.maximum(self.peak_shell,
                                            self.part_force(self.cfg.shell_sensor) * live))


__all__ = ["GentleRollTracker", "GentleRollTrackerCfg"]
