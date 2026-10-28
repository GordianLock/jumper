"""The roll tracker for jumper.horned_roll: jumper.showcase_roll's, plus the roll clock of
jumper.gentle_momentum_roll (copied, so that task can change without changing this one),
plus what the bonus is scaled by here.

From jumper.showcase_roll (`showcase_roll/mdp/state.py`): `peak_limb` and `peak_sensor`,
the hardest floor contact of the episode on the limbs and on the display / dToF / camera
(each sensor sums its parts: `netforce`, one slot).

Copied from jumper.gentle_momentum_roll (`gentle_momentum_roll/mdp/state.py`, 2026-10-28):
`begun` and `t0`, a roll clock that starts the step the body has turned `start_turn` rad,
and that a spawn partway through the demonstration inherits.

Added here:

- `peak_shell`: the hardest contact on the **stock** shell and base only. The carapace
  is not in it.
- `peak_carapace`: the hardest contact on any one **carapace region** (`env_cfg.REGIONS`).
  Each region's sensor sums the pieces in it, so a hit that straddles the seam between
  two collision pieces of one horn counts at full strength; contact spread over several
  regions counts only its hardest region, so a load shared across the carapace reads low.
- `peak_ground`: the hardest **total** floor force, every part summed, feet included --
  a hard landing whatever touches down first.
- `flight` and `longest_flight`: control steps with nothing on the floor, now and at most
  this episode (jumper.roll's `airborne` test: body and feet both under 0.5 N).

All peaks and flights skip a spawn's settle window, as jumper.showcase_roll's do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ...showcase_roll.mdp.state import ShowcaseTracker, ShowcaseTrackerCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HornedRollTrackerCfg(ShowcaseTrackerCfg):
    shell_sensor: str = "stock_shell_ground_contact"
    #: One sensor per carapace region.
    region_sensors: tuple[str, ...] = field(default_factory=tuple)
    #: Everything but the feet (jumper.roll's sensor) and the feet.
    body_sensor: str = "body_ground_contact"
    feet_sensor: str = "feet_ground_contact"

    def build(self, env: ManagerBasedRlEnv) -> HornedRollTracker:
        return HornedRollTracker(self, env)


class HornedRollTracker(ShowcaseTracker):
    cfg: HornedRollTrackerCfg

    def __init__(self, cfg: HornedRollTrackerCfg, env: ManagerBasedRlEnv):
        n = env.num_envs
        # Before super().__init__: the parent's constructor may already call
        # roll_seconds() through its metrics.
        self.begun = torch.zeros(n, dtype=torch.bool, device=env.device)
        #: The episode step the roll began; negative for a spawn already partway.
        self.t0 = torch.zeros(n, device=env.device)
        super().__init__(cfg, env)
        self.peak_shell = torch.zeros(n, device=self.device)
        self.peak_carapace = torch.zeros(n, device=self.device)
        self.peak_ground = torch.zeros(n, device=self.device)
        self.flight = torch.zeros(n, device=self.device)
        self.longest_flight = torch.zeros(n, device=self.device)
        for k in ("peak_shell_N", "peak_carapace_N", "peak_ground_N", "longest_flight_s"):
            self.metrics[k] = torch.zeros(n, device=self.device)
        # The demonstration's time at which it passed `start_turn` (gentle_momentum_roll's
        # reasoning: the state after step k is at (k + 1) * dt).
        ref = self.reference
        passed = (ref.total >= cfg.start_turn).nonzero()
        k = int(passed[0]) if len(passed) else ref.frames - 1
        self.demo_start_t = (k + 1) * ref.dt

    # ── Forces ────────────────────────────────────────────────────────────────
    def _force(self, sensor: str) -> torch.Tensor:
        f = self._env.scene[sensor].data.force
        assert f is not None
        return f.norm(dim=-1).sum(dim=1)

    def carapace_force(self) -> torch.Tensor:
        """The hardest carapace region's floor force this step, N."""
        if not self.cfg.region_sensors:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.stack([self._force(s) for s in self.cfg.region_sensors], dim=1) \
            .max(dim=1).values

    def ground_force(self) -> torch.Tensor:
        """Every part's floor force this step, summed, N."""
        return self._force(self.cfg.body_sensor) + self._force(self.cfg.feet_sensor)

    def in_air(self) -> torch.Tensor:
        return (self._force(self.cfg.body_sensor) < 0.5) & \
            (self._force(self.cfg.feet_sensor) < 0.5)

    # ── The roll clock (jumper.gentle_momentum_roll's) ─────────────────────────
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
        for buf in (self.peak_shell, self.peak_carapace, self.peak_ground,
                    self.flight, self.longest_flight):
            buf[env_ids] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["peak_shell_N"] = self.peak_shell.clone()
        self.metrics["peak_carapace_N"] = self.peak_carapace.clone()
        self.metrics["peak_ground_N"] = self.peak_ground.clone()
        self.metrics["longest_flight_s"] = self.longest_flight * self._env.step_dt

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        steps = self._env.episode_length_buf.float()
        start = ~self.begun & (self.total >= self.cfg.start_turn)
        self.t0.copy_(torch.where(start, steps, self.t0))
        self.begun.copy_(self.begun | start)
        live = ~self.settling
        lf = live.float()
        self.peak_shell.copy_(torch.maximum(self.peak_shell,
                                            self._force(self.cfg.shell_sensor) * lf))
        self.peak_carapace.copy_(torch.maximum(self.peak_carapace, self.carapace_force() * lf))
        self.peak_ground.copy_(torch.maximum(self.peak_ground, self.ground_force() * lf))
        air = self.in_air() & live
        self.flight.copy_(torch.where(air, self.flight + 1.0, torch.zeros_like(self.flight)))
        self.longest_flight.copy_(torch.maximum(self.longest_flight, self.flight))


__all__ = ["HornedRollTracker", "HornedRollTrackerCfg"]
