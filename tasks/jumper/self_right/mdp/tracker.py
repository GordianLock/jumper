"""How upright the body is, and the episode's hardest hits, for jumper.self_right.

## Upright, not turned

The robot starts on its back and may right itself any way: over either side, end
over end, or diagonally. So progress is measured by how upright the body is, not by a
turn about a chosen axis:

    up = -g_z  (g the gravity direction in the base frame)

-1 lying flat on its back, 0 on its side or end, 1 standing level. `best` is the
episode's high-water mark; the progress reward pays new ground above it, so rocking
back and forth earns nothing twice.

## The settle window

Every episode starts with the robot set down on its back (`events.reset_on_back`).
It then lies there, its joints held at HOME whatever the policy asks
(`actions.HoldWhileSettlingAction`), until it is still: at least `settle_steps`
control steps, then until the body has turned slower than `still_ang_vel` and moved
slower than `still_lin_vel` for `still_steps` steps running, at most
`max_settle_steps`. Only then does the policy act. The hits and the high-water mark of
that window are the spawn's, not the policy's, and rewards that judge the policy are
multiplied by `~settling`. (Runs 1 and 2 had a fixed window in which the policy acted
unjudged, and learnt to rock and pump in it.)

## Contact, per episode

As jumper.horn_roll's tracker (`horn_roll/mdp/tracker.py`): peak force on the limbs
(feet excepted) and the display / dToF / camera; **impact**, not load, on the stock
shell and on the floor as a whole, impact being the largest rise in force within one
control step, read from the sensors' 5 ms substeps; and the longest time in the air.
On the stock robot the shell is what it lies on, so its load is not charged, only how
sharply it is hit. And the torque floor contact puts on each driven joint, |J^T f|,
sampled at every physics step (`common/floor_load.py`), as `peak_load`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ...common.floor_load import FloorJointLoad

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class SelfRightTrackerCfg(CommandTermCfg):
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    #: Settling (`SelfRightTracker.settling`): at least `settle_steps` control steps,
    #: then until the body has been still -- turning slower than `still_ang_vel` rad/s
    #: and moving slower than `still_lin_vel` m/s -- for `still_steps` steps running,
    #: and never longer than `max_settle_steps`.
    settle_steps: int = 25
    max_settle_steps: int = 100
    still_ang_vel: float = 0.17
    still_lin_vel: float = 0.02
    still_steps: int = 5
    limb_sensor: str = "limb_ground_contact"
    sensor_sensor: str = "sensor_ground_contact"
    shell_sensor: str = "shell_ground_contact"
    ground_sensor: str = "all_ground_contact"
    load_sensor: str = "leg_floor_contact"
    #: The driven joints whose floor-contact torque is tracked (`common/floor_load.py`).
    load_joints: tuple[str, ...] = ()

    def build(self, env: ManagerBasedRlEnv) -> SelfRightTracker:
        return SelfRightTracker(self, env)


class SelfRightTracker(CommandTerm):
    cfg: SelfRightTrackerCfg

    def __init__(self, cfg: SelfRightTrackerCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        n = self.num_envs
        z = lambda: torch.zeros(n, device=self.device)  # noqa: E731
        #: How upright the body was at the end of the settle window, and at best since.
        self.spawn_up, self.best = z(), z()
        self.peak_limb, self.peak_sensor = z(), z()
        self.peak_shell_impact, self.peak_ground_impact = z(), z()
        self.flight, self.longest_flight = z(), z()
        self._last: dict[str, torch.Tensor] = {}
        self._released = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._still = torch.zeros(n, dtype=torch.long, device=self.device)
        self._dummy = torch.zeros(n, 1, device=self.device)
        self.peak_load = z()
        self.floor_load = FloorJointLoad(env, cfg.load_sensor, cfg.load_joints,
                                         cfg.entity_name)
        for k in ("up", "rose", "peak_limb_N", "peak_shell_impact_N",
                  "peak_ground_impact_N", "longest_flight_s", "peak_joint_load_Nm"):
            self.metrics[k] = z()

    @property
    def command(self) -> torch.Tensor:
        return self._dummy

    def up_now(self) -> torch.Tensor:
        return -self.robot.data.projected_gravity_b[:, 2]

    @property
    def settling(self) -> torch.Tensor:
        """True until the robot is released: lying still on its back after the spawn
        (the cfg's settle fields). The action is held at HOME meanwhile
        (`actions.HoldWhileSettlingAction`), and nothing is paid or charged."""
        return ~self._released

    def _update_release(self) -> None:
        w = self.robot.data.root_link_ang_vel_b.norm(dim=1)
        v = self.robot.data.root_link_lin_vel_b.norm(dim=1)
        still = (w < self.cfg.still_ang_vel) & (v < self.cfg.still_lin_vel)
        self._still.copy_(torch.where(still, self._still + 1, torch.zeros_like(self._still)))
        steps = self._env.episode_length_buf
        go = ((steps >= self.cfg.settle_steps) & (self._still >= self.cfg.still_steps)) \
            | (steps >= self.cfg.max_settle_steps)
        self._released |= go

    # ── Forces ────────────────────────────────────────────────────────────────
    def force(self, sensor: str) -> torch.Tensor:
        f = self._env.scene[sensor].data.force
        assert f is not None
        return f.norm(dim=-1).sum(dim=1)

    def impact(self, sensor: str) -> torch.Tensor:
        """The largest rise in the sensor's summed force within this control step's
        substeps and the last substep of the step before, N."""
        h = self._env.scene[sensor].data.force_history          # [B, N, H, 3], 0 newest
        assert h is not None
        f = h.norm(dim=-1).sum(dim=1).flip(dims=[1])
        prev = self._last.get(sensor, f[:, :1])
        seq = torch.cat([prev, f], dim=1)
        low = torch.cummin(seq, dim=1).values
        return (seq[:, 1:] - low[:, :-1]).clamp(min=0.0).max(dim=1).values

    def in_air(self) -> torch.Tensor:
        return self.force(self.cfg.ground_sensor) < 0.5

    # ── Episode bookkeeping ───────────────────────────────────────────────────
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        up = self.up_now()[env_ids]
        self.spawn_up[env_ids] = up
        self.best[env_ids] = up
        for buf in (self.peak_limb, self.peak_sensor, self.peak_shell_impact,
                    self.peak_ground_impact, self.flight, self.longest_flight,
                    self.peak_load):
            buf[env_ids] = 0.0
        self.floor_load.reset(env_ids)
        self._released[env_ids] = False
        self._still[env_ids] = 0
        for last in self._last.values():
            last[env_ids] = float("inf")             # no rise across the reset

    def _update_metrics(self) -> None:
        self.metrics["up"] = self.up_now()
        self.metrics["rose"] = self.best - self.spawn_up
        self.metrics["peak_limb_N"] = self.peak_limb.clone()
        self.metrics["peak_shell_impact_N"] = self.peak_shell_impact.clone()
        self.metrics["peak_ground_impact_N"] = self.peak_ground_impact.clone()
        self.metrics["longest_flight_s"] = self.longest_flight * self._env.step_dt
        self.metrics["peak_joint_load_Nm"] = self.peak_load.clone()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            return
        up = self.up_now()
        settling = self.settling
        # In the settle window, the spawn's pose is the baseline.
        self.spawn_up.copy_(torch.where(settling, up, self.spawn_up))
        self.best.copy_(torch.where(settling, up, torch.maximum(self.best, up)))
        lf = (~settling).float()
        self.peak_limb.copy_(torch.maximum(self.peak_limb,
                                           self.force(self.cfg.limb_sensor) * lf))
        self.peak_sensor.copy_(torch.maximum(self.peak_sensor,
                                             self.force(self.cfg.sensor_sensor) * lf))
        self.peak_shell_impact.copy_(torch.maximum(self.peak_shell_impact,
                                                   self.impact(self.cfg.shell_sensor) * lf))
        self.peak_ground_impact.copy_(torch.maximum(self.peak_ground_impact,
                                                    self.impact(self.cfg.ground_sensor) * lf))
        air = self.in_air() & ~settling
        self.flight.copy_(torch.where(air, self.flight + 1.0, torch.zeros_like(self.flight)))
        self.longest_flight.copy_(torch.maximum(self.longest_flight, self.flight))
        for s in (self.cfg.shell_sensor, self.cfg.ground_sensor):
            h = self._env.scene[s].data.force_history
            self._last[s] = h[:, :, 0].norm(dim=-1).sum(dim=1, keepdim=True)
        self.peak_load.copy_(torch.maximum(self.peak_load, self.floor_load.worst() * lf))
        self.floor_load.end_step()
        # Last: a robot released now is still settling for this step's bookkeeping.
        self._update_release()


__all__ = ["SelfRightTracker", "SelfRightTrackerCfg"]
