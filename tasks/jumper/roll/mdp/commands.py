"""The roll's bookkeeping: how far the body has turned about its long axis.

There is nothing to command -- the roll starts when the policy does -- so this
term exists to hold per-episode state that the rewards, the termination and
the critic all read, and to keep it in one place.

## The angle

With g the gravity direction in the base frame, a body turned by theta about
its own x axis (forward) reads g = (0, -sin theta, -cos theta). So

    theta = atan2(-g_y, -g_z)

is 0 standing, +pi/2 lying on its right side, pi on its back, 3pi/2 on its left
side. Pitch is ignored: it moves g_x, which this does not read.

**The total is unwrapped.** Each control step adds the wrapped change in theta,
so the total is the net turn this episode in radians -- 2pi is one full roll,
and rocking back and forth adds nothing, because every step back is subtracted.

## Why the rewards see the angle before this term does

mjlab runs terminations and rewards after the physics and *before* the command
update (`ManagerBasedRlEnv.step`), so the state here is one step behind the body
they are judging. Rather than reward last step's turn, `pending_turn()` gives the
change from the stored angle to the body's angle now, and `total_now()` the total
including it; the update then folds it in. Rewards and the termination read the
same numbers in the same step.

## Spawns

`events.reset_rolled` turns a share of the spawns partway round and records the
angle on the env (`_roll_spawn_angle`); a reset reads it here, so a spawn that
starts on its back starts with half the roll already counted. It is paid only for
the turn it makes itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

TWO_PI = 2.0 * math.pi


def roll_angle(g_b: torch.Tensor) -> torch.Tensor:
    """theta in (-pi, pi]: the body's turn about its long axis, right side down
    positive. `g_b` is the projected gravity, [N, 3]."""
    return torch.atan2(-g_b[:, 1], -g_b[:, 2])


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, TWO_PI) - math.pi


@dataclass(kw_only=True)
class RollTrackerCfg(CommandTermCfg):
    entity_name: str = "robot"
    #: Never resampled mid-episode: the bookkeeping is the episode's.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    #: Control steps after a *turned* spawn during which the body is left to
    #: settle and nothing is paid or charged for it. Measured with the legs held
    #: at HOME: a body set down on its side does not stay there -- spawns from 90
    #: to 345 degrees all flopped onto the back, the rest onto the feet, with
    #: ground forces up to 170 N -- and that turn and that thump are the spawn's,
    #: not the policy's. The angle is still tracked throughout.
    settle_steps: int = 25

    def build(self, env: ManagerBasedRlEnv) -> RollTracker:
        return RollTracker(self, env)


class RollTracker(CommandTerm):
    cfg: RollTrackerCfg

    def __init__(self, cfg: RollTrackerCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        n = self.num_envs
        #: The body's angle as of the last update.
        self.angle = torch.zeros(n, device=self.device)
        #: Net turn this episode, radians, spawn angle included.
        self.total = torch.zeros(n, device=self.device)
        #: Where the episode started, for the metrics.
        self.spawn = torch.zeros(n, device=self.device)
        #: The episode started turned (see `settle_steps`).
        self.rolled_spawn = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._turned = torch.zeros(n, device=self.device)
        self._done = torch.zeros(n, device=self.device)
        self.metrics["turns"] = self._turned
        self.metrics["full_roll"] = self._done
        self._dummy = torch.zeros(n, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        # Nothing is commanded; the manager wants a tensor.
        return self._dummy

    def angle_now(self) -> torch.Tensor:
        return roll_angle(self.robot.data.projected_gravity_b)

    def pending_turn(self) -> torch.Tensor:
        """The turn since the last update, wrapped to (-pi, pi]."""
        return _wrap(self.angle_now() - self.angle)

    def total_now(self) -> torch.Tensor:
        """The episode's net turn including the step just simulated."""
        return self.total + self.pending_turn()

    @property
    def settling(self) -> torch.Tensor:
        """True for a turned spawn still inside its settle window. Rewards that
        judge the policy are multiplied by `~settling`."""
        steps = self._env.episode_length_buf
        return self.rolled_spawn & (steps <= self.cfg.settle_steps)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        spawn = getattr(self._env, "_roll_spawn_angle", None)
        a = (
            spawn[env_ids].clone()
            if spawn is not None
            else torch.zeros(len(env_ids), device=self.device)
        )
        self.angle[env_ids] = _wrap(a)
        self.total[env_ids] = a
        self.spawn[env_ids] = a
        self.rolled_spawn[env_ids] = a != 0.0

    def _update_metrics(self) -> None:
        # Turned by the policy: from where the body settled, not where it spawned.
        self._turned[:] = (self.total - self.spawn) / TWO_PI
        self._done[:] = (self.total >= TWO_PI).float()
        self.metrics["turns"] = self._turned
        self.metrics["full_roll"] = self._done

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            # Reset path: `_resample_command` has set the spawn's angle.
            return
        now = self.angle_now()
        self.total.add_(_wrap(now - self.angle))
        self.angle.copy_(now)
        # Until the settle window closes, the turn so far is gravity's.
        self.spawn.copy_(torch.where(self.settling, self.total, self.spawn))


__all__ = ["RollTracker", "RollTrackerCfg", "TWO_PI", "roll_angle"]
