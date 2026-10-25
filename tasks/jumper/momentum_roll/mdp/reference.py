"""The demonstration: one recorded momentum roll, and reading it by time.

`ref/momentum_roll.npz` was recorded in this task's own physics (jumper.roll's: 50 Hz
control, 5 ms physics, kp 10, the servo curve) by playing back four keyframes found
by a cross-entropy search -- crouch, lopsided push, tuck, land -- then HOME. It runs
from standing to standing again: 1.96 s, 361 degrees.

Measured on the recording's own keyframes under this task's randomisation (foot
friction, base CoM, encoder bias; 64 envs, native CPU, 2026-10-24): it completed the
roll in 60 of 64, with a median 108 N peak of body on the floor, 0.26 s with nothing
touching the floor and 3.7 cm of forward drift. So it is a demonstration, not the
goal: training is asked to land softer and stay lower than it does.

## What each array is, per control step k (`dt` apart)

    q_cmd[k]   the joint targets sent *during* step k
    q[k], qd[k]   the 20 driven joints *after* step k
    root_*[k]  the base's pose and velocity after step k, position relative to the
               env origin, quaternion (w, x, y, z), velocities in the world frame
    total[k]   the body's net turn about its long axis after step k, radians

Joints are in `joint_names` order, which is `GAIT_JOINTS`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...common.constants import GAIT_JOINTS

REF_NPZ = Path(__file__).resolve().parent.parent / "ref" / "momentum_roll.npz"


class RollReference:
    def __init__(self, path: Path, device: str):
        data = np.load(path)
        names = tuple(str(n) for n in data["joint_names"])
        if names != tuple(GAIT_JOINTS):
            raise ValueError(f"{path}: joints are {names}, expected GAIT_JOINTS")
        self.dt = float(data["dt"])

        def t(key: str) -> torch.Tensor:
            return torch.as_tensor(data[key], dtype=torch.float32, device=device)

        self.q_cmd, self.q, self.qd = t("q_cmd"), t("q"), t("qd")
        self.root_pos, self.root_quat = t("root_pos"), t("root_quat")
        self.root_lin, self.root_ang = t("root_lin"), t("root_ang")
        self.total = t("total")
        self.frames = self.q.shape[0]
        self.span = self.frames * self.dt

    def index(self, time: torch.Tensor) -> torch.Tensor:
        """Frame index for a time since the roll began, clamped to the recording."""
        return (time / self.dt).round().long().clamp(0, self.frames - 1)

    def last_spawn_frame(self, max_turn: float) -> int:
        """The last frame whose turn is still under `max_turn`, over the first pass."""
        over = (self.total >= max_turn).nonzero()
        return int(over[0]) - 1 if len(over) else self.frames - 1


__all__ = ["REF_NPZ", "RollReference"]
