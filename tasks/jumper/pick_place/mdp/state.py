"""The task's bookkeeping: where the claw, the debris and the receptacle are, and
how far the job has got.

A command term, as `jumper.roll`'s tracker is, because it is the one manager term
that holds per-episode state and is updated once per step. Nothing is
commanded; the layout is sampled by the reset events (`events.py`) and recorded
on the env, and read here.

## Read fresh, latched once

Rewards and terminations run after the physics and **before** the command update
(`ManagerBasedRlEnv.step`). So everything they need is a method that reads the
simulation as it is now -- `grasped()`, `lift()`, `in_bin()` -- and only the one
thing that has to remember the past is latched in `_update_command`:
`was_lifted`, that the debris has been off the ground in the claw this episode.
A placement counts only after it; pushing the debris into a pile without
picking it up is not the task.

## The claw's mouth

The anvil, the closing direction and the way into the mouth are constants of the
claw in `LF_palm_link`'s frame, solved by `five_foot/mdp/grasp.py` from the jaws'
drawn meshes. They are solved here on a plain copy of `jumper.xml` rather than on
the simulation's model, because the native backend strips visual meshes at
training-sized environment counts and the solver needs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from ...common.assets import JUMPER_XML
from ...five_foot.mdp.grasp import _closing_in_palm, _inward_in_palm, _mouth_in_palm
from ..props import BIN_INNER_HALF

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_CLAW_CACHE: dict[str, tuple] = {}


def claw_constants() -> tuple:
    """(mouth, closing, inward) in the palm frame, from the plain model."""
    if "c" not in _CLAW_CACHE:
        m = mujoco.MjModel.from_xml_path(str(JUMPER_XML))
        _CLAW_CACHE["c"] = (_mouth_in_palm(m), _closing_in_palm(m), _inward_in_palm(m))
    return _CLAW_CACHE["c"]


@dataclass(kw_only=True)
class PickPlaceStateCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    palm_sensor: str = "palm_debris"
    finger_sensor: str = "finger_debris"
    #: Newtons on *each* jaw for the debris to count as gripped.
    grip_force: float = 0.5
    #: Metres the debris's lowest point must clear the floor, in the claw, to
    #: count as lifted.
    lift_min: float = 0.015
    #: The debris centre must be this far inside the tray's inner wall, m.
    bin_margin: float = 0.005
    #: And moving slower than this, m/s, to count as put down.
    rest_speed: float = 0.05
    #: Control steps after a held spawn during which the claw is closed by the
    #: spawn, not by the policy (`events.HeldSpawn`).
    held_steps: int = 30

    def build(self, env: ManagerBasedRlEnv) -> PickPlaceState:
        return PickPlaceState(self, env)


class PickPlaceState(CommandTerm):
    cfg: PickPlaceStateCfg

    def __init__(self, cfg: PickPlaceStateCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.robot = env.scene["robot"]
        self.debris = env.scene["debris"]
        self.palm = self.robot.find_bodies(["LF_palm_link"])[0][0]
        mouth, closing, inward = claw_constants()
        self.mouth_palm = torch.tensor(mouth, device=dev, dtype=torch.float32)
        self.closing_palm = torch.tensor(closing, device=dev, dtype=torch.float32)
        self.inward_palm = torch.tensor(inward, device=dev, dtype=torch.float32)
        self._debris_geom = int(self.debris.indexing.geom_ids[0])

        #: The layout, written by `events.reset_layout`.
        self.bin_pos = torch.zeros(n, 3, device=dev)  # tray centre on the floor, world
        self.rim = torch.zeros(n, device=dev)         # rim height above the floor
        self.bin_floor = torch.zeros(n, device=dev)   # tray floor's top above the floor
        self.spawn_xy = torch.zeros(n, 2, device=dev)  # the robot's base at reset
        self.held_spawn = torch.zeros(n, dtype=torch.bool, device=dev)

        self.was_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dummy = torch.zeros(n, 1, device=dev)
        for k in ("gripped", "lifted", "in_bin", "placed"):
            self.metrics[k] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._dummy

    # ── The claw ──────────────────────────────────────────────────────────
    def _palm_rot(self, vec: torch.Tensor) -> torch.Tensor:
        q = self.robot.data.body_link_quat_w[:, self.palm]
        return quat_apply(q, vec.expand(q.shape[0], 3))

    def mouth(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.palm] + self._palm_rot(self.mouth_palm)

    def closing(self) -> torch.Tensor:
        return self._palm_rot(self.closing_palm)

    def inward(self) -> torch.Tensor:
        return self._palm_rot(self.inward_palm)

    # ── The debris ────────────────────────────────────────────────────────
    def half(self) -> torch.Tensor:
        """The debris's half-sizes per env, [N, 3] -- randomised at startup."""
        size = self._env.sim.model.geom_size
        if size.dim() == 2:
            half = size[self._debris_geom].expand(self.num_envs, 3)
        else:
            half = size[:, self._debris_geom]
        # A plain tensor: the native backend's model fields are a tensor subclass,
        # and the subclass would ride through every reward into the logger.
        return half.as_subclass(torch.Tensor)

    def debris_pos(self) -> torch.Tensor:
        return self.debris.data.root_link_pos_w

    def debris_quat(self) -> torch.Tensor:
        return self.debris.data.root_link_quat_w

    def bottom(self) -> torch.Tensor:
        """Height of the debris's lowest corner above the floor."""
        q = self.debris_quat()
        h = self.half()
        # Support of a box along world -z: sum over axes of |R e_i . z| h_i.
        n = q.shape[0]
        axes = torch.eye(3, device=q.device).expand(n, 3, 3)
        z = torch.stack([quat_apply(q, axes[:, i]) for i in range(3)], dim=1)[..., 2]
        reach = (z.abs() * h).sum(dim=1)
        floor = self._env.scene.env_origins[:, 2]
        return self.debris_pos()[:, 2] - reach - floor

    def grip_axis_b(self) -> torch.Tensor:
        """(cos 2 psi, sin 2 psi): the debris's x axis about the vertical, in the
        robot's base frame. Doubled because a box turned half a turn is the same."""
        ex = torch.zeros(self.num_envs, 3, device=self.device)
        ex[:, 0] = 1.0
        a = quat_apply_inverse(self.robot.data.root_link_quat_w,
                               quat_apply(self.debris_quat(), ex))
        psi = torch.atan2(a[:, 1], a[:, 0])
        return torch.stack([torch.cos(2 * psi), torch.sin(2 * psi)], dim=1)

    # ── How far the job has got ───────────────────────────────────────────
    def _force(self, name: str) -> torch.Tensor:
        f = self._env.scene[name].data.force
        assert f is not None
        return f.norm(dim=-1).sum(dim=1)

    def grasped(self) -> torch.Tensor:
        return (self._force(self.cfg.palm_sensor) > self.cfg.grip_force) & (
            self._force(self.cfg.finger_sensor) > self.cfg.grip_force
        )

    def touching(self) -> torch.Tensor:
        return (self._force(self.cfg.palm_sensor) > self.cfg.grip_force) | (
            self._force(self.cfg.finger_sensor) > self.cfg.grip_force
        )

    def lifted(self) -> torch.Tensor:
        return self.grasped() & (self.bottom() > self.cfg.lift_min)

    def to_bin_xy(self) -> torch.Tensor:
        return self.bin_pos[:, :2] - self.debris_pos()[:, :2]

    def in_bin(self) -> torch.Tensor:
        """The debris's centre over the tray's inside, and its lowest point down
        on the tray's floor: in the tray, or on the pile spot."""
        d = (self.debris_pos()[:, :2] - self.bin_pos[:, :2]).abs().max(dim=1).values
        inside = d < BIN_INNER_HALF - self.cfg.bin_margin
        low = self.bottom() < self.bin_floor + 0.01
        return inside & low

    def resting(self) -> torch.Tensor:
        return self.debris.data.root_link_lin_vel_w.norm(dim=1) < self.cfg.rest_speed

    def placed(self) -> torch.Tensor:
        """Picked up, put in the tray, let go of, and at rest there."""
        return self.was_lifted & self.in_bin() & ~self.touching() & self.resting()

    @property
    def settling(self) -> torch.Tensor:
        """A held spawn whose claw is still being closed for it."""
        return self.held_spawn & (self._env.episode_length_buf <= self.cfg.held_steps)

    # ── The manager's side ────────────────────────────────────────────────
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.was_lifted[env_ids] = False
        layout = getattr(self._env, "_pick_layout", None)
        if layout is None:
            return
        self.bin_pos[env_ids] = layout["bin_pos"][env_ids]
        self.rim[env_ids] = layout["rim"][env_ids]
        self.bin_floor[env_ids] = layout["bin_floor"][env_ids]
        self.spawn_xy[env_ids] = layout["spawn_xy"][env_ids]
        self.held_spawn[env_ids] = layout["held"][env_ids]

    def _update_metrics(self) -> None:
        self.metrics["gripped"] = self.grasped().float()
        self.metrics["lifted"] = self.was_lifted.float()
        self.metrics["in_bin"] = (self.in_bin() & self.was_lifted).float()
        self.metrics["placed"] = self.placed().float()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            return
        self.was_lifted |= self.lifted() & ~self.settling


__all__ = ["PickPlaceState", "PickPlaceStateCfg", "claw_constants"]
