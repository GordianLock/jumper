"""Laying out an episode, and starting some with the debris already in the claw.

## The layout (`reset_layout`)

Every episode the robot stands at its env origin facing +x, and

- **the debris** lies on the floor somewhere in `DEBRIS_ZONE` -- in front of the
  trunk, where the claw's mouth reaches the floor (measured in `env_cfg.py`) --
  turned to any heading;
- **the receptacle** stands in `BIN_ZONE`, to the left, sunk into the floor by a
  random amount so its rim is anywhere from flush (a pile spot) to its full
  `BIN_WALL_H`; `pile_fraction` of episodes get exactly flush.

Positions are in the robot's frame at spawn and the robot spawns unrotated, so
they are written straight off the env origin.

## Held spawns (`HeldSpawn`)

`held_fraction` of episodes start with the debris in the claw, so carrying and
putting down are practised from the first iteration rather than only once the
policy has learnt to pick something up. The five-foot task's `play --hold`
is the recipe, and its two rules are kept because they were measured there:
the debris is held in the mouth only **while the claw is open**, and the claw is
then closed on it **after it is let go** -- closing on a body whose pose is being
written every step throws it out of the jaws (`five_foot/mdp/grasp.py`,
`PLACE_STEPS`). The claw itself is driven by the action term during those
steps (`actions.py`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_from_matrix

from ..props import BIN_WALL_H, BIN_WALL_T

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: How far off the anvil the debris's face is put, and how far into the mouth.
#: The five-foot task seats 20 mm in (`five_foot/mdp/grasp.py::HOLD_DEPTH`) with
#: a world-aligned prop; a box square to the jaws there runs into the knuckle at
#: some sizes (measured: overlapping at widths 30 and 58 mm), and at 10 mm every
#: width from 22 to 58 mm sits clear with 6.5 mm to the fixed jaw.
SEAT_CLEARANCE = 0.005
HOLD_DEPTH = 0.010


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


def reset_layout(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    debris_zone: dict[str, tuple[float, float]],
    bin_zone: dict[str, tuple[float, float]],
    pile_fraction: float,
    held_fraction: float,
    state_name: str = "task",
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, dev = len(env_ids), env.device
    if not hasattr(env, "_pick_layout"):
        N = env.num_envs
        env._pick_layout = {
            "bin_pos": torch.zeros(N, 3, device=dev),
            "rim": torch.zeros(N, device=dev),
            "bin_floor": torch.zeros(N, device=dev),
            "spawn_xy": torch.zeros(N, 2, device=dev),
            "held": torch.zeros(N, dtype=torch.bool, device=dev),
        }
    layout = env._pick_layout
    origin = env.scene.env_origins[env_ids]
    state = env.command_manager.get_term(state_name)

    # ── The debris, on the floor ──────────────────────────────────────────
    debris = env.scene["debris"]
    half = state.half()[env_ids]
    yaw = _uniform(-math.pi / 2, math.pi / 2, n, dev)
    zero = torch.zeros(n, device=dev)
    pos = torch.stack([
        origin[:, 0] + _uniform(*debris_zone["x"], n, dev),
        origin[:, 1] + _uniform(*debris_zone["y"], n, dev),
        origin[:, 2] + half[:, 2] + 0.001,
    ], dim=1)
    quat = quat_from_euler_xyz(zero, zero, yaw)
    debris.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)
    debris.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    # ── The receptacle, sunk to its rim height ────────────────────────────
    full = BIN_WALL_T + BIN_WALL_H
    rim = _uniform(0.0, full, n, dev)
    rim = torch.where(torch.rand(n, device=dev) < pile_fraction, torch.zeros_like(rim), rim)
    sink = full - rim
    bin_xy = torch.stack([
        origin[:, 0] + _uniform(*bin_zone["x"], n, dev),
        origin[:, 1] + _uniform(*bin_zone["y"], n, dev),
    ], dim=1)
    bin_pose = torch.zeros(n, 7, device=dev)
    bin_pose[:, :2] = bin_xy
    bin_pose[:, 2] = origin[:, 2] - sink
    bin_pose[:, 3] = 1.0
    env.scene["bin"].write_mocap_pose_to_sim(bin_pose, env_ids=env_ids)

    layout["bin_pos"][env_ids] = torch.cat([bin_xy, origin[:, 2:3]], dim=1)
    layout["rim"][env_ids] = rim
    layout["bin_floor"][env_ids] = (BIN_WALL_T - sink).clamp(min=0.0)
    layout["spawn_xy"][env_ids] = origin[:, :2]
    layout["held"][env_ids] = torch.rand(n, device=dev) < held_fraction


@requires_model_fields("geom_size", "geom_rbound")
def debris_size(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale: dict[int, tuple[float, float]],
    geom_name: str = "debris_geom",
) -> None:
    """Shrink each environment's debris box by a per-axis factor in `scale`.

    **Shrink only.** `dr.geom_size` would also rewrite the geom's broadphase box,
    `geom_aabb`, which neither backend can hold per environment
    (`mjrl/backend/native_sim.py::WORLD_BATCHED_FIELDS`). So the box is authored at
    its largest (`props.DEBRIS_HALF`) and every factor here is at most 1: the
    shared `geom_aabb` then stays a bound on every environment's box, which is all
    the broadphase needs of it. `geom_rbound` is per environment and is updated.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if any(hi > 1.0 for _, hi in scale.values()):
        raise ValueError("debris_size only shrinks: every factor must be <= 1")
    debris = env.scene["debris"]
    g = int(debris.indexing.geom_ids[0])
    model = env.sim.model
    n = len(env_ids)
    size = model.geom_size[env_ids, g].clone()
    for axis, (lo, hi) in scale.items():
        size[:, axis] = size[:, axis] * _uniform(lo, hi, n, env.device)
    model.geom_size[env_ids, g] = size
    model.geom_rbound[env_ids, g] = size.norm(dim=1)


class HeldSpawn:
    """A step event: for a held spawn's first `place_steps`, keep the debris
    seated in the claw; the action term closes the claw on it afterwards.

    **Weightless until the grip is made.** Let go of 50 mm up, the debris fell
    20 mm before the jaws reached it and came to rest on the fixed jaw's hook
    rather than between the jaws (measured, native backend). So for the whole of
    the held window the debris's gravity is compensated (`body_gravcomp` 1, a
    per-environment field on both backends) and the jaws close on a body that
    stays where it was put; gravity returns when the window ends, and the grip
    has to hold it from there.
    """

    #: Read by the event manager, as `requires_model_fields` sets it on a function.
    model_fields = ("body_gravcomp",)

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._body = int(env.scene["debris"].indexing.body_ids[0])

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None = None,
        *,
        place_steps: int = 10,
        state_name: str = "task",
    ) -> None:
        del env_ids
        state = env.command_manager.get_term(state_name)
        env.sim.model.body_gravcomp[:, self._body] = state.settling.float()
        ids = (state.held_spawn & (env.episode_length_buf <= place_steps)).nonzero().flatten()
        if ids.numel() == 0:
            return
        debris = env.scene["debris"]
        mouth = state.mouth()[ids]
        closing = state.closing()[ids]
        inward = state.inward()[ids]
        half = state.half()[ids]
        # Square to the jaws: the debris's x axis along the closing direction
        # (which dips about 14 degrees below level), its z as near world-up as
        # that allows. Laid level instead, the slanted jaws meet it on two edges
        # and squeeze it out downward -- measured, a 28 mm block slid 19 mm and
        # fell as the claw closed.
        x = closing / closing.norm(dim=1, keepdim=True).clamp(min=1e-6)
        up = torch.zeros_like(x)
        up[:, 2] = 1.0
        z = up - (up * x).sum(dim=1, keepdim=True) * x
        z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-6)
        y = torch.cross(z, x, dim=1)
        quat = quat_from_matrix(torch.stack([x, y, z], dim=2))
        centre = mouth + x * (half[:, 0:1] + SEAT_CLEARANCE) + inward * HOLD_DEPTH
        floor = env.scene.env_origins[ids, 2]
        centre[:, 2] = torch.maximum(centre[:, 2], floor + half[:, 2] + 0.005)
        debris.write_root_link_pose_to_sim(torch.cat([centre, quat], dim=1), env_ids=ids)
        debris.write_root_link_velocity_to_sim(
            torch.zeros(ids.numel(), 6, device=env.device), env_ids=ids
        )


__all__ = ["HeldSpawn", "debris_size", "reset_layout"]
