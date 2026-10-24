"""Spawning partway through the roll.

From HOME the roll is a long way to the first reward the policy can find: it has
to get onto its side before it has learnt to. So a share of the episodes start
already turned -- on the side, on the back, on the other side -- and every stage
of the roll is practised from the start, the way reference-state initialisation
works for the jump. The rest start at HOME, where the robot will.

## Setting a turned body down

A body rolled by theta with its legs at HOME has its lowest point somewhere else
for every theta: the feet standing, the hip on its side, the shell on its back.
Spawned at the standing height it would start buried, spawned high it would start
with a fall. So the height is solved per spawn: the collision meshes' vertices at
HOME, in the base frame, are computed once from the compiled model, and for each
theta the base is set where the lowest of them, rotated, clears the ground by
`clearance`. The legs then get a little noise, which can lower a point by a few
millimetres; the contact solver resolves that in the first substeps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_CACHE: dict[int, torch.Tensor] = {}


def _home_points(env: ManagerBasedRlEnv, robot) -> torch.Tensor:
    """Collision-mesh vertices at HOME in the base frame, [P, 3], subsampled."""
    key = id(env)
    if key in _CACHE:
        return _CACHE[key]
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    ix = robot.indexing
    root = robot.data.default_root_state[0, :7].cpu().numpy().copy()
    root[:3] = 0.0
    root[3:7] = (1.0, 0.0, 0.0, 0.0)
    d.qpos[ix.free_joint_q_adr.cpu().numpy()] = root
    d.qpos[ix.joint_q_adr.cpu().numpy()] = robot.data.default_joint_pos[0].cpu().numpy()
    mujoco.mj_kinematics(m, d)
    pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.endswith("_meshcol"):
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a : a + n]
        pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    p = np.vstack(pts)
    if len(p) > 4000:
        p = p[np.random.default_rng(0).choice(len(p), 4000, replace=False)]
    out = torch.as_tensor(p, dtype=torch.float32, device=env.device)
    _CACHE[key] = out
    return out


def reset_rolled(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    rolled_fraction: float,
    angle_range: tuple[float, float],
    joint_noise: float = 0.1,
    clearance: float = 0.005,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Turn `rolled_fraction` of the resetting envs by theta, uniform in
    `angle_range` (radians, right side down), about the body's long axis, and set
    them down on the ground. The rest are left as `reset_scene_to_default` put
    them. Records every resetting env's spawn angle as `env._roll_spawn_angle`,
    which the roll tracker reads."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_roll_spawn_angle"):
        env._roll_spawn_angle = torch.zeros(env.num_envs, device=env.device)
    env._roll_spawn_angle[env_ids] = 0.0

    pick = torch.rand(len(env_ids), device=env.device) < rolled_fraction
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    robot = env.scene[asset_cfg.name]
    lo, hi = angle_range
    theta = lo + (hi - lo) * torch.rand(len(ids), device=env.device)
    env._roll_spawn_angle[ids] = theta

    # Lowest point of the HOME pose turned by theta: z' = y sin(theta) + z cos(theta).
    p = _home_points(env, robot)
    s, c = torch.sin(theta)[:, None], torch.cos(theta)[:, None]
    zmin = (p[None, :, 1] * s + p[None, :, 2] * c).min(dim=1).values

    root = robot.data.default_root_state[ids].clone()
    root[:, 0:2] = env.scene.env_origins[ids, 0:2]
    root[:, 2] = env.scene.env_origins[ids, 2] - zmin + clearance
    zero = torch.zeros_like(theta)
    root[:, 3:7] = quat_mul(root[:, 3:7], quat_from_euler_xyz(theta, zero, zero))
    root[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root, env_ids=ids)

    q = robot.data.default_joint_pos[ids].clone()
    q += joint_noise * (2.0 * torch.rand_like(q) - 1.0)
    limits = robot.data.soft_joint_pos_limits[ids]
    q = q.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)


__all__ = ["reset_rolled"]
