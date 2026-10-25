"""Picking each episode's start: from standing, partway through the demonstration,
or on the robot's back.

Mid-roll spawns write the demonstration's recorded state at a frame -- base pose
and velocity, joint angles and speeds -- so the policy meets the spin and the
landing with the momentum the demonstration had there. Only frames up to the
last one under `max_spawn_turn` are used: completion needs a turn of the policy's
own (`roll/mdp/terminations.py`, `min_own_turn`), and a spawn nearly round could
not make one.

Back spawns reuse jumper.roll's `reset_rolled`, which sets a turned body down on
the floor at the height its collision meshes need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...roll.mdp.events import reset_rolled

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_momentum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    demo_fraction: float,
    back_fraction: float,
    back_angle_range: tuple[float, float],
    back_joint_noise: float,
    max_spawn_turn: float,
    command_name: str = "roll",
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, dev = len(env_ids), env.device
    if not hasattr(env, "_momentum_spawn"):
        env._momentum_spawn = {
            "demo_t": torch.zeros(env.num_envs, device=dev),
            "self_right": torch.zeros(env.num_envs, dtype=torch.bool, device=dev),
        }
    if not hasattr(env, "_roll_spawn_angle"):
        env._roll_spawn_angle = torch.zeros(env.num_envs, device=dev)
    spawn = env._momentum_spawn
    ref = env.command_manager.get_term(command_name).reference

    u = torch.rand(n, device=dev)
    mid = u < demo_fraction
    back = (u >= demo_fraction) & (u < demo_fraction + back_fraction)
    spawn["demo_t"][env_ids] = 0.0
    spawn["self_right"][env_ids] = False
    env._roll_spawn_angle[env_ids] = 0.0

    # ── Partway through the demonstration ────────────────────────────────────
    ids = env_ids[mid]
    if len(ids):
        last = ref.last_spawn_frame(max_spawn_turn)
        k = torch.randint(0, last + 1, (len(ids),), device=dev)
        robot = env.scene["robot"]
        root = torch.cat([
            ref.root_pos[k] + torch.cat([env.scene.env_origins[ids, :2],
                                         torch.zeros(len(ids), 1, device=dev)], dim=1),
            ref.root_quat[k], ref.root_lin[k], ref.root_ang[k],
        ], dim=1)
        robot.write_root_state_to_sim(root, env_ids=ids)
        q = robot.data.default_joint_pos[ids].clone()
        qd = torch.zeros_like(q)
        action = env.action_manager.get_term("joint_pos")
        q[:, action._target_ids] = ref.q[k]
        qd[:, action._target_ids] = ref.qd[k]
        robot.write_joint_state_to_sim(q, qd, env_ids=ids)
        # The recorded state is the state *after* step k; the next target is k+1's.
        spawn["demo_t"][ids] = (k + 1).float() * ref.dt
        env._roll_spawn_angle[ids] = ref.total[k]

    # ── On its back: self-righting ────────────────────────────────────────────
    ids = env_ids[back]
    if len(ids):
        reset_rolled(env, ids, rolled_fraction=1.0, angle_range=back_angle_range,
                     joint_noise=back_joint_noise)
        spawn["demo_t"][ids] = -1.0
        spawn["self_right"][ids] = True


__all__ = ["reset_momentum"]
