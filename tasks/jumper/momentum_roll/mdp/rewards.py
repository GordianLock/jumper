"""Rewards for the momentum roll.

**The prior** reads the demonstration and fades to nothing (`env_cfg.py`'s
schedule): the joint angles and the body's turn the demonstration had at the
same moment. It asks for the motion while the policy has none of its own.

**The objective** knows nothing of the demonstration:

- `progress` -- jumper.roll's new-ground reward on a roll episode; on a
  self-righting episode, new ground toward the nearest upright, either way round.
  One term, so the two kinds of episode are paid on one scale.
- `finishing` and `complete` -- level, legs home, and standing.

**The costs** are what the demonstration does worse than the goal: it lands with
about 100 N of body on the floor and spends a quarter of a second with nothing
touching it. `body_impact` and `airborne` are charged accordingly.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...roll.mdp.commands import TWO_PI
from ...roll.mdp.rewards import airborne


def _cmd(env, name: str):
    return env.command_manager.get_term(name)


def prior_joint_pos(env, asset_cfg: SceneEntityCfg, command_name: str = "roll",
                    sigma_sq: float = 0.2) -> torch.Tensor:
    """exp(-mean squared joint error / sigma_sq) against the demonstration at this
    moment, inside it; 0 outside."""
    cmd = _cmd(env, command_name)
    ref = cmd.reference
    robot: Entity = env.scene[asset_cfg.name]
    want = ref.q[ref.index(cmd.demo_t.clamp(min=0.0))]
    have = robot.data.joint_pos[:, asset_cfg.joint_ids]
    e = ((have - want) ** 2).mean(dim=1)
    return torch.exp(-e / sigma_sq) * cmd.in_demo.float() * (~cmd.settling).float()


def prior_turn(env, command_name: str = "roll", sigma_sq: float = 0.5) -> torch.Tensor:
    """exp(-(turn error)^2 / sigma_sq) against the demonstration's turn at this
    moment: where the body is in the roll, not how the legs got it there."""
    cmd = _cmd(env, command_name)
    ref = cmd.reference
    want = ref.total[ref.index(cmd.demo_t.clamp(min=0.0))]
    e = (cmd.total_now() - want) ** 2
    return torch.exp(-e / sigma_sq) * cmd.in_demo.float() * (~cmd.settling).float()


def progress(env, command_name: str = "roll", max_rate: float = 6.0) -> torch.Tensor:
    """New ground per second, capped at `max_rate`: round the roll on a roll
    episode, toward the nearest upright on a self-righting one."""
    cmd = _cmd(env, command_name)
    after = cmd.total_now().clamp(max=TWO_PI)
    roll = (after - cmd.best).clamp(min=0.0)
    right = (cmd.best_gap - cmd.upright_gap()).clamp(min=0.0)
    gain = torch.where(cmd.self_right, right, roll)
    return (gain / env.step_dt).clamp(max=max_rate) * (~cmd.settling).float()


def nearly_done(env, command_name: str, tolerance: float) -> torch.Tensor:
    """Roll episode: within `tolerance` of a full turn. Self-righting: within it of
    upright, having turned at least that far itself."""
    cmd = _cmd(env, command_name)
    roll = cmd.total_now() >= TWO_PI - tolerance
    own = (cmd.total_now() - cmd.spawn).abs() >= tolerance
    right = (cmd.upright_gap() < tolerance) & own
    return torch.where(cmd.self_right, right, roll)


def finish_upright(env, command_name: str = "roll", tolerance: float = 0.5,
                   sigma_sq: float = 0.10) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    g = robot.data.projected_gravity_b[:, :2]
    return torch.exp(-(g**2).sum(dim=1) / sigma_sq) * nearly_done(env, command_name, tolerance).float()


def finish_home(env, asset_cfg: SceneEntityCfg, command_name: str = "roll",
                tolerance: float = 0.5, sigma_sq: float = 0.10) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    k = torch.exp(-((q - home) ** 2).mean(dim=1) / sigma_sq)
    return k * nearly_done(env, command_name, tolerance).float()


def complete(env, **kwargs) -> torch.Tensor:
    from .terminations import standing_done

    return standing_done(env, **kwargs).float()


__all__ = ["airborne", "complete", "finish_home", "finish_upright", "nearly_done",
           "prior_joint_pos", "prior_turn", "progress"]
