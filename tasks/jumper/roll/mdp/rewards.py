"""Rewards for the sideways roll.

**The objective** is the turn itself: `roll_progress` pays for every radian of
*new ground* -- each step the body gets further round than it has been this
episode, up to one full roll. Falling back costs nothing and re-covering old
ground pays nothing, so rocking cannot farm it and a failed push is free.

That last point is the fix the first trained run asked for. It paid the net
turn, so a push that tipped the body from its back to 240 degrees and let it fall
back was worth zero before its drift and heading costs and less than zero after
them; the policy learnt to lie on its back and stay there. The motion is
possible: a cross-entropy search over two held leg postures (64 samples x 10
rounds, native CPU, 2026-10-24) took the body from its back past 360 degrees in
the first round, and its elite averaged 504 degrees by the tenth.

It is a rate capped at `max_rate`, so a fling that gets the turn over in a few
frames pays no more than a controlled roll at that speed.

Once the turn is (nearly) complete, `finish_upright` and `finish_home` pull the
body level and the legs back to HOME, and `roll_complete` pays a one-off bonus
the step it is standing -- which also ends the episode (`terminations.py`).
None of the three pays before the turn, so standing still at HOME earns nothing.

**The costs** keep the roll a roll: thumping the body into the ground
(`body_impact`, the shell's sake), time with nothing touching the floor
(`airborne`), turning away from the start heading or
creeping forward instead of sideways (`heading_error`, `forward_drift`), legs
knocking each other, joint limits and jittery targets.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from .commands import TWO_PI


def _tracker(env, command_name: str):
    return env.command_manager.get_term(command_name)


def roll_progress(env, command_name: str = "roll", max_rate: float = 4.0) -> torch.Tensor:
    """New ground about the long axis, rad/s: how far the body got past its
    furthest point so far this episode (capped at one full roll), per second,
    capped at `max_rate`. Never negative."""
    cmd = _tracker(env, command_name)
    after = cmd.total_now().clamp(max=TWO_PI)
    rate = ((after - cmd.best).clamp(min=0.0) / env.step_dt).clamp(max=max_rate)
    return rate * (~cmd.settling).float()


def _turned(env, command_name: str, tolerance: float) -> torch.Tensor:
    return cmd_total(env, command_name) >= TWO_PI - tolerance


def cmd_total(env, command_name: str) -> torch.Tensor:
    return _tracker(env, command_name).total_now()


def finish_upright(
    env, command_name: str = "roll", tolerance: float = 0.5, sigma_sq: float = 0.10
) -> torch.Tensor:
    """exp(-|g_xy|^2 / sigma_sq) once the turn is within `tolerance` of a full
    roll; 0 before."""
    robot: Entity = env.scene["robot"]
    g = robot.data.projected_gravity_b[:, :2]
    k = torch.exp(-(g**2).sum(dim=1) / sigma_sq)
    return k * _turned(env, command_name, tolerance).float()


def finish_home(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "roll",
    tolerance: float = 0.5,
    sigma_sq: float = 0.10,
) -> torch.Tensor:
    """exp(-e / sigma_sq), e the mean squared distance of the driven joints from
    HOME, once the turn is within `tolerance` of a full roll; 0 before.

    `asset_cfg` has no default: the reward manager resolves joint ids only for a
    `SceneEntityCfg` it finds in the params."""
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    k = torch.exp(-((q - home) ** 2).mean(dim=1) / sigma_sq)
    return k * _turned(env, command_name, tolerance).float()


def roll_complete(env, **kwargs) -> torch.Tensor:
    """1 on the step the robot is standing after a full roll, else 0. The same
    predicate ends the episode, so this pays once."""
    from .terminations import standing_after_roll

    return standing_after_roll(env, **kwargs).float()


def body_impact(
    env,
    sensor_name: str = "body_ground_contact",
    threshold: float = 40.0,
    cap: float = 10.0,
    command_name: str = "roll",
) -> torch.Tensor:
    """((F - threshold) / threshold)^2 for the net ground force on everything
    that is not a foot, 0 below `threshold` newtons, capped at `cap`.

    The roll has to put the body on the ground -- that is not what this charges.
    The robot's own weight is about 25 N; it charges landing on the shell harder
    than lying on it."""
    f = env.scene[sensor_name].data.force
    assert f is not None
    total = f.norm(dim=-1).sum(dim=1)
    cost = (((total - threshold).clamp(min=0.0) / threshold) ** 2).clamp(max=cap)
    return cost * (~_tracker(env, command_name).settling).float()


def airborne(env, command_name: str = "roll") -> torch.Tensor:
    """1 while nothing of the robot touches the floor (no body, no foot over 0.5 N),
    outside a spawn's settle window."""
    body = env.scene["body_ground_contact"].data.force
    feet = env.scene["feet_ground_contact"].data.force
    assert body is not None and feet is not None
    off = (body.norm(dim=-1).sum(dim=1) < 0.5) & (feet.norm(dim=-1).sum(dim=1) < 0.5)
    return off.float() * (~_tracker(env, command_name).settling).float()


def heading_error(env) -> torch.Tensor:
    """y^2 of the body's forward axis in the world, normalised over its
    horizontal part: 0 facing the spawn heading (+x, or -x), 1 turned 90 degrees.
    A roll about the long axis leaves the axis where it was."""
    robot: Entity = env.scene["robot"]
    q = robot.data.root_link_quat_w
    ex = torch.zeros(q.shape[0], 3, device=q.device)
    ex[:, 0] = 1.0
    fwd = quat_apply(q, ex)[:, :2]
    n = fwd.norm(dim=1).clamp(min=1e-3)
    return (fwd[:, 1] / n) ** 2


def forward_drift(env, scale: float = 0.10, cap: float = 4.0) -> torch.Tensor:
    """(dx / scale)^2 along the spawn heading, capped. A sideways roll travels in
    y; travel in x is the roll going wrong."""
    robot: Entity = env.scene["robot"]
    dx = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return ((dx / scale) ** 2).clamp(max=cap)


__all__ = [
    "airborne",
    "body_impact",
    "finish_home",
    "finish_upright",
    "forward_drift",
    "heading_error",
    "roll_complete",
    "roll_progress",
]
