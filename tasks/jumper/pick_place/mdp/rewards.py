"""Rewards for picking up the debris and putting it in the receptacle.

A ladder, each rung paid only once the one below it holds, so the policy is
always paid more for the next step of the job than for staying where it is:

    reach      the claw's mouth to the debris            always
    grip       both jaws pressing on it                  always
    lift       its lowest point off the floor, capped    while gripped
    carry      over the receptacle, horizontally         while gripped and clear of the rim
    in_bin     down inside the receptacle                once it has been lifted this episode
    placed     let go of, at rest in it                  one-off bonus; ends the episode

`in_bin` is gated on the debris having been lifted, so shoving it across the
floor onto a pile spot pays nothing. Everything is suspended while a held spawn
is still closing the claw (`state.settling`): that grip is the spawn's.

The costs keep it a careful, standing job: the robot's own body or its stance
legs anywhere but on their feet, the arm hitting the floor hard, the trunk
walking off its spot, legs knocking, joint limits and jittery targets.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _s(env, name: str):
    return env.command_manager.get_term(name)


def _live(s) -> torch.Tensor:
    return (~s.settling).float()


def reach(env, state_name: str = "task", std: float = 0.05) -> torch.Tensor:
    s = _s(env, state_name)
    d = (s.mouth() - s.debris_pos()).norm(dim=1)
    return (1.0 - torch.tanh(d / std)) * _live(s)


def grip(env, state_name: str = "task") -> torch.Tensor:
    s = _s(env, state_name)
    return s.grasped().float() * _live(s)


def lift(env, state_name: str = "task", clearance: float = 0.03) -> torch.Tensor:
    """Fraction of the way to `clearance` above the receptacle's rim, while
    gripped. The rim is 0 for a pile spot, so there it asks for `clearance`."""
    s = _s(env, state_name)
    goal = s.rim + clearance
    frac = (s.bottom() / goal).clamp(0.0, 1.0)
    return frac * s.grasped().float() * _live(s)


def carry(env, state_name: str = "task", std: float = 0.08, clearance: float = 0.01) -> torch.Tensor:
    """Over the receptacle, horizontally, while gripped and above its rim."""
    s = _s(env, state_name)
    d = s.to_bin_xy().norm(dim=1)
    clear = s.bottom() > s.rim + clearance
    return (1.0 - torch.tanh(d / std)) * (s.grasped() & clear).float() * _live(s)


def in_bin(env, state_name: str = "task") -> torch.Tensor:
    s = _s(env, state_name)
    return (s.in_bin() & s.was_lifted).float() * _live(s)


def placed(env, state_name: str = "task") -> torch.Tensor:
    """1 on the step the job is done; the same predicate ends the episode."""
    s = _s(env, state_name)
    return s.placed().float() * _live(s)


def ground_force(env, sensor_name: str, threshold: float, cap: float = 10.0) -> torch.Tensor:
    """((F - threshold) / threshold)^2 of the sensor's net ground force, capped."""
    f = env.scene[sensor_name].data.force
    assert f is not None
    total = f.norm(dim=-1).sum(dim=1)
    return (((total - threshold).clamp(min=0.0) / threshold) ** 2).clamp(max=cap)


def any_contact(env, sensor_name: str, force_threshold: float = 1.0) -> torch.Tensor:
    f = env.scene[sensor_name].data.force
    assert f is not None
    return (f.norm(dim=-1) > force_threshold).any(dim=1).float()


def base_drift(env, state_name: str = "task", scale: float = 0.05, cap: float = 4.0) -> torch.Tensor:
    """(d / scale)^2, capped, d the base's horizontal distance from where it spawned."""
    s = _s(env, state_name)
    robot = env.scene["robot"]
    d = robot.data.root_link_pos_w[:, :2] - s.spawn_xy
    return ((d**2).sum(dim=1) / scale**2).clamp(max=cap)


def failed(env) -> torch.Tensor:
    """1 on the step the debris is lost or the robot falls: the same predicates
    that end the episode (`terminations.py`)."""
    from .terminations import debris_lost, fell

    return (debris_lost(env) | fell(env)).float()


def tilt(env) -> torch.Tensor:
    """|g_xy|^2: zero level, 1 lying on its side. Leaning to reach is allowed, so
    this is a light cost rather than a gate."""
    g = env.scene["robot"].data.projected_gravity_b[:, :2]
    return (g**2).sum(dim=1)


def stance_feet_off(env, sensor_name: str = "feet_ground_contact",
                    force_threshold: float = 1.0) -> torch.Tensor:
    """Number of the five stance feet off the ground."""
    f = env.scene[sensor_name].data.force
    assert f is not None
    return (f.norm(dim=-1) <= force_threshold).float().sum(dim=1)


def finger_effort(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The claw servo's torque squared, N*m^2: squeezing harder than holding
    needs costs the servo and, on the robot, its battery."""
    robot = env.scene[asset_cfg.name]
    tau = robot.data.actuator_force[:, asset_cfg.actuator_ids]
    return (tau**2).sum(dim=1)


__all__ = [
    "any_contact", "base_drift", "carry", "failed", "finger_effort", "grip", "ground_force",
    "in_bin", "lift", "placed", "reach", "stance_feet_off", "tilt",
]
