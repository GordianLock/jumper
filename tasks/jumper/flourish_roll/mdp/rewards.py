"""Rewards for jumper.flourish_roll: a roll that is safe for the hardware, slow, and
interesting to watch.

## Style -- what makes a roll worth watching, as things a reward can see

- `flourish`: limbs off the floor that are *moving*. A leg in the air gesturing,
  sweeping or waving is most of what makes a body look alive; one held still
  in the air is not. Mean joint speed of the legs whose foot is off the floor,
  each joint capped, so one leg thrashing does not buy it.
- `wobble`: the body pitching while it rolls -- a corkscrew rather than a barrel.
  Paid up to a bound, and only mid-roll.
- The tempo lives in `env_cfg.py`: progress is paid up to a turn rate that makes
  a full roll take several seconds, so a slow roll earns everything and a fast
  one earns less.

## Safety -- contact where the hardware is not built for it

`assets/jumper/jumper.xml` has 41 collision parts in four kinds, and they are
charged differently (`env_cfg.py` builds one contact sensor per kind):

- **feet** (the six tips the robot stands on): built to take load; only landings
  harder than `foot_threshold` per foot are charged;
- **limbs** (every other leg and arm part -- shins, thighs, shoulders, palms,
  fingers, grip inserts): their floor contact twists the joints from the side.
  Measured on jumper4 (jumper.styled_roll, model_1999), the largest joint loads
  came from the claws (2.7 N*m median peak, 4.5 worst) and the shins (2.5, 4.5),
  not the feet (2.1) -- see `env_cfg.py`. Charged above a small force;
- **sensors** (display module, dToF, camera): should never touch the floor;
- **shell** (base and upper shell): what the printed shell will cover, charged
  only for hard hits.

Plus the servo's own limits: driving into a joint stop at speed, and holding
torque near the plateau.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common.constants import FEET, LEGS


def _cmd(env, name: str):
    return env.command_manager.get_term(name)


# ── Style ──────────────────────────────────────────────────────────────────────


class Flourish:
    """Mean joint speed, rad/s, over the legs whose foot is off the floor, each
    joint capped at `cap`; divided by `cap`, so it is 0-1. A class term so the
    foot-to-joint mapping is worked out once."""

    def __init__(self, cfg, env):
        sensor = env.scene[cfg.params.get("sensor_name", "feet_ground_contact")]
        names = sensor.primary_names
        robot = env.scene["robot"]
        order = [names.index(f"{foot}_meshcol") for foot in FEET]
        self._order = torch.tensor(order, device=env.device)
        cols = []
        joint_names = list(robot.joint_names)
        for leg in LEGS:
            ids = [i for i, j in enumerate(joint_names)
                   if j.startswith(f"{leg}_J") and not j.endswith("J4_joint")]
            cols.append(ids)
        width = max(len(c) for c in cols)
        self._ids = torch.tensor([c + [c[0]] * (width - len(c)) for c in cols], device=env.device)
        self._mask = torch.tensor([[1.0] * len(c) + [0.0] * (width - len(c)) for c in cols],
                                  device=env.device)

    def __call__(self, env, sensor_name: str = "feet_ground_contact", cap: float = 4.0,
                 command_name: str = "roll") -> torch.Tensor:
        f = env.scene[sensor_name].data.force
        assert f is not None
        off = f.norm(dim=-1)[:, self._order] < 1.0  # [N, 6], LEGS order
        qd = env.scene["robot"].data.joint_vel.abs().clamp(max=cap) / cap
        per_leg = (qd[:, self._ids] * self._mask).sum(-1) / self._mask.sum(-1)  # [N, 6]
        n_off = off.float().sum(dim=1).clamp(min=1.0)
        score = (per_leg * off.float()).sum(dim=1) / n_off
        return score * (~_cmd(env, command_name).settling).float()


def wobble(env, command_name: str = "roll", bound: float = 0.5,
           margin: float = 0.3) -> torch.Tensor:
    """|sin pitch| up to `bound`, over `bound`, while mid-roll: more than `margin`
    rad into it and more than `margin` short of done."""
    cmd = _cmd(env, command_name)
    g = env.scene["robot"].data.projected_gravity_b
    pitch = g[:, 0].abs().clamp(max=bound) / bound
    total = cmd.total_now()
    mid = (total > margin) & (total < 2 * torch.pi - margin)
    return pitch * mid.float() * (~cmd.settling).float()


# ── Safety ─────────────────────────────────────────────────────────────────────


def contact_overload(env, sensor_name: str, threshold: float, cap: float = 10.0,
                     command_name: str = "roll") -> torch.Tensor:
    """Sum over the sensor's parts of ((F - threshold) / threshold)^2 for each part
    pressing on the floor harder than `threshold` newtons; capped at `cap`."""
    f = env.scene[sensor_name].data.force
    assert f is not None
    over = ((f.norm(dim=-1) - threshold).clamp(min=0.0) / threshold) ** 2
    return over.sum(dim=1).clamp(max=cap) * (~_cmd(env, command_name).settling).float()


def stop_strike(env, asset_cfg: SceneEntityCfg, margin: float = 0.05) -> torch.Tensor:
    """Sum of squared joint speeds, (rad/s)^2, of joints within `margin` rad of
    either end of their range and still moving toward it: arriving at a stop."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    qd = robot.data.joint_vel[:, asset_cfg.joint_ids]
    lim = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    near_lo = (q - lim[..., 0] < margin) & (qd < 0)
    near_hi = (lim[..., 1] - q < margin) & (qd > 0)
    return ((qd**2) * (near_lo | near_hi).float()).sum(dim=1)


def torque_strain(env, asset_cfg: SceneEntityCfg, level: float = 1.5) -> torch.Tensor:
    """Share of the driven joints holding more than `level` N*m: the servo's plateau
    is 1.7464 and its continuous rating 1.2."""
    robot = env.scene[asset_cfg.name]
    tau = robot.data.actuator_force[:, asset_cfg.actuator_ids].abs()
    return (tau > level).float().mean(dim=1)


__all__ = ["Flourish", "contact_overload", "stop_strike", "torque_strain", "wobble"]
