"""The rewards of jumper.tuck_roll that are its own.

- `tuck`: while the nose points down, the claws side by side under the face.
- `front_contact`: per step, floor force on the front of the body over a threshold.
- `arm_contact`: per step, floor force on the arms over a (higher) threshold.
- `joint_load`: per step, the worst joint's floor-contact torque |J^T f| over a
  threshold.
- `gentle_complete`: jumper.horn_roll's scaled finishing bonus with two more factors,
  the arms' load and the front of the body's.
- `prior_joint_pos_legs`: jumper.momentum_roll's joint prior on the six walking legs'
  joints only; the arms are free to leave the demonstration.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from ...horn_roll.mdp.rewards import _factor, _overload
from ...momentum_roll.mdp.terminations import standing_done


def _cmd(env, name: str = "roll"):
    return env.command_manager.get_term(name)


def nose_down(env, min_down: float = 0.3) -> torch.Tensor:
    """The body's forward axis points down by more than `min_down` (its world z below
    -min_down): from about 17 degrees nose-down until the body is nearly on its back."""
    robot: Entity = env.scene["robot"]
    fwd = torch.zeros(env.num_envs, 3, device=env.device)
    fwd[:, 0] = 1.0
    return quat_apply(robot.data.root_link_quat_w, fwd)[:, 2] < -min_down


def tuck(env, asset_cfg: SceneEntityCfg, min_x: float = 0.12, max_y: float = 0.08,
         sigma_sq: float = 0.01, min_down: float = 0.3, either: bool = False,
         command_name: str = "roll") -> torch.Tensor:
    """While `nose_down`: exp(-d^2 / sigma_sq) for each claw -- the better of the two
    with `either`, else their mean -- d how far
    the palm is from the box under the face -- at least `min_x` ahead of the base origin
    and within `max_y` of the centreline (body frame). 1 tucked; about 0.09 at
    jumper.horn_roll's splay (palms 5 cm ahead, 22 cm out). 0 otherwise."""
    robot: Entity = env.scene[asset_cfg.name]
    p = robot.data.body_link_pos_w[:, asset_cfg.body_ids] - \
        robot.data.root_link_pos_w.unsqueeze(1)                       # [B, 2, 3]
    q = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, p.shape[1], -1)
    b = quat_apply_inverse(q.reshape(-1, 4), p.reshape(-1, 3)).reshape(p.shape)
    dx = (min_x - b[..., 0]).clamp(min=0.0)
    dy = (b[..., 1].abs() - max_y).clamp(min=0.0)
    k = torch.exp(-(dx ** 2 + dy ** 2) / sigma_sq)
    k = k.max(dim=1).values if either else k.mean(dim=1)
    cmd = _cmd(env, command_name)
    return k * nose_down(env, min_down).float() * (~cmd.settling).float()


def joint_load(env, threshold: float, cap: float = 10.0,
               command_name: str = "roll") -> torch.Tensor:
    """Per step: the worst driven joint's floor-contact torque |J^T f| over `threshold`
    N*m, ((tau - threshold) / threshold)^2, capped."""
    cmd = _cmd(env, command_name)
    return _overload(cmd.floor_joint_load().max(dim=1).values, threshold, cap) * \
        (~cmd.settling).float()


def front_contact(env, threshold: float, cap: float = 10.0,
                  command_name: str = "roll") -> torch.Tensor:
    cmd = _cmd(env, command_name)
    return _overload(cmd.front_force(), threshold, cap) * (~cmd.settling).float()


def arm_contact(env, threshold: float, cap: float = 10.0,
                command_name: str = "roll") -> torch.Tensor:
    cmd = _cmd(env, command_name)
    return _overload(cmd.force(cmd.cfg.arm_sensor), threshold, cap) * \
        (~cmd.settling).float()


def gentle_complete(env, command_name: str = "roll", min_own_turn: float = 1.0,
                    gentle_floor: float = 0.5,
                    limb: tuple[float, float] = (20.0, 100.0),
                    arm: tuple[float, float] = (60.0, 150.0),
                    front: tuple[float, float] = (5.0, 40.0),
                    load: tuple[float, float] | None = None,
                    sensor: tuple[float, float] = (2.0, 20.0),
                    shell: tuple[float, float] = (30.0, 150.0),
                    carapace_impact: tuple[float, float] = (50.0, 200.0),
                    ground_impact: tuple[float, float] = (60.0, 250.0),
                    flight: tuple[float, float] = (0.06, 0.20)) -> torch.Tensor:
    """jumper.horn_roll's bonus (its `rewards.gentle_complete`), the arms' load and the
    front of the body's floor force among the factors the lowest of which scales it."""
    cmd = _cmd(env, command_name)
    done = standing_done(env, command_name=command_name, min_own_turn=min_own_turn).float()
    worst = {
        "limb": torch.maximum(cmd.peak_limb, cmd.force(cmd.cfg.limb_sensor)),
        "arm": torch.maximum(cmd.peak_arm, cmd.force(cmd.cfg.arm_sensor)),
        "front": torch.maximum(cmd.peak_front, cmd.front_force()),
        "sensor": torch.maximum(cmd.peak_sensor, cmd.force(cmd.cfg.sensor_sensor)),
        "shell": torch.maximum(cmd.peak_shell, cmd.force(cmd.cfg.shell_sensor)),
        "carapace_impact": torch.maximum(cmd.peak_carapace_impact, cmd.carapace_impact()),
        "ground_impact": torch.maximum(cmd.peak_ground_impact,
                                       cmd.impact(cmd.cfg.ground_sensor)),
        "flight": cmd.longest_flight * env.step_dt,
    }
    bands = {"limb": limb, "arm": arm, "front": front, "sensor": sensor, "shell": shell,
             "carapace_impact": carapace_impact, "ground_impact": ground_impact,
             "flight": flight}
    if load is not None:
        worst["load"] = torch.maximum(cmd.peak_load, cmd.floor_joint_load().max(dim=1).values)
        bands["load"] = load
    g = torch.stack([_factor(worst[k], *bands[k]) for k in bands], dim=1).min(dim=1).values
    return done * (gentle_floor + (1.0 - gentle_floor) * g)


def prior_joint_pos_legs(env, asset_cfg: SceneEntityCfg, columns: tuple[int, ...],
                         command_name: str = "roll", sigma_sq: float = 0.2) -> torch.Tensor:
    """jumper.momentum_roll's `prior_joint_pos` over the demonstration's `columns` only
    (`asset_cfg` the same joints, in the same order)."""
    cmd = _cmd(env, command_name)
    ref = cmd.reference
    robot: Entity = env.scene[asset_cfg.name]
    want = ref.q[ref.index(cmd.demo_t.clamp(min=0.0))][:, list(columns)]
    have = robot.data.joint_pos[:, asset_cfg.joint_ids]
    e = ((have - want) ** 2).mean(dim=1)
    return torch.exp(-e / sigma_sq) * cmd.in_demo.float() * (~cmd.settling).float()


__all__ = ["arm_contact", "front_contact", "gentle_complete", "joint_load",
           "nose_down", "prior_joint_pos_legs", "tuck"]
