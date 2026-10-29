"""The rewards of jumper.horn_roll that are its own.

## The finishing bonus, scaled by how gently the roll went

    complete x (GENTLE_FLOOR + (1 - GENTLE_FLOOR) x gentleness)

`complete` is jumper.momentum_roll's (1 on the step it stands after the roll). The
floor is paid for finishing however hard the roll was, so a rough roll is always worth
more than not rolling: a bonus that a hard landing can take to nothing teaches the
policy to stand still. gentleness is the **lowest** of six factors, each 1 at or below
its free level and 0 at its zero level, from the episode's worst:

- force on the limbs (feet excepted), on the sensors, on the stock shell and base;
- impact on the carapace (hardest region) and on the floor as a whole;
- time in the air.

The lowest rather than the product: the worst offence decides, and six products would
leave almost nothing for any real roll. The per-step charges give the rest a gradient.

**Load on the carapace costs nothing**, here or per step: only its impact does. The
horns and their supports are built to be pivoted on and to carry the robot.

## Axis-relative heading and drift

jumper.roll's heading and drift terms assume a sideways barrel roll: drift is
penalised along the body's forward direction, the way an end-over-end roll must go.
Here both are measured against the roll's axis in the world: the axis should stay
where it was at the spawn (heading), and the body should travel across it, not along
it (drift).
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply

from ...momentum_roll.mdp.terminations import standing_done


def _cmd(env, name: str = "roll"):
    return env.command_manager.get_term(name)


def _factor(worst: torch.Tensor, free: float, zero: float) -> torch.Tensor:
    return (1.0 - (worst - free) / (zero - free)).clamp(0.0, 1.0)


def gentle_complete(env, command_name: str = "roll", min_own_turn: float = 1.0,
                    gentle_floor: float = 0.5,
                    limb: tuple[float, float] = (20.0, 100.0),
                    sensor: tuple[float, float] = (2.0, 20.0),
                    shell: tuple[float, float] = (30.0, 150.0),
                    carapace_impact: tuple[float, float] = (50.0, 200.0),
                    ground_impact: tuple[float, float] = (60.0, 250.0),
                    flight: tuple[float, float] = (0.06, 0.20)) -> torch.Tensor:
    """On the step the robot stands after its roll: the scaled bonus (module docstring).
    0 on every other step. The episode's worsts include this step."""
    cmd = _cmd(env, command_name)
    done = standing_done(env, command_name=command_name, min_own_turn=min_own_turn).float()
    worst = {
        "limb": torch.maximum(cmd.peak_limb, cmd.force(cmd.cfg.limb_sensor)),
        "sensor": torch.maximum(cmd.peak_sensor, cmd.force(cmd.cfg.sensor_sensor)),
        "shell": torch.maximum(cmd.peak_shell, cmd.force(cmd.cfg.shell_sensor)),
        "carapace_impact": torch.maximum(cmd.peak_carapace_impact, cmd.carapace_impact()),
        "ground_impact": torch.maximum(cmd.peak_ground_impact,
                                       cmd.impact(cmd.cfg.ground_sensor)),
        "flight": cmd.longest_flight * env.step_dt,
    }
    bands = {"limb": limb, "sensor": sensor, "shell": shell,
             "carapace_impact": carapace_impact, "ground_impact": ground_impact,
             "flight": flight}
    gentleness = torch.stack([_factor(worst[k], *bands[k]) for k in bands], dim=1) \
        .min(dim=1).values
    return done * (gentle_floor + (1.0 - gentle_floor) * gentleness)


def _overload(force: torch.Tensor, threshold: float, cap: float) -> torch.Tensor:
    """jumper.flourish_roll's contact charge: ((F - threshold) / threshold)^2, capped."""
    return (((force - threshold).clamp(min=0.0) / threshold) ** 2).clamp(max=cap)


def carapace_impact(env, threshold: float, cap: float = 10.0,
                    command_name: str = "roll") -> torch.Tensor:
    """Per step: the hardest carapace region's impact over `threshold` N."""
    cmd = _cmd(env, command_name)
    return _overload(cmd.carapace_impact(), threshold, cap) * (~cmd.settling).float()


def ground_impact(env, threshold: float, cap: float = 10.0,
                  command_name: str = "roll") -> torch.Tensor:
    """Per step: the impact on the floor as a whole (every part summed) over
    `threshold` N, whatever touches down."""
    cmd = _cmd(env, command_name)
    return _overload(cmd.impact(cmd.cfg.ground_sensor), threshold, cap) * \
        (~cmd.settling).float()


def stock_shell_contact(env, threshold: float, cap: float = 10.0,
                        command_name: str = "roll") -> torch.Tensor:
    """Per step: force on the stock shell and base over `threshold` N."""
    cmd = _cmd(env, command_name)
    return _overload(cmd.force(cmd.cfg.shell_sensor), threshold, cap) * \
        (~cmd.settling).float()


def _axis_world(env, command_name: str) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    robot: Entity = env.scene["robot"]
    a = cmd.axis.expand(env.num_envs, 3)
    return quat_apply(robot.data.root_link_quat_w, a)


def axis_heading_error(env, command_name: str = "roll") -> torch.Tensor:
    """1 - cos^2 of the angle between the roll axis's horizontal direction now and at
    the start (the demonstration's axis turned to the spawn heading): 0 on course,
    1 turned 90 degrees. In a clean roll the axis stays put while the body turns."""
    a = _axis_world(env, command_name)[:, :2]
    a0 = _cmd(env, command_name).axis[:2].expand_as(a)
    n = a.norm(dim=1).clamp(min=1e-3) * a0.norm(dim=1).clamp(min=1e-3)
    return 1.0 - ((a * a0).sum(dim=1) / n) ** 2


def axis_drift(env, scale: float = 0.10, cap: float = 4.0,
               command_name: str = "roll") -> torch.Tensor:
    """(d / scale)^2, d the base's travel from its spawn along the roll axis's start
    direction, capped. A roll travels across its axis; travel along it is the roll
    going wrong."""
    robot: Entity = env.scene["robot"]
    a0 = _cmd(env, command_name).axis[:2]
    a0 = a0 / a0.norm().clamp(min=1e-6)
    dp = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    d = dp @ a0
    return ((d / scale) ** 2).clamp(max=cap)


__all__ = ["axis_drift", "axis_heading_error", "carapace_impact", "gentle_complete",
           "ground_impact", "stock_shell_contact"]
