"""The torque floor contact puts on each driven joint, |J^T f|, estimated in training.

The servo model (`actuator.py`) limits what a motor can push. Nothing in the
framework limits what the floor pushes back into a joint when the robot lands on a
limb, and that load reaches several times the 1.7464 N*m plateau: measured on replays
(MuJoCo's `mj_mulJacTVec` over every floor contact, 5 ms physics steps), jumper.roll's
variants peaked at 5.3-7.6 N*m, jumper.self_right's first policy at 6.6-8.4 (right
claw pushing off), KingKong's reference-free jump at 4.8.

For a hinge, a contact force F at point p loads every joint above the touching link
by axis . ((p - anchor) x F). `FloorJointLoad` sums that over the legs' floor contacts
(`leg_floor_sensor`: every leg geom against the floor, world-frame forces and
positions, four contacts per geom) for every listed joint, at **every physics step**
-- driven by `floor_load_sampler`, a metric term run inside the decimation loop -- and
keeps each control step's worst. Read once per control step instead, it misses a knock
that starts and ends between two control steps: on jumper.tuck_roll's second policy
that read 5.5 N*m against 6.7-7.9 on replay. Sampled per physics step with four
contacts per geom it matched `mj_mulJacTVec` at that policy's peak (7.25 against 7.25
N*m, same joint) and step by step (correlation 0.97).

A task that uses it: give its command term a `floor_load` attribute
(`FloorJointLoad(env, sensor, joints)`), call its `reset(env_ids)` on resample and its
`end_step()` once the control step's bookkeeping is done, add `leg_floor_sensor` to
the scene, `floor_load_sampler` to `cfg.metrics` (`per_substep=True`) and `joint_load`
to the rewards. Mechanism only: the thresholds are the task's.
"""

from __future__ import annotations

import numpy as np
import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply

LEG_GEOMS = r"(LF|RF|LM|RM|LR|RR)_.*_meshcol"


def leg_floor_sensor(decimation: int, name: str = "leg_floor_contact") -> ContactSensorCfg:
    """Every leg geom against the floor: contact found, world-frame force and point,
    four contacts per geom."""
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=LEG_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"), global_frame=True,
        reduce="maxforce", num_slots=4, history_length=decimation,
    )


class FloorJointLoad:
    """|J^T f| from floor contact per joint in `joints`, N*m (module docstring)."""

    def __init__(self, env, sensor: str, joints: tuple[str, ...], entity: str = "robot"):
        self._env, self.sensor, self.joints, self.entity = env, sensor, joints, entity
        self._chain: tuple | None = None
        self._step: torch.Tensor | None = None

    def _build_chain(self) -> tuple:
        """Per sensor contact slot, which joints it loads; the joints' bodies, and
        their anchors and axes in their body frames."""
        m = self._env.sim.mj_model
        sensor = self._env.scene[self.sensor]
        slots = len(sensor.data.found[0]) // len(sensor.primary_names)

        def mj(kind, name):
            for n in (name, f"{self.entity}/{name}"):
                try:
                    return getattr(m, kind)(n)
                except KeyError:
                    continue
            raise KeyError(name)

        joints = [mj("joint", j) for j in self.joints]
        jbody = [int(j.bodyid[0]) for j in joints]
        mask = torch.zeros(len(sensor.primary_names) * slots, len(joints))
        for i, name in enumerate(sensor.primary_names):
            b, chain = int(mj("geom", name.split("/")[-1]).bodyid[0]), set()
            while b > 0:
                chain.add(b)
                b = int(m.body_parentid[b])
            for k, jb in enumerate(jbody):
                if jb in chain:
                    mask[i * slots:(i + 1) * slots, k] = 1.0
        dev = self._env.device
        return (torch.tensor(jbody, device=dev), mask.to(dev),
                torch.tensor(np.array([j.pos for j in joints]), dtype=torch.float32, device=dev),
                torch.tensor(np.array([j.axis for j in joints]), dtype=torch.float32, device=dev))

    def substep(self) -> torch.Tensor:
        """This physics step's load, [B, J], folded into the control step's worst."""
        if self._chain is None:
            self._chain = self._build_chain()
        jbody, mask, jpos, jaxis = self._chain
        data = self._env.scene[self.sensor].data
        sim = self._env.sim.data
        bq = sim.xquat[:, jbody]                                       # [B, J, 4]
        n = bq.shape[0]
        flat = bq.reshape(-1, 4)
        anchor = sim.xpos[:, jbody] + quat_apply(
            flat, jpos.expand(n, -1, -1).reshape(-1, 3)).reshape(n, -1, 3)
        axis = quat_apply(flat, jaxis.expand(n, -1, -1).reshape(-1, 3)).reshape(n, -1, 3)
        live = (data.found > 0).float().unsqueeze(-1)                  # [B, N, 1]
        r = data.pos.unsqueeze(2) - anchor.unsqueeze(1)                # [B, N, J, 3]
        f = (data.force * live).unsqueeze(2).expand_as(r)              # [B, N, J, 3]
        # tau[b, j] = sum_n mask[n, j] * axis_j . (r_nj x f_n)
        t = (torch.linalg.cross(r, f, dim=-1) * axis.unsqueeze(1)).sum(-1) * mask
        load = t.sum(1).abs()
        self._step = load if self._step is None else torch.maximum(self._step, load)
        return load

    def step(self) -> torch.Tensor:
        """The worst over this control step's physics steps, [B, J]."""
        if self._step is None:
            return torch.zeros(self._env.num_envs, len(self.joints), device=self._env.device)
        return self._step

    def worst(self) -> torch.Tensor:
        """The worst joint this control step, [B]."""
        return self.step().max(dim=1).values

    def reset(self, env_ids: torch.Tensor) -> None:
        if self._step is not None:
            self._step[env_ids] = 0.0

    def end_step(self) -> None:
        """The control step is accounted for; the next one starts afresh."""
        if self._step is not None:
            self._step.zero_()


def floor_load_sampler(env, command_name: str = "roll") -> torch.Tensor:
    """Metric term, `per_substep=True`: samples the floor load at every physics step.
    Its own logged value is a per-step mean of those samples, not the peak; the
    command term's own peak metric is the one to read."""
    return env.command_manager.get_term(command_name).floor_load.substep().max(dim=1).values


def joint_load(env, threshold: float, cap: float = 10.0,
               command_name: str = "roll") -> torch.Tensor:
    """Per control step: ((tau - threshold) / threshold)^2, capped, tau the worst
    joint's floor load this step; 0 while the command term's `settling` is True."""
    cmd = env.command_manager.get_term(command_name)
    over = ((cmd.floor_load.worst() - threshold).clamp(min=0.0) / threshold) ** 2
    return over.clamp(max=cap) * (~cmd.settling).float()


__all__ = ["FloorJointLoad", "LEG_GEOMS", "floor_load_sampler", "joint_load",
           "leg_floor_sensor"]
