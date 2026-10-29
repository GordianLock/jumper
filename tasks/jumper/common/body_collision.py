"""The claws against the body, and leg-against-leg charged pair by pair.

The repository's collision masks (`constants._collision_cfg`) let every leg pass
through the shell, base and sensors: legs meet the floor and each other, never the
body. A task whose motion reaches across the body can learn to put a claw where the
body is -- measured 2026-10-29: jumper.horn_roll's first policy swung its right claw
4.6 cm through the shell and display, jumper.self_right's put its left claw 1.6 cm
into the base -- so such tasks add `arm_body_collision`.

The body is the convex hulls of its parts, which fill its hollows; a claw kept out of
them stays further clear than the real parts need. At HOME the claws clear the body
by 3 mm or more (measured on prototype 3).

With the body in the robot's own contacts, the whole-robot `self_collision` sensor
would see a claw resting on the body as a leg collision. `leg_pair_sensors` watches
leg-against-leg only, one sensor per leg against the legs before it in `LEGS`, and
`leg_collision_pairs` charges it like jumper.ref_free_jump's `leg_collision`.
"""

from __future__ import annotations

import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from . import constants as K
from .constants import FOOT_GEOMS, LEGS

ARMS = ("LF", "RF")
#: A bit no other geom uses: the floor holds 1, the legs 2-64.
BODY_BIT = 1 << 7
BODY_GEOMS = (r"^base_link_meshcol$", r"^upper_shell_link_meshcol$",
              r"^display_module_link_meshcol$", r"^tof_sensor_link_meshcol$",
              r"^camera_link_meshcol$")
LEG_ROOT = {"LF": "LF_shoulder_link", "RF": "RF_shoulder_link", "LM": "LM_hip_link",
            "RM": "RM_hip_link", "LR": "LR_hip_link", "RR": "RR_hip_link"}


def arm_body_collision(carapace_geoms: str | None = None,
                       carapace_friction: tuple[float, ...] | None = None
                       ) -> tuple[CollisionCfg, ...]:
    """CollisionCfgs to append after the robot's own: the claws meet the body (and the
    carapace pieces matching `carapace_geoms`, given their floor friction); nothing
    else changes.

    The body gains `BODY_BIT` in contype and the claws gain it in conaffinity, so only
    that pair is new: the walking legs' conaffinity lacks the bit, the body's
    conaffinity stays 0, and the floor still meets everything through bit 1.
    """
    feet = "|".join(f"^{n}$" for n in FOOT_GEOMS)
    out = [
        CollisionCfg(
            geom_names_expr=tuple(f"^{a}_.*_meshcol$" for a in ARMS),
            contype={f"^{a}_": K._GROUND_BIT | K._LEG_BIT[a] for a in ARMS},
            conaffinity={f"^{a}_": (K._ALL_LEG_BITS & ~K._LEG_BIT[a]) | BODY_BIT
                         for a in ARMS},
            condim={feet: 3, ".*": 1},
            priority={feet: 1, ".*": 0},
            disable_other_geoms=False,
        ),
        CollisionCfg(
            geom_names_expr=BODY_GEOMS,
            contype=K._GROUND_BIT | BODY_BIT, conaffinity=0, condim=1, priority=0,
            disable_other_geoms=False,
        ),
    ]
    if carapace_geoms is not None:
        out.append(CollisionCfg(
            geom_names_expr=(carapace_geoms,),
            contype=K._GROUND_BIT | BODY_BIT, conaffinity=0, condim=3, priority=0,
            friction=carapace_friction, solref=K._CONTACT_SOLREF, solimp=K._CONTACT_SOLIMP,
            disable_other_geoms=False,
        ))
    return tuple(out)


def leg_pair_sensors(decimation: int, free_pairs: tuple[frozenset, ...] = ()
                     ) -> tuple[ContactSensorCfg, ...]:
    """One sensor per leg against every leg before it in `LEGS`, pairs in `free_pairs`
    left out, named `leg_pair_<leg>`, keeping `decimation` substeps of force."""
    out = []
    for j, leg in enumerate(LEGS):
        others = [o for o in LEGS[:j] if frozenset((o, leg)) not in free_pairs]
        if not others:
            continue
        out.append(ContactSensorCfg(
            name=f"leg_pair_{leg}",
            primary=ContactMatch(mode="subtree",
                                 pattern=tuple(f"^{LEG_ROOT[o]}$" for o in others),
                                 entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern=LEG_ROOT[leg], entity="robot"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            history_length=decimation,
        ))
    return tuple(out)


def leg_collision_pairs(env, sensor_names: tuple[str, ...],
                        force_threshold: float = 1.0) -> torch.Tensor:
    """1 in a step in which any watched pair of legs pushed on each other harder than
    `force_threshold` in any substep, 0 otherwise."""
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in sensor_names:
        h = env.scene[name].data.force_history
        assert h is not None, f"{name} keeps no force history"
        hit |= (h.norm(dim=-1) > force_threshold).flatten(1).any(dim=1)
    return hit.float()


__all__ = ["ARMS", "BODY_BIT", "BODY_GEOMS", "LEG_ROOT", "arm_body_collision",
           "leg_collision_pairs", "leg_pair_sensors"]
