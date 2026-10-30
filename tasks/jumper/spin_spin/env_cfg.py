"""jumper.spin_spin -- the spin roll, free to pivot on the spot.

jumper.spin_roll (the version that lets the roll drift off its demonstration's axis
once the guide fades) with one term removed: `axis_heading`, jumper.horn_roll's charge
on the roll axis turning in the world, which is what a robot pivoting about the
vertical does. So the roll may wind up into a spin about the vertical as well.

Still in force: the turn is counted about the demonstration's axis and a roll finishes
only when that count reaches a full turn and the robot stands (so the roll itself
stays within roughly 45-60 deg of that axis); `axis_drift`, the charge on sliding
along the roll axis rather than across it; and every gentleness term -- face, claw and
limb load, impacts, air time, floor-contact joint torque.
"""

from __future__ import annotations

from pathlib import Path

from ..spin_roll import env_cfg as spin


def env_cfg(asset: Path | None = None, play: bool = False):
    """jumper.spin_roll without the charge on pivoting."""
    cfg = spin.env_cfg(asset=asset, play=play)
    cfg.rewards.pop("axis_heading")
    return cfg


__all__ = ["env_cfg"]
