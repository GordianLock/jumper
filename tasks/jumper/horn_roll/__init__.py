"""jumper.horn_roll -- a gentle roll on a horned carapace, from a demonstration found on
the carapace itself, with the horns free to be pivoted on. See `env_cfg.py`."""

from __future__ import annotations

from ...paths import ASSETS_DIR
from ...registry import AssetSpec, register
from ..common.assets import JUMPER_ASSETS

#: Carapace prototypes this task trains on, from design/carapace (the carapace design
#: session). The first is the default. Any model with `upper_shell_carapace_NN_meshcol`
#: pieces also works by path (`--model assets/jumper/....xml`).
HORN_ROLL_ASSETS = (
    AssetSpec(
        name="CF1-tri30m-rb-v3",
        path=ASSETS_DIR / "jumper" / "jumper_CF1-tri30m-rb-v3.xml",
        description="carapace prototype 3: centre horn 25 deg, front corner horns 28 deg, "
                    "three rear bumpers a side; belly-up rest 24 deg head-up",
    ),
    AssetSpec(
        name="CF1-tri30m-rb-v3b",
        path=ASSETS_DIR / "jumper" / "jumper_CF1-tri30m-rb-v3b.xml",
        description="prototype 3 with the front corner horns at 20 deg",
    ),
    AssetSpec(
        name="CF1-tri30m-rb-v4",
        path=ASSETS_DIR / "jumper" / "jumper_CF1-tri30m-rb-v4.xml",
        description="carapace prototype 4: prototype 3b with the centre horn at 15 deg "
                    "ending in a rocker, the spine scalloped",
    ),
    AssetSpec(
        name="CF1-tri30m-rb-v4b",
        path=ASSETS_DIR / "jumper" / "jumper_CF1-tri30m-rb-v4b.xml",
        description="prototype 4 with the spine's back end eased onto the shell",
    ),
    AssetSpec(
        name="CF1-tri30m-rb-v4c",
        path=ASSETS_DIR / "jumper" / "jumper_CF1-tri30m-rb-v4c.xml",
        description="prototype 4b with the faces beside the spine's end eased",
    ),
    *JUMPER_ASSETS,
)

register(
    id="jumper.horn_roll",
    assets=HORN_ROLL_ASSETS,
    description=(
        "jumper hexapod roll on a horned carapace: a fading demonstration found on the "
        "carapace, horns free to pivot on, impacts kept gentle, no reward for airtime"
    ),
    tags=("jumper", "roll", "one-shot", "action-prior", "carapace"),
)
