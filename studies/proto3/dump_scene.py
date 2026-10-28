"""Write scene.xml and options.json: jumper.roll's training scene on carapace prototype 3
(CF1-tri30m-rb-v3), with the carapace colliding, dumped from the environment.

The XML does not carry the solver options (mjlab sets them on the compiled model), so
they are written to options.json from the environment's compiled model and applied
after compiling (`search.build`). The horn study's scene lacked them and ran at
MuJoCo's defaults (2 ms Euler); this one runs the training physics.

    python studies/proto3/dump_scene.py        # from the repo root
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tasks  # noqa: E402
from mjrl.backend.resolve import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402

use_backend(resolve("native", "cpu", num_envs=1, training=False))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

MODEL = ROOT / "assets" / "jumper" / "jumper_CF1-tri30m-rb-v3.xml"
cfg = tasks.load_env_cfg("jumper.roll", asset=MODEL)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
here = os.path.dirname(os.path.abspath(__file__))
xml = env.scene.spec.to_xml().replace("  <default>\n    <default/>\n", "  <default>\n", 1)
open(os.path.join(here, "scene.xml"), "w").write(xml)
o = env.sim.mj_model.opt
opts = {k: (getattr(o, k).tolist() if hasattr(getattr(o, k), "tolist") else getattr(o, k))
        for k in ("timestep", "integrator", "solver", "cone", "iterations", "ls_iterations",
                  "tolerance", "ls_tolerance", "ccd_iterations", "ccd_tolerance", "impratio",
                  "noslip_iterations", "gravity", "disableflags", "enableflags")}
opts = {k: int(v) if hasattr(v, "__int__") and not isinstance(v, (float, list)) else v
        for k, v in opts.items()}
json.dump(opts, open(os.path.join(here, "options.json"), "w"), indent=1)
print("wrote scene.xml and options.json:", opts)
