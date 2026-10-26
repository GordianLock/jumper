"""Regenerate scene.xml: jumper.roll's training scene (floor, contact settings, 5 ms
step, sensors), dumped from the environment so plain MuJoCo can use it.

usage (from the repo root, after `pip install -e .`):
    python studies/carapace/dump_scene.py
"""
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

cfg = tasks.load_env_cfg("jumper.roll")
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
xml = env.scene.spec.to_xml()
# mjlab writes an unnamed nested <default/>, which MjSpec.from_string rejects
xml = xml.replace("  <default>\n    <default/>\n", "  <default>\n", 1)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.xml")
open(out, "w").write(xml)
print("wrote", out)
