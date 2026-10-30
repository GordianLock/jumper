"""PPO hyperparameters for jumper.spin_spin: jumper.spin_roll's, under this task's
experiment name."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..spin_roll import rl_cfg as spin


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = spin.agent_cfg()
    cfg.experiment_name = "jumper.spin_spin"
    return cfg
