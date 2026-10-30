"""PPO hyperparameters for jumper.spin_roll_axis: jumper.spin_drift's, under this task's
experiment name."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..spin_drift import rl_cfg as spin


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = spin.agent_cfg()
    cfg.experiment_name = "jumper.spin_roll_axis"
    return cfg
