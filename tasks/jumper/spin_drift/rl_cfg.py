"""PPO hyperparameters for jumper.spin_drift: jumper.tuck_roll's, under this task's
experiment name."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..tuck_roll import rl_cfg as tuck


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = tuck.agent_cfg()
    cfg.experiment_name = "jumper.spin_drift"
    return cfg
