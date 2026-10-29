"""PPO hyperparameters for jumper.tuck_roll: jumper.horn_roll's (jumper.momentum_roll's),
under this task's experiment name."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..horn_roll import rl_cfg as horn


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = horn.agent_cfg()
    cfg.experiment_name = "jumper.tuck_roll"
    return cfg
