"""PPO hyperparameters for jumper.horn_roll: jumper.momentum_roll's, under this task's
experiment name. The guide's schedule is counted in jumper.momentum_roll's
`NUM_STEPS_PER_ENV`, which this task shares by building on its config."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..momentum_roll import rl_cfg as momentum


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = momentum.agent_cfg()
    cfg.experiment_name = "jumper.horn_roll"
    return cfg
