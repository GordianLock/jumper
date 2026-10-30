"""PPO hyperparameters for jumper.lie_down: jumper.self_right's, under this task's
name."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..self_right import rl_cfg as sr


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = sr.agent_cfg()
    cfg.experiment_name = "jumper.lie_down"
    return cfg
