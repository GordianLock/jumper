"""PPO hyperparameters for jumper.self_right: jumper.roll's (no demonstration, wide
exploration), under this task's name, with jumper.roll's run length cut to 2500."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..roll import rl_cfg as roll


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = roll.agent_cfg()
    cfg.experiment_name = "jumper.self_right"
    cfg.max_iterations = 2500
    # Symmetry stays off (jumper.roll's reason): mirrored samples would teach both
    # ways over from the same observation, and the policy may pick either side.
    return cfg
