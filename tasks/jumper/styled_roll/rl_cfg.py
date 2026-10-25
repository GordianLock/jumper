"""PPO hyperparameters for jumper.styled_roll: jumper.momentum_roll's, under this
task's experiment name. The feedforward schedule is counted in its
`NUM_STEPS_PER_ENV`, which this task shares by building on its config."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline
from ..momentum_roll.rl_cfg import NUM_STEPS_PER_ENV


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.styled_roll",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=0.4,
        entropy_coef=0.003,
        gamma=0.99,
        lam=0.95,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=2500,
        # Off: the roll goes right side first (jumper.momentum_roll's reason).
        symmetry=False,
    )
