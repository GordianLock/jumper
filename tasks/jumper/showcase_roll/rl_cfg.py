"""PPO hyperparameters for jumper.showcase_roll."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline
from ..flourish_roll.rl_cfg import NUM_STEPS_PER_ENV


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.showcase_roll",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=0.5,
        entropy_coef=0.006,
        # 0.998 at 50 Hz is a 10 s horizon. At jumper.flourish_roll's 0.995 a
        # bonus 4.2 s away was worth 0.35 of one 0.8 s away; here 0.66.
        gamma=0.998,
        lam=0.95,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=2500,
        # Every 250 iterations rather than 50: jumper5's Colab backup was 452 MB,
        # 50 checkpoints of 9.2 MB each, too large to hand over whole.
        save_interval=250,
        symmetry=False,
    )
