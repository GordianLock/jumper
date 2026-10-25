"""PPO hyperparameters for jumper.momentum_roll."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update. A module constant because `env_cfg.py`
#: converts the prior's schedule from iterations into the steps the curriculum
#: counts; a literal in both places could drift apart in silence.
NUM_STEPS_PER_ENV = 24


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.momentum_roll",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # The feedforward carries the motion at first, so exploration rides on
        # top of a roll rather than having to find one: jumper.roll's 0.4.
        init_std=0.4,
        entropy_coef=0.003,
        gamma=0.99,
        lam=0.95,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        # The prior is gone by 1200 (`PRIOR_END_ITERATIONS`); the rest is the
        # objective on its own.
        max_iterations=2500,
        # Every 250 iterations rather than 50: a Colab backup of 50 checkpoints
        # (jumper5's, 452 MB) is too large to hand over whole.
        save_interval=250,
        # Off: the roll goes right side first, and a mirrored sample would teach
        # the other way round from the same observation.
        symmetry=False,
    )
