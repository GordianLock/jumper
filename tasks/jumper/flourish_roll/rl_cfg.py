"""PPO hyperparameters for jumper.flourish_roll."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update; `env_cfg.py` converts the bootstrap's
#: schedule from iterations with it.
NUM_STEPS_PER_ENV = 24


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.flourish_roll",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # Wider exploration and a larger entropy bonus than the efficient rolls:
        # this task wants the policy to keep trying things.
        init_std=0.5,
        entropy_coef=0.006,
        # A slow roll pays off seconds after its moves: 0.995 at 50 Hz is a 4 s
        # horizon, against 2 s for the efficient rolls' 0.99.
        gamma=0.995,
        lam=0.95,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=3000,
        # Off: the roll goes right side first.
        symmetry=False,
    )
