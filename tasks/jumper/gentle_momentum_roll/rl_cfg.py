"""PPO hyperparameters for jumper.gentle_momentum_roll: jumper.momentum_roll's."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update. A module constant because `env_cfg.py`
#: converts the guide's schedule from iterations into the steps the curriculum
#: counts (jumper.momentum_roll's reason).
NUM_STEPS_PER_ENV = 24

#: The discount. A module constant because the progress shaping in `env_cfg.py`
#: must use the same one, or it stops being neutral about time (`mdp/rewards.py`).
GAMMA = 0.99


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.gentle_momentum_roll",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=0.4,
        entropy_coef=0.003,
        gamma=GAMMA,
        lam=0.95,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        # The guide is gone by 1200 (`PRIOR_END_ITERATIONS`); the rest is the
        # objective on its own.
        max_iterations=2500,
        save_interval=250,
        # Off: the roll goes right side first, as the demonstration does.
        symmetry=False,
    )
