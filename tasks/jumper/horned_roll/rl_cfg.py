"""PPO hyperparameters for jumper.horned_roll: jumper.gentle_momentum_roll's (2026-10-28),
written out rather than imported so that task can change without moving this one."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update; `env_cfg.py` converts the guide's schedule
#: from iterations with it.
NUM_STEPS_PER_ENV = 24

#: A 10 s horizon at 50 Hz, so progress paid as the roll is made barely favours a
#: quicker roll (jumper.gentle_momentum_roll's reasoning).
GAMMA = 0.998


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.horned_roll",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=0.4,
        entropy_coef=0.003,
        gamma=GAMMA,
        lam=0.95,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        # The guide is gone by 1200 (`PRIOR_END_ITERATIONS`).
        max_iterations=2500,
        save_interval=250,
        # Off: the roll goes right side first, as the demonstration does.
        symmetry=False,
    )
