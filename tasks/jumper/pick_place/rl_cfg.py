"""PPO hyperparameters for jumper.pick_place."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.pick_place",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # 0.5 explores 0.375 rad on the arm and 0.125 on the legs: the arm has to
        # find the floor and the tray on its own.
        init_std=0.5,
        # Per joint, so the policy can quiet its stance legs while the arm keeps
        # exploring. With one shared std, leg noise had the robot shuffling: the
        # stance and drift costs were 95% of the penalty in a 300-iteration CPU check.
        std_type="log",
        entropy_coef=0.005,
        gamma=0.99,
        lam=0.95,
        num_steps_per_env=24,
        max_iterations=8000,
        # Off: the robot works with its left claw only, so it is not left-right
        # symmetric (the five-foot task's reason too).
        symmetry=False,
    )
