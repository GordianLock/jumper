"""PPO hyperparameters for jumper.roll."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.roll",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # 0.4 at the action scale of 0.75 explores 0.3 rad of target: wide, because
        # the roll needs joints far from HOME and nothing feeds it a recording.
        init_std=0.4,
        entropy_coef=0.005,
        # 50 Hz, as the walking tasks: 0.99 is a 2 s horizon, about a slow roll.
        gamma=0.99,
        lam=0.95,
        num_steps_per_env=24,
        max_iterations=5000,
        # Every 250 iterations rather than 50: a Colab backup of 50 checkpoints
        # (jumper5's, 452 MB) is too large to hand over whole.
        save_interval=250,
        # **Off, and it has to be.** The roll goes right side first; its mirror
        # image goes left side first. Augmenting with mirrored samples would teach
        # the actor both from the same observation. A left roll is the mirrored
        # policy on the robot, not a second training run.
        symmetry=False,
    )
