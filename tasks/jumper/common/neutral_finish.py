"""Finish, then settle at HOME: an episode that does not end the moment the move is done.

A one-shot task (a roll, a self-right) ends its episode on the step its finish
predicate first holds -- standing after the roll -- and pays its finishing bonus
there. Nothing after that step is ever trained, so the policy finishes in whatever
pose got it there: jumper.self_right's run 3 stood with a leg in the air, jumper.
spin_drift's run 1 hopped once 0.2 s after standing (both replayed 2026-10-30, past
the finish).

`NeutralFinish`, a class-based termination term, keeps the episode going after the
finish for at most `window_steps` control steps, and ends it once the robot is
**settled**: the finish predicate still holding, every listed joint within
`home_tol` of HOME, the body turning slower than `still_ang_vel` and moving slower
than `still_lin_vel`, all of it for `hold_steps` running. Its sibling
`settle_timeout` ends the episodes that run out of window unsettled, so the training
log shows the two apart (`Episode_Termination/<name>`).

The rewards read its state through `term(env, name)`:

- `once`: the wrapped reward (the task's finishing bonus) only on the finish step, so
  a bonus that pays whenever the predicate holds still pays once.
- `until_finish`: the wrapped reward (approach shaping) up to and including the
  finish step, and not after, as when the episode ended there.
- `settle_home`: after the finish, while the predicate holds, legs near HOME --
  exp(-mean squared distance / `sigma_sq`) per step. Small: its total over the window
  must stay below `settled`, or lingering out the window would pay better.
- `settled`: 1 on the step the robot counts as settled, else 0.

The task's main finish and bonus are untouched, so the move itself is learnt as
before; settling is an addition on top. Mechanism only: every threshold is the
task's, passed in the term's params.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


class NeutralFinish:
    """Termination: settled at HOME after the finish (module docstring)."""

    def __init__(self, cfg, env):
        n, dev = env.num_envs, env.device
        self._env = env
        self.first = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.held = torch.zeros(n, dtype=torch.long, device=dev)
        self.just = torch.zeros(n, dtype=torch.bool, device=dev)
        self.holding = torch.zeros(n, dtype=torch.bool, device=dev)
        self.timed_out = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hold_steps = int(cfg.params["hold_steps"])
        self._stamp = -1

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.first[ids] = -1
        self.held[ids] = 0
        self.just[ids] = False
        self.holding[ids] = False
        self.timed_out[ids] = False

    @property
    def after(self) -> torch.Tensor:
        """The finish has happened this episode (this step included)."""
        return self.first >= 0

    def __call__(self, env, finish, finish_params: dict, asset_cfg: SceneEntityCfg,
                 home_tol: float, still_ang_vel: float, still_lin_vel: float,
                 hold_steps: int, window_steps: int) -> torch.Tensor:
        if self._stamp != env.common_step_counter:       # once per control step
            self._stamp = env.common_step_counter
            steps = env.episode_length_buf
            fin = finish(env, **finish_params)
            self.just = fin & (self.first < 0)
            self.first = torch.where(self.just, steps, self.first)
            self.holding = fin
            robot = env.scene[asset_cfg.name]
            q = robot.data.joint_pos[:, asset_cfg.joint_ids]
            home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
            near = (q - home).abs().max(dim=1).values < home_tol
            still = ((robot.data.root_link_ang_vel_b.norm(dim=1) < still_ang_vel)
                     & (robot.data.root_link_lin_vel_b.norm(dim=1) < still_lin_vel))
            ok = self.after & fin & near & still
            self.held = torch.where(ok, self.held + 1, torch.zeros_like(self.held))
            settled = self.held >= hold_steps
            self.timed_out = self.after & (steps - self.first >= window_steps) & ~settled
        return self.held >= hold_steps


def term(env, name: str) -> NeutralFinish:
    return env.termination_manager.get_term_cfg(name).func


def settle_timeout(env, name: str = "settled") -> torch.Tensor:
    """Termination, listed after `name`: the window ran out unsettled."""
    return term(env, name).timed_out


def once(env, func, name: str = "settled", **params) -> torch.Tensor:
    return func(env, **params) * term(env, name).just.float()


def until_finish(env, func, name: str = "settled", **params) -> torch.Tensor:
    t = term(env, name)
    return func(env, **params) * (~t.after | t.just).float()


def settle_home(env, asset_cfg: SceneEntityCfg, sigma_sq: float,
                name: str = "settled") -> torch.Tensor:
    t = term(env, name)
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    k = torch.exp(-((q - home) ** 2).mean(dim=1) / sigma_sq)
    return k * (t.after & ~t.just & t.holding).float()


def settled(env, name: str = "settled") -> torch.Tensor:
    t = term(env, name)
    return (t.held == t.hold_steps).float()



def settle_after_finish(cfg, finish: str, asset_cfg: SceneEntityCfg, *, home_tol: float,
                        still_ang_vel: float, still_lin_vel: float, hold_steps: int,
                        window_steps: int, settle_home_weight: float,
                        settle_sigma_sq: float, settled_weight: float,
                        once_rewards: tuple[str, ...], until_rewards: tuple[str, ...],
                        name: str = "settled") -> None:
    """Rewire a task's cfg in place: the termination `finish` (its predicate and
    params) becomes `NeutralFinish`'s finish and is removed; `name` and
    `settle_timeout` end the episode instead; the rewards in `once_rewards` pay on the
    finish step only, those in `until_rewards` up to it; `settle_home` and `settled`
    are added with the given weights (the caller divides a one-shot bonus by the step,
    as the rest of the task does)."""
    from mjlab.managers import TerminationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg

    old = cfg.terminations.pop(finish)
    terms = dict(cfg.terminations)
    terms[name] = TerminationTermCfg(
        func=NeutralFinish,
        params={"finish": old.func, "finish_params": dict(old.params), "asset_cfg": asset_cfg,
                "home_tol": home_tol, "still_ang_vel": still_ang_vel,
                "still_lin_vel": still_lin_vel, "hold_steps": hold_steps,
                "window_steps": window_steps})
    terms["settle_timeout"] = TerminationTermCfg(func=settle_timeout, params={"name": name})
    cfg.terminations = terms
    r = dict(cfg.rewards)
    for key, wrap in [(k, once) for k in once_rewards] + [(k, until_finish) for k in until_rewards]:
        t = r[key]
        r[key] = RewardTermCfg(func=wrap, weight=t.weight,
                               params={**t.params, "func": t.func, "name": name})
    r["settle_home"] = RewardTermCfg(func=settle_home, weight=settle_home_weight,
                                     params={"asset_cfg": asset_cfg,
                                             "sigma_sq": settle_sigma_sq, "name": name})
    r["settled"] = RewardTermCfg(func=settled, weight=settled_weight, params={"name": name})
    cfg.rewards = r


__all__ = ["NeutralFinish", "once", "settle_after_finish", "settle_home", "settle_timeout",
           "settled", "term", "until_finish"]
