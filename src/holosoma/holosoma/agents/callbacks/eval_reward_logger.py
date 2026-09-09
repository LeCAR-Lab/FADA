"""Minimal eval reward logger callback.

Aggregates per-step rewards and per-episode reward sums during eval, prints
periodic summaries, and dumps a final JSON. Compatible with both single-actor
PPO and multi-actor PPO-MA (uses only ``actor_state["rewards"]`` /
``actor_state["dones"]`` /  ``actor_state["extras"]`` which both algos
populate).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback


class EvalRewardLogger(RLEvalCallback):
    def __init__(self, config, training_loop):
        super().__init__(config, training_loop)
        self.print_every = int(getattr(config, "print_every", 100))
        self.env = self.training_loop.env
        self.num_envs = self.env.num_envs
        log_dir = getattr(config, "log_dir", None) or self.training_loop.log_dir or "."
        self._out_path = Path(log_dir) / "eval_rewards.json"

        self._step_count = 0
        self._step_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._step_reward_count = 0
        self._running_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._completed_returns: list[float] = []
        self._episode_lengths: list[int] = []
        self._cur_ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-term episode reward accumulator: term_name → list of per-episode means.
        self._completed_term_returns: dict[str, list[float]] = {}

    def on_post_eval_env_step(self, actor_state):
        rewards = actor_state["rewards"]  # (N,)
        dones = actor_state["dones"]      # (N,)
        extras = actor_state.get("extras", {})

        self._step_reward_sum += rewards.float()
        self._step_reward_count += 1
        self._running_return += rewards.float()
        self._cur_ep_len += 1

        done_mask = dones.bool() if dones.dtype != torch.bool else dones
        if done_mask.any():
            done_returns = self._running_return[done_mask].detach().cpu().numpy().tolist()
            done_lens = self._cur_ep_len[done_mask].detach().cpu().numpy().tolist()
            self._completed_returns.extend(done_returns)
            self._episode_lengths.extend(done_lens)
            self._running_return[done_mask] = 0.0
            self._cur_ep_len[done_mask] = 0

            # Episode-level per-term rewards (populated by reward_manager.reset on done).
            ep_extras = extras.get("episode", {}) or {}
            for k, v in ep_extras.items():
                if isinstance(v, torch.Tensor):
                    self._completed_term_returns.setdefault(k, []).append(float(v.mean().detach().cpu()))
                elif isinstance(v, (int, float)):
                    self._completed_term_returns.setdefault(k, []).append(float(v))

        self._step_count += 1
        if self._step_count % self.print_every == 0:
            mean_step = (self._step_reward_sum.mean() / max(self._step_reward_count, 1)).item()
            mean_ret = (sum(self._completed_returns) / len(self._completed_returns)) if self._completed_returns else float("nan")
            mean_len = (sum(self._episode_lengths) / len(self._episode_lengths)) if self._episode_lengths else float("nan")
            logger.info(
                f"[eval-rew] step={self._step_count} mean_step_rew={mean_step:.4f} "
                f"completed_eps={len(self._completed_returns)} mean_ep_return={mean_ret:.3f} mean_ep_len={mean_len:.1f}"
            )
        return actor_state

    def on_post_evaluate_policy(self):
        mean_step = (self._step_reward_sum.mean() / max(self._step_reward_count, 1)).item()
        mean_ret = (sum(self._completed_returns) / len(self._completed_returns)) if self._completed_returns else float("nan")
        mean_len = (sum(self._episode_lengths) / len(self._episode_lengths)) if self._episode_lengths else float("nan")
        per_term = {k: (sum(v) / len(v)) for k, v in self._completed_term_returns.items() if v}
        if "rew_tracking_lin_vel" not in per_term:
            lin_parts = [
                per_term[key]
                for key in ("rew_tracking_lin_vel_x", "rew_tracking_lin_vel_y")
                if key in per_term
            ]
            if lin_parts:
                per_term["rew_tracking_lin_vel"] = float(sum(lin_parts))
        summary = {
            "total_steps": self._step_count,
            "mean_step_reward": mean_step,
            "completed_episodes": len(self._completed_returns),
            "mean_episode_return": mean_ret,
            "mean_episode_length": mean_len,
            "per_term_episode_rewards": per_term,
        }
        try:
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._out_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"[eval-rew] summary written to {self._out_path}")
        except Exception as e:
            logger.warning(f"[eval-rew] failed to write summary: {e}")
        logger.info(f"[eval-rew] FINAL: {summary}")
