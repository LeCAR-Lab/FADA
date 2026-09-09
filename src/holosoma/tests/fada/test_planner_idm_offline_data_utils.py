from __future__ import annotations

from pathlib import Path

import numpy as np

from holosoma.fada.planner_idm.offline_data_utils import (
    OfflineRolloutData,
    flatten_offline_rollout_to_replay_batch,
    list_model_checkpoints,
    select_optimal_checkpoint,
    select_suboptimal_checkpoints,
)


def _build_rollout_data() -> OfflineRolloutData:
    horizon = 4
    num_envs = 2
    obs_dim = 2
    act_dim = 1
    cmd_dim = 1

    obs = np.arange(horizon * num_envs * obs_dim, dtype=np.float32).reshape(horizon, num_envs, obs_dim)
    current_command = (np.arange(horizon * num_envs * cmd_dim, dtype=np.float32) + 100.0).reshape(
        horizon,
        num_envs,
        cmd_dim,
    )
    executed_act = (np.arange(horizon * num_envs * act_dim, dtype=np.float32) + 200.0).reshape(
        horizon,
        num_envs,
        act_dim,
    )
    expert_act = executed_act + 10.0
    next_obs = obs + 1000.0
    reward = np.ones((horizon, num_envs), dtype=np.float32)
    done = np.zeros((horizon, num_envs), dtype=np.bool_)
    done[-1, :] = True
    done[1, 0] = True
    failure = np.zeros((horizon, num_envs), dtype=np.bool_)
    failure[1, 0] = True

    return OfflineRolloutData(
        obs=obs,
        current_command=current_command,
        executed_act=executed_act,
        expert_act=expert_act,
        next_obs=next_obs,
        reward=reward,
        done=done,
        failure=failure,
    )


def test_optimal_filter_drops_early_done_envs() -> None:
    filtered = flatten_offline_rollout_to_replay_batch(
        _build_rollout_data(),
        mode="optimal",
        pred_horizon=1,
        episode_ids=np.asarray([0, 0], dtype=np.int64),
    )
    assert filtered.kept_episodes == 1
    assert filtered.dropped_episodes == 1
    assert filtered.kept_steps == 4
    assert np.all(filtered.env_id == 1)
    assert filtered.strict_label_valid.tolist() == [True, True, True, True]


def test_suboptimal_filter_keeps_truncated_episode() -> None:
    filtered = flatten_offline_rollout_to_replay_batch(
        _build_rollout_data(),
        mode="suboptimal",
        pred_horizon=1,
        episode_ids=np.asarray([7, 11], dtype=np.int64),
    )
    assert filtered.kept_episodes == 2
    assert filtered.dropped_episodes == 0
    assert filtered.kept_steps == 6
    assert filtered.env_id[:2].tolist() == [0, 0]
    assert filtered.episode_id[:2].tolist() == [7, 7]
    assert filtered.strict_label_valid[:2].tolist() == [True, False]
    assert filtered.done[:2].tolist() == [False, True]


def test_checkpoint_selection_uses_last_for_optimal_and_excludes_last_for_suboptimal(tmp_path) -> None:
    checkpoints = []
    for step in (10, 20, 30, 40, 50):
        checkpoint = tmp_path / f"model_{step}.pt"
        checkpoint.write_text("x", encoding="utf-8")
        checkpoints.append(checkpoint)

    discovered = list_model_checkpoints(tmp_path)
    assert discovered == checkpoints
    assert select_optimal_checkpoint(discovered) == checkpoints[-1]

    reward_map = {10: 0.1, 20: 0.2, 30: 0.95, 40: 0.8, 50: 1.2}
    sampled = select_suboptimal_checkpoints(
        discovered,
        n_target=2,
        sampling_mode="reward_aware",
        reward_map=reward_map,
    )
    assert len(sampled) == 2
    assert checkpoints[-1] not in sampled


def test_suboptimal_reward_aware_falls_back_to_logspace_when_reward_curve_missing(tmp_path) -> None:
    checkpoints = []
    for step in (10, 20, 30, 40):
        checkpoint = tmp_path / f"model_{step}.pt"
        checkpoint.write_text("x", encoding="utf-8")
        checkpoints.append(checkpoint)

    sampled = select_suboptimal_checkpoints(
        checkpoints,
        n_target=2,
        sampling_mode="reward_aware",
        reward_map={},
    )
    expected = select_suboptimal_checkpoints(
        checkpoints,
        n_target=2,
        sampling_mode="logspace",
        reward_map={},
    )
    assert sampled == expected
