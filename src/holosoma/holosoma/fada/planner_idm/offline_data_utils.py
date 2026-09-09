from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


OfflineFilterMode = Literal["optimal", "suboptimal"]
CheckpointSamplingMode = Literal["logspace", "reward_aware"]


@dataclasses.dataclass(frozen=True)
class OfflineRolloutData:
    obs: np.ndarray
    current_command: np.ndarray
    executed_act: np.ndarray
    expert_act: np.ndarray
    next_obs: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    failure: np.ndarray


@dataclasses.dataclass(frozen=True)
class OracleRelabeledChunks:
    """Pre-computed oracle future obs/actions for each step in a suboptimal rollout.

    Shapes: expert_chunk [T, E, K, act_dim], expert_future_obs_chunk [T, E, K, obs_dim],
    valid [T, E] (True if oracle K-step rollout succeeded without terminal).
    """

    expert_chunk: np.ndarray
    expert_future_obs_chunk: np.ndarray
    valid: np.ndarray


@dataclasses.dataclass(frozen=True)
class FilteredReplayBatch:
    obs: np.ndarray
    current_command: np.ndarray
    executed_act: np.ndarray
    expert_act: np.ndarray
    expert_chunk: np.ndarray
    expert_future_obs_chunk: np.ndarray
    strict_label_valid: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    env_id: np.ndarray
    episode_id: np.ndarray
    kept_episodes: int
    dropped_episodes: int
    kept_steps: int


def extract_checkpoint_step(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def list_model_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("model_*.pt"), key=extract_checkpoint_step)
    return [checkpoint for checkpoint in checkpoints if extract_checkpoint_step(checkpoint) >= 0]


def select_optimal_checkpoint(checkpoints: Sequence[Path]) -> Path:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    return sorted(checkpoints, key=extract_checkpoint_step)[-1]


def sample_checkpoints_logspace(checkpoints: Sequence[Path], n_target: int) -> list[Path]:
    checkpoints = sorted(checkpoints, key=extract_checkpoint_step)
    if not checkpoints:
        return []
    if n_target <= 0:
        raise ValueError("n_target must be positive")
    if n_target >= len(checkpoints):
        return list(checkpoints)

    base = np.geomspace(1, len(checkpoints), num=n_target)
    indices = sorted({max(0, min(len(checkpoints) - 1, int(round(value - 1)))) for value in base})
    indices.append(len(checkpoints) - 1)
    indices = sorted(set(indices))

    sampled = [checkpoints[index] for index in indices]
    while len(sampled) > n_target:
        sampled.pop(-2)
    while len(sampled) < n_target:
        chosen_steps = [extract_checkpoint_step(checkpoint) for checkpoint in sampled]
        candidates = [checkpoint for checkpoint in checkpoints if checkpoint not in sampled]
        if not candidates:
            break
        checkpoint = max(
            candidates,
            key=lambda candidate: min(
                abs(extract_checkpoint_step(candidate) - chosen_step)
                for chosen_step in chosen_steps
            ),
        )
        sampled.append(checkpoint)
        sampled = sorted(sampled, key=extract_checkpoint_step)
    return sampled


def sample_checkpoints_reward_aware(
    checkpoints: Sequence[Path],
    n_target: int,
    reward_map: dict[int, float],
) -> list[Path]:
    checkpoints = sorted(checkpoints, key=extract_checkpoint_step)
    if not checkpoints:
        return []
    if not reward_map:
        return []

    def _nearest_checkpoint(step: int) -> Path:
        return min(checkpoints, key=lambda checkpoint: abs(extract_checkpoint_step(checkpoint) - step))

    chosen: set[Path] = set(sample_checkpoints_logspace(checkpoints, n_target))
    rewards = np.asarray(list(reward_map.values()), dtype=np.float64)
    thresholds = [float(np.quantile(rewards, quantile)) for quantile in (0.50, 0.70, 0.85, 0.95, 1.00)]
    reward_items = sorted(reward_map.items(), key=lambda item: item[0])
    for threshold in thresholds:
        candidates = [step for step, reward in reward_items if reward >= threshold]
        if not candidates:
            continue
        chosen.add(_nearest_checkpoint(candidates[0]))

    sampled = sorted(chosen, key=extract_checkpoint_step)
    if len(sampled) > n_target:
        sampled = sample_checkpoints_logspace(sampled, n_target)
    return sampled


def select_suboptimal_checkpoints(
    checkpoints: Sequence[Path],
    *,
    n_target: int,
    sampling_mode: CheckpointSamplingMode,
    reward_map: dict[int, float] | None = None,
) -> list[Path]:
    checkpoints = sorted(checkpoints, key=extract_checkpoint_step)
    if n_target <= 0:
        raise ValueError("n_target must be positive")
    if len(checkpoints) <= 1:
        return []

    middle_checkpoints = checkpoints[:-1]
    if n_target >= len(middle_checkpoints):
        return list(middle_checkpoints)
    if sampling_mode == "reward_aware":
        sampled = sample_checkpoints_reward_aware(middle_checkpoints, n_target, reward_map or {})
        if sampled:
            return sampled
        return sample_checkpoints_logspace(middle_checkpoints, n_target)
    if sampling_mode == "logspace":
        return sample_checkpoints_logspace(middle_checkpoints, n_target)
    raise ValueError(f"unsupported checkpoint sampling mode: {sampling_mode}")


def _validate_rollout_data(data: OfflineRolloutData) -> tuple[int, int, int, int]:
    if data.obs.ndim != 3:
        raise ValueError(f"obs must have shape [T, E, obs_dim], got {data.obs.shape}")
    horizon, num_envs, obs_dim = data.obs.shape
    if horizon <= 0 or num_envs <= 0 or obs_dim <= 0:
        raise ValueError("obs must have positive [T, E, obs_dim]")
    if data.current_command.ndim != 3 or data.current_command.shape[:2] != (horizon, num_envs):
        raise ValueError("current_command shape mismatch with obs")
    if data.executed_act.ndim != 3 or data.executed_act.shape[:2] != (horizon, num_envs):
        raise ValueError("executed_act shape mismatch with obs")
    if data.expert_act.ndim != 3 or data.expert_act.shape != data.executed_act.shape:
        raise ValueError("expert_act shape mismatch with executed_act")
    if data.next_obs.ndim != 3 or data.next_obs.shape != data.obs.shape:
        raise ValueError("next_obs shape mismatch with obs")
    if data.reward.shape != (horizon, num_envs):
        raise ValueError("reward shape mismatch with obs")
    if data.done.shape != (horizon, num_envs):
        raise ValueError("done shape mismatch with obs")
    if data.failure.shape != (horizon, num_envs):
        raise ValueError("failure shape mismatch with obs")
    act_dim = int(data.executed_act.shape[2])
    cmd_dim = int(data.current_command.shape[2])
    return horizon, num_envs, obs_dim, act_dim + cmd_dim


def segment_lengths_from_failure_mask(failure: np.ndarray, mode: OfflineFilterMode) -> tuple[np.ndarray, np.ndarray]:
    if failure.ndim != 2:
        raise ValueError(f"failure must have shape [T, E], got {failure.shape}")
    horizon, num_envs = failure.shape
    lengths = np.zeros((num_envs,), dtype=np.int64)
    keep_mask = np.zeros((num_envs,), dtype=np.bool_)
    for env_id in range(num_envs):
        done_indices = np.flatnonzero(failure[:, env_id])
        first_done = int(done_indices[0]) if done_indices.size > 0 else None
        if mode == "optimal":
            early_done = first_done is not None and first_done < horizon - 1
            keep_mask[env_id] = not early_done
            lengths[env_id] = horizon if keep_mask[env_id] else 0
            continue
        if mode == "suboptimal":
            lengths[env_id] = horizon if first_done is None else first_done + 1
            keep_mask[env_id] = lengths[env_id] > 0
            continue
        raise ValueError(f"unsupported filter mode: {mode}")
    return lengths, keep_mask


def strict_valid_count_for_segment(*, segment_len: int, pred_horizon: int, terminal_failure: bool) -> int:
    if pred_horizon <= 0:
        raise ValueError("pred_horizon must be positive")
    if segment_len <= 0:
        return 0
    if terminal_failure:
        return max(0, segment_len - pred_horizon)
    return max(0, segment_len - pred_horizon + 1)


def flatten_offline_rollout_to_replay_batch(
    data: OfflineRolloutData,
    *,
    mode: OfflineFilterMode,
    pred_horizon: int,
    episode_ids: np.ndarray,
) -> FilteredReplayBatch:
    horizon, num_envs, obs_dim, _ = _validate_rollout_data(data)
    if pred_horizon <= 0:
        raise ValueError("pred_horizon must be positive")
    if episode_ids.shape != (num_envs,):
        raise ValueError(f"episode_ids must have shape ({num_envs},), got {episode_ids.shape}")

    lengths, keep_mask = segment_lengths_from_failure_mask(data.failure, mode)
    act_dim = int(data.executed_act.shape[2])
    cmd_dim = int(data.current_command.shape[2])

    obs_list: list[np.ndarray] = []
    current_command_list: list[np.ndarray] = []
    executed_act_list: list[np.ndarray] = []
    expert_act_list: list[np.ndarray] = []
    expert_chunk_list: list[np.ndarray] = []
    future_obs_chunk_list: list[np.ndarray] = []
    strict_valid_list: list[np.ndarray] = []
    reward_list: list[np.ndarray] = []
    done_list: list[np.ndarray] = []
    env_id_list: list[np.ndarray] = []
    episode_id_list: list[np.ndarray] = []

    kept_episodes = 0
    kept_steps = 0
    for env_id in range(num_envs):
        if not bool(keep_mask[env_id]):
            continue
        segment_len = int(lengths[env_id])
        if segment_len <= 0:
            continue
        terminal_failure = bool(data.failure[segment_len - 1, env_id])
        valid_count = strict_valid_count_for_segment(
            segment_len=segment_len,
            pred_horizon=pred_horizon,
            terminal_failure=terminal_failure,
        )

        expert_chunk = np.zeros((segment_len, pred_horizon, act_dim), dtype=np.float32)
        future_obs_chunk = np.zeros((segment_len, pred_horizon, obs_dim), dtype=np.float32)
        strict_valid = np.zeros((segment_len,), dtype=np.bool_)
        for anchor in range(valid_count):
            expert_chunk[anchor] = data.executed_act[anchor : anchor + pred_horizon, env_id].astype(
                np.float32,
                copy=False,
            )
            future_obs_chunk[anchor] = data.next_obs[anchor : anchor + pred_horizon, env_id].astype(
                np.float32,
                copy=False,
            )
            strict_valid[anchor] = True

        obs_list.append(data.obs[:segment_len, env_id].astype(np.float32, copy=False))
        current_command_list.append(data.current_command[:segment_len, env_id].astype(np.float32, copy=False))
        executed_act_list.append(data.executed_act[:segment_len, env_id].astype(np.float32, copy=False))
        expert_act_list.append(data.expert_act[:segment_len, env_id].astype(np.float32, copy=False))
        expert_chunk_list.append(expert_chunk)
        future_obs_chunk_list.append(future_obs_chunk)
        strict_valid_list.append(strict_valid)
        reward_list.append(data.reward[:segment_len, env_id].astype(np.float32, copy=False))
        done_list.append(data.done[:segment_len, env_id].astype(np.bool_, copy=False))
        env_id_list.append(np.full((segment_len,), env_id, dtype=np.int32))
        episode_id_list.append(np.full((segment_len,), int(episode_ids[env_id]), dtype=np.int64))
        kept_episodes += 1
        kept_steps += segment_len

    dropped_episodes = int(num_envs - kept_episodes)
    if kept_episodes <= 0:
        return FilteredReplayBatch(
            obs=np.empty((0, obs_dim), dtype=np.float32),
            current_command=np.empty((0, cmd_dim), dtype=np.float32),
            executed_act=np.empty((0, act_dim), dtype=np.float32),
            expert_act=np.empty((0, act_dim), dtype=np.float32),
            expert_chunk=np.empty((0, pred_horizon, act_dim), dtype=np.float32),
            expert_future_obs_chunk=np.empty((0, pred_horizon, obs_dim), dtype=np.float32),
            strict_label_valid=np.empty((0,), dtype=np.bool_),
            reward=np.empty((0,), dtype=np.float32),
            done=np.empty((0,), dtype=np.bool_),
            env_id=np.empty((0,), dtype=np.int32),
            episode_id=np.empty((0,), dtype=np.int64),
            kept_episodes=0,
            dropped_episodes=dropped_episodes,
            kept_steps=0,
        )

    return FilteredReplayBatch(
        obs=np.concatenate(obs_list, axis=0).astype(np.float32, copy=False),
        current_command=np.concatenate(current_command_list, axis=0).astype(np.float32, copy=False),
        executed_act=np.concatenate(executed_act_list, axis=0).astype(np.float32, copy=False),
        expert_act=np.concatenate(expert_act_list, axis=0).astype(np.float32, copy=False),
        expert_chunk=np.concatenate(expert_chunk_list, axis=0).astype(np.float32, copy=False),
        expert_future_obs_chunk=np.concatenate(future_obs_chunk_list, axis=0).astype(np.float32, copy=False),
        strict_label_valid=np.concatenate(strict_valid_list, axis=0).astype(np.bool_, copy=False),
        reward=np.concatenate(reward_list, axis=0).astype(np.float32, copy=False),
        done=np.concatenate(done_list, axis=0).astype(np.bool_, copy=False),
        env_id=np.concatenate(env_id_list, axis=0).astype(np.int32, copy=False),
        episode_id=np.concatenate(episode_id_list, axis=0).astype(np.int64, copy=False),
        kept_episodes=kept_episodes,
        dropped_episodes=dropped_episodes,
        kept_steps=kept_steps,
    )


def flatten_offline_rollout_with_oracle_chunks(
    data: OfflineRolloutData,
    oracle_chunks: OracleRelabeledChunks,
    *,
    mode: OfflineFilterMode,
    pred_horizon: int,
    episode_ids: np.ndarray,
) -> FilteredReplayBatch:
    """Like flatten_offline_rollout_to_replay_batch but substitutes oracle chunks.

    The obs/commands/rewards come from the suboptimal trajectory while expert_chunk
    and expert_future_obs_chunk come from the pre-computed oracle relabeling.
    """
    horizon, num_envs, obs_dim, _ = _validate_rollout_data(data)
    if pred_horizon <= 0:
        raise ValueError("pred_horizon must be positive")
    if episode_ids.shape != (num_envs,):
        raise ValueError(f"episode_ids must have shape ({num_envs},), got {episode_ids.shape}")
    if oracle_chunks.expert_chunk.shape[:2] != (horizon, num_envs):
        raise ValueError("oracle_chunks expert_chunk shape mismatch with rollout data")
    if oracle_chunks.expert_future_obs_chunk.shape[:2] != (horizon, num_envs):
        raise ValueError("oracle_chunks future_obs shape mismatch with rollout data")
    if oracle_chunks.valid.shape != (horizon, num_envs):
        raise ValueError("oracle_chunks valid shape mismatch with rollout data")

    lengths, keep_mask = segment_lengths_from_failure_mask(data.failure, mode)
    act_dim = int(data.executed_act.shape[2])
    cmd_dim = int(data.current_command.shape[2])

    obs_list: list[np.ndarray] = []
    current_command_list: list[np.ndarray] = []
    executed_act_list: list[np.ndarray] = []
    expert_act_list: list[np.ndarray] = []
    expert_chunk_list: list[np.ndarray] = []
    future_obs_chunk_list: list[np.ndarray] = []
    strict_valid_list: list[np.ndarray] = []
    reward_list: list[np.ndarray] = []
    done_list: list[np.ndarray] = []
    env_id_list: list[np.ndarray] = []
    episode_id_list: list[np.ndarray] = []

    kept_episodes = 0
    kept_steps = 0
    for env_id in range(num_envs):
        if not bool(keep_mask[env_id]):
            continue
        segment_len = int(lengths[env_id])
        if segment_len <= 0:
            continue

        # Use pre-computed oracle chunks and validity.
        # Combined: oracle K-step valid AND trajectory has K+1 future positions
        # (so IDM/FDM can reconstruct K-step targets from per-step fields).
        expert_chunk = oracle_chunks.expert_chunk[:segment_len, env_id].astype(np.float32, copy=False)
        future_obs_chunk = oracle_chunks.expert_future_obs_chunk[:segment_len, env_id].astype(
            np.float32, copy=False,
        )
        oracle_valid = oracle_chunks.valid[:segment_len, env_id].astype(np.bool_, copy=False)
        traj_valid_count = max(0, segment_len - pred_horizon)
        traj_valid = np.zeros((segment_len,), dtype=np.bool_)
        traj_valid[:traj_valid_count] = True
        strict_valid = np.logical_and(oracle_valid, traj_valid)

        obs_list.append(data.obs[:segment_len, env_id].astype(np.float32, copy=False))
        current_command_list.append(data.current_command[:segment_len, env_id].astype(np.float32, copy=False))
        executed_act_list.append(data.executed_act[:segment_len, env_id].astype(np.float32, copy=False))
        expert_act_list.append(data.expert_act[:segment_len, env_id].astype(np.float32, copy=False))
        expert_chunk_list.append(expert_chunk)
        future_obs_chunk_list.append(future_obs_chunk)
        strict_valid_list.append(strict_valid)
        reward_list.append(data.reward[:segment_len, env_id].astype(np.float32, copy=False))
        done_list.append(data.done[:segment_len, env_id].astype(np.bool_, copy=False))
        env_id_list.append(np.full((segment_len,), env_id, dtype=np.int32))
        episode_id_list.append(np.full((segment_len,), int(episode_ids[env_id]), dtype=np.int64))
        kept_episodes += 1
        kept_steps += segment_len

    dropped_episodes = int(num_envs - kept_episodes)
    if kept_episodes <= 0:
        return FilteredReplayBatch(
            obs=np.empty((0, obs_dim), dtype=np.float32),
            current_command=np.empty((0, cmd_dim), dtype=np.float32),
            executed_act=np.empty((0, act_dim), dtype=np.float32),
            expert_act=np.empty((0, act_dim), dtype=np.float32),
            expert_chunk=np.empty((0, pred_horizon, act_dim), dtype=np.float32),
            expert_future_obs_chunk=np.empty((0, pred_horizon, obs_dim), dtype=np.float32),
            strict_label_valid=np.empty((0,), dtype=np.bool_),
            reward=np.empty((0,), dtype=np.float32),
            done=np.empty((0,), dtype=np.bool_),
            env_id=np.empty((0,), dtype=np.int32),
            episode_id=np.empty((0,), dtype=np.int64),
            kept_episodes=0,
            dropped_episodes=dropped_episodes,
            kept_steps=0,
        )

    return FilteredReplayBatch(
        obs=np.concatenate(obs_list, axis=0).astype(np.float32, copy=False),
        current_command=np.concatenate(current_command_list, axis=0).astype(np.float32, copy=False),
        executed_act=np.concatenate(executed_act_list, axis=0).astype(np.float32, copy=False),
        expert_act=np.concatenate(expert_act_list, axis=0).astype(np.float32, copy=False),
        expert_chunk=np.concatenate(expert_chunk_list, axis=0).astype(np.float32, copy=False),
        expert_future_obs_chunk=np.concatenate(future_obs_chunk_list, axis=0).astype(np.float32, copy=False),
        strict_label_valid=np.concatenate(strict_valid_list, axis=0).astype(np.bool_, copy=False),
        reward=np.concatenate(reward_list, axis=0).astype(np.float32, copy=False),
        done=np.concatenate(done_list, axis=0).astype(np.bool_, copy=False),
        env_id=np.concatenate(env_id_list, axis=0).astype(np.int32, copy=False),
        episode_id=np.concatenate(episode_id_list, axis=0).astype(np.int64, copy=False),
        kept_episodes=kept_episodes,
        dropped_episodes=dropped_episodes,
        kept_steps=kept_steps,
    )


def target_suboptimal_steps(*, optimal_kept_steps: int, ratio: float) -> int:
    if optimal_kept_steps < 0:
        raise ValueError("optimal_kept_steps must be non-negative")
    if ratio < 0.0:
        raise ValueError("ratio must be non-negative")
    return int(math.ceil(float(optimal_kept_steps) * float(ratio)))
