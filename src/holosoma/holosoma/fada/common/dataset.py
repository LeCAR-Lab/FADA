from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ReplayCacheMetadataMismatchError(RuntimeError):
    """Raised when replay cache metadata conflicts with current runtime settings."""


class ReplayCacheLegacyMetadataMissingError(ReplayCacheMetadataMismatchError):
    """Raised when loading a legacy replay cache that lacks required metadata."""


_COMPACT_OBS_METADATA_JSON_KEY = "compact_obs_metadata_json"


def _normalize_checkpoint_path(value: object) -> str:
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return str(Path(text).expanduser())


def _canonicalize_compact_obs_metadata(metadata: dict[str, object]) -> dict[str, object]:
    required = ("term_scale", "term_noise", "add_noise", "noise_seed", "source_checkpoint_abs")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"compact_obs metadata missing required keys: {missing}")

    term_scale_raw = metadata.get("term_scale")
    term_noise_raw = metadata.get("term_noise")
    if not isinstance(term_scale_raw, dict):
        raise ValueError("compact_obs metadata key 'term_scale' must be a dict")
    if not isinstance(term_noise_raw, dict):
        raise ValueError("compact_obs metadata key 'term_noise' must be a dict")

    term_scale = {str(k): float(v) for k, v in term_scale_raw.items()}
    term_noise = {str(k): float(v) for k, v in term_noise_raw.items()}
    add_noise = bool(metadata.get("add_noise"))
    noise_seed_raw = metadata.get("noise_seed")
    noise_seed = int(noise_seed_raw) if noise_seed_raw is not None else None
    source_checkpoint_abs = _normalize_checkpoint_path(metadata.get("source_checkpoint_abs", ""))

    return {
        "term_scale": term_scale,
        "term_noise": term_noise,
        "add_noise": add_noise,
        "noise_seed": noise_seed,
        "source_checkpoint_abs": source_checkpoint_abs,
    }


def _float_dicts_close(lhs: dict[str, float], rhs: dict[str, float], *, atol: float = 1e-8) -> bool:
    if set(lhs.keys()) != set(rhs.keys()):
        return False
    for key in lhs.keys():
        if abs(float(lhs[key]) - float(rhs[key])) > float(atol):
            return False
    return True


def _compact_obs_metadata_matches(
    expected: dict[str, object],
    loaded: dict[str, object],
    *,
    atol: float = 1e-8,
) -> bool:
    if bool(expected["add_noise"]) != bool(loaded["add_noise"]):
        return False
    if expected["noise_seed"] != loaded["noise_seed"]:
        return False
    if str(expected["source_checkpoint_abs"]) != str(loaded["source_checkpoint_abs"]):
        return False
    expected_scale = expected["term_scale"]
    loaded_scale = loaded["term_scale"]
    expected_noise = expected["term_noise"]
    loaded_noise = loaded["term_noise"]
    if not isinstance(expected_scale, dict) or not isinstance(loaded_scale, dict):
        return False
    if not isinstance(expected_noise, dict) or not isinstance(loaded_noise, dict):
        return False
    return _float_dicts_close(expected_scale, loaded_scale, atol=atol) and _float_dicts_close(
        expected_noise,
        loaded_noise,
        atol=atol,
    )


class ReplayBuffer:
    """Replay buffer that supports history/prediction chunk sampling.

    Stored transition fields:
    - obs
    - current command
    - executed_act
    - expert_act
    - expert_chunk (optional per-anchor strict chunk label)
    - expert_future_obs_chunk (optional per-anchor strict future-observation label)
    - has_expert_chunk
    - has_expert_future_obs_chunk
    - strict_label_valid (whether this anchor can be used for chunk training)
    - reward
    - done
    - env_id
    - episode_id
    """

    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        history_len: int,
        pred_horizon: int,
        require_future_obs_targets: bool = False,
        growable: bool = False,
        read_only: bool = False,
        eviction_policy: str = "fifo",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if obs_dim <= 0 or act_dim <= 0 or cmd_dim <= 0:
            raise ValueError("obs_dim, act_dim, cmd_dim must be positive")
        if history_len <= 0 or pred_horizon <= 0:
            raise ValueError("history_len and pred_horizon must be positive")

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cmd_dim = int(cmd_dim)
        self.history_len = int(history_len)
        self.pred_horizon = int(pred_horizon)
        self.require_future_obs_targets = bool(require_future_obs_targets)
        self.growable = bool(growable)
        self._read_only = bool(read_only)
        self.eviction_policy = str(eviction_policy).strip().lower()
        if self.eviction_policy not in {"fifo", "random"}:
            raise ValueError("eviction_policy must be 'fifo' or 'random'")
        if self.eviction_policy == "random" and self.growable:
            raise ValueError("random eviction is supported only for fixed-capacity buffers (growable=False)")

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.current_command = np.zeros((self.capacity, self.cmd_dim), dtype=np.float32)
        self.executed_act = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.expert_act = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.expert_chunk = np.zeros((self.capacity, self.pred_horizon, self.act_dim), dtype=np.float32)
        self.expert_future_obs_chunk = np.zeros((self.capacity, self.pred_horizon, self.obs_dim), dtype=np.float32)
        self.has_expert_chunk = np.zeros((self.capacity,), dtype=np.bool_)
        self.has_expert_future_obs_chunk = np.zeros((self.capacity,), dtype=np.bool_)
        self.strict_label_valid = np.ones((self.capacity,), dtype=np.bool_)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.bool_)
        self.env_id = np.zeros((self.capacity,), dtype=np.int32)
        self.episode_id = np.zeros((self.capacity,), dtype=np.int64)
        self._arrival_order = np.zeros((self.capacity,), dtype=np.int64)
        self._arrival_counter = 0
        self._occupied_mask = np.zeros((self.capacity,), dtype=np.bool_)
        self._episode_index_map: dict[tuple[int, int], set[int]] = {}
        self._episode_has_done: dict[tuple[int, int], bool] = {}

        self._head = 0  # next write index
        self._size = 0

        # Sampling cache for fast repeated sample_chunk() calls while buffer is unchanged.
        self._sampling_cache_dirty = True
        self._cached_chrono_idx: np.ndarray | None = None
        self._cached_seq_positions: list[np.ndarray] = []
        self._cached_seq_anchor_positions: list[np.ndarray] = []
        self._cached_seq_usable_cumsum = np.zeros((0,), dtype=np.int64)
        self._cached_num_valid_chunks = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    def freeze(self) -> None:
        self._read_only = True

    def _mark_sampling_cache_dirty(self) -> None:
        self._sampling_cache_dirty = True
        self._cached_chrono_idx = None
        self._cached_seq_positions = []
        self._cached_seq_anchor_positions = []
        self._cached_seq_usable_cumsum = np.zeros((0,), dtype=np.int64)
        self._cached_num_valid_chunks = 0

    def clear(self) -> None:
        if self._read_only:
            raise RuntimeError("ReplayBuffer is read-only and cannot be cleared")
        self._head = 0
        self._size = 0
        self._arrival_order.fill(0)
        self._arrival_counter = 0
        self._occupied_mask.fill(False)
        self._episode_index_map.clear()
        self._episode_has_done.clear()
        self._mark_sampling_cache_dirty()

    def add_from_buffer(self, source: "ReplayBuffer") -> None:
        """Copy all occupied data from *source* into this buffer via add_batch."""
        n = len(source)
        if n <= 0:
            return
        idx = source._chronological_indices()
        self.add_batch(
            obs=source.obs[idx],
            current_command=source.current_command[idx],
            executed_act=source.executed_act[idx],
            expert_act=source.expert_act[idx],
            expert_chunk=source.expert_chunk[idx] if source.has_expert_chunk[idx].any() else None,
            expert_future_obs_chunk=(
                source.expert_future_obs_chunk[idx] if source.has_expert_future_obs_chunk[idx].any() else None
            ),
            strict_label_valid=source.strict_label_valid[idx],
            reward=source.reward[idx],
            done=source.done[idx],
            env_id=source.env_id[idx],
            episode_id=source.episode_id[idx],
        )

    def _write_rows(
        self,
        dst: np.ndarray | slice,
        *,
        obs: np.ndarray,
        current_command: np.ndarray,
        executed_act: np.ndarray,
        expert_act: np.ndarray,
        expert_chunk: np.ndarray | None,
        expert_future_obs_chunk: np.ndarray | None,
        has_expert_chunk: np.ndarray,
        has_expert_future_obs_chunk: np.ndarray,
        strict_label_valid: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        env_id: np.ndarray,
        episode_id: np.ndarray,
    ) -> None:
        self.obs[dst] = obs
        self.current_command[dst] = current_command
        self.executed_act[dst] = executed_act
        self.expert_act[dst] = expert_act
        if expert_chunk is not None:
            self.expert_chunk[dst] = expert_chunk
        else:
            self.expert_chunk[dst] = 0.0
        if expert_future_obs_chunk is not None:
            self.expert_future_obs_chunk[dst] = expert_future_obs_chunk
        else:
            self.expert_future_obs_chunk[dst] = 0.0
        self.has_expert_chunk[dst] = has_expert_chunk
        self.has_expert_future_obs_chunk[dst] = has_expert_future_obs_chunk
        self.strict_label_valid[dst] = strict_label_valid
        self.reward[dst] = reward
        self.done[dst] = done
        self.env_id[dst] = env_id
        self.episode_id[dst] = episode_id

    def _stamp_arrival_order(self, dst: np.ndarray | slice, count: int) -> None:
        if count <= 0:
            return
        values = np.arange(self._arrival_counter, self._arrival_counter + count, dtype=np.int64)
        self._arrival_order[dst] = values
        self._arrival_counter += int(count)

    def _random_rebuild_episode_index_state(self) -> None:
        self._episode_index_map.clear()
        self._episode_has_done.clear()
        if self.eviction_policy != "random":
            return
        occupied = np.flatnonzero(self._occupied_mask)
        for idx in occupied.tolist():
            key = (int(self.env_id[idx]), int(self.episode_id[idx]))
            bucket = self._episode_index_map.get(key)
            if bucket is None:
                bucket = set()
                self._episode_index_map[key] = bucket
                self._episode_has_done[key] = False
            bucket.add(int(idx))
            if bool(self.done[idx]):
                self._episode_has_done[key] = True

    def _random_register_rows(
        self,
        *,
        dst_indices: np.ndarray,
        env_id: np.ndarray,
        episode_id: np.ndarray,
        done: np.ndarray,
    ) -> None:
        if dst_indices.ndim != 1:
            raise ValueError("dst_indices must be a rank-1 array")
        count = int(dst_indices.shape[0])
        if count == 0:
            return
        if env_id.shape != (count,) or episode_id.shape != (count,) or done.shape != (count,):
            raise ValueError("random register rows shape mismatch")
        for i in range(count):
            idx = int(dst_indices[i])
            key = (int(env_id[i]), int(episode_id[i]))
            bucket = self._episode_index_map.get(key)
            if bucket is None:
                bucket = set()
                self._episode_index_map[key] = bucket
                self._episode_has_done[key] = False
            bucket.add(idx)
            if bool(done[i]):
                self._episode_has_done[key] = True
            self._occupied_mask[idx] = True

    def _random_pop_episode(self, key: tuple[int, int]) -> np.ndarray:
        bucket = self._episode_index_map.pop(key, None)
        self._episode_has_done.pop(key, None)
        if not bucket:
            return np.zeros((0,), dtype=np.int64)
        indices = np.fromiter(bucket, dtype=np.int64, count=len(bucket))
        if indices.size > 0:
            self._occupied_mask[indices] = False
        return indices

    def _random_evict_episode_units(self, required_slots: int) -> np.ndarray:
        """Evict random trajectory units until enough free slots are available.

        Eviction is always performed in (env_id, episode_id) units: one random
        episode key at a time, removed as a whole.
        """
        if required_slots <= 0:
            return np.zeros((0,), dtype=np.int64)

        evicted_chunks: list[np.ndarray] = []
        freed = 0

        while freed < required_slots and self._episode_index_map:
            keys = list(self._episode_index_map.keys())
            key = keys[int(np.random.randint(0, len(keys)))]
            indices = self._random_pop_episode(key)
            if indices.size <= 0:
                continue
            evicted_chunks.append(indices)
            freed += int(indices.size)

        if not evicted_chunks:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(evicted_chunks, axis=0).astype(np.int64, copy=False)

    def add_batch(
        self,
        *,
        obs: np.ndarray,
        current_command: np.ndarray,
        executed_act: np.ndarray,
        expert_act: np.ndarray,
        expert_chunk: np.ndarray | None = None,
        expert_future_obs_chunk: np.ndarray | None = None,
        strict_label_valid: np.ndarray | None = None,
        reward: np.ndarray,
        done: np.ndarray,
        env_id: np.ndarray,
        episode_id: np.ndarray,
    ) -> None:
        if self._read_only:
            raise RuntimeError("ReplayBuffer is read-only and cannot be modified")

        num = int(obs.shape[0])
        self._validate_batch_shapes(
            obs=obs,
            current_command=current_command,
            executed_act=executed_act,
            expert_act=expert_act,
            expert_chunk=expert_chunk,
            expert_future_obs_chunk=expert_future_obs_chunk,
            strict_label_valid=strict_label_valid,
            reward=reward,
            done=done,
            env_id=env_id,
            episode_id=episode_id,
        )
        if num <= 0:
            return

        obs = obs.astype(np.float32, copy=False)
        current_command = current_command.astype(np.float32, copy=False)
        executed_act = executed_act.astype(np.float32, copy=False)
        expert_act = expert_act.astype(np.float32, copy=False)
        if expert_chunk is not None:
            expert_chunk = expert_chunk.astype(np.float32, copy=False)
            has_expert_chunk = np.ones((num,), dtype=np.bool_)
        else:
            has_expert_chunk = np.zeros((num,), dtype=np.bool_)
        if expert_future_obs_chunk is not None:
            expert_future_obs_chunk = expert_future_obs_chunk.astype(np.float32, copy=False)
            has_expert_future_obs_chunk = np.ones((num,), dtype=np.bool_)
        else:
            has_expert_future_obs_chunk = np.zeros((num,), dtype=np.bool_)
        if strict_label_valid is None:
            strict_label_valid = np.ones((num,), dtype=np.bool_)
        else:
            strict_label_valid = strict_label_valid.astype(np.bool_, copy=False)
        reward = reward.astype(np.float32, copy=False)
        done = done.astype(np.bool_, copy=False)
        env_id = env_id.astype(np.int32, copy=False)
        episode_id = episode_id.astype(np.int64, copy=False)

        if self.growable and (self._size + num > self.capacity):
            new_capacity = int(self.capacity)
            required = int(self._size + num)
            while new_capacity < required:
                new_capacity = max(new_capacity * 2, new_capacity + 1)
            self._resize(new_capacity)

        if self.eviction_policy == "random" and not self.growable:
            write_pos = 0
            while write_pos < num:
                free_indices = np.flatnonzero(~self._occupied_mask)
                if free_indices.size <= 0:
                    needed = int(num - write_pos)
                    evicted = self._random_evict_episode_units(needed)
                    if evicted.size <= 0:
                        raise RuntimeError("Random episode eviction failed to free slots for incoming batch")
                    self._size = max(0, int(self._size - evicted.size))
                    free_indices = evicted

                write_count = int(min(num - write_pos, free_indices.size))
                dst = free_indices[:write_count]
                src = slice(write_pos, write_pos + write_count)
                self._write_rows(
                    dst,
                    obs=obs[src],
                    current_command=current_command[src],
                    executed_act=executed_act[src],
                    expert_act=expert_act[src],
                    expert_chunk=(expert_chunk[src] if expert_chunk is not None else None),
                    expert_future_obs_chunk=(
                        expert_future_obs_chunk[src]
                        if expert_future_obs_chunk is not None
                        else None
                    ),
                    has_expert_chunk=has_expert_chunk[src],
                    has_expert_future_obs_chunk=has_expert_future_obs_chunk[src],
                    strict_label_valid=strict_label_valid[src],
                    reward=reward[src],
                    done=done[src],
                    env_id=env_id[src],
                    episode_id=episode_id[src],
                )
                self._stamp_arrival_order(dst, write_count)
                self._random_register_rows(
                    dst_indices=dst,
                    env_id=env_id[src],
                    episode_id=episode_id[src],
                    done=done[src],
                )
                self._size = min(self.capacity, int(self._size + write_count))
                write_pos += write_count

            self._head = int(self._size % self.capacity)
            self._mark_sampling_cache_dirty()
            return

        if num >= self.capacity and not self.growable:
            # Keep only the most recent part that fits buffer capacity.
            tail = slice(num - self.capacity, num)
            self._write_rows(
                slice(None),
                obs=obs[tail],
                current_command=current_command[tail],
                executed_act=executed_act[tail],
                expert_act=expert_act[tail],
                expert_chunk=(expert_chunk[tail] if expert_chunk is not None else None),
                expert_future_obs_chunk=(
                    expert_future_obs_chunk[tail]
                    if expert_future_obs_chunk is not None
                    else None
                ),
                has_expert_chunk=has_expert_chunk[tail],
                has_expert_future_obs_chunk=has_expert_future_obs_chunk[tail],
                strict_label_valid=strict_label_valid[tail],
                reward=reward[tail],
                done=done[tail],
                env_id=env_id[tail],
                episode_id=episode_id[tail],
            )
            self._stamp_arrival_order(slice(None), self.capacity)
            self._head = 0
            self._size = self.capacity
            self._mark_sampling_cache_dirty()
            return

        head = int(self._head)
        first = min(num, self.capacity - head)
        if first > 0:
            dst = slice(head, head + first)
            self._write_rows(
                dst,
                obs=obs[:first],
                current_command=current_command[:first],
                executed_act=executed_act[:first],
                expert_act=expert_act[:first],
                expert_chunk=(expert_chunk[:first] if expert_chunk is not None else None),
                expert_future_obs_chunk=(
                    expert_future_obs_chunk[:first]
                    if expert_future_obs_chunk is not None
                    else None
                ),
                has_expert_chunk=has_expert_chunk[:first],
                has_expert_future_obs_chunk=has_expert_future_obs_chunk[:first],
                strict_label_valid=strict_label_valid[:first],
                reward=reward[:first],
                done=done[:first],
                env_id=env_id[:first],
                episode_id=episode_id[:first],
            )
            self._stamp_arrival_order(dst, first)

        remaining = num - first
        if remaining > 0:
            dst = slice(0, remaining)
            self._write_rows(
                dst,
                obs=obs[first:],
                current_command=current_command[first:],
                executed_act=executed_act[first:],
                expert_act=expert_act[first:],
                expert_chunk=(expert_chunk[first:] if expert_chunk is not None else None),
                expert_future_obs_chunk=(
                    expert_future_obs_chunk[first:]
                    if expert_future_obs_chunk is not None
                    else None
                ),
                has_expert_chunk=has_expert_chunk[first:],
                has_expert_future_obs_chunk=has_expert_future_obs_chunk[first:],
                strict_label_valid=strict_label_valid[first:],
                reward=reward[first:],
                done=done[first:],
                env_id=env_id[first:],
                episode_id=episode_id[first:],
            )
            self._stamp_arrival_order(dst, remaining)

        self._head = (head + num) % self.capacity
        self._size = min(self._size + num, self.capacity)
        self._mark_sampling_cache_dirty()

    def _validate_batch_shapes(
        self,
        *,
        obs: np.ndarray,
        current_command: np.ndarray,
        executed_act: np.ndarray,
        expert_act: np.ndarray,
        expert_chunk: np.ndarray | None,
        expert_future_obs_chunk: np.ndarray | None,
        strict_label_valid: np.ndarray | None,
        reward: np.ndarray,
        done: np.ndarray,
        env_id: np.ndarray,
        episode_id: np.ndarray,
    ) -> None:
        num = int(obs.shape[0])
        if obs.shape != (num, self.obs_dim):
            raise ValueError(f"obs shape mismatch: got {obs.shape}, expected {(num, self.obs_dim)}")
        if current_command.shape != (num, self.cmd_dim):
            raise ValueError(
                f"current_command shape mismatch: got {current_command.shape}, expected {(num, self.cmd_dim)}"
            )
        if executed_act.shape != (num, self.act_dim):
            raise ValueError(
                f"executed_act shape mismatch: got {executed_act.shape}, expected {(num, self.act_dim)}"
            )
        if expert_act.shape != (num, self.act_dim):
            raise ValueError(f"expert_act shape mismatch: got {expert_act.shape}, expected {(num, self.act_dim)}")
        if expert_chunk is not None and expert_chunk.shape != (num, self.pred_horizon, self.act_dim):
            raise ValueError(
                "expert_chunk shape mismatch: "
                f"got {expert_chunk.shape}, expected {(num, self.pred_horizon, self.act_dim)}"
            )
        if expert_future_obs_chunk is not None and expert_future_obs_chunk.shape != (num, self.pred_horizon, self.obs_dim):
            raise ValueError(
                "expert_future_obs_chunk shape mismatch: "
                f"got {expert_future_obs_chunk.shape}, expected {(num, self.pred_horizon, self.obs_dim)}"
            )
        if strict_label_valid is not None and strict_label_valid.shape != (num,):
            raise ValueError(
                "strict_label_valid shape mismatch: "
                f"got {strict_label_valid.shape}, expected {(num,)}"
            )
        if reward.shape != (num,):
            raise ValueError(f"reward shape mismatch: got {reward.shape}, expected {(num,)}")
        if done.shape != (num,):
            raise ValueError(f"done shape mismatch: got {done.shape}, expected {(num,)}")
        if env_id.shape != (num,):
            raise ValueError(f"env_id shape mismatch: got {env_id.shape}, expected {(num,)}")
        if episode_id.shape != (num,):
            raise ValueError(f"episode_id shape mismatch: got {episode_id.shape}, expected {(num,)}")

    def _resize(self, new_capacity: int) -> None:
        if new_capacity <= self.capacity:
            return

        idx = self._chronological_indices()
        new_obs = np.zeros((new_capacity, self.obs_dim), dtype=np.float32)
        new_current_command = np.zeros((new_capacity, self.cmd_dim), dtype=np.float32)
        new_executed_act = np.zeros((new_capacity, self.act_dim), dtype=np.float32)
        new_expert_act = np.zeros((new_capacity, self.act_dim), dtype=np.float32)
        new_expert_chunk = np.zeros((new_capacity, self.pred_horizon, self.act_dim), dtype=np.float32)
        new_expert_future_obs_chunk = np.zeros((new_capacity, self.pred_horizon, self.obs_dim), dtype=np.float32)
        new_has_expert_chunk = np.zeros((new_capacity,), dtype=np.bool_)
        new_has_expert_future_obs_chunk = np.zeros((new_capacity,), dtype=np.bool_)
        new_strict_label_valid = np.ones((new_capacity,), dtype=np.bool_)
        new_reward = np.zeros((new_capacity,), dtype=np.float32)
        new_done = np.zeros((new_capacity,), dtype=np.bool_)
        new_env_id = np.zeros((new_capacity,), dtype=np.int32)
        new_episode_id = np.zeros((new_capacity,), dtype=np.int64)
        new_arrival_order = np.zeros((new_capacity,), dtype=np.int64)

        size = int(self._size)
        if size > 0:
            new_obs[:size] = self.obs[idx]
            new_current_command[:size] = self.current_command[idx]
            new_executed_act[:size] = self.executed_act[idx]
            new_expert_act[:size] = self.expert_act[idx]
            new_expert_chunk[:size] = self.expert_chunk[idx]
            new_expert_future_obs_chunk[:size] = self.expert_future_obs_chunk[idx]
            new_has_expert_chunk[:size] = self.has_expert_chunk[idx]
            new_has_expert_future_obs_chunk[:size] = self.has_expert_future_obs_chunk[idx]
            new_strict_label_valid[:size] = self.strict_label_valid[idx]
            new_reward[:size] = self.reward[idx]
            new_done[:size] = self.done[idx]
            new_env_id[:size] = self.env_id[idx]
            new_episode_id[:size] = self.episode_id[idx]
            new_arrival_order[:size] = self._arrival_order[idx]

        self.capacity = int(new_capacity)
        self.obs = new_obs
        self.current_command = new_current_command
        self.executed_act = new_executed_act
        self.expert_act = new_expert_act
        self.expert_chunk = new_expert_chunk
        self.expert_future_obs_chunk = new_expert_future_obs_chunk
        self.has_expert_chunk = new_has_expert_chunk
        self.has_expert_future_obs_chunk = new_has_expert_future_obs_chunk
        self.strict_label_valid = new_strict_label_valid
        self.reward = new_reward
        self.done = new_done
        self.env_id = new_env_id
        self.episode_id = new_episode_id
        self._arrival_order = new_arrival_order
        self._occupied_mask = np.zeros((new_capacity,), dtype=np.bool_)
        self._head = size
        self._size = size
        self._arrival_counter = int(new_arrival_order[:size].max()) + 1 if size > 0 else 0
        if self.eviction_policy == "random" and size > 0:
            self._occupied_mask[:size] = True
            self._random_rebuild_episode_index_state()
        else:
            self._episode_index_map.clear()
            self._episode_has_done.clear()
        self._mark_sampling_cache_dirty()

    def _chronological_indices(self) -> np.ndarray:
        if self._size == 0:
            return np.zeros((0,), dtype=np.int64)
        if self.eviction_policy == "random":
            idx = np.flatnonzero(self._occupied_mask).astype(np.int64, copy=False)
            if idx.size <= 0:
                return np.zeros((0,), dtype=np.int64)
            order = self._arrival_order[idx]
            return idx[np.argsort(order, kind="stable")]
        if self._size < self.capacity:
            return np.arange(self._size, dtype=np.int64)

        # Buffer full: oldest item is at self._head.
        return np.concatenate(
            [
                np.arange(self._head, self.capacity, dtype=np.int64),
                np.arange(0, self._head, dtype=np.int64),
            ]
        )

    def _as_chronological_dict(self) -> dict[str, np.ndarray]:
        idx = self._chronological_indices()
        return {
            "obs": self.obs[idx],
            "current_command": self.current_command[idx],
            "executed_act": self.executed_act[idx],
            "expert_act": self.expert_act[idx],
            "expert_chunk": self.expert_chunk[idx],
            "expert_future_obs_chunk": self.expert_future_obs_chunk[idx],
            "has_expert_chunk": self.has_expert_chunk[idx],
            "has_expert_future_obs_chunk": self.has_expert_future_obs_chunk[idx],
            "strict_label_valid": self.strict_label_valid[idx],
            "reward": self.reward[idx],
            "done": self.done[idx],
            "env_id": self.env_id[idx],
            "episode_id": self.episode_id[idx],
        }

    def get_obs_mean_std(self, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self._size <= 0:
            raise RuntimeError("Cannot compute observation statistics from an empty replay buffer")
        idx = self._chronological_indices()
        obs_data = self.obs[idx]
        mean = obs_data.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = obs_data.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = np.maximum(std, np.float32(eps))
        return mean, std

    def get_expert_action_mean_std(self, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self._size <= 0:
            raise RuntimeError("Cannot compute action statistics from an empty replay buffer")
        idx = self._chronological_indices()
        action_data = self.expert_act[idx]
        mean = action_data.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = action_data.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = np.maximum(std, np.float32(eps))
        return mean, std

    def get_current_command_mean_std(self, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self._size <= 0:
            raise RuntimeError("Cannot compute current_command statistics from an empty replay buffer")
        idx = self._chronological_indices()
        current_command_data = self.current_command[idx]
        mean = current_command_data.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = current_command_data.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = np.maximum(std, np.float32(eps))
        return mean, std

    def get_command_mean_std(self, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        return self.get_current_command_mean_std(eps=eps)

    def _refresh_sampling_cache(self) -> None:
        if not self._sampling_cache_dirty:
            return

        if self._size <= 0:
            self._cached_chrono_idx = np.zeros((0,), dtype=np.int64)
            self._cached_seq_positions = []
            self._cached_seq_anchor_positions = []
            self._cached_seq_usable_cumsum = np.zeros((0,), dtype=np.int64)
            self._cached_num_valid_chunks = 0
            self._sampling_cache_dirty = False
            return

        chrono_idx = self._chronological_indices()
        env_ids = self.env_id[chrono_idx]
        episode_ids = self.episode_id[chrono_idx]

        grouped: dict[tuple[int, int], list[int]] = {}
        for pos, (env, epi) in enumerate(zip(env_ids.tolist(), episode_ids.tolist())):
            key = (int(env), int(epi))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(pos)

        seq_positions: list[np.ndarray] = []
        seq_anchor_positions: list[np.ndarray] = []
        usable_counts: list[int] = []
        for positions_list in grouped.values():
            seq = np.asarray(positions_list, dtype=np.int64)
            seq_len = int(seq.shape[0])
            if seq_len <= 0:
                continue

            seq_raw = chrono_idx[seq]
            seq_done = self.done[seq_raw].astype(np.bool_, copy=False)
            seq_done_prefix = np.concatenate(
                [np.zeros((1,), dtype=np.int64), np.cumsum(seq_done.astype(np.int64, copy=False), dtype=np.int64)]
            )

            # Allow tail anchors when strict chunk labels are available; otherwise
            # require enough in-trajectory future timesteps for fallback targets.
            candidate_anchors = np.arange(seq_len, dtype=np.int64)
            candidate_raw = seq_raw[candidate_anchors]
            valid_mask = self.strict_label_valid[candidate_raw]

            has_strict_action_chunk = self.has_expert_chunk[candidate_raw]
            traj_action_available = candidate_anchors <= int(seq_len - self.pred_horizon)
            traj_action_no_done_cross = np.zeros((seq_len,), dtype=np.bool_)
            action_max_anchor = int(seq_len - self.pred_horizon)
            if action_max_anchor >= 0:
                if self.pred_horizon <= 1:
                    traj_action_no_done_cross[: action_max_anchor + 1] = True
                else:
                    action_starts = np.arange(action_max_anchor + 1, dtype=np.int64)
                    # Require no done in [anchor, anchor + pred_horizon - 2], so
                    # trajectory fallback never crosses episode boundaries.
                    action_done_counts = (
                        seq_done_prefix[action_starts + self.pred_horizon - 1] - seq_done_prefix[action_starts]
                    )
                    traj_action_no_done_cross[: action_max_anchor + 1] = action_done_counts == 0
            traj_action_usable = np.logical_and(traj_action_available, traj_action_no_done_cross)
            action_target_available = np.logical_or(has_strict_action_chunk, traj_action_usable)
            valid_mask = np.logical_and(valid_mask, action_target_available)

            if self.require_future_obs_targets:
                has_strict_future_obs_chunk = self.has_expert_future_obs_chunk[candidate_raw]
                # No trajectory fallback for next-observation targets: only strict
                # expert future-observation chunks are allowed.
                valid_mask = np.logical_and(valid_mask, has_strict_future_obs_chunk)

            valid_anchors = candidate_anchors[valid_mask]
            usable = int(valid_anchors.shape[0])
            if usable <= 0:
                continue
            seq_positions.append(seq)
            seq_anchor_positions.append(valid_anchors)
            usable_counts.append(usable)

        if usable_counts:
            usable_arr = np.asarray(usable_counts, dtype=np.int64)
            usable_cumsum = np.cumsum(usable_arr, dtype=np.int64)
            num_valid_chunks = int(usable_cumsum[-1])
        else:
            usable_cumsum = np.zeros((0,), dtype=np.int64)
            num_valid_chunks = 0

        self._cached_chrono_idx = chrono_idx
        self._cached_seq_positions = seq_positions
        self._cached_seq_anchor_positions = seq_anchor_positions
        self._cached_seq_usable_cumsum = usable_cumsum
        self._cached_num_valid_chunks = int(num_valid_chunks)
        self._sampling_cache_dirty = False

    def num_valid_chunks(self) -> int:
        self._refresh_sampling_cache()
        return int(self._cached_num_valid_chunks)

    def sample_chunk(
        self,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        self._refresh_sampling_cache()
        total_valid = int(self._cached_num_valid_chunks)
        if total_valid <= 0 or self._cached_chrono_idx is None:
            raise RuntimeError(
                "No valid chunks available in replay buffer. "
                "Collect more transitions or reduce pred_horizon."
            )

        replace = total_valid < batch_size
        sampled_global = np.random.choice(total_valid, size=batch_size, replace=replace).astype(np.int64, copy=False)
        seq_ids = np.searchsorted(self._cached_seq_usable_cumsum, sampled_global, side="right")
        prev_cumsum = np.zeros_like(sampled_global)
        has_prev = seq_ids > 0
        prev_cumsum[has_prev] = self._cached_seq_usable_cumsum[seq_ids[has_prev] - 1]
        anchors_in_seq = sampled_global - prev_cumsum

        history_obs = np.zeros((batch_size, self.history_len, self.obs_dim), dtype=np.float32)
        history_act = np.zeros((batch_size, self.history_len, self.act_dim), dtype=np.float32)
        history_valid_mask = np.zeros((batch_size, self.history_len), dtype=np.bool_)
        current_command = np.zeros((batch_size, self.cmd_dim), dtype=np.float32)
        target_actions = np.zeros((batch_size, self.pred_horizon, self.act_dim), dtype=np.float32)
        target_next_observations: np.ndarray | None = None
        if self.require_future_obs_targets:
            target_next_observations = np.zeros((batch_size, self.pred_horizon, self.obs_dim), dtype=np.float32)
        # Trajectory-derived targets: always reconstructed from per-step fields.
        # When expert_chunk contains oracle-relabeled data, these provide the raw
        # trajectory targets for IDM/FDM which need real causal dynamics.
        trajectory_target_actions = np.zeros((batch_size, self.pred_horizon, self.act_dim), dtype=np.float32)
        trajectory_target_next_observations: np.ndarray | None = None
        if self.require_future_obs_targets:
            trajectory_target_next_observations = np.zeros(
                (batch_size, self.pred_horizon, self.obs_dim), dtype=np.float32,
            )
        # Tracks whether trajectory-derived targets are fully filled for each sample.
        # Boundary anchors near the end of an episode may lack enough consecutive
        # steps, leaving trajectory targets as zeros — those samples must be excluded
        # from trajectory-target losses (IDM/FDM real-causal mode).
        trajectory_target_valid = np.ones((batch_size,), dtype=np.bool_)

        chrono_idx = self._cached_chrono_idx
        for b in range(batch_size):
            seq_positions = self._cached_seq_positions[int(seq_ids[b])]
            anchor_candidates = self._cached_seq_anchor_positions[int(seq_ids[b])]
            anchor_in_seq = int(anchor_candidates[int(anchors_in_seq[b])])
            seq_raw = chrono_idx[seq_positions]

            anchor_raw = int(chrono_idx[seq_positions[anchor_in_seq]])
            if self.has_expert_chunk[anchor_raw]:
                target_actions[b] = self.expert_chunk[anchor_raw]
            else:
                target_chrono = seq_positions[anchor_in_seq : anchor_in_seq + self.pred_horizon]
                target_raw = chrono_idx[target_chrono]
                target_actions[b] = self.expert_act[target_raw]
            if target_next_observations is not None:
                if not self.has_expert_future_obs_chunk[anchor_raw]:
                    raise RuntimeError(
                        "Missing strict expert future-observation chunk for sampled anchor "
                        f"(raw_index={anchor_raw}). Next-observation fallback is disabled."
                    )
                target_next_observations[b] = self.expert_future_obs_chunk[anchor_raw]

            # Trajectory-derived targets from consecutive per-step fields.
            # Use executed_act (the action actually taken) rather than expert_act
            # (oracle label) so the (action, next_obs) pair is causally consistent.
            seq_len = int(seq_positions.shape[0])
            traj_end = anchor_in_seq + self.pred_horizon
            if traj_end <= seq_len:
                traj_chrono = seq_positions[anchor_in_seq : traj_end]
                traj_raw = chrono_idx[traj_chrono]
                trajectory_target_actions[b] = self.executed_act[traj_raw]
            else:
                trajectory_target_valid[b] = False
            if trajectory_target_next_observations is not None:
                future_end = anchor_in_seq + 1 + self.pred_horizon
                if future_end <= seq_len:
                    future_chrono = seq_positions[anchor_in_seq + 1 : future_end]
                    future_raw = chrono_idx[future_chrono]
                    trajectory_target_next_observations[b] = self.obs[future_raw]
                else:
                    trajectory_target_valid[b] = False

            hist_seq_start = max(0, anchor_in_seq - self.history_len + 1)
            hist_seq_indices = np.arange(hist_seq_start, anchor_in_seq + 1, dtype=np.int64)
            offset = self.history_len - int(hist_seq_indices.shape[0])
            hist_chrono = seq_positions[hist_seq_indices]
            hist_raw = chrono_idx[hist_chrono]
            history_obs[b, offset:] = self.obs[hist_raw]
            history_valid_mask[b, offset:] = True
            current_command[b] = self.current_command[anchor_raw]

            # Action history token: action at previous step within same episode sequence.
            prev_seq_indices = hist_seq_indices - 1
            valid_prev = prev_seq_indices >= 0
            if np.any(valid_prev):
                valid_positions = np.flatnonzero(valid_prev)
                prev_chrono = seq_positions[prev_seq_indices[valid_prev]]
                prev_raw = chrono_idx[prev_chrono]
                history_act[b, offset + valid_positions] = self.executed_act[prev_raw]

        out = {
            "history_obs": history_obs,
            "history_act": history_act,
            "current_command": current_command,
            "history_valid_mask": history_valid_mask,
            # Always [B, K, act_dim], even when K=1.
            "target_actions": target_actions,
            "trajectory_target_actions": trajectory_target_actions,
            "trajectory_target_valid": trajectory_target_valid,
        }
        if target_next_observations is not None:
            out["target_next_observations"] = target_next_observations
        if trajectory_target_next_observations is not None:
            out["trajectory_target_next_observations"] = trajectory_target_next_observations
        return out

    @classmethod
    def from_npz_mmap(
        cls,
        path: "str | Path",
        *,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        history_len: int,
        pred_horizon: int,
    ) -> "ReplayBuffer":
        """Create a read-only ReplayBuffer backed by a memory-mapped NPZ file.

        Unlike load_npz(), this bypasses __init__ pre-allocation entirely:
        no np.zeros arrays are created, and the NPZ is not copied into RAM.
        The OS page cache handles physical memory; only sampled pages are resident.
        Sampling logic (sample_chunk, _refresh_sampling_cache) is identical.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Buffer NPZ not found: {path}")

        data = np.load(str(path), allow_pickle=False, mmap_mode="r")
        n = int(data["obs"].shape[0])

        obj: "ReplayBuffer" = object.__new__(cls)
        # Keep the NpzFile open — mmap arrays are invalid once it closes.
        obj._mmap_npz = data  # type: ignore[attr-defined]

        # Scalar metadata
        obj.capacity = n
        obj.obs_dim = int(obs_dim)
        obj.act_dim = int(act_dim)
        obj.cmd_dim = int(cmd_dim)
        obj.history_len = int(history_len)
        obj.pred_horizon = int(pred_horizon)
        obj.require_future_obs_targets = True
        obj.growable = False
        obj._read_only = True
        obj.eviction_policy = "fifo"
        obj._head = 0
        obj._size = n
        obj._arrival_counter = n
        obj._episode_index_map = {}
        obj._episode_has_done = {}

        # Large float32 arrays: mmap views (no copy if already float32 in NPZ).
        obj.obs = data["obs"].astype(np.float32, copy=False)
        obj.current_command = data["current_command"].astype(np.float32, copy=False)
        obj.executed_act = data["executed_act"].astype(np.float32, copy=False)
        obj.expert_act = data["expert_act"].astype(np.float32, copy=False)

        _ec_shape = (n, pred_horizon, act_dim)
        if "expert_chunk" in data:
            raw = data["expert_chunk"].astype(np.float32, copy=False)
            obj.expert_chunk = raw if raw.shape == _ec_shape else np.zeros(_ec_shape, dtype=np.float32)
        else:
            obj.expert_chunk = np.zeros(_ec_shape, dtype=np.float32)

        _efo_shape = (n, pred_horizon, obs_dim)
        if "expert_future_obs_chunk" in data:
            raw = data["expert_future_obs_chunk"].astype(np.float32, copy=False)
            obj.expert_future_obs_chunk = raw if raw.shape == _efo_shape else np.zeros(_efo_shape, dtype=np.float32)
        else:
            obj.expert_future_obs_chunk = np.zeros(_efo_shape, dtype=np.float32)

        # Small bool/int arrays: copied into RAM (needed by _refresh_sampling_cache).
        obj.has_expert_chunk = (
            np.asarray(data["has_expert_chunk"], dtype=np.bool_)
            if "has_expert_chunk" in data
            else np.zeros((n,), dtype=np.bool_)
        )
        obj.has_expert_future_obs_chunk = (
            np.asarray(data["has_expert_future_obs_chunk"], dtype=np.bool_)
            if "has_expert_future_obs_chunk" in data
            else np.zeros((n,), dtype=np.bool_)
        )
        obj.strict_label_valid = (
            np.asarray(data["strict_label_valid"], dtype=np.bool_)
            if "strict_label_valid" in data
            else np.ones((n,), dtype=np.bool_)
        )
        obj.reward = np.asarray(data["reward"], dtype=np.float32)
        obj.done = np.asarray(data["done"], dtype=np.bool_)
        obj.env_id = np.asarray(data["env_id"], dtype=np.int32)
        obj.episode_id = np.asarray(data["episode_id"], dtype=np.int64)
        obj._arrival_order = np.arange(n, dtype=np.int64)
        obj._occupied_mask = np.zeros((n,), dtype=np.bool_)

        # Sampling cache (computed lazily on first sample_chunk call).
        obj._sampling_cache_dirty = True
        obj._cached_chrono_idx = None
        obj._cached_seq_positions = []
        obj._cached_seq_anchor_positions = []
        obj._cached_seq_usable_cumsum = np.zeros((0,), dtype=np.int64)
        obj._cached_num_valid_chunks = 0

        return obj

    def save_npz(
        self,
        path: str | Path,
        *,
        compact_obs_metadata: dict[str, object] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._as_chronological_dict()
        # Keep offline caches horizon-agnostic: when strict chunk labels are absent,
        # do not persist expert_chunk fields that encode pred_horizon.
        if "has_expert_chunk" in payload and not np.any(payload["has_expert_chunk"]):
            payload.pop("expert_chunk", None)
            payload.pop("has_expert_chunk", None)
        if "has_expert_future_obs_chunk" in payload and not np.any(payload["has_expert_future_obs_chunk"]):
            payload.pop("expert_future_obs_chunk", None)
            payload.pop("has_expert_future_obs_chunk", None)
        if compact_obs_metadata is not None:
            canonical_metadata = _canonicalize_compact_obs_metadata(compact_obs_metadata)
            payload[_COMPACT_OBS_METADATA_JSON_KEY] = np.asarray(
                [json.dumps(canonical_metadata, sort_keys=True)],
                dtype=np.str_,
            )
        np.savez(
            path,
            **payload,
            capacity=np.array([self.capacity], dtype=np.int64),
            obs_dim=np.array([self.obs_dim], dtype=np.int64),
            act_dim=np.array([self.act_dim], dtype=np.int64),
            cmd_dim=np.array([self.cmd_dim], dtype=np.int64),
            history_len=np.array([self.history_len], dtype=np.int64),
            pred_horizon=np.array([self.pred_horizon], dtype=np.int64),
        )
        return path

    def load_npz(
        self,
        path: str | Path,
        *,
        expected_compact_obs_metadata: dict[str, object] | None = None,
    ) -> int:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Replay buffer cache not found: {path}")

        data = np.load(path, allow_pickle=False)
        if expected_compact_obs_metadata is not None:
            expected_metadata = _canonicalize_compact_obs_metadata(expected_compact_obs_metadata)
            if _COMPACT_OBS_METADATA_JSON_KEY not in data:
                raise ReplayCacheLegacyMetadataMissingError(
                    "Replay cache is missing compact-obs metadata. "
                    "This cache was likely generated by an older code version."
                )
            raw_metadata = data[_COMPACT_OBS_METADATA_JSON_KEY]
            raw_metadata_text = str(np.asarray(raw_metadata).reshape(-1)[0]) if np.asarray(raw_metadata).size > 0 else ""
            try:
                loaded_metadata = _canonicalize_compact_obs_metadata(json.loads(raw_metadata_text))
            except Exception as exc:
                raise ReplayCacheMetadataMismatchError(
                    "Replay cache compact-obs metadata is invalid or corrupted."
                ) from exc
            if not _compact_obs_metadata_matches(expected_metadata, loaded_metadata):
                raise ReplayCacheMetadataMismatchError(
                    "Replay cache compact-obs metadata mismatch. "
                    f"expected={expected_metadata}, loaded={loaded_metadata}"
                )

        required_keys = (
            "obs",
            "current_command",
            "executed_act",
            "expert_act",
            "reward",
            "done",
            "env_id",
            "episode_id",
        )
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing key '{key}' in replay cache: {path}")

        obs = data["obs"].astype(np.float32, copy=False)
        current_command = data["current_command"].astype(np.float32, copy=False)
        executed_act = data["executed_act"].astype(np.float32, copy=False)
        expert_act = data["expert_act"].astype(np.float32, copy=False)
        has_expert_chunk = (
            data["has_expert_chunk"].astype(np.bool_, copy=False)
            if "has_expert_chunk" in data
            else np.zeros((obs.shape[0],), dtype=np.bool_)
        )
        if has_expert_chunk.shape != (obs.shape[0],):
            raise ValueError(
                "has_expert_chunk shape mismatch in cache: "
                f"got {has_expert_chunk.shape}, expected {(obs.shape[0],)}"
            )
        strict_label_valid = (
            data["strict_label_valid"].astype(np.bool_, copy=False)
            if "strict_label_valid" in data
            else np.ones((obs.shape[0],), dtype=np.bool_)
        )
        if strict_label_valid.shape != (obs.shape[0],):
            raise ValueError(
                "strict_label_valid shape mismatch in cache: "
                f"got {strict_label_valid.shape}, expected {(obs.shape[0],)}"
            )

        has_expert_future_obs_chunk = (
            data["has_expert_future_obs_chunk"].astype(np.bool_, copy=False)
            if "has_expert_future_obs_chunk" in data
            else np.zeros((obs.shape[0],), dtype=np.bool_)
        )
        if has_expert_future_obs_chunk.shape != (obs.shape[0],):
            raise ValueError(
                "has_expert_future_obs_chunk shape mismatch in cache: "
                f"got {has_expert_future_obs_chunk.shape}, expected {(obs.shape[0],)}"
            )

        expert_chunk = np.zeros((obs.shape[0], self.pred_horizon, self.act_dim), dtype=np.float32)
        if "expert_chunk" in data:
            raw_expert_chunk = data["expert_chunk"].astype(np.float32, copy=False)
            base_shape_ok = (
                raw_expert_chunk.ndim == 3
                and raw_expert_chunk.shape[0] == obs.shape[0]
                and raw_expert_chunk.shape[2] == self.act_dim
            )
            if base_shape_ok and raw_expert_chunk.shape[1] == self.pred_horizon:
                expert_chunk = raw_expert_chunk
            else:
                # Backward-compatible path for offline caches:
                # old files may contain expert_chunk with a different horizon while
                # has_expert_chunk is all False (labels actually unused).
                if np.any(has_expert_chunk):
                    raise ValueError(
                        "expert_chunk shape mismatch in cache with active chunk labels: "
                        f"got {raw_expert_chunk.shape}, expected (*, {self.pred_horizon}, {self.act_dim})"
                    )
                # Ignore mismatched stale chunk field and keep zero placeholder.
                has_expert_chunk = np.zeros((obs.shape[0],), dtype=np.bool_)
        elif np.any(has_expert_chunk):
            raise ValueError("Cache marks active chunk labels but expert_chunk field is missing")

        expert_future_obs_chunk = np.zeros((obs.shape[0], self.pred_horizon, self.obs_dim), dtype=np.float32)
        if "expert_future_obs_chunk" in data:
            raw_expert_future_obs_chunk = data["expert_future_obs_chunk"].astype(np.float32, copy=False)
            base_shape_ok = (
                raw_expert_future_obs_chunk.ndim == 3
                and raw_expert_future_obs_chunk.shape[0] == obs.shape[0]
                and raw_expert_future_obs_chunk.shape[2] == self.obs_dim
            )
            if base_shape_ok and raw_expert_future_obs_chunk.shape[1] == self.pred_horizon:
                expert_future_obs_chunk = raw_expert_future_obs_chunk
            else:
                if np.any(has_expert_future_obs_chunk):
                    raise ValueError(
                        "expert_future_obs_chunk shape mismatch in cache with active labels: "
                        f"got {raw_expert_future_obs_chunk.shape}, expected (*, {self.pred_horizon}, {self.obs_dim})"
                    )
                has_expert_future_obs_chunk = np.zeros((obs.shape[0],), dtype=np.bool_)
        elif np.any(has_expert_future_obs_chunk):
            raise ValueError("Cache marks active future-obs labels but expert_future_obs_chunk field is missing")
        reward = data["reward"].astype(np.float32, copy=False)
        done = data["done"].astype(np.bool_, copy=False)
        env_id = data["env_id"].astype(np.int32, copy=False)
        episode_id = data["episode_id"].astype(np.int64, copy=False)

        if obs.shape[1] != self.obs_dim:
            raise ValueError(f"obs_dim mismatch in cache: got {obs.shape[1]}, expected {self.obs_dim}")
        if current_command.shape[1] != self.cmd_dim:
            raise ValueError(f"cmd_dim mismatch in cache: got {current_command.shape[1]}, expected {self.cmd_dim}")
        if executed_act.shape[1] != self.act_dim:
            raise ValueError(f"act_dim mismatch in cache: got {executed_act.shape[1]}, expected {self.act_dim}")
        if expert_act.shape[1] != self.act_dim:
            raise ValueError(f"act_dim mismatch in cache: got {expert_act.shape[1]}, expected {self.act_dim}")
        if expert_chunk.ndim != 3 or expert_chunk.shape != (obs.shape[0], self.pred_horizon, self.act_dim):
            raise ValueError(
                "expert_chunk shape mismatch in cache: "
                f"got {expert_chunk.shape}, expected (*, {self.pred_horizon}, {self.act_dim})"
            )
        if expert_future_obs_chunk.ndim != 3 or expert_future_obs_chunk.shape != (obs.shape[0], self.pred_horizon, self.obs_dim):
            raise ValueError(
                "expert_future_obs_chunk shape mismatch in cache: "
                f"got {expert_future_obs_chunk.shape}, expected (*, {self.pred_horizon}, {self.obs_dim})"
            )

        self.clear()

        num = int(obs.shape[0])
        if self.growable and num > self.capacity:
            self._resize(num)
            start = 0
        else:
            start = max(0, num - self.capacity)

        kept = int(num - start)
        if kept <= 0:
            self._mark_sampling_cache_dirty()
            return 0

        tail = slice(start, num)
        self.obs[:kept] = obs[tail]
        self.current_command[:kept] = current_command[tail]
        self.executed_act[:kept] = executed_act[tail]
        self.expert_act[:kept] = expert_act[tail]
        self.expert_chunk[:kept] = expert_chunk[tail]
        self.expert_future_obs_chunk[:kept] = expert_future_obs_chunk[tail]
        self.has_expert_chunk[:kept] = has_expert_chunk[tail]
        self.has_expert_future_obs_chunk[:kept] = has_expert_future_obs_chunk[tail]
        self.strict_label_valid[:kept] = strict_label_valid[tail]
        self.reward[:kept] = reward[tail]
        self.done[:kept] = done[tail]
        self.env_id[:kept] = env_id[tail]
        self.episode_id[:kept] = episode_id[tail]
        self._arrival_order.fill(0)
        self._arrival_order[:kept] = np.arange(kept, dtype=np.int64)
        self._arrival_counter = int(kept)
        self._size = kept
        self._head = kept % self.capacity
        self._occupied_mask.fill(False)
        if self.eviction_policy == "random" and kept > 0:
            self._occupied_mask[:kept] = True
            self._random_rebuild_episode_index_state()
        else:
            self._episode_index_map.clear()
            self._episode_has_done.clear()
        self._mark_sampling_cache_dirty()
        return int(self._size)

    def split_by_trajectory(
        self,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple["ReplayBuffer", "ReplayBuffer"]:
        """Split this buffer into train and val buffers by trajectory.

        Trajectories are identified by unique (env_id, episode_id) pairs.
        A fraction ``val_ratio`` of trajectories (by count) is assigned to
        the validation buffer; the rest go to training.

        Both returned buffers are frozen (read-only).  The original buffer
        is **not** modified.

        Returns:
            (train_buffer, val_buffer)
        """
        if self._size <= 0:
            empty_kwargs = dict(
                capacity=1,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                cmd_dim=self.cmd_dim,
                history_len=self.history_len,
                pred_horizon=self.pred_horizon,
                require_future_obs_targets=self.require_future_obs_targets,
                read_only=True,
            )
            return ReplayBuffer(**empty_kwargs), ReplayBuffer(**empty_kwargs)

        chrono_idx = self._chronological_indices()
        env_ids = self.env_id[chrono_idx]
        episode_ids = self.episode_id[chrono_idx]

        # Discover unique trajectory keys, preserving first-appearance order.
        seen: dict[tuple[int, int], None] = {}
        for env, epi in zip(env_ids.tolist(), episode_ids.tolist()):
            key = (int(env), int(epi))
            if key not in seen:
                seen[key] = None
        all_keys = list(seen.keys())

        num_val = max(1, int(round(len(all_keys) * val_ratio)))
        num_val = min(num_val, len(all_keys) - 1)  # keep at least 1 for train

        rng = np.random.default_rng(seed)
        shuffled = list(range(len(all_keys)))
        rng.shuffle(shuffled)
        val_key_set = {all_keys[i] for i in shuffled[:num_val]}

        # Build per-sample mask
        val_mask = np.zeros(len(chrono_idx), dtype=np.bool_)
        for pos, (env, epi) in enumerate(zip(env_ids.tolist(), episode_ids.tolist())):
            if (int(env), int(epi)) in val_key_set:
                val_mask[pos] = True
        train_mask = ~val_mask

        train_buf = self._subset_buffer(chrono_idx, train_mask)
        val_buf = self._subset_buffer(chrono_idx, val_mask)
        train_buf.freeze()
        val_buf.freeze()
        return train_buf, val_buf

    def _subset_buffer(
        self,
        chrono_idx: np.ndarray,
        mask: np.ndarray,
    ) -> "ReplayBuffer":
        """Create a new buffer containing the masked subset of chronological data."""
        selected = chrono_idx[mask]
        n = int(selected.shape[0])
        buf = ReplayBuffer(
            capacity=max(1, n),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            cmd_dim=self.cmd_dim,
            history_len=self.history_len,
            pred_horizon=self.pred_horizon,
            require_future_obs_targets=self.require_future_obs_targets,
            growable=False,
        )
        if n <= 0:
            return buf
        sl = slice(0, n)
        buf.obs[sl] = self.obs[selected]
        buf.current_command[sl] = self.current_command[selected]
        buf.executed_act[sl] = self.executed_act[selected]
        buf.expert_act[sl] = self.expert_act[selected]
        buf.expert_chunk[sl] = self.expert_chunk[selected]
        buf.expert_future_obs_chunk[sl] = self.expert_future_obs_chunk[selected]
        buf.has_expert_chunk[sl] = self.has_expert_chunk[selected]
        buf.has_expert_future_obs_chunk[sl] = self.has_expert_future_obs_chunk[selected]
        buf.strict_label_valid[sl] = self.strict_label_valid[selected]
        buf.reward[sl] = self.reward[selected]
        buf.done[sl] = self.done[selected]
        buf.env_id[sl] = self.env_id[selected]
        buf.episode_id[sl] = self.episode_id[selected]
        buf._arrival_order[:n] = np.arange(n, dtype=np.int64)
        buf._arrival_counter = n
        buf._size = n
        buf._head = n % buf.capacity
        buf._mark_sampling_cache_dirty()
        return buf
