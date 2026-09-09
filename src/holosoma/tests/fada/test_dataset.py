from __future__ import annotations

import numpy as np
import pytest

from holosoma.fada.common.dataset import (
    ReplayBuffer,
    ReplayCacheLegacyMetadataMissingError,
    ReplayCacheMetadataMismatchError,
)


def _build_buffer(*, require_future_obs_targets: bool = True, pred_horizon: int = 1) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=16,
        obs_dim=2,
        act_dim=1,
        cmd_dim=1,
        history_len=2,
        pred_horizon=pred_horizon,
        require_future_obs_targets=require_future_obs_targets,
    )


def _base_batch(*, num_steps: int = 3) -> dict[str, np.ndarray]:
    obs = np.arange(num_steps * 2, dtype=np.float32).reshape(num_steps, 2) + 1.0
    current_command = (np.arange(num_steps, dtype=np.float32).reshape(num_steps, 1) + 1.0) / 10.0
    executed_act = (np.arange(num_steps, dtype=np.float32).reshape(num_steps, 1) + 5.0) / 10.0
    expert_act = (np.arange(num_steps, dtype=np.float32).reshape(num_steps, 1) + 15.0) / 10.0
    obs = np.asarray(
        obs,
        dtype=np.float32,
    )
    return {
        "obs": obs,
        "current_command": np.asarray(current_command, dtype=np.float32),
        "executed_act": np.asarray(executed_act, dtype=np.float32),
        "expert_act": np.asarray(expert_act, dtype=np.float32),
        "reward": np.zeros((num_steps,), dtype=np.float32),
        "done": np.zeros((num_steps,), dtype=np.bool_),
        "env_id": np.zeros((num_steps,), dtype=np.int32),
        "episode_id": np.zeros((num_steps,), dtype=np.int64),
    }


def _compact_metadata(*, source_checkpoint_abs: str) -> dict[str, object]:
    return {
        "term_scale": {
            "base_ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "projected_gravity": 1.0,
        },
        "term_noise": {
            "base_ang_vel": 0.3,
            "dof_pos": 0.01,
            "dof_vel": 1.0,
            "projected_gravity": 0.2,
        },
        "add_noise": False,
        "noise_seed": None,
        "source_checkpoint_abs": source_checkpoint_abs,
    }


@pytest.mark.parametrize("pred_horizon", [1, 2, 4])
def test_sample_chunk_prefers_expert_future_obs_chunk(pred_horizon: int):
    buffer = _build_buffer(require_future_obs_targets=True, pred_horizon=pred_horizon)
    num_steps = pred_horizon + 2
    batch = _base_batch(num_steps=num_steps)
    expert_chunk = np.zeros((num_steps, pred_horizon, 1), dtype=np.float32)
    expert_chunk[0, :, 0] = np.arange(pred_horizon, dtype=np.float32) + 9.0
    expert_future_obs_chunk = np.zeros((num_steps, pred_horizon, 2), dtype=np.float32)
    expert_future_obs_chunk[0, :, :] = np.stack(
        [
            np.arange(pred_horizon, dtype=np.float32) + 100.0,
            np.arange(pred_horizon, dtype=np.float32) + 200.0,
        ],
        axis=1,
    )
    strict_label_valid = np.zeros((num_steps,), dtype=np.bool_)
    strict_label_valid[0] = True
    buffer.add_batch(
        **batch,
        expert_chunk=expert_chunk,
        expert_future_obs_chunk=expert_future_obs_chunk,
        strict_label_valid=strict_label_valid,
    )

    sampled = buffer.sample_chunk(batch_size=1)
    assert sampled["target_actions"].shape == (1, pred_horizon, 1)
    assert sampled["target_next_observations"].shape == (1, pred_horizon, 2)
    assert np.allclose(sampled["target_actions"][0, :, 0], expert_chunk[0, :, 0])
    assert np.allclose(sampled["target_next_observations"][0], expert_future_obs_chunk[0])


@pytest.mark.parametrize("pred_horizon", [1, 2, 4])
def test_sample_chunk_raises_when_no_strict_future_obs(pred_horizon: int):
    buffer = _build_buffer(require_future_obs_targets=True, pred_horizon=pred_horizon)
    num_steps = pred_horizon + 2
    batch = _base_batch(num_steps=num_steps)
    expert_chunk = np.zeros((num_steps, pred_horizon, 1), dtype=np.float32)
    expert_chunk[0, :, 0] = np.arange(pred_horizon, dtype=np.float32) + 8.0
    strict_label_valid = np.zeros((num_steps,), dtype=np.bool_)
    strict_label_valid[0] = True
    buffer.add_batch(
        **batch,
        expert_chunk=expert_chunk,
        expert_future_obs_chunk=None,
        strict_label_valid=strict_label_valid,
    )

    with pytest.raises(RuntimeError, match="No valid chunks available in replay buffer"):
        buffer.sample_chunk(batch_size=1)


def test_load_npz_metadata_mismatch_raises(tmp_path):
    path = tmp_path / "cache_with_meta.npz"
    writer = _build_buffer(require_future_obs_targets=True)
    writer.add_batch(
        **_base_batch(),
        expert_chunk=None,
        expert_future_obs_chunk=None,
        strict_label_valid=None,
    )
    writer.save_npz(path, compact_obs_metadata=_compact_metadata(source_checkpoint_abs="/tmp/source_a.pt"))

    reader = _build_buffer(require_future_obs_targets=True)
    with pytest.raises(ReplayCacheMetadataMismatchError):
        reader.load_npz(
            path,
            expected_compact_obs_metadata=_compact_metadata(source_checkpoint_abs="/tmp/source_b.pt"),
        )


def test_load_npz_legacy_missing_metadata_raises(tmp_path):
    path = tmp_path / "legacy_cache.npz"
    writer = _build_buffer(require_future_obs_targets=True)
    writer.add_batch(
        **_base_batch(),
        expert_chunk=None,
        expert_future_obs_chunk=None,
        strict_label_valid=None,
    )
    writer.save_npz(path)

    reader = _build_buffer(require_future_obs_targets=True)
    with pytest.raises(ReplayCacheLegacyMetadataMissingError):
        reader.load_npz(
            path,
            expected_compact_obs_metadata=_compact_metadata(source_checkpoint_abs="/tmp/source_a.pt"),
        )


def test_load_npz_metadata_match_succeeds(tmp_path):
    path = tmp_path / "cache_match.npz"
    metadata = _compact_metadata(source_checkpoint_abs="/tmp/source_ok.pt")
    writer = _build_buffer(require_future_obs_targets=True)
    writer.add_batch(
        **_base_batch(),
        expert_chunk=None,
        expert_future_obs_chunk=None,
        strict_label_valid=None,
    )
    writer.save_npz(path, compact_obs_metadata=metadata)

    reader = _build_buffer(require_future_obs_targets=True)
    loaded = reader.load_npz(path, expected_compact_obs_metadata=metadata)
    assert loaded == 3


def test_num_valid_chunks_allows_tail_anchors_with_strict_future_obs_chunks():
    pred_horizon = 2
    num_steps = 3
    buffer = _build_buffer(require_future_obs_targets=True, pred_horizon=pred_horizon)
    batch = _base_batch(num_steps=num_steps)
    strict_label_valid = np.ones((num_steps,), dtype=np.bool_)
    expert_chunk = np.zeros((num_steps, pred_horizon, 1), dtype=np.float32)
    expert_future_obs_chunk = np.zeros((num_steps, pred_horizon, 2), dtype=np.float32)
    buffer.add_batch(
        **batch,
        expert_chunk=expert_chunk,
        expert_future_obs_chunk=expert_future_obs_chunk,
        strict_label_valid=strict_label_valid,
    )
    assert buffer.num_valid_chunks() == 3


def test_num_valid_chunks_allows_tail_anchors_with_strict_action_chunks_only():
    pred_horizon = 4
    num_steps = 4
    buffer = _build_buffer(require_future_obs_targets=False, pred_horizon=pred_horizon)
    batch = _base_batch(num_steps=num_steps)
    strict_label_valid = np.ones((num_steps,), dtype=np.bool_)
    expert_chunk = np.zeros((num_steps, pred_horizon, 1), dtype=np.float32)
    buffer.add_batch(
        **batch,
        expert_chunk=expert_chunk,
        expert_future_obs_chunk=None,
        strict_label_valid=strict_label_valid,
    )
    assert buffer.num_valid_chunks() == 4


def test_sample_chunk_returns_anchor_current_command():
    buffer = _build_buffer(require_future_obs_targets=False, pred_horizon=1)
    buffer.add_batch(
        **_base_batch(num_steps=5),
        expert_chunk=None,
        expert_future_obs_chunk=None,
        strict_label_valid=None,
    )

    sampled = buffer.sample_chunk(batch_size=8)
    current_command = sampled["current_command"]
    history_valid = sampled["history_valid_mask"]

    assert current_command.shape == (8, 1)
    valid_command_values = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    for batch_idx in range(current_command.shape[0]):
        valid_slots = np.flatnonzero(history_valid[batch_idx])
        assert valid_slots.size > 0
        assert np.any(np.isclose(current_command[batch_idx, 0], valid_command_values))


def test_load_npz_metadata_path_normalization_succeeds(tmp_path):
    path = tmp_path / "cache_path_norm.npz"
    writer = _build_buffer(require_future_obs_targets=True)
    writer.add_batch(
        **_base_batch(),
        expert_chunk=None,
        expert_future_obs_chunk=None,
        strict_label_valid=None,
    )
    writer.save_npz(path, compact_obs_metadata=_compact_metadata(source_checkpoint_abs="/tmp/source_norm.pt"))

    reader = _build_buffer(require_future_obs_targets=True)
    loaded = reader.load_npz(
        path,
        expected_compact_obs_metadata=_compact_metadata(source_checkpoint_abs="/tmp/../tmp/source_norm.pt"),
    )
    assert loaded == 3


def test_load_npz_metadata_float_tolerance_succeeds(tmp_path):
    path = tmp_path / "cache_float_tol.npz"
    metadata = _compact_metadata(source_checkpoint_abs="/tmp/source_float_tol.pt")
    writer = _build_buffer(require_future_obs_targets=True)
    writer.add_batch(
        **_base_batch(),
        expert_chunk=None,
        expert_future_obs_chunk=None,
        strict_label_valid=None,
    )
    writer.save_npz(path, compact_obs_metadata=metadata)

    expected = _compact_metadata(source_checkpoint_abs="/tmp/source_float_tol.pt")
    expected["term_noise"]["dof_vel"] = float(expected["term_noise"]["dof_vel"]) + 1e-10
    reader = _build_buffer(require_future_obs_targets=True)
    loaded = reader.load_npz(path, expected_compact_obs_metadata=expected)
    assert loaded == 3


def test_random_eviction_policy_evicts_completed_episodes_as_units():
    np.random.seed(1)
    buffer = ReplayBuffer(
        capacity=4,
        obs_dim=1,
        act_dim=1,
        cmd_dim=1,
        history_len=1,
        pred_horizon=1,
        require_future_obs_targets=False,
        eviction_policy="random",
    )

    buffer.add_batch(
        obs=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        current_command=np.zeros((4, 1), dtype=np.float32),
        executed_act=np.zeros((4, 1), dtype=np.float32),
        expert_act=np.zeros((4, 1), dtype=np.float32),
        reward=np.zeros((4,), dtype=np.float32),
        done=np.asarray([False, True, False, True], dtype=np.bool_),
        env_id=np.zeros((4,), dtype=np.int32),
        episode_id=np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    buffer.add_batch(
        obs=np.asarray([[10.0], [11.0]], dtype=np.float32),
        current_command=np.zeros((2, 1), dtype=np.float32),
        executed_act=np.zeros((2, 1), dtype=np.float32),
        expert_act=np.zeros((2, 1), dtype=np.float32),
        reward=np.zeros((2,), dtype=np.float32),
        done=np.asarray([False, True], dtype=np.bool_),
        env_id=np.zeros((2,), dtype=np.int32),
        episode_id=np.asarray([2, 2], dtype=np.int64),
    )

    chrono = buffer._chronological_indices()
    kept_episode_ids = buffer.episode_id[chrono]
    assert buffer._size == 4
    assert int(np.sum(kept_episode_ids == 2)) == 2
    assert int(np.sum(kept_episode_ids == 0)) in {0, 2}
    assert int(np.sum(kept_episode_ids == 1)) in {0, 2}


def test_random_eviction_policy_evicts_ongoing_episodes_as_whole_units():
    np.random.seed(0)
    buffer = ReplayBuffer(
        capacity=6,
        obs_dim=1,
        act_dim=1,
        cmd_dim=1,
        history_len=1,
        pred_horizon=1,
        require_future_obs_targets=False,
        eviction_policy="random",
    )

    # Two ongoing trajectory units (no done marker) with fixed lengths.
    buffer.add_batch(
        obs=np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32),
        current_command=np.zeros((6, 1), dtype=np.float32),
        executed_act=np.zeros((6, 1), dtype=np.float32),
        expert_act=np.zeros((6, 1), dtype=np.float32),
        reward=np.zeros((6,), dtype=np.float32),
        done=np.asarray([False, False, False, False, False, False], dtype=np.bool_),
        env_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        episode_id=np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64),
    )

    # Force eviction; required slots=2 while each existing episode unit has length 3.
    # The evicted unit must be removed as a whole (0 or 3 kept from old units).
    buffer.add_batch(
        obs=np.asarray([[10.0], [11.0]], dtype=np.float32),
        current_command=np.zeros((2, 1), dtype=np.float32),
        executed_act=np.zeros((2, 1), dtype=np.float32),
        expert_act=np.zeros((2, 1), dtype=np.float32),
        reward=np.zeros((2,), dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        env_id=np.asarray([2, 2], dtype=np.int32),
        episode_id=np.asarray([0, 0], dtype=np.int64),
    )

    chrono = buffer._chronological_indices()
    kept_env = buffer.env_id[chrono]
    # Old env units are either kept entirely (3) or evicted entirely (0), never partially.
    assert int(np.sum(kept_env == 0)) in {0, 3}
    assert int(np.sum(kept_env == 1)) in {0, 3}
    assert int(np.sum(kept_env == 2)) == 2


def test_num_valid_chunks_ignores_fallback_horizon_crossing_invalid_tail():
    buffer = ReplayBuffer(
        capacity=16,
        obs_dim=1,
        act_dim=1,
        cmd_dim=1,
        history_len=1,
        pred_horizon=2,
        require_future_obs_targets=False,
    )
    buffer.add_batch(
        obs=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        current_command=np.zeros((4, 1), dtype=np.float32),
        executed_act=np.zeros((4, 1), dtype=np.float32),
        expert_act=np.zeros((4, 1), dtype=np.float32),
        reward=np.zeros((4,), dtype=np.float32),
        done=np.asarray([False, True, False, True], dtype=np.bool_),
        env_id=np.asarray([0, 0, 0, 0], dtype=np.int32),
        episode_id=np.asarray([0, 0, 0, 0], dtype=np.int64),
        strict_label_valid=np.asarray([True, True, False, False], dtype=np.bool_),
    )
    assert buffer.num_valid_chunks() == 1
