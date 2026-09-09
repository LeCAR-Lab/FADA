from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from holosoma.agents.modules.logging_utils import LoggingHelper


@pytest.fixture
def mock_writer():
    """Fixture providing a mock SummaryWriter."""
    return MagicMock(spec=SummaryWriter)


@pytest.fixture
def mock_wandb():
    """Fixture providing mocked wandb module."""
    with patch("holosoma.agents.modules.logging_utils.wandb") as mock_wandb:
        yield mock_wandb


@pytest.fixture
def logging_helper(mock_writer):
    """Fixture providing a LoggingHelper instance with default parameters."""
    return LoggingHelper(
        writer=mock_writer,
        log_dir="/tmp/test_logs",
        num_envs=2,
        num_steps_per_env=10,
        num_learning_iterations=100,
        device="cpu",
    )


@pytest.fixture
def prefixed_logging_helper(mock_writer):
    """Fixture providing a LoggingHelper instance with a prefix."""
    return LoggingHelper(
        writer=mock_writer,
        log_dir="/tmp/test_logs",
        num_envs=2,
        num_steps_per_env=10,
        num_learning_iterations=100,
        device="cpu",
        prefix="test_prefix/",
    )


def test_prefix_in_logging(prefixed_logging_helper, mock_writer, mock_wandb):
    """Test that the prefix is properly added to all logged metrics."""
    # Add episode info
    prefixed_logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=prefixed_logging_helper.device)}]

    # Call post_epoch_logging with some test data
    prefixed_logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Check that the prefix was added to all logged metrics
    expected_calls = [
        "test_prefix/Loss/test_loss",
        "test_prefix/Perf/total_fps",
        "test_prefix/Perf/collection_time",
        "test_prefix/Perf/learning_time",
        "test_prefix/Train/num_samples",
        "test_prefix/test_section/test_metric",
        "test_prefix/Episode/test_metric",
    ]

    # Verify all expected calls were made to writer.add_scalar
    actual_calls = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
    for expected in expected_calls:
        assert expected in actual_calls


def test_no_prefix_logging(logging_helper, mock_writer, mock_wandb):
    """Test that logging works correctly without a prefix."""
    # Add episode info
    logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=logging_helper.device)}]

    # Call post_epoch_logging with some test data
    logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Check that metrics were logged without prefix
    expected_calls = [
        "Loss/test_loss",
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
        "Train/num_samples",
        "test_section/test_metric",
        "Episode/test_metric",
    ]

    # Verify all expected calls were made to writer.add_scalar
    actual_calls = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
    for expected in expected_calls:
        assert expected in actual_calls


def test_episode_stats_update(logging_helper):
    """Test that episode statistics are properly updated."""
    # Create test data
    rewards = torch.tensor([1.0, 2.0], device=logging_helper.device)
    dones = torch.tensor([1.0, 0.0], device=logging_helper.device)
    infos = {
        "episode": {
            "test_metric": torch.tensor([1.0], device=logging_helper.device),
        },
        "raw_episode": {
            "raw_test_metric": torch.tensor([2.0], device=logging_helper.device),
        },
        "to_log": {
            "env_metric": torch.tensor([2.0], device=logging_helper.device),
        },
    }

    # Update episode stats
    logging_helper.update_episode_stats(rewards, dones, infos)

    # Verify reward buffer was updated
    assert len(logging_helper.rewbuffer) == 1
    assert logging_helper.rewbuffer[0] == 1.0  # First environment's reward

    # Verify length buffer was updated
    assert len(logging_helper.lenbuffer) == 1
    assert logging_helper.lenbuffer[0] == 1.0  # First environment's length

    # Verify episode info was stored
    assert len(logging_helper.ep_infos) == 1
    assert logging_helper.ep_infos[0]["test_metric"].item() == 1.0

    # Verify raw episode info was stored
    assert len(logging_helper.raw_ep_infos) == 1
    assert logging_helper.raw_ep_infos[0]["raw_test_metric"].item() == 2.0


def test_wandb_logging(prefixed_logging_helper, mock_wandb):
    """Test that metrics are properly logged to wandb when available."""
    # Add some episode info to avoid empty list error
    prefixed_logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=prefixed_logging_helper.device)}]
    prefixed_logging_helper.raw_ep_infos = [
        {"raw_test_metric": torch.tensor([2.0], device=prefixed_logging_helper.device)}
    ]

    # Call post_epoch_logging with some test data
    prefixed_logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Verify wandb.log was called with the correct data
    mock_wandb.log.assert_called_once()
    logged_data = mock_wandb.log.call_args[0][0]
    assert "test_prefix/Loss/test_loss" in logged_data
    assert "test_prefix/test_section/test_metric" in logged_data
    assert "test_prefix/Episode/test_metric" in logged_data
    assert "test_prefix/RawEpisode/raw_test_metric" in logged_data
    assert logged_data["global_step"] == 0


def test_save_checkpoint_artifact(prefixed_logging_helper, mock_wandb, tmp_path):
    """Test that checkpoints are properly saved and logged to wandb."""
    # Create a temporary directory for the test
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir()
    prefixed_logging_helper.log_dir = str(log_dir)

    # Create test state dict
    state_dict = {"test_param": torch.tensor([1.0])}
    checkpoint_path = log_dir / "checkpoint.pt"

    # Save checkpoint
    prefixed_logging_helper.save_checkpoint_artifact(state_dict, str(checkpoint_path))

    # Verify wandb.save was called with correct path
    mock_wandb.save.assert_called_once()
    assert mock_wandb.save.call_args[0][0] == str(checkpoint_path)
    assert mock_wandb.save.call_args[1]["base_path"] == str(log_dir)
