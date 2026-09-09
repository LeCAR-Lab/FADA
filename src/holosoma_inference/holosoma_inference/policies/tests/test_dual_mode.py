from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from holosoma.utils.data_collector import DataCollector

from holosoma_inference.policies.dual_mode import DualModePolicy, _select_policy_class
from holosoma_inference.policies.locomotion import LocomotionPolicy, LocomotionPolicy_Deploy
from holosoma_inference.policies.locomotion_fada import LocomotionPolicy_FADA
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.sdk.vel_state_processor.basic_vel_state_processor import BasicVelStateProcessor
from holosoma_inference.utils.inference_logger import InferenceLogger


class _NoopMeasure:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyLatencyTracker:
    def measure(self, _name: str):
        return _NoopMeasure()


class _DummyRate:
    def sleep(self) -> None:
        return


class _DummyInterface:
    def __init__(self, state: object | None = None):
        self._state = state if state is not None else object()

    def get_low_state(self):
        return self._state


class _FakePolicy:
    def __init__(self, name: str):
        self.name = name
        self.handle_joystick_button = lambda key: None
        self.handle_keyboard_button = lambda key: None
        self.interface = _DummyInterface()
        self.rate = _DummyRate()
        self.config = SimpleNamespace(task=SimpleNamespace(done_signal_file=None, ready_signal_file=None))
        self.desired_base_height = 0.72
        self.lin_vel_command = np.ones((1, 2), dtype=np.float32)
        self.ang_vel_command = np.ones((1, 1), dtype=np.float32)
        self.stand_command = np.ones((1, 1), dtype=np.float32)
        self.base_height_command = np.zeros((1, 1), dtype=np.float32)
        self.key_states = {}
        self.last_key_states = {}
        self.prepare_recorded_calls: list[bool] = []
        self.prepare_unrecorded_calls = 0
        self.reset_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.after_run_calls = 0
        self._resolve_control_gains_calls = 0
        self.projected_gravity_z = -1.0
        self.iteration_returns: list[bool] = []

    def _resolve_control_gains(self):
        self._resolve_control_gains_calls += 1

    def _handle_start_policy(self):
        self.start_calls += 1

    def _handle_stop_policy(self):
        self.stop_calls += 1

    def reset_runtime_state_for_new_session(self):
        self.reset_calls += 1
        self.lin_vel_command[:] = 0.0
        self.ang_vel_command[:] = 0.0
        self.stand_command[:] = 0.0

    def prepare_recorded_session(self, reset_outputs: bool):
        self.prepare_recorded_calls.append(bool(reset_outputs))

    def prepare_unrecorded_session(self):
        self.prepare_unrecorded_calls += 1

    def get_projected_gravity_z(self, _robot_state) -> float:
        return float(self.projected_gravity_z)

    def _run_iteration(self, _it: int) -> bool:
        if self.iteration_returns:
            return self.iteration_returns.pop(0)
        return False

    def _after_run_loop(self):
        self.after_run_calls += 1


def _make_controller() -> DualModePolicy:
    controller = object.__new__(DualModePolicy)
    controller.primary = _FakePolicy("primary")
    controller.secondary = _FakePolicy("secondary")
    controller.active = controller.secondary
    controller.active_label = "secondary"
    controller._session_mode = True
    controller._start_in_secondary = True
    controller._test_start_delay_s = -1.0
    controller._post_secondary_hold_s = 0.0
    controller._exit_after_post_hold = False
    controller._projected_gravity_z_min = -0.7
    controller._primary_iteration = 0
    controller._secondary_iteration = 0
    controller._auto_test_deadline = None
    controller._post_hold_deadline = None
    return controller


def test_session_toggle_resets_primary_from_command_zero() -> None:
    controller = _make_controller()

    controller._handle_session_toggle()

    assert controller.active is controller.primary
    assert controller.primary.reset_calls == 1
    assert controller.primary.prepare_recorded_calls == [True]
    assert controller.primary.start_calls == 1
    assert controller.secondary.stop_calls == 1
    assert controller._primary_iteration == 0

    controller._handle_session_toggle()
    controller._handle_session_toggle()

    assert controller.active is controller.primary
    assert controller.primary.reset_calls == 2
    assert controller.primary.prepare_recorded_calls == [True, True]
    assert controller.primary.prepare_unrecorded_calls == 1


def test_select_policy_class_respects_explicit_policy_mode() -> None:
    transformer_cfg = SimpleNamespace(
        task=SimpleNamespace(policy_mode="transformer"),
        robot=SimpleNamespace(robot_type="t1-23dof"),
        observation=SimpleNamespace(obs_dict={"actor_obs": []}),
    )
    robust_cfg = SimpleNamespace(
        task=SimpleNamespace(policy_mode="deploy"),
        robot=SimpleNamespace(robot_type="t1-23dof"),
        observation=SimpleNamespace(obs_dict={"actor_obs": []}),
    )

    assert _select_policy_class(transformer_cfg).__name__ == "LocomotionPolicy_FADA"
    assert _select_policy_class(robust_cfg).__name__ == "LocomotionPolicy_Deploy"


def test_projected_gravity_guard_switches_back_to_secondary() -> None:
    controller = _make_controller()
    controller._handle_session_toggle()
    controller.primary.projected_gravity_z = -0.6

    switched = controller._check_primary_safety()

    assert switched is True
    assert controller.active is controller.secondary
    assert controller.primary.prepare_unrecorded_calls == 1
    assert controller.secondary.start_calls == 1


def test_auto_session_runs_primary_then_returns_to_secondary() -> None:
    controller = _make_controller()
    controller._test_start_delay_s = 0.0
    controller._exit_after_post_hold = True
    controller._auto_test_deadline = time.perf_counter() - 1.0
    controller.primary.iteration_returns = [True]

    controller._run_session_mode()

    assert controller.primary.prepare_recorded_calls == [True]
    assert controller.active is controller.secondary


def test_random_eval_commands_hold_zero_before_first_resample() -> None:
    class _DummyLogger:
        def info(self, *_args, **_kwargs):
            return

    class _RandomCommandPolicy:
        _maybe_resample_commands = LocomotionPolicy_Deploy._maybe_resample_commands
        _hold_initial_zero_random_commands = LocomotionPolicy_Deploy._hold_initial_zero_random_commands
        _zero_out_commands = LocomotionPolicy_Deploy._zero_out_commands

        def __init__(self):
            self.randomize_commands = True
            self.command_resample_steps = 4
            self.initial_zero_command_steps = 5
            self.last_random_resample_iteration = -1
            self._initial_zero_command_hold_logged = False
            self._initial_zero_command_release_logged = False
            self.logger = _DummyLogger()
            self.lin_vel_command = np.full((1, 2), 99.0, dtype=np.float32)
            self.ang_vel_command = np.full((1, 1), 99.0, dtype=np.float32)
            self.stand_command = np.full((1, 1), 99.0, dtype=np.float32)
            self.resample_calls: list[int] = []

        def _resample_commands(self):
            self.resample_calls.append(self._active_iteration)
            self.stand_command[0, 0] = 1.0
            self.lin_vel_command[0, :] = np.asarray([0.3, -0.2], dtype=np.float32)
            self.ang_vel_command[0, 0] = 0.4

    policy = _RandomCommandPolicy()

    for it in range(5):
        policy._active_iteration = it
        policy._maybe_resample_commands(it)
        np.testing.assert_allclose(policy.lin_vel_command[0], [0.0, 0.0])
        np.testing.assert_allclose(policy.ang_vel_command[0], [0.0])
        np.testing.assert_allclose(policy.stand_command[0], [0.0])

    policy._active_iteration = 5
    policy._maybe_resample_commands(5)
    assert policy.resample_calls == [5]
    np.testing.assert_allclose(policy.lin_vel_command[0], [0.3, -0.2])
    np.testing.assert_allclose(policy.ang_vel_command[0], [0.4])
    np.testing.assert_allclose(policy.stand_command[0], [1.0])

    for it in range(6, 9):
        policy._active_iteration = it
        policy._maybe_resample_commands(it)
    assert policy.resample_calls == [5]

    policy._active_iteration = 9
    policy._maybe_resample_commands(9)
    assert policy.resample_calls == [5, 9]


def test_inference_logger_reset_session_keeps_only_latest_session(tmp_path: Path) -> None:
    logger = InferenceLogger(str(tmp_path), "state_log")

    logger.log_states({"reward": 1.0, "done": False})
    logger.reset_session()
    logger.log_states({"reward": 2.5, "done": True})

    save_path = logger.save()
    assert save_path is not None

    with np.load(save_path) as data:
        assert data["reward"].tolist() == [2.5]
        assert data["done"].tolist() == [True]


def test_vel_state_processor_session_recording_gate_keeps_live_pose_cache() -> None:
    processor = BasicVelStateProcessor("base", record_mocap_history=True)

    processor.update([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0], timestamp=1.0)
    processor.record_command(1.0, 0.4, 0.0, 0.1)
    processor.record_reconstruction(1.0, np.array([1.0, 2.0]), np.array([1.5, 2.5]))

    processor.set_recording_enabled(False)
    processor.update([1.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0], timestamp=2.0)
    processor.record_command(2.0, 0.0, 0.0, 0.0)
    processor.record_reconstruction(2.0, np.array([3.0, 4.0]), np.array([3.5, 4.5]))

    mocap_ts, mocap_pos, _ = processor.get_mocap_history()
    cmd_ts, cmd_x, _, _ = processor.get_command_history()
    recon_ts, actual_target, predicted_target = processor.get_recon_history()

    assert mocap_ts is not None and mocap_ts.tolist() == [1.0]
    assert mocap_pos is not None and mocap_pos.tolist() == [[0.0, 0.0, 0.5]]
    assert cmd_ts is not None and cmd_ts.tolist() == [1.0]
    assert cmd_x is not None and cmd_x.tolist() == [0.4]
    assert recon_ts is not None and recon_ts.tolist() == [1.0]
    assert actual_target is not None and actual_target.tolist() == [[1.0, 2.0]]
    assert predicted_target is not None and predicted_target.tolist() == [[[1.5, 2.5]]]
    assert processor.get_last_mocap_timestamp() == 2.0
    assert processor.get_pose_state() is not None
    assert processor.get_pose_state().tolist() == [[1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]]

    processor.reset_session_history()
    assert processor.get_mocap_history() == (None, None, None)
    assert processor.get_command_history() == (None, None, None, None)
    assert processor.get_recon_history() == (None, None, None)
    assert processor.get_last_mocap_timestamp() == 2.0


def test_vel_state_processor_flattens_ragged_idm_and_fdm_windows() -> None:
    processor = BasicVelStateProcessor("base", record_mocap_history=True)

    processor.record_idm_inverse(1.0, np.zeros((1, 2, 3)), np.ones((1, 2, 3)))
    processor.record_idm_inverse(2.0, np.zeros((2, 2, 3)), np.ones((2, 2, 3)))
    processor.record_fdm_forward(3.0, np.zeros((1, 2, 4)), np.ones((1, 2, 4)))
    processor.record_fdm_forward(4.0, np.zeros((3, 2, 4)), np.ones((3, 2, 4)))

    idm_ts, idm_actual, idm_predicted = processor.get_idm_inverse_history()
    fdm_ts, fdm_actual, fdm_predicted = processor.get_fdm_forward_history()

    assert idm_ts is not None and idm_ts.tolist() == [1.0, 2.0, 2.0]
    assert idm_actual is not None and idm_actual.shape == (3, 2, 3)
    assert idm_predicted is not None and idm_predicted.shape == (3, 2, 3)

    assert fdm_ts is not None and fdm_ts.tolist() == [3.0, 4.0, 4.0, 4.0]
    assert fdm_actual is not None and fdm_actual.shape == (4, 2, 4)
    assert fdm_predicted is not None and fdm_predicted.shape == (4, 2, 4)


def test_data_collector_reset_session_replaces_previous_dataset(tmp_path: Path) -> None:
    collector = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="session_dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )

    collector.collect_step(
        obs_dict={"base_obs": np.array([[1.0]], dtype=np.float32)},
        actions=np.array([[0.1, 0.2]], dtype=np.float32),
        dones=np.array([False]),
    )
    collector.flush()

    save_path = tmp_path / "session_dataset.h5"
    with h5py.File(save_path, "r") as handle:
        assert list(handle["episodes"].keys()) == ["episode_0"]
        np.testing.assert_allclose(handle["episodes"]["episode_0"]["actions"][0, 0], [0.1, 0.2])

    collector.reset_session()
    assert collector.has_rewards is None
    assert collector.current_episode is None
    assert collector.saved_episode_count == 0
    assert list(collector.h5_episodes_group.keys()) == []

    collector.collect_step(
        obs_dict={"base_obs": np.array([[9.0]], dtype=np.float32)},
        actions=np.array([[0.9, 1.1]], dtype=np.float32),
        dones=np.array([False]),
        rewards=np.array([3.5], dtype=np.float32),
    )
    collector.close()

    with h5py.File(save_path, "r") as handle:
        assert bool(handle.attrs["has_rewards"]) is True
        assert list(handle["episodes"].keys()) == ["episode_0"]
        episode = handle["episodes"]["episode_0"]
        np.testing.assert_allclose(episode["base_obs"][0, 0], [9.0])
        np.testing.assert_allclose(episode["actions"][0, 0], [0.9, 1.1])
        np.testing.assert_allclose(episode["rewards"][0, 0], [3.5])


# ---------------------------------------------------------------------------
# A run that "completes" without producing a valid H5 dataset must not exit 0.
# _validate_data_collection_output() (called from
# LocomotionPolicy_Deploy._after_run_loop after data_collector.close()) checks the
# file was flushed/closed, exists on disk, and has >=1 saved episode.
# ---------------------------------------------------------------------------


def _minimal_deploy_policy_for_after_run_loop(data_collector) -> LocomotionPolicy_Deploy:
    policy = object.__new__(LocomotionPolicy_Deploy)
    policy.data_collector = data_collector
    policy.logger = logging.getLogger("test_dual_mode")
    policy.state_logger = None
    policy.config = SimpleNamespace(task=SimpleNamespace(done_signal_file=None))
    return policy


def test_validate_data_collection_output_passes_with_one_saved_episode(tmp_path: Path) -> None:
    collector = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )
    collector.collect_step(
        obs_dict={"base_obs": np.array([[1.0]], dtype=np.float32)},
        actions=np.array([[0.1, 0.2]], dtype=np.float32),
        dones=np.array([False]),
    )
    collector.close()

    policy = _minimal_deploy_policy_for_after_run_loop(collector)
    policy._validate_data_collection_output()  # must not raise


def test_validate_data_collection_output_raises_on_zero_episodes(tmp_path: Path) -> None:
    """No collect_step() before close(): the H5 file exists but is empty, and validation
    raises."""
    collector = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )
    collector.close()

    policy = _minimal_deploy_policy_for_after_run_loop(collector)
    with pytest.raises(RuntimeError, match="zero new episodes/steps"):
        policy._validate_data_collection_output()


# ---------------------------------------------------------------------------
# _validate_data_collection_output() checks what this run itself produced, not
# `saved_episode_count` (the H5 file's *total* episode count, which includes episodes a
# prior run appended). DataCollector tracks new_episode_count/new_step_count (reset to 0
# on construction/reset_session, incremented only by this session's own work) and
# validation checks those. Appending to an existing file across multiple runs passes when
# this run does add data.
# ---------------------------------------------------------------------------


def test_validate_data_collection_output_raises_when_reopened_run_collects_nothing_new(
    tmp_path: Path,
) -> None:
    """Run 1 writes one episode and closes; run 2 opens the same dataset.h5 path but
    never calls collect_step() before closing. The file's total episode count is still 1
    (from run 1) while run 2 added zero, and validation raises."""
    first_run = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )
    first_run.collect_step(
        obs_dict={"base_obs": np.array([[1.0]], dtype=np.float32)},
        actions=np.array([[0.1, 0.2]], dtype=np.float32),
        dones=np.array([False]),
    )
    first_run.close()
    assert first_run.saved_episode_count == 1

    second_run = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )
    # Open the existing file, inherit its total count, and never collect anything before
    # close().
    assert second_run.saved_episode_count == 1
    assert second_run.new_episode_count == 0
    second_run.close()
    assert second_run.saved_episode_count == 1  # unchanged: no new episode was appended

    policy = _minimal_deploy_policy_for_after_run_loop(second_run)
    with pytest.raises(RuntimeError, match="zero new episodes/steps"):
        policy._validate_data_collection_output()


def test_validate_data_collection_output_passes_when_reopened_run_appends_new_episode(
    tmp_path: Path,
) -> None:
    """Run 2 opens the same dataset.h5 as run 1 and *does* collect a new episode before
    closing: validation passes and the total episode count covers both runs."""
    first_run = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )
    first_run.collect_step(
        obs_dict={"base_obs": np.array([[1.0]], dtype=np.float32)},
        actions=np.array([[0.1, 0.2]], dtype=np.float32),
        dones=np.array([False]),
    )
    first_run.close()

    second_run = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )
    assert second_run.saved_episode_count == 1  # inherited from run 1's file
    second_run.collect_step(
        obs_dict={"base_obs": np.array([[2.0]], dtype=np.float32)},
        actions=np.array([[0.3, 0.4]], dtype=np.float32),
        dones=np.array([False]),
    )
    second_run.close()
    assert second_run.new_episode_count == 1
    assert second_run.saved_episode_count == 2  # both runs' episodes now in the file

    policy = _minimal_deploy_policy_for_after_run_loop(second_run)
    policy._validate_data_collection_output()  # must not raise

    with h5py.File(tmp_path / "dataset.h5", "r") as handle:
        assert sorted(handle["episodes"].keys()) == ["episode_0", "episode_1"]


def test_validate_data_collection_output_raises_when_not_closed(tmp_path: Path) -> None:
    collector = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )
    # close() is not called, so the h5_file handle is still open.
    policy = _minimal_deploy_policy_for_after_run_loop(collector)
    try:
        with pytest.raises(RuntimeError, match="not cleanly flushed/closed"):
            policy._validate_data_collection_output()
    finally:
        collector.close()


def test_after_run_loop_raises_when_collection_invalid_but_still_writes_done_signal(
    tmp_path: Path,
) -> None:
    """A data collector that produced zero episodes makes _after_run_loop raise (->
    non-zero process exit via run_policy.py's top-level except) after the rest of teardown
    has run, including the orchestrator done-signal write."""
    collector = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )

    done_signal_file = tmp_path / "done_signal"
    policy = object.__new__(LocomotionPolicy_Deploy)
    policy.data_collector = collector
    policy.logger = logging.getLogger("test_dual_mode")
    policy.state_logger = None
    policy.config = SimpleNamespace(task=SimpleNamespace(done_signal_file=str(done_signal_file)))

    with pytest.raises(RuntimeError, match="collect-data requested but data collection"):
        policy._after_run_loop()

    assert done_signal_file.exists(), "teardown (done-signal write) must still run before the raise"


def test_after_run_loop_does_not_raise_for_a_valid_collection(tmp_path: Path) -> None:
    """A run with >=1 saved episode does not raise from _after_run_loop."""
    collector = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        num_envs=1,
        obs_dict={"base_obs": ["base_lin_vel"]},
    )
    collector.collect_step(
        obs_dict={"base_obs": np.array([[1.0]], dtype=np.float32)},
        actions=np.array([[0.1, 0.2]], dtype=np.float32),
        dones=np.array([False]),
    )

    policy = object.__new__(LocomotionPolicy_Deploy)
    policy.data_collector = collector
    policy.logger = logging.getLogger("test_dual_mode")
    policy.state_logger = None
    policy.config = SimpleNamespace(task=SimpleNamespace(done_signal_file=None))

    policy._after_run_loop()  # must not raise


def test_deploy_policy_action_uses_snapshot_when_manual_toggle_happens_mid_cycle() -> None:
    sent_commands: list[np.ndarray] = []

    class _RaceInterface:
        def get_low_state(self):
            return np.zeros((1, 7 + 2), dtype=np.float32)

        def send_low_command(self, cmd_q, *_args, **_kwargs):
            sent_commands.append(np.array(cmd_q, copy=True))

    class _RacePolicy:
        policy_action = LocomotionPolicy_Deploy.policy_action

        def __init__(self):
            self.use_policy_action = True
            self.get_ready_state = False
            self.latency_tracker = _DummyLatencyTracker()
            self.interface = _RaceInterface()
            self.num_dofs = 2
            self.default_dof_angles = np.array([[0.1, -0.2]], dtype=np.float32)
            self.upper_body_controller = False
            self.cmd_q = np.zeros((2,), dtype=np.float32)
            self.cmd_dq = np.zeros((2,), dtype=np.float32)
            self.cmd_tau = np.zeros((2,), dtype=np.float32)
            self.data_collector = None
            self._session_recording_enabled = False
            self._current_step_recon = None
            self._current_step_planner_first_step = None
            self._current_step_idm_inverse = None
            self._pending_dynamics_prediction = None

        def prepare_obs_for_rl(self, _robot_state_data):
            self.use_policy_action = False
            return {"actor_obs": np.zeros((1, 4), dtype=np.float32)}

        def _log_pending_dynamics_targets(self, _obs_for_rl):
            return

        def rl_inference(self, _robot_state_data, _obs_for_rl):
            scaled = np.array([[0.3, 0.4]], dtype=np.float32)
            raw = np.array([[0.3, 0.4]], dtype=np.float32)
            return scaled, raw

        def _maybe_log_inference_state(self, _robot_state_data, _q_target, _obs_for_rl=None):
            return

        def _get_manual_command(self, _robot_state_data):
            return None

        def _apply_residual_upper_body(self, q_target, _scaled_action):
            return q_target

    policy = _RacePolicy()
    policy.policy_action()

    assert policy.use_policy_action is False
    assert len(sent_commands) == 1
    np.testing.assert_allclose(sent_commands[0], [0.4, 0.2])


# ---------------------------------------------------------------------------
# The predefined-command-sequence feature (the two `TaskConfig` fields named in
# `_REMOVED_PREDEFINED_ATTRS` below and the JSON sequence files they read) is not part of
# this release. These tests assert that none of the four policy classes `run_policy.py`
# can select defines either name, and that `dual_mode.py` carries no guard on them.
# ---------------------------------------------------------------------------

_REMOVED_PREDEFINED_ATTRS = ("use_predefined_commands", "_prepare_predefined_command_session")


def test_no_selectable_policy_class_defines_the_removed_predefined_command_hooks() -> None:
    selectable = (LocomotionPolicy, LocomotionPolicy_Deploy, LocomotionPolicy_FADA, WholeBodyTrackingPolicy)
    for policy_class in selectable:
        for attr in _REMOVED_PREDEFINED_ATTRS:
            assert not hasattr(policy_class, attr), f"{policy_class.__name__} still defines {attr}"
            assert not any(attr in vars(base) for base in policy_class.__mro__), (
                f"{policy_class.__name__} inherits {attr}"
            )


def test_dual_mode_carries_no_guard_for_the_removed_feature() -> None:
    text = Path(inspect.getsourcefile(DualModePolicy)).read_text()
    for attr in _REMOVED_PREDEFINED_ATTRS:
        assert attr not in text, f"dual_mode.py still guards on {attr}"
