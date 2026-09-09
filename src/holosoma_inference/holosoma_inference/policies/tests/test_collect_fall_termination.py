"""End-to-end guard on fall accounting in collected target-domain data.

A MuJoCo/real rollout has no environment to terminate it: `_should_stop` only checks
`max_steps` / `max_eval_time`, so a robot that falls after the gantry release keeps being
recorded to the same episode.

By default `dones` stays all-False and `load_trajectories_from_h5` (which truncates a
trajectory at its first `done`) yields the whole episode, post-fall tail included. The fall
is detected and reported either way (`collection_fall_report()` /
`_log_collection_fall_report()`); `--task.collect-mark-fall-terminal true` truncates.

These tests drive the real `LocomotionPolicy_Deploy.policy_action` into a real
`DataCollector`, write a real HDF5 file, and load it back through the real FADA finetune
loader, so the assertions are on the training windows the finetune would actually see.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from holosoma.fada.common.lora_utils import load_trajectories_from_h5
from holosoma.utils.data_collector import DataCollector
from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.policies.locomotion import (
    LocomotionPolicy_Deploy,
    _step6_fall_projected_gravity_z_max,
)

_NUM_DOFS = 2
_ACT_DIM = 2
# base_pos(3) + quat(4) + dof_pos(N) + lin_vel(3) + ang_vel(3) + dof_vel(N), plus the
# optional trailing pre-computed projected_gravity(3) the sim interface supplies.
_STATE_LEN = 7 + _NUM_DOFS + 6 + _NUM_DOFS


class _NoopMeasure:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _LatencyTracker:
    def measure(self, _name: str):
        return _NoopMeasure()


class _ScriptedInterface:
    """Feeds a scripted projected-gravity z, so "the robot fell at step k" is exact."""

    def __init__(self, gravity_z_by_step: list[float]):
        self._gravity_z_by_step = gravity_z_by_step
        self.step = 0

    def get_low_state(self):
        gravity_z = self._gravity_z_by_step[min(self.step, len(self._gravity_z_by_step) - 1)]
        self.step += 1
        state = np.zeros((1, _STATE_LEN + 3), dtype=np.float32)
        # Joint positions are the step index so post-fall payload is distinguishable.
        state[0, 7 : 7 + _NUM_DOFS] = float(self.step)
        state[0, _STATE_LEN : _STATE_LEN + 3] = (0.0, 0.0, gravity_z)
        return state

    def send_low_command(self, *_args, **_kwargs):
        return


class _CollectingPolicy(LocomotionPolicy_Deploy):
    """Real `LocomotionPolicy_Deploy` with only the SDK/IO surface replaced.

    Subclassing (rather than copying the bound methods onto a bare object) keeps
    `policy_action`, `_collection_done_flag`, `get_projected_gravity_z` and
    `get_current_obs_buffer_dict` -- including their zero-arg `super()` calls -- exactly
    the production ones. `BasePolicy.__init__` is skipped because it wants a real robot.
    """

    def __init__(
        self,
        collector: DataCollector,
        interface: _ScriptedInterface,
        *,
        mark_terminal: bool = False,
    ):
        self.use_policy_action = True
        self.get_ready_state = False
        self.latency_tracker = _LatencyTracker()
        self.interface = interface
        self.num_dofs = _NUM_DOFS
        self.default_dof_angles = np.zeros((1, _NUM_DOFS), dtype=np.float32)
        self.upper_body_controller = False
        self.cmd_q = np.zeros((_NUM_DOFS,), dtype=np.float32)
        self.cmd_dq = np.zeros((_NUM_DOFS,), dtype=np.float32)
        self.cmd_tau = np.zeros((_NUM_DOFS,), dtype=np.float32)
        self.data_collector = collector
        self._session_recording_enabled = True
        self._current_step_recon = None
        self._current_step_planner_first_step = None
        self._current_step_idm_inverse = None
        self._pending_dynamics_prediction = None
        self.logger = logging.getLogger("test_collect_fall_termination")
        # Attributes `LocomotionPolicy_Deploy.get_current_obs_buffer_dict` reads on its way
        # to projected_gravity; supplying them keeps the production method in the loop.
        self.last_policy_action = np.zeros((1, _ACT_DIM), dtype=np.float32)
        self.lin_vel_command = np.zeros((1, 2), dtype=np.float32)
        self.ang_vel_command = np.zeros((1, 1), dtype=np.float32)
        self.stand_command = np.zeros((1, 1), dtype=np.float32)
        self.obs_dict: dict[str, list[str]] = {"actor_obs": []}
        self.obs_buf_dict: dict[str, np.ndarray] = {}
        self.collect_mark_fall_terminal = mark_terminal
        self.collect_fall_projected_gravity_z_max = -0.7
        # Mirrors production `__init__`: the second, report-only criterion (step 6's), so
        # these tests exercise the same cross-check a real collection run does.
        self._step6_fall_projected_gravity_z_max = _step6_fall_projected_gravity_z_max()
        self._collection_step6_fall_step = None
        self._collection_fall_signal = None
        self._collection_fall_latched = False
        self._collection_fall_detectable = True
        self._collection_steps = 0
        self._collection_post_fall_steps = 0
        self._collection_fall_step = None
        self.config = SimpleNamespace(
            task=SimpleNamespace(
                debug=SimpleNamespace(force_upright_imu=False, force_zero_angular_velocity=False),
            )
        )

    def prepare_obs_for_rl(self, robot_state_data):
        self.obs_buf_dict = {
            "dynamics_obs": np.asarray(robot_state_data[:, 7 : 7 + _NUM_DOFS], dtype=np.float32).copy(),
            "current_command": np.zeros((1, 1), dtype=np.float32),
        }
        return {"actor_obs": np.zeros((1, 4), dtype=np.float32)}

    def _log_pending_dynamics_targets(self, _obs_for_rl):
        return

    def rl_inference(self, _robot_state_data, _obs_for_rl):
        action = np.full((1, _ACT_DIM), float(self.interface.step), dtype=np.float32)
        return action, action.copy()

    def _maybe_log_inference_state(self, *_args, **_kwargs):
        return

    def _get_manual_command(self, _robot_state_data):
        return None

    def _apply_residual_upper_body(self, q_target, _scaled_action):
        return q_target


def _make_collector(tmp_path: Path) -> DataCollector:
    return DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        compress=False,
        num_envs=1,
        obs_dict={"dynamics_obs": ["dof_pos"], "current_command": ["command_lin_vel"]},
    )


def _run_collection(
    tmp_path: Path, *, mark_terminal: bool, fall_at: int, total_steps: int
) -> tuple[Path, _CollectingPolicy]:
    """Record `total_steps` steps, the robot falling at index `fall_at`."""
    gravity = [-1.0] * fall_at + [-0.2] * (total_steps - fall_at)
    collector = _make_collector(tmp_path)
    collector.start_episode()
    policy = _CollectingPolicy(collector, _ScriptedInterface(gravity), mark_terminal=mark_terminal)
    for _ in range(total_steps):
        policy.policy_action()
    collector.close()
    return tmp_path / "dataset.h5", policy


def _load(h5_path: Path, *, pred_horizon_k: int = 2):
    return load_trajectories_from_h5(
        [str(h5_path)],
        obs_key="dynamics_obs",
        command_key="current_command",
        obs_dim=_NUM_DOFS,
        act_dim=_ACT_DIM,
        cmd_dim=1,
        raw_obs_preprocess=None,
        pred_horizon_k=pred_horizon_k,
    )


def _dones(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        group = handle["episodes"]["episode_0"]
        return np.asarray(group["dones"]).reshape(-1)


@pytest.mark.parametrize("fall_at", [4, 9])
def test_default_keeps_the_released_behavior_but_reports_the_contamination(tmp_path: Path, fall_at: int) -> None:
    """Default run: every collected step stays non-terminal.

    The finetune loader yields the whole episode, and the run's report states how many of
    those steps were recorded after the robot fell.
    """
    total_steps = 20
    h5_path, policy = _run_collection(tmp_path, mark_terminal=False, fall_at=fall_at, total_steps=total_steps)

    assert not _dones(h5_path).any(), "the default must not mark anything terminal"

    trajectories, _stats = _load(h5_path)
    assert len(trajectories) == 1
    assert trajectories[0].obs.shape[0] == total_steps
    # dof_pos was written as the 1-based step index, so the tail really is post-fall data.
    assert float(trajectories[0].obs.max()) == float(total_steps)

    report = policy.collection_fall_report()
    assert report["fall_detection_available"] is True
    assert report["fall_step"] == fall_at
    assert report["collected_steps"] == total_steps
    assert report["post_fall_steps"] == total_steps - fall_at
    assert report["marked_terminal"] is False


@pytest.mark.parametrize("fall_at", [4, 9])
def test_opt_in_flag_truncates_the_episode_at_the_fall(tmp_path: Path, fall_at: int) -> None:
    """`--task.collect-mark-fall-terminal true`: post-fall steps never reach the finetune."""
    total_steps = 20
    h5_path, policy = _run_collection(tmp_path, mark_terminal=True, fall_at=fall_at, total_steps=total_steps)

    dones = _dones(h5_path)
    assert dones.shape[0] == total_steps
    assert not dones[:fall_at].any(), "steps before the fall must stay non-terminal"
    assert dones[fall_at:].all(), "the fall step and every later one must be terminal"

    trajectories, _stats = _load(h5_path)
    assert len(trajectories) == 1
    # The loader truncates at the first done (exclusive), so exactly the pre-fall steps
    # survive -- this is the data the IDM finetune actually trains on.
    assert trajectories[0].obs.shape[0] == fall_at
    assert float(trajectories[0].obs.max()) <= float(fall_at)
    assert policy.collection_fall_report()["marked_terminal"] is True


def test_a_clean_rollout_is_identical_with_and_without_the_opt_in(tmp_path: Path) -> None:
    """When nothing falls, `--task.collect-mark-fall-terminal` changes no loaded window."""
    baseline_dir = tmp_path / "baseline"
    optin_dir = tmp_path / "optin"
    baseline_dir.mkdir()
    optin_dir.mkdir()
    baseline, _ = _run_collection(baseline_dir, mark_terminal=False, fall_at=12, total_steps=12)
    optin, _ = _run_collection(optin_dir, mark_terminal=True, fall_at=12, total_steps=12)

    np.testing.assert_array_equal(_dones(baseline), _dones(optin))
    baseline_trajectories, _ = _load(baseline)
    optin_trajectories, _ = _load(optin)
    assert len(baseline_trajectories) == len(optin_trajectories) == 1
    np.testing.assert_array_equal(baseline_trajectories[0].obs, optin_trajectories[0].obs)
    np.testing.assert_array_equal(baseline_trajectories[0].actions, optin_trajectories[0].actions)


def test_fall_flag_is_sticky_across_a_recovery(tmp_path: Path) -> None:
    """A robot that returns upright for a few steps does not un-terminate the episode.

    `DataCollector` ORs each `dones` row into a sticky mask and zeroes every later step's
    payload, so the flag must stay True once raised. The post-fall count latches the same
    way.
    """
    gravity = [-1.0] * 3 + [-0.2] * 2 + [-1.0] * 5
    collector = _make_collector(tmp_path)
    collector.start_episode()
    policy = _CollectingPolicy(collector, _ScriptedInterface(gravity), mark_terminal=True)
    for _ in range(len(gravity)):
        policy.policy_action()
    collector.close()

    np.testing.assert_array_equal(_dones(tmp_path / "dataset.h5"), np.array([False] * 3 + [True] * 7))
    assert policy.collection_fall_report()["post_fall_steps"] == 7


def test_force_upright_imu_makes_the_report_say_unavailable_rather_than_zero(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`--task.debug.force-upright-imu` pins projected gravity upright, so fall detection
    cannot fire; the report says "unavailable" once rather than "0 post-fall steps"."""
    collector = _make_collector(tmp_path)
    collector.start_episode()
    policy = _CollectingPolicy(collector, _ScriptedInterface([-0.2] * 5), mark_terminal=True)
    policy.config.task.debug.force_upright_imu = True
    with caplog.at_level(logging.WARNING, logger="test_collect_fall_termination"):
        for _ in range(5):
            policy.policy_action()
    collector.close()

    assert not _dones(tmp_path / "dataset.h5").any()
    report = policy.collection_fall_report()
    assert report["fall_detection_available"] is False
    assert report["post_fall_steps"] == 0
    warnings = [record.message for record in caplog.records if "force-upright-imu" in record.message]
    assert len(warnings) == 1, "the blinding condition must be reported exactly once, not per step"


def test_collection_fall_report_is_logged_and_written_next_to_the_dataset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The count must reach the run's own output, not only an in-memory dict."""
    total_steps = 10
    fall_at = 6
    gravity = [-1.0] * fall_at + [-0.2] * (total_steps - fall_at)
    collector = _make_collector(tmp_path)
    collector.start_episode()
    policy = _CollectingPolicy(collector, _ScriptedInterface(gravity), mark_terminal=False)
    for _ in range(total_steps):
        policy.policy_action()
    collector.close()

    with caplog.at_level(logging.INFO, logger="test_collect_fall_termination"):
        policy._log_collection_fall_report()

    messages = "\n".join(record.message for record in caplog.records)
    assert "4/10" in messages, messages
    assert "AFTER the fall" in messages, messages
    assert "collect-mark-fall-terminal" in messages, messages

    written = json.loads((tmp_path / "collection_fall_report.json").read_text())
    assert written["post_fall_steps"] == 4
    assert written["fall_step"] == fall_at
    assert written["marked_terminal"] is False


def test_marking_is_opt_in_and_the_threshold_matches_the_dual_mode_guard() -> None:
    """`collect_mark_fall_terminal` defaults to False, and the collector's fall threshold
    equals the dual-mode safety guard's."""
    defaults = TaskConfig(model_path="unused.onnx")
    assert defaults.collect_mark_fall_terminal is False, "truncation must never be the default"
    assert defaults.collect_fall_projected_gravity_z_max == defaults.dual_mode_projected_gravity_z_min


# ---------------------------------------------------------------------------------------
# The collection report names step 6's criterion as well as its own
# ---------------------------------------------------------------------------------------
#
# The two thresholds are 45.573 deg here and 60.000 deg in step 6, so a rollout tilted
# between them is FALLEN in this report and UPRIGHT in step 6's figure and metrics. These
# tests pin that the collection report says so in both channels: the JSON written next to
# the dataset, and the console.
#
# `-0.6` is inside the band: 53.13 deg, past this collector's -0.7 and short of step 6's
# -0.5. `-0.2` (78.46 deg) is past both.


def test_a_rollout_in_the_disputed_band_reports_both_verdicts(tmp_path: Path) -> None:
    gravity = [-1.0] * 5 + [-0.6] * 15
    collector = _make_collector(tmp_path)
    collector.start_episode()
    policy = _CollectingPolicy(collector, _ScriptedInterface(gravity), mark_terminal=False)
    for _ in range(len(gravity)):
        policy.policy_action()
    collector.close()

    report = policy.collection_fall_report()
    # The verdict the dataset is based on.
    assert report["fall_step"] == 5
    assert report["post_fall_steps"] == 15
    # ... and the other criterion is reported alongside it.
    assert report["collector_verdict"] == "fell"
    assert report["step6_verdict"] == "no_fall"
    assert report["fall_criteria_disagree"] is True
    assert report["collector_fall_threshold_projected_gravity_z"] == -0.7
    assert report["step6_fall_threshold_projected_gravity_z"] == -0.5
    assert report["collector_fall_tilt_deg"] == pytest.approx(45.573, abs=0.01)
    assert report["step6_fall_tilt_deg"] == pytest.approx(60.0, abs=1e-9)


def test_a_rollout_past_both_thresholds_reports_agreement(tmp_path: Path) -> None:
    _h5, policy = _run_collection(tmp_path, mark_terminal=False, fall_at=5, total_steps=20)
    report = policy.collection_fall_report()
    assert report["collector_verdict"] == "fell"
    assert report["step6_verdict"] == "fell"
    assert report["fall_criteria_disagree"] is False


def test_a_clean_rollout_reports_agreement_on_no_fall(tmp_path: Path) -> None:
    collector = _make_collector(tmp_path)
    collector.start_episode()
    policy = _CollectingPolicy(collector, _ScriptedInterface([-1.0] * 12), mark_terminal=False)
    for _ in range(12):
        policy.policy_action()
    collector.close()

    report = policy.collection_fall_report()
    assert report["collector_verdict"] == "no_fall"
    assert report["step6_verdict"] == "no_fall"
    assert report["fall_criteria_disagree"] is False


def test_the_console_names_both_thresholds_and_shouts_on_disagreement(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gravity = [-1.0] * 5 + [-0.6] * 15
    collector = _make_collector(tmp_path)
    collector.start_episode()
    policy = _CollectingPolicy(collector, _ScriptedInterface(gravity), mark_terminal=False)
    with caplog.at_level(logging.WARNING, logger="test_collect_fall_termination"):
        for _ in range(len(gravity)):
            policy.policy_action()
        policy._log_collection_fall_report()
    collector.close()

    text = " ".join(record.getMessage() for record in caplog.records)
    assert "FALL CRITERIA DISAGREE" in text, text
    assert "-0.700" in text and "45.573 deg" in text, text
    assert "-0.500" in text and "60.000 deg" in text, text
    assert "would NOT call this rollout fallen" in text, text

    written = json.loads((tmp_path / "collection_fall_report.json").read_text())
    assert written["fall_criteria_disagree"] is True
    assert written["step6_verdict"] == "no_fall"


def test_both_verdicts_are_printed_even_when_they_agree(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Both verdicts are printed on a run where the two criteria agree."""
    collector = _make_collector(tmp_path)
    collector.start_episode()
    policy = _CollectingPolicy(collector, _ScriptedInterface([-1.0] * 12), mark_terminal=False)
    with caplog.at_level(logging.WARNING, logger="test_collect_fall_termination"):
        for _ in range(12):
            policy.policy_action()
        policy._log_collection_fall_report()
    collector.close()

    text = " ".join(record.getMessage() for record in caplog.records)
    assert text.count("FALL CRITERIA:") == 2, text
    assert "FALL CRITERIA DISAGREE" not in text, text


def test_the_cross_check_never_touches_the_collected_data(tmp_path: Path) -> None:
    """The step-6 criterion is reporting only: `dones` and the loaded windows are unchanged."""
    gravity = [-1.0] * 5 + [-0.6] * 15
    with_dir, without_dir = tmp_path / "with", tmp_path / "without"
    with_dir.mkdir()
    without_dir.mkdir()

    def _collect(out_dir: Path, cross_check: bool) -> Path:
        collector = _make_collector(out_dir)
        collector.start_episode()
        policy = _CollectingPolicy(collector, _ScriptedInterface(gravity), mark_terminal=False)
        if not cross_check:
            policy._step6_fall_projected_gravity_z_max = None
        for _ in range(len(gravity)):
            policy.policy_action()
        collector.close()
        return out_dir / "dataset.h5"

    on = _collect(with_dir, True)
    off = _collect(without_dir, False)
    np.testing.assert_array_equal(_dones(on), _dones(off))
    on_traj, _ = _load(on)
    off_traj, _ = _load(off)
    np.testing.assert_array_equal(on_traj[0].obs, off_traj[0].obs)
    np.testing.assert_array_equal(on_traj[0].actions, off_traj[0].actions)
