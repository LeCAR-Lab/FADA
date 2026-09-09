"""Guards on the step-6 metric window.

Collection/plotting starts at policy start and the virtual gantry is released afterwards, so
the first seconds of every run are the robot hanging, then descending, then landing.
`plot_trim_head` / `plot_trim_tail` default to 0 all the way down the chain (`TaskConfig` ->
`task.locomotion` -> the `*-loco-fada` presets -> `BasicVelStateProcessor` ->
`PLOT_TRIM_STEPS`), so those steps are inside every reported RMSE and tracking return.

These tests pin that the defaults are 0 and that the window used is stated on the console, on
the velocity figure, and in the metrics dict, including the case where a requested trim did
not fit.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest
from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.config.config_values import inference as inference_presets
from holosoma_inference.utils.compute_mocap_velocity_metrics import (
    compute_velocity_errors_from_mocap_unified,
)
from holosoma_inference.utils.plot_mocap_raw import (
    PLOT_TRIM_STEPS,
    _plot_velocity_tracking,
    _trim_initial_final_steps,
    describe_metric_window,
    format_metric_window,
    metric_window_log_lines,
    save_mocap_unified_log,
)
from matplotlib.figure import Figure

# Headless: these tests render real figures.
mpl.use("Agg")


def test_released_defaults_are_still_zero_trim() -> None:
    """The behavior contract: nothing is trimmed unless the caller asks."""
    assert PLOT_TRIM_STEPS == 0
    defaults = TaskConfig(model_path="unused.onnx")
    assert defaults.plot_trim_head == 0
    assert defaults.plot_trim_tail == 0
    for preset_name in ("t1_23dof_loco_fada", "g1_29dof_loco_fada"):
        preset = getattr(inference_presets, preset_name)
        assert preset.task.plot_trim_head == 0, preset_name
        assert preset.task.plot_trim_tail == 0, preset_name


def test_window_reports_the_whole_run_when_nothing_is_trimmed() -> None:
    window = describe_metric_window(3000, 0, 0, rate_hz=50.0)
    assert window["steps_total"] == 3000
    assert window["steps_used"] == 3000
    assert window["trim_head_applied"] == 0
    assert window["trim_requested_but_not_applied"] is False

    stamp = format_metric_window(window)
    assert "ALL 3000 steps" in stamp
    assert "INCLUDED" in stamp

    lines = " ".join(metric_window_log_lines(window))
    assert "all 3000 recorded steps are included" in lines
    assert "--task.plot-trim-head" in lines, "the console must name the flag that fixes it"
    assert "50 steps" in lines, "the console must translate seconds into steps at the run's rate"


def test_window_reports_the_applied_slice_when_trimmed() -> None:
    window = describe_metric_window(3000, 1000, 25, rate_hz=50.0)
    assert window["trim_head_applied"] == 1000
    assert window["trim_tail_applied"] == 25
    assert window["steps_used"] == 1975
    assert "steps [1000, 2975) of 3000" in format_metric_window(window)
    assert "1000 head and 25 tail steps excluded" in " ".join(metric_window_log_lines(window))


def test_a_requested_trim_that_does_not_fit_is_reported_not_silently_dropped() -> None:
    """`_trim_initial_final_steps` no-ops when `n <= head + tail`, and the window says so."""
    n = 800
    arrays = (np.arange(n, dtype=np.float64),)
    (untouched,) = _trim_initial_final_steps(n, 1000, *arrays, trim_tail=0)
    assert untouched.shape[0] == n, "precondition: the trim really is silently skipped"

    window = describe_metric_window(n, 1000, 0, rate_hz=50.0)
    assert window["trim_requested_but_not_applied"] is True
    assert window["trim_head_applied"] == 0
    assert window["steps_used"] == n
    assert "NOT applied" in format_metric_window(window)
    assert "was NOT applied" in " ".join(metric_window_log_lines(window))


@pytest.mark.parametrize(
    ("total", "head", "tail"),
    [(100, 0, 0), (100, 10, 5), (100, 200, 0), (0, 0, 0), (10, -5, -5)],
)
def test_window_matches_what_the_trim_primitive_actually_does(total: int, head: int, tail: int) -> None:
    """The reported window must equal the slice `_trim_initial_final_steps` really produces."""
    window = describe_metric_window(total, head, tail)
    arrays = (np.arange(max(total, 0), dtype=np.float64),)
    (trimmed,) = _trim_initial_final_steps(max(total, 0), head, *arrays, trim_tail=tail)
    assert trimmed.shape[0] == window["steps_used"]
    if trimmed.shape[0] > 0:
        assert float(trimmed[0]) == float(window["trim_head_applied"])


def test_velocity_figure_carries_the_window_stamp(tmp_path) -> None:
    """The rendered velocity PNG carries the metric-window stamp."""
    n = 50
    t = np.linspace(0.0, 1.0, n)
    zeros = np.zeros(n)
    window = describe_metric_window(n, 0, 0, rate_hz=50.0)
    save_path = tmp_path / "mocap_velocity.png"
    _plot_velocity_tracking(
        t,
        zeros,
        zeros,
        zeros,
        None,
        None,
        None,
        None,
        save_path=str(save_path),
        show=False,
        mocap_ts=t,
        metric_window=window,
    )
    assert save_path.exists()

    # Read the text actually rendered onto the figure, not the helper's return value.
    captured: list[str] = []
    original_text = Figure.text

    def _recording_text(self, x, y, s, *args, **kwargs):
        captured.append(str(s))
        return original_text(self, x, y, s, *args, **kwargs)

    Figure.text = _recording_text
    try:
        _plot_velocity_tracking(
            t, zeros, zeros, zeros, None, None, None, None,
            save_path=str(tmp_path / "again.png"), show=False, mocap_ts=t, metric_window=window,
        )
    finally:
        Figure.text = original_text

    stamps = [text for text in captured if "Metric window" in text]
    assert stamps, f"no metric-window stamp on the figure; texts were {captured}"
    assert "ALL 50 steps" in stamps[0]
    assert "INCLUDED" in stamps[0]


def test_unified_metrics_carry_the_window(tmp_path) -> None:
    """`compute_velocity_errors_from_mocap_unified` records the window it aggregated over."""
    n = 40
    ts = np.linspace(0.0, n / 50.0, n)
    pos = np.zeros((n, 3))
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    npz_path = tmp_path / "mocap_unified.npz"
    save_mocap_unified_log(
        ts, pos, quat, None, None, None, None, None, None, None, path=str(npz_path)
    )

    metrics = compute_velocity_errors_from_mocap_unified(npz_path, mocap_plot_hz=50.0)
    window = metrics["metric_window"]
    assert window["steps_total"] == n
    assert window["trim_head_applied"] == 0
    assert window["steps_used"] == n


def _render_velocity_figure_texts(tmp_path, *, tracking_summary, n: int) -> list[str]:
    """Render the velocity figure and return every string it drew via `Figure.text`."""
    t = np.linspace(0.0, n / 50.0, n)
    zeros = np.zeros(n)
    captured: list[str] = []
    original_text = Figure.text

    def _recording_text(self, x, y, s, *args, **kwargs):
        captured.append(str(s))
        return original_text(self, x, y, s, *args, **kwargs)

    Figure.text = _recording_text
    try:
        _plot_velocity_tracking(
            t, zeros, zeros, zeros, None, None, None, None,
            save_path=str(tmp_path / "fall_window.png"), show=False, mocap_ts=t,
            tracking_summary=tracking_summary,
            metric_window=describe_metric_window(n, 0, 0, rate_hz=50.0),
        )
    finally:
        Figure.text = original_text
    return captured


def _fall_tracking_summary(n: int, fall_step: int) -> dict:
    """A rollout that tracks perfectly until `fall_step`, then is wildly wrong.

    The pre-fall and full-sequence windows therefore give very different RMSEs. The numeric
    parity contract for this shape lives in `test_plot_rmse_parity.py`.
    """
    actual = np.zeros((n, 2), dtype=np.float64)
    command = np.zeros((n, 2), dtype=np.float64)
    command[fall_step:, 0] = 4.0  # post-fall error, excluded from every reported metric
    return {
        "available": True,
        "lin_vel_tracking_return": 1.0,
        "ang_vel_tracking_return": 1.0,
        "tracking_total_return": 2.0,
        "plot_payload": {
            "step_axis": np.arange(n, dtype=np.float64),
            "lin_actual_xy": actual,
            "ang_actual_z": np.zeros(n),
            "lin_command_xy": command,
            "ang_command_z": np.zeros(n),
            "lin_frame": "body_xy",
            "step_hz": 50.0,
            "fall_step": fall_step,
        },
    }


def test_the_figure_states_the_window_split_instead_of_closing_it(tmp_path) -> None:
    """The figure states which of its three windows each number covers.

    The exported `lin/ang_vel_tracking_err` stop at the fall, the tracking return zeroes
    reward after it, and the plotted RMSE covers the whole post-trim series. The figure
    labels the fall marker and states the plotted RMSE's window in words.

    With a post-fall command error of 4 m/s on 60% of the run, the full-sequence RMSE is
    ~3.098; a pre-fall window would give 0.
    """
    captured = _render_velocity_figure_texts(tmp_path, tracking_summary=_fall_tracking_summary(50, 20), n=50)
    rmse_lines = [text for text in captured if "Linear Vel RMSE" in text]
    assert rmse_lines, f"no RMSE stamp on the figure; texts were {captured}"
    assert "3.09" in rmse_lines[0], rmse_lines[0]
    assert "fall at step 20 INCLUDED" in rmse_lines[0], rmse_lines[0]
    assert "ALL 50 plotted steps" in rmse_lines[0], rmse_lines[0]


def test_a_rollout_without_a_fall_carries_no_fall_note(tmp_path) -> None:
    """No fall detected: same number, and no fall wording added to the stamp."""
    summary = _fall_tracking_summary(50, 20)
    summary["plot_payload"]["fall_step"] = None
    captured = _render_velocity_figure_texts(tmp_path, tracking_summary=summary, n=50)
    rmse_lines = [text for text in captured if "Linear Vel RMSE" in text]
    assert rmse_lines
    assert "fall" not in rmse_lines[0]
    # 30 of 50 steps carry a 4 m/s error: sqrt(30/50 * 16) ~= 3.098 -- identical to the
    # falling fixture above.
    assert "3.09" in rmse_lines[0], rmse_lines[0]
