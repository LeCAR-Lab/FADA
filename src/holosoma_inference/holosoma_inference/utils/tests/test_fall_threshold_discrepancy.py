"""Pins the size and direction of the two fall criteria's disagreement.

Two code paths decide "the robot has fallen", with different thresholds:

* the collector and the dual-mode safety guard -- ``projected_gravity_z > -0.7``
  (``TaskConfig.collect_fall_projected_gravity_z_max`` / ``dual_mode_projected_gravity_z_min``),
  which is 45.57 degrees of body tilt, watched from the first step;
* step 6's pre/post-finetune comparison -- ``R[2,2] < 0.5``
  (``plot_mocap_raw.STEP6_FALL_R22_MAX``), which is 60 degrees, and only after the first 5%
  of the run.

They measure the same physical quantity: ``projected_gravity_z == -R[2,2]`` exactly. So a
rollout whose tilt lands between those two angles is fallen to one path and upright to the
other, on the same data, at the same instant.

What ``fall_step`` controls
---------------------------
Step 6's ``fall_step`` truncates the exported ``lin_vel_tracking_err`` /
``ang_vel_tracking_err`` and zeroes the tracking return after the fall, so the step-6
threshold selects which steps those numbers cover.

These tests do not assert that the two criteria agree; they assert the size and direction of
the disagreement, and that both verdicts are reported. Unifying the thresholds is what fails
this file.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.utils.math.quat import quat_rotate_inverse
from holosoma_inference.utils.plot_mocap_raw import (
    STEP6_FALL_R22_MAX,
    STEP6_FALL_WARMUP_FRACTION,
    _compute_velocity_tracking_summary,
    fall_criteria_log_lines,
)


def _tilted_quat_wxyz(tilt_deg: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for a pure roll of *tilt_deg* about the body x axis."""
    half = math.radians(tilt_deg) / 2.0
    return np.array([[math.cos(half), math.sin(half), 0.0, 0.0]], dtype=np.float64)


def _r22_from_xyzw(quat_xyzw: np.ndarray) -> float:
    """`plot_mocap_raw`'s own expression, which reads the array as xyzw."""
    return float(1.0 - 2.0 * (quat_xyzw[0, 0] ** 2 + quat_xyzw[0, 1] ** 2))


@pytest.mark.parametrize("tilt_deg", [0.0, 20.0, 45.57, 46.0, 55.0, 60.0, 75.0, 90.0])
def test_the_two_signals_are_the_same_number_with_opposite_signs(tilt_deg: float) -> None:
    """`projected_gravity_z == -R[2,2]`, so the thresholds are directly comparable."""
    quat_wxyz = _tilted_quat_wxyz(tilt_deg)
    projected_gravity_z = float(quat_rotate_inverse(quat_wxyz, np.array([[0.0, 0.0, -1.0]]))[0, 2])
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    assert projected_gravity_z == pytest.approx(-_r22_from_xyzw(quat_xyzw), abs=1e-12)
    assert projected_gravity_z == pytest.approx(-math.cos(math.radians(tilt_deg)), abs=1e-12)


def test_the_two_thresholds_are_the_documented_values() -> None:
    """Pinned so a silent edit to either side shows up here."""
    assert TaskConfig(model_path="unused.onnx").collect_fall_projected_gravity_z_max == -0.7
    assert TaskConfig(model_path="unused.onnx").dual_mode_projected_gravity_z_min == -0.7
    assert STEP6_FALL_R22_MAX == 0.5
    assert STEP6_FALL_WARMUP_FRACTION == 0.05


def test_the_disagreement_band_is_45_57_to_60_degrees() -> None:
    """The exact band where the two paths give opposite answers."""
    collector_deg = math.degrees(math.acos(0.7))
    step6_deg = math.degrees(math.acos(STEP6_FALL_R22_MAX))
    assert collector_deg == pytest.approx(45.573, abs=0.01)
    assert step6_deg == pytest.approx(60.0, abs=1e-9)
    assert collector_deg < step6_deg, "the collector is the stricter of the two"

    # A rollout in the band: fallen to the collector, upright to step 6.
    mid_deg = 52.0
    projected_gravity_z = -math.cos(math.radians(mid_deg))
    r22 = math.cos(math.radians(mid_deg))
    assert projected_gravity_z > -0.7, "collector calls this fallen"
    assert not (r22 < STEP6_FALL_R22_MAX), "step 6 calls the same pose upright"


def test_step_6_would_move_a_reported_number_if_unified() -> None:
    """Unifying the thresholds would move a step-6 number.

    A rollout that tilts to 52 degrees halfway through has no `fall_step` under the shipped
    criterion; under the collector's -0.7 it acquires one, and `fall_step` truncates the
    exported tracking error and zeroes the tracking return.
    """
    n = 200
    quats_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    half = math.radians(52.0) / 2.0
    quats_xyzw[n // 2 :, 0] = math.sin(half)
    quats_xyzw[n // 2 :, 3] = math.cos(half)

    r22 = 1.0 - 2.0 * (quats_xyzw[:, 0] ** 2 + quats_xyzw[:, 1] ** 2)
    skip = max(1, int(len(r22) * STEP6_FALL_WARMUP_FRACTION))

    shipped = np.where(r22[skip:] < STEP6_FALL_R22_MAX)[0]
    unified = np.where(r22[skip:] < 0.7)[0]
    assert shipped.size == 0, "shipped criterion: no fall, whole run counted"
    assert unified.size > 0, "unified criterion: a fall, and the run gets truncated"
    assert int(skip + unified[0]) == n // 2


def test_step_6_detection_is_reachable_and_still_uses_the_shipped_constant() -> None:
    """The detector reads the module-level constants, not literals.

    Drives the real summary helper: with no command series it returns None before any fall
    logic runs, so the constants are checked against the function's source instead.
    """
    assert _compute_velocity_tracking_summary(
        np.zeros(4), np.zeros((4, 3)), np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (4, 1)),
        None, None, None, None, checkpoint_path=None,
    ) is None

    source = inspect.getsource(_compute_velocity_tracking_summary)
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert "np.where(r22[skip:] < STEP6_FALL_R22_MAX)" in code, (
        "the detector must read the named constant, not a literal"
    )
    assert "int(len(r22) * STEP6_FALL_WARMUP_FRACTION)" in code


# ---------------------------------------------------------------------------------------
# Both sides state both verdicts, every run
# ---------------------------------------------------------------------------------------
#
# These tests pin that reporting. None of them asserts anything about a computed number;
# `test_step_6_would_move_a_reported_number_if_unified` above covers the numbers.


def _summary(*, step6: int | None, collector: int | None) -> dict:
    disagree = (step6 is None) != (collector is None) or (
        step6 is not None and collector is not None and step6 != collector
    )
    return {
        "available": True,
        "fell": step6 is not None,
        "fall_step_step6": step6,
        "collector_fall_step": collector,
        "collector_fell": collector is not None,
        "fall_criteria_disagree": disagree,
    }


def test_both_verdicts_and_both_thresholds_are_named_even_when_they_agree() -> None:
    from holosoma_inference.utils.plot_mocap_raw import fall_criteria_log_lines

    text = " ".join(fall_criteria_log_lines(_summary(step6=None, collector=None)))
    assert "0.500" in text and "60.000 deg" in text, text
    assert "-0.700" in text and "45.573 deg" in text, text
    assert text.count("NO FALL") == 2, text
    assert "DISAGREE" not in text, "agreement must not be reported as a disagreement"


def test_the_collector_only_fall_is_called_out_as_a_disagreement() -> None:
    from holosoma_inference.utils.plot_mocap_raw import fall_criteria_log_lines

    lines = fall_criteria_log_lines(_summary(step6=None, collector=120))
    text = " ".join(lines)
    assert "FALL CRITERIA DISAGREE" in text
    assert "NO FALL" in text and "FELL at step 120" in text
    assert "WHOLE run" in text, "the reader must be told what the exported metrics cover"


def test_the_step6_only_fall_is_called_out_as_a_disagreement() -> None:
    from holosoma_inference.utils.plot_mocap_raw import fall_criteria_log_lines

    text = " ".join(fall_criteria_log_lines(_summary(step6=90, collector=None)))
    assert "FALL CRITERIA DISAGREE" in text
    assert "FELL at step 90" in text and "NO FALL" in text


def test_nothing_is_reported_when_the_summary_is_unavailable() -> None:
    from holosoma_inference.utils.plot_mocap_raw import fall_criteria_log_lines

    assert fall_criteria_log_lines(None) == []
    assert fall_criteria_log_lines({"available": False, "reason": "missing checkpoint"}) == []
    # A summary with no cross-criterion keys produces no report rather than a partial one.
    assert fall_criteria_log_lines({"available": True, "fell": False}) == []


def test_the_summary_reports_the_other_criterion_without_moving_fall_step() -> None:
    """A 52-degree rollout: fallen to the collector, upright to step 6, on the same array."""
    n = 200
    quats_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    half = math.radians(52.0) / 2.0
    quats_xyzw[n // 2 :, 0] = math.sin(half)
    quats_xyzw[n // 2 :, 3] = math.cos(half)
    r22 = 1.0 - 2.0 * (quats_xyzw[:, 0] ** 2 + quats_xyzw[:, 1] ** 2)

    from holosoma_inference.utils.plot_mocap_raw import COLLECTOR_FALL_R22_MAX, _first_index_below

    skip = max(1, int(len(r22) * STEP6_FALL_WARMUP_FRACTION))
    step6_step = _first_index_below(r22, STEP6_FALL_R22_MAX, skip=skip)
    collector_step = _first_index_below(r22, COLLECTOR_FALL_R22_MAX)
    assert step6_step is None, "the shipped fall_step is unchanged by any of this"
    assert collector_step == n // 2

    text = " ".join(fall_criteria_log_lines(_summary(step6=step6_step, collector=collector_step)))
    assert "FALL CRITERIA DISAGREE" in text


def _plot_summary(step6: int | None, collector: int | None, n: int = 8) -> dict:
    summary = _summary(step6=step6, collector=collector)
    summary.update(
        {
            "lin_vel_tracking_return": 1.0,
            "ang_vel_tracking_return": 1.0,
            "tracking_total_return": 2.0,
            "plot_payload": {
                "step_axis": np.arange(n, dtype=np.float64),
                "lin_actual_xy": np.zeros((n, 2)),
                "ang_actual_z": np.zeros(n),
                "lin_command_xy": np.zeros((n, 2)),
                "ang_command_z": np.zeros(n),
                "lin_frame": "body_xy",
                "step_hz": 50.0,
                "fall_step": step6,
            },
        }
    )
    return summary


def _rendered_figure_text(summary: dict, tmp_path) -> str:
    """Every string the velocity figure draws, captured the way the parity harness does."""
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib.figure import Figure

    from holosoma_inference.utils import plot_mocap_raw

    n = int(summary["plot_payload"]["step_axis"].shape[0])
    zeros = np.zeros(n)
    captured: list[str] = []
    original_text = Figure.text

    def _recording_text(self, x, y, s, *args, **kwargs):
        captured.append(str(s))
        return original_text(self, x, y, s, *args, **kwargs)

    Figure.text = _recording_text
    try:
        plot_mocap_raw._plot_velocity_tracking(
            np.arange(n, dtype=np.float64), zeros, zeros, zeros,
            None, None, None, None,
            save_path=str(tmp_path / "vel.png"), show=False, tracking_summary=summary,
        )
    finally:
        Figure.text = original_text
    return " ".join(captured)


def test_the_figure_carries_the_disagreement_too(tmp_path) -> None:
    """`mocap_velocity.png` carries the disagreement banner as well as the console."""
    text = _rendered_figure_text(_plot_summary(step6=None, collector=4), tmp_path)
    assert "FALL CRITERIA DISAGREE" in text, text
    assert "60.00" in text and "45.57" in text, text
    assert "no fall" in text and "fell at step 4" in text, text


def test_the_figure_stays_quiet_when_the_two_criteria_agree(tmp_path) -> None:
    """No banner when the two criteria agree, falling or not."""
    for step6, collector in ((None, None), (4, 4)):
        text = _rendered_figure_text(_plot_summary(step6=step6, collector=collector), tmp_path)
        assert "FALL CRITERIA DISAGREE" not in text, text
