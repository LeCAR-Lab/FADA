"""Parity harness for the step-6 velocity figure: every plotted number must equal the baseline's.

Runs both a falling and a non-falling fixture, so the fall-truncation branch is exercised.

What is compared
----------------
Two independent references, both on the four RMSEs the figure renders (the linear-xy stamp
and the three subplot titles):

1. **Pinned literals.** The falling fixture is ``actual vx = [0, 0, 10, 10]``,
   ``command = [0, 0, 0, 0]``, ``fall_step = 2``, ``trim = 0``; the full-sequence answer is
   ``sqrt(mean([0, 0, 100, 100])) = 7.0711``, and truncating at the fall gives ``0.0000``.
   No baseline checkout is needed for this half.
2. **The baseline implementation itself**, loaded out of git (``FADA_PLOT_PARITY_BASELINE_REF``,
   default ``HEAD``) and executed on the same fixtures. This compares the *rendered strings*,
   so a change of formatting, rounding or window all fail alike. Skipped -- not silently
   passed -- when the ref or the file at that ref is unavailable.

Presentation is NOT pinned here: the figure may add window notes, label the fall marker, or
restyle freely. Only the numbers are frozen.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
from holosoma_inference.utils import plot_mocap_raw
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Headless: these tests render real figures.
mpl.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_REL_PATH = "src/holosoma_inference/holosoma_inference/utils/plot_mocap_raw.py"

# `Linear Vel RMSE (xy): 7.0711 m/s...` and `... (RMSE=7.0711, incl. post-fall)`.
_RMSE_RE = re.compile(r"RMSE(?:\s*\(xy\))?[=:]\s*(-?\d+\.\d+)")


def _fixture_falling(n: int = 4, fall_step: int = 2) -> dict:
    """Perfect tracking, then a 10 m/s error from the fall onward.

    Full-sequence RMSE = sqrt((0 + 0 + 100 + 100) / 4) = 7.0710678...; truncating at the
    fall gives 0, so the two windows are distinguishable.
    """
    actual = np.zeros((n, 2), dtype=np.float64)
    actual[fall_step:, 0] = 10.0
    return _tracking_summary(actual, np.zeros((n, 2), dtype=np.float64), n, fall_step)


def _fixture_no_fall(n: int = 4) -> dict:
    """The same series with `fall_step = None`."""
    actual = np.zeros((n, 2), dtype=np.float64)
    actual[2:, 0] = 10.0
    return _tracking_summary(actual, np.zeros((n, 2), dtype=np.float64), n, None)


def _tracking_summary(actual_xy: np.ndarray, command_xy: np.ndarray, n: int, fall_step: int | None) -> dict:
    return {
        "available": True,
        "lin_vel_tracking_return": 1.0,
        "ang_vel_tracking_return": 1.0,
        "tracking_total_return": 2.0,
        "plot_payload": {
            "step_axis": np.arange(n, dtype=np.float64),
            "lin_actual_xy": actual_xy,
            "ang_actual_z": np.zeros(n),
            "lin_command_xy": command_xy,
            "ang_command_z": np.zeros(n),
            "lin_frame": "body_xy",
            "step_hz": 50.0,
            "fall_step": fall_step,
        },
    }


FIXTURES = {"falling": _fixture_falling, "no_fall": _fixture_no_fall}


def _render_rmse_strings(module: types.ModuleType, summary: dict, tmp_path: Path) -> list[str]:
    """Render the velocity figure with *module* and return every RMSE number it drew.

    Both `Figure.text` (the linear-xy stamp) and `Axes.set_title` (the three per-axis RMSEs)
    are captured, so a number that moves from one to the other is still seen. Values are
    returned as the formatted strings the figure carries, not as floats.
    """
    n = int(summary["plot_payload"]["step_axis"].shape[0])
    t = np.linspace(0.0, n / 50.0, n)
    zeros = np.zeros(n)
    captured: list[str] = []

    original_text, original_title = Figure.text, Axes.set_title

    def _recording_text(self, x, y, s, *args, **kwargs):
        captured.append(str(s))
        return original_text(self, x, y, s, *args, **kwargs)

    def _recording_title(self, label, *args, **kwargs):
        captured.append(str(label))
        return original_title(self, label, *args, **kwargs)

    Figure.text, Axes.set_title = _recording_text, _recording_title
    try:
        module._plot_velocity_tracking(
            t, zeros, zeros, zeros, None, None, None, None,
            save_path=str(tmp_path / "parity.png"), show=False, mocap_ts=t,
            tracking_summary=summary,
        )
    finally:
        Figure.text, Axes.set_title = original_text, original_title

    return [match.group(1) for text in captured for match in _RMSE_RE.finditer(text)]


def _load_baseline_module(tmp_path: Path) -> types.ModuleType | None:
    """Import ``plot_mocap_raw`` as it exists at the baseline git ref, or None if unavailable."""
    ref = os.environ.get("FADA_PLOT_PARITY_BASELINE_REF", "HEAD")
    blob = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{ref}:{MODULE_REL_PATH}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if blob.returncode != 0:
        return None
    path = tmp_path / "plot_mocap_raw_baseline.py"
    path.write_text(blob.stdout)
    spec = importlib.util.spec_from_file_location("_plot_mocap_raw_baseline", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Pinned per fixture: the four RMSEs the figure renders, in the order it renders them
# (linear-xy stamp, then Linear X, Linear Y, Angular Yaw). Both fixtures carry the same
# numbers -- detecting a fall must not move a plotted number.
EXPECTED_RMSE = {
    "falling": ["7.0711", "7.0711", "0.0000", "0.0000"],
    "no_fall": ["7.0711", "7.0711", "0.0000", "0.0000"],
}


@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
def test_plotted_rmse_matches_the_released_full_sequence_values(fixture_name: str, tmp_path: Path) -> None:
    """The figure renders the full-sequence RMSE, fall or no fall."""
    rendered = _render_rmse_strings(plot_mocap_raw, FIXTURES[fixture_name](), tmp_path)
    assert rendered == EXPECTED_RMSE[fixture_name], f"{fixture_name}: figure rendered {rendered}"


@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
def test_plotted_rmse_is_byte_identical_to_the_baseline_implementation(fixture_name: str, tmp_path: Path) -> None:
    """Same fixtures, run through the module as it exists at the baseline git ref."""
    baseline = _load_baseline_module(tmp_path)
    if baseline is None:
        pytest.skip("baseline ref unavailable (no git checkout, or the file is absent at that ref)")

    (tmp_path / "cur").mkdir(exist_ok=True)
    (tmp_path / "ref").mkdir(exist_ok=True)
    current = _render_rmse_strings(plot_mocap_raw, FIXTURES[fixture_name](), tmp_path / "cur")
    reference = _render_rmse_strings(baseline, FIXTURES[fixture_name](), tmp_path / "ref")
    assert current == reference, f"{fixture_name}: current {current} != baseline {reference}"


def test_the_falling_fixture_would_actually_catch_the_regression(tmp_path: Path) -> None:
    """The falling fixture's truncated and untruncated RMSEs differ, so it separates the
    two windows."""
    payload = _fixture_falling()["plot_payload"]
    actual_vx = payload["lin_actual_xy"][:, 0]
    command_vx = payload["lin_command_xy"][:, 0]
    fall_step = int(payload["fall_step"])

    full_sequence = float(np.sqrt(np.nanmean(np.square(actual_vx - command_vx))))
    truncated = float(np.sqrt(np.nanmean(np.square(actual_vx[:fall_step] - command_vx[:fall_step]))))
    assert f"{full_sequence:.4f}" == "7.0711"
    assert f"{truncated:.4f}" == "0.0000"
    assert full_sequence != truncated, "fixture cannot distinguish the two windows"


def test_the_figure_still_says_a_fall_was_detected(tmp_path: Path) -> None:
    """The figure states the fall and the window its RMSE covers, alongside the number."""
    captured: list[str] = []
    original_text, original_title = Figure.text, Axes.set_title
    Figure.text = lambda self, x, y, s, *a, **k: (captured.append(str(s)), original_text(self, x, y, s, *a, **k))[1]
    Axes.set_title = lambda self, label, *a, **k: (captured.append(str(label)), original_title(self, label, *a, **k))[1]
    try:
        n = 4
        t = np.linspace(0.0, n / 50.0, n)
        zeros = np.zeros(n)
        plot_mocap_raw._plot_velocity_tracking(
            t, zeros, zeros, zeros, None, None, None, None,
            save_path=str(tmp_path / "reported.png"), show=False, mocap_ts=t,
            tracking_summary=_fixture_falling(),
        )
    finally:
        Figure.text, Axes.set_title = original_text, original_title

    joined = " ".join(captured)
    assert "fall at step 2" in joined, joined
    assert "INCLUDED" in joined, joined
    assert "7.0711" in joined, "the reported number is still the full-sequence one"
