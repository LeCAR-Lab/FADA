"""Plot raw mocap data (position and quaternion) vs time. Used on program exit when enabled.

Works for both MuJoCo (mocap from sim ZMQ, timestamps = sim_time in seconds) and real robot
(mocap from external mocap ZMQ; timestamps should be in seconds, e.g. wall clock). Command
timestamps are policy time.time() in seconds. Alignment uses elapsed time from each stream's
first sample so displacement/velocity plots show command vs base correctly.

Implementation only -- no standalone CLI entry point is shipped for this module (see
tools/check_entrypoints.py). Call `plot_from_mocap_raw_log(...)` directly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from loguru import logger

from holosoma_inference.utils.math.quat import (
    quat_apply,
    quat_inverse,
    quat_mul,
    quat_rotate_inverse,
    xyzw_to_wxyz,
)
from holosoma_inference.utils.chunk_twin_publisher import build_compact_term_slices

# Trim this many steps from start and end when plotting and computing metrics (RMSE, MSE).
# 0 means metrics aggregate from the first recorded step; describe_metric_window() /
# metric_window_log_lines() report the window actually used.
PLOT_TRIM_STEPS = 0

# --- step-6 fall detection vs. the collector's criterion -------------------------------
#
# Two code paths decide "the robot has fallen", with different thresholds:
#
#   this file (step 6, the pre/post-finetune comparison)
#       R[2,2] < 0.5                       -> tilt > 60.00 deg, ignoring the first 5% of
#                                             the run
#   the collector and the dual-mode safety guard
#       projected_gravity_z > -0.7         -> tilt > 45.57 deg, from the first step
#       (`collect_fall_projected_gravity_z_max`, `dual_mode_projected_gravity_z_min`)
#
# The two quantities are the same number with opposite signs: projected_gravity_z is
# exactly -R[2,2]. So a rollout whose worst tilt lands between 45.57 and 60 degrees is
# FALLEN to the collector and UPRIGHT to step 6.
#
# `fall_step` truncates the exported `lin_vel_tracking_err` and `ang_vel_tracking_err` and
# zeroes the tracking return after the fall.
STEP6_FALL_R22_MAX = 0.5
#: Fraction of the run skipped before fall detection starts. The collector watches from
#: step 0.
STEP6_FALL_WARMUP_FRACTION = 0.05
#: The *collector's* threshold expressed in this module's R[2,2] convention, so step 6 can
#: evaluate the other criterion on the same rollout.
#: `projected_gravity_z == -R[2,2]`, so the collector's `projected_gravity_z > -0.7` is
#: `R[2,2] < 0.7`. REPORTING ONLY -- `fall_step`, and therefore every exported number, is
#: decided by STEP6_FALL_R22_MAX alone.
COLLECTOR_FALL_R22_MAX = 0.7


def _tilt_deg_from_r22(r22_threshold: float) -> float:
    """Body tilt in degrees at which an `R[2,2] < threshold` test starts firing."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r22_threshold)))))


def _first_index_below(values: np.ndarray, threshold: float, *, skip: int = 0) -> int | None:
    hits = np.where(np.asarray(values)[skip:] < threshold)[0]
    return int(skip + int(hits[0])) if hits.size > 0 else None


def fall_criteria_log_lines(summary: dict[str, Any] | None) -> list[str]:
    """Console block stating both fall verdicts (step 6 and the collector) for this rollout.

    See STEP6_FALL_R22_MAX for the two criteria. Returns [] when the summary is missing or
    carries no `collector_fall_step`. Changes no computed value.
    """
    if summary is None or not summary.get("available", False):
        return []
    if "collector_fall_step" not in summary:
        return []
    step6_step = summary.get("fall_step_step6")
    collector_step = summary.get("collector_fall_step")
    step6_deg = _tilt_deg_from_r22(STEP6_FALL_R22_MAX)
    collector_deg = _tilt_deg_from_r22(COLLECTOR_FALL_R22_MAX)
    step6_verdict = f"FELL at step {int(step6_step)}" if step6_step is not None else "NO FALL"
    collector_verdict = f"FELL at step {int(collector_step)}" if collector_step is not None else "NO FALL"
    lines = [
        f"FALL CRITERIA: step 6 (R[2,2] < {STEP6_FALL_R22_MAX:.3f}, tilt > {step6_deg:.3f} deg, "
        f"first {STEP6_FALL_WARMUP_FRACTION:.0%} of the run ignored) says {step6_verdict}.",
        f"FALL CRITERIA: the collector / dual-mode guard (projected_gravity_z > "
        f"{-COLLECTOR_FALL_R22_MAX:.3f}, tilt > {collector_deg:.3f} deg, watched from step 0) "
        f"says {collector_verdict} on this same rollout.",
    ]
    if not summary.get("fall_criteria_disagree", False):
        return lines
    if step6_step is None:
        lines.append(
            "FALL CRITERIA DISAGREE: only the collector's criterion fired, so this figure has no "
            "fall marker and the exported lin/ang_vel_tracking_err and tracking return cover the "
            "WHOLE run -- including steps the collector would have called post-fall. Only "
            "STEP6_FALL_R22_MAX decides fall_step; the collector's verdict is reported, not applied."
        )
    elif collector_step is None:
        lines.append(
            "FALL CRITERIA DISAGREE: only step 6's criterion fired. The collected dataset from the "
            "same rollout carries no fall report, so a finetune on it will treat these steps as "
            "ordinary windows."
        )
    else:
        lines.append(
            f"FALL CRITERIA DISAGREE: the two criteria fire at different steps "
            f"({int(collector_step)} vs {int(step6_step)}); the exported metrics stop at "
            f"{int(step6_step)}, the collected dataset's post-fall count starts at {int(collector_step)}."
        )
    return lines


def describe_metric_window(
    n_total: int,
    trim_head: int,
    trim_tail: int,
    *,
    rate_hz: float = 50.0,
) -> dict[str, Any]:
    """Return exactly which steps the metrics/plots aggregate over.

    `_trim_initial_final_steps` returns the arrays untouched when the requested trim does
    not fit (`n <= head + tail`), so requested and applied trims can differ. Both are
    recorded in the returned dict.
    """
    head = max(int(trim_head), 0)
    tail = max(int(trim_tail), 0)
    total = max(int(n_total), 0)
    applied = (head + tail) > 0 and total > (head + tail)
    head_applied = head if applied else 0
    tail_applied = tail if applied else 0
    return {
        "steps_total": total,
        "trim_head_requested": head,
        "trim_tail_requested": tail,
        "trim_head_applied": head_applied,
        "trim_tail_applied": tail_applied,
        "steps_used": total - head_applied - tail_applied,
        "trim_requested_but_not_applied": (head + tail) > 0 and not applied,
        "rate_hz": float(rate_hz),
    }


def format_metric_window(window: dict[str, Any]) -> str:
    """One-line provenance stamped onto the velocity figure."""
    head = int(window["trim_head_applied"])
    tail = int(window["trim_tail_applied"])
    total = int(window["steps_total"])
    rate = float(window.get("rate_hz") or 0.0)
    seconds = f" = {head / rate:.1f}s" if rate > 0 else ""
    if window["trim_requested_but_not_applied"]:
        return (
            f"Metric window: ALL {total} steps (trim of "
            f"{window['trim_head_requested']}+{window['trim_tail_requested']} NOT applied: "
            "series too short)"
        )
    if head == 0 and tail == 0:
        return f"Metric window: ALL {total} steps from policy start (plot_trim_head=0; startup transient INCLUDED)"
    return f"Metric window: steps [{head}, {total - tail}) of {total} — head trim {head}{seconds}, tail trim {tail}"


def metric_window_log_lines(window: dict[str, Any]) -> list[str]:
    """Console block describing the metric window.

    Recording starts at the policy's first step, so with no head trim the run's startup --
    gantry release, the robot settling onto its feet -- is inside every RMSE and tracking
    return reported below.
    """
    total = int(window["steps_total"])
    rate = float(window.get("rate_hz") or 0.0)
    if window["trim_requested_but_not_applied"]:
        return [
            f"METRIC WINDOW: requested head/tail trim of {window['trim_head_requested']}/"
            f"{window['trim_tail_requested']} steps was NOT applied — the run is only {total} steps long. "
            "Every reported metric covers the whole run, including the startup transient.",
        ]
    head = int(window["trim_head_applied"])
    tail = int(window["trim_tail_applied"])
    if head == 0 and tail == 0:
        hint = ""
        if rate > 0:
            hint = f" At {rate:g} Hz, each second of it is {round(rate)} steps."
        return [
            f"METRIC WINDOW: all {total} recorded steps are included, starting at the first step after "
            "policy start (plot_trim_head=0, plot_trim_tail=0).",
            "Recording begins at policy start, so whatever this run began with -- the gantry release, the "
            "robot settling onto its feet, or a drop if the band was still carrying it -- is measured here "
            "as tracking. Check the first seconds of this run; nothing in these numbers separates them."
            + hint,
            "Pass --task.plot-trim-head N (and the same N to every run you compare) to exclude it.",
        ]
    return [
        f"METRIC WINDOW: steps [{head}, {total - tail}) of {total} — {window['steps_used']} steps used, "
        f"{head} head and {tail} tail steps excluded.",
    ]


def _trim_initial_final_steps(
    n: int, trim_steps: int, *arrays: np.ndarray, trim_tail: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Drop *trim_steps* from the start and *trim_tail* from the end of each array.

    If *trim_tail* is None it defaults to *trim_steps* (symmetric trim).
    Returns arrays unchanged when the total trim exceeds *n*.
    """
    head = max(trim_steps, 0)
    tail = max(trim_tail if trim_tail is not None else trim_steps, 0)
    if head + tail <= 0 or n <= head + tail:
        return arrays
    end = n - tail if tail > 0 else n
    return tuple(a[head:end] for a in arrays)


def _load_optional_string(data: Any, key: str) -> str | None:
    if key not in data:
        return None
    value = np.asarray(data[key]).reshape(-1)
    if value.size <= 0:
        return None
    text = str(value[0]).strip()
    return text or None


def _yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Extract yaw (rad) from quaternion (N, 4) in xyzw format."""
    qx, qy, qz, qw = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _quat_delta_to_ang_vel(prev_q: np.ndarray, curr_q: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from two quaternions (wxyz). Single sample."""
    if dt <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    dq = quat_mul(curr_q.reshape(1, 4), quat_inverse(prev_q.reshape(1, 4))).reshape(-1)
    vec = dq[1:]
    vec_norm = np.linalg.norm(vec)
    scalar = np.clip(dq[0], -1.0, 1.0)
    angle = 2.0 * np.arctan2(vec_norm, scalar)
    if vec_norm < 1e-9 or angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = vec / vec_norm
    return axis * (angle / dt)


def resample_mocap_to_hz(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample mocap to a regular grid at `hz` (e.g. 50). Uses linear interpolation.
    Returns (ts_new, pos_new, quat_new) with length M ≈ (t_end - t0) * hz.
    """
    if hz <= 0 or len(timestamps) < 2:
        return timestamps, positions, quaternions_xyzw
    t0 = float(timestamps[0])
    t_end = float(timestamps[-1])
    duration = t_end - t0
    if duration <= 0:
        return timestamps, positions, quaternions_xyzw
    t_new = np.arange(t0, t_end, 1.0 / hz)
    if len(t_new) == 0:
        return timestamps, positions, quaternions_xyzw
    pos_new = np.column_stack([
        np.interp(t_new, timestamps, positions[:, j]) for j in range(3)
    ])
    quat_new = np.column_stack([
        np.interp(t_new, timestamps, quaternions_xyzw[:, j]) for j in range(4)
    ])
    # Normalize quaternions
    norms = np.linalg.norm(quat_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    quat_new = quat_new / norms
    return t_new, pos_new, quat_new


def _ensure_quat_continuous_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Ensure consecutive quaternions take the short path (negate q if dot(prev, q) < 0) to avoid angular velocity spikes from q vs -q flips."""
    out = np.array(quat_wxyz, dtype=np.float64)
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0:
            out[i] = -out[i]
    return out


def compute_velocity_and_displacement_from_raw(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From raw mocap (ts, pos, quat_xyzw), compute displacement in initial base frame and
    velocity in base frame. Same logic as BasicVelStateProcessor but in batch.
    Quaternions are normalized to short path (no sign flip) to avoid angular velocity spikes.

    Returns
    -------
    t : (N,) relative time in seconds
    base_pos_x, base_pos_y : (N,) position in initial base frame (m)
    base_angle : (N,) yaw in initial base frame (rad), unwrapped
    base_vx, base_vy, base_yaw : (N,) velocity in base frame (m/s, m/s, rad/s)
    """
    n = len(timestamps)
    t = timestamps - timestamps[0]
    quat_wxyz = xyzw_to_wxyz(quaternions_xyzw.reshape(n, 4))  # (n, 4)
    quat_wxyz = _ensure_quat_continuous_wxyz(quat_wxyz)

    # Displacement: position and yaw in initial base frame
    initial_pos = positions[0:1]
    initial_quat_wxyz = quat_wxyz[0:1]
    pos_world = positions - initial_pos
    initial_quat_repeated = np.tile(initial_quat_wxyz, (n, 1))
    pos_initial_base = quat_rotate_inverse(initial_quat_repeated, pos_world)
    base_pos_x = pos_initial_base[:, 0]
    base_pos_y = pos_initial_base[:, 1]
    yaw_world = _yaw_from_quat_xyzw(quaternions_xyzw)
    base_angle = np.unwrap(yaw_world - yaw_world[0])

    # Velocity: world-frame finite difference then transform to base frame per sample
    base_vx = np.zeros(n, dtype=np.float64)
    base_vy = np.zeros(n, dtype=np.float64)
    base_yaw_vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 1e-9:
            continue
        lin_vel_world = (positions[i] - positions[i - 1]) / dt
        ang_vel_world = _quat_delta_to_ang_vel(quat_wxyz[i - 1], quat_wxyz[i], dt)
        q_curr = quat_wxyz[i].reshape(1, 4)
        base_lin = quat_rotate_inverse(q_curr, lin_vel_world.reshape(1, 3))
        base_ang = quat_rotate_inverse(q_curr, ang_vel_world.reshape(1, 3))
        base_vx[i] = base_lin[0, 0]
        base_vy[i] = base_lin[0, 1]
        base_yaw_vel[i] = base_ang[0, 2]
    return t, base_pos_x, base_pos_y, base_angle, base_vx, base_vy, base_yaw_vel


def _command_elapsed_sorted(
    cmd_ts: np.ndarray,
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    cmd_yaw: np.ndarray,
    t0_reference: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize command to elapsed seconds (since t0_reference or since first sample), sort by time,
    and deduplicate so np.interp gets strictly increasing xp.
    When t0_reference is set (e.g. common t0 with mocap), cmd_t_rel = cmd_ts - t0_reference.
    Returns (cmd_t_rel, cx, cy, cyaw) or empty arrays if < 2 samples.
    """
    if len(cmd_ts) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    cmd_ts = np.asarray(cmd_ts, dtype=np.float64).reshape(-1)
    cmd_x = np.asarray(cmd_x, dtype=np.float64).reshape(-1)
    cmd_y = np.asarray(cmd_y, dtype=np.float64).reshape(-1)
    cmd_yaw = np.asarray(cmd_yaw, dtype=np.float64).reshape(-1)
    t0 = float(t0_reference) if t0_reference is not None else float(cmd_ts[0])
    cmd_t_rel = cmd_ts - t0
    sort_idx = np.argsort(cmd_t_rel)
    cmd_t_rel = cmd_t_rel[sort_idx]
    cx = cmd_x[sort_idx]
    cy = cmd_y[sort_idx]
    cyaw = cmd_yaw[sort_idx]
    # Deduplicate by time (keep last value per unique t) so interp xp is strictly increasing
    uniq = np.unique(cmd_t_rel)
    if len(uniq) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    # Last index of each unique time in sorted cmd_t_rel
    last_idx = np.searchsorted(cmd_t_rel, uniq, side="right") - 1
    return uniq, cx[last_idx], cy[last_idx], cyaw[last_idx]


def _command_to_initial_base_integrated(
    t_mocap: np.ndarray,
    quaternions_xyzw: np.ndarray,
    cmd_ts: np.ndarray,
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    cmd_yaw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate command to mocap times, convert to initial base frame, integrate to get
    cmd_pos_x, cmd_pos_y, cmd_angle in initial base frame.

    Works for both MuJoCo (mocap_ts = sim_time in seconds) and real (mocap_ts = wall clock
    in seconds from mocap ZMQ). Command timestamps are policy time.time() in seconds.
    Alignment uses elapsed time since each stream's first sample so timelines are comparable.

    Integration uses the actual time steps between mocap samples (dt = diff(t_mocap)).
    When mocap and command share the same time base (both sim_time or both wall clock),
    use common t0 so the same instant aligns.
    """
    n = len(t_mocap)
    t0_m = float(t_mocap[0])
    t0_c = float(np.asarray(cmd_ts).reshape(-1)[0])
    if (t0_m > 1e6) == (t0_c > 1e6):
        t0_common = min(t0_m, t0_c)
        t_rel = t_mocap - t0_common
        cmd_t_rel, cx, cy, cyaw = _command_elapsed_sorted(cmd_ts, cmd_x, cmd_y, cmd_yaw, t0_reference=t0_common)
    else:
        t_rel = t_mocap - t0_m
        cmd_t_rel, cx, cy, cyaw = _command_elapsed_sorted(cmd_ts, cmd_x, cmd_y, cmd_yaw)
    if len(cmd_t_rel) < 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    # Interpolate command to each mocap time
    cmd_x_i = np.interp(t_rel, cmd_t_rel, cx)
    cmd_y_i = np.interp(t_rel, cmd_t_rel, cy)
    cmd_yaw_i = np.interp(t_rel, cmd_t_rel, cyaw)
    quat_wxyz = xyzw_to_wxyz(quaternions_xyzw.reshape(n, 4))
    initial_quat_wxyz = quat_wxyz[0:1]
    initial_quat_repeated = np.tile(initial_quat_wxyz, (n, 1))
    cmd_vel_base = np.stack([cmd_x_i, cmd_y_i, np.zeros(n)], axis=1)
    cmd_vel_world = quat_apply(quat_wxyz.reshape(n, 4), cmd_vel_base.reshape(n, 3))
    cmd_vel_initial_base = quat_rotate_inverse(initial_quat_repeated, cmd_vel_world)
    # Use actual elapsed time between mocap samples for integration (handles variable rate)
    dt = np.diff(t_mocap)
    dt = np.maximum(dt, 0.0)  # guard against out-of-order timestamps
    if len(dt) == 0:
        dt = np.array([0.0])
    cmd_pos_x = np.concatenate([[0.0], np.cumsum(cmd_vel_initial_base[:-1, 0] * dt)])
    cmd_pos_y = np.concatenate([[0.0], np.cumsum(cmd_vel_initial_base[:-1, 1] * dt)])
    cmd_angle = np.unwrap(np.concatenate([[0.0], np.cumsum(cmd_yaw_i[:-1] * dt)]))
    return cmd_pos_x, cmd_pos_y, cmd_angle


def _command_to_ideal_trajectory(
    t_mocap: np.ndarray,
    cmd_ts: np.ndarray,
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    cmd_yaw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate commands assuming perfect tracking of both translation AND yaw.

    Unlike _command_to_initial_base_integrated which uses the measured yaw,
    this function integrates the commanded yaw rate to obtain an ideal yaw
    angle and uses that to rotate body-frame velocity into the initial base
    frame. The result is the pure geometric trajectory the commands describe
    -- a perfect circle for our circular command sequence.
    """
    n = len(t_mocap)
    t0_m = float(t_mocap[0])
    t0_c = float(np.asarray(cmd_ts).reshape(-1)[0])
    if (t0_m > 1e6) == (t0_c > 1e6):
        t0_common = min(t0_m, t0_c)
        t_rel = t_mocap - t0_common
        cmd_t_rel, cx, cy, cyaw = _command_elapsed_sorted(
            cmd_ts, cmd_x, cmd_y, cmd_yaw, t0_reference=t0_common
        )
    else:
        t_rel = t_mocap - t0_m
        cmd_t_rel, cx, cy, cyaw = _command_elapsed_sorted(cmd_ts, cmd_x, cmd_y, cmd_yaw)
    if len(cmd_t_rel) < 2:
        return np.zeros(n), np.zeros(n)
    cmd_x_i = np.interp(t_rel, cmd_t_rel, cx)
    cmd_y_i = np.interp(t_rel, cmd_t_rel, cy)
    cmd_yaw_i = np.interp(t_rel, cmd_t_rel, cyaw)
    dt = np.diff(t_mocap)
    dt = np.maximum(dt, 0.0)
    if len(dt) == 0:
        return np.zeros(n), np.zeros(n)
    ideal_yaw = np.concatenate([[0.0], np.cumsum(cmd_yaw_i[:-1] * dt)])
    cos_y = np.cos(ideal_yaw)
    sin_y = np.sin(ideal_yaw)
    vx_world = cmd_x_i * cos_y - cmd_y_i * sin_y
    vy_world = cmd_x_i * sin_y + cmd_y_i * cos_y
    ref_x = np.concatenate([[0.0], np.cumsum(vx_world[:-1] * dt)])
    ref_y = np.concatenate([[0.0], np.cumsum(vy_world[:-1] * dt)])
    return ref_x, ref_y


def _plot_displacement_tracking(
    t: np.ndarray,
    base_pos_x: np.ndarray,
    base_pos_y: np.ndarray,
    base_angle: np.ndarray,
    cmd_ts: np.ndarray | None,
    cmd_x: np.ndarray | None,
    cmd_y: np.ndarray | None,
    cmd_yaw: np.ndarray | None,
    mocap_ts: np.ndarray | None,
    quat: np.ndarray | None,
    save_path: str | None,
    title: str = "Mocap displacement (initial base frame)",
    show: bool = False,
) -> None:
    """Plot displacement with optional command curves and RMSE."""
    import matplotlib.pyplot as plt

    has_cmd = (
        cmd_ts is not None and cmd_x is not None and cmd_y is not None and cmd_yaw is not None
        and mocap_ts is not None and quat is not None
        and len(cmd_ts) >= 2 and len(mocap_ts) == len(t)
    )
    t_plot = np.asarray(t, dtype=np.float64)
    if has_cmd:
        t0_m = float(mocap_ts[0])
        t0_c = float(np.asarray(cmd_ts).reshape(-1)[0])
        if (t0_m > 1e6) == (t0_c > 1e6):
            t_plot = (mocap_ts - min(t0_m, t0_c)).astype(np.float64)
    if has_cmd:
        cmd_pos_x, cmd_pos_y, cmd_angle = _command_to_initial_base_integrated(
            mocap_ts, quat, cmd_ts, cmd_x, cmd_y, cmd_yaw
        )
        err_x = base_pos_x - cmd_pos_x
        err_y = base_pos_y - cmd_pos_y
        err_yaw = base_angle - cmd_angle
        rmse_x = float(np.sqrt(np.nanmean(err_x**2)))
        rmse_y = float(np.sqrt(np.nanmean(err_y**2)))
        rmse_yaw = float(np.sqrt(np.nanmean(err_yaw**2)))
        rmse_linear = float(np.sqrt(np.nanmean(err_x**2 + err_y**2)))
    else:
        rmse_x = rmse_y = rmse_yaw = rmse_linear = 0.0

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    if has_cmd:
        fig.text(
            0.01, 0.99, f"Linear Pos RMSE (xy): {rmse_linear:.4f} m",
            ha="left", va="top", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    axs[0, 0].plot(t_plot, base_pos_x, label="base_pos_x", color="tab:blue")
    if has_cmd:
        axs[0, 0].plot(t_plot, cmd_pos_x, "--", label="cmd_pos_x", color="tab:orange")
    axs[0, 0].set_title(f"Position X (Initial Base Frame)" + (f" (RMSE={rmse_x:.4f})" if has_cmd else ""), fontsize=14)
    axs[0, 0].set_ylabel("m", fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)

    axs[0, 1].plot(t_plot, base_pos_y, label="base_pos_y", color="tab:green")
    if has_cmd:
        axs[0, 1].plot(t_plot, cmd_pos_y, "--", label="cmd_pos_y", color="tab:red")
    axs[0, 1].set_title(f"Position Y (Initial Base Frame)" + (f" (RMSE={rmse_y:.4f})" if has_cmd else ""), fontsize=14)
    axs[0, 1].set_ylabel("m", fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)

    axs[1, 0].plot(t_plot, base_angle, label="base_angle (yaw)", color="tab:blue")
    if has_cmd:
        axs[1, 0].plot(t_plot, cmd_angle, "--", label="cmd_angle", color="tab:orange")
    axs[1, 0].set_title(f"Angle Yaw (Initial Base Frame)" + (f" (RMSE={rmse_yaw:.4f})" if has_cmd else ""), fontsize=14)
    axs[1, 0].set_xlabel("Time (s)", fontsize=12)
    axs[1, 0].set_ylabel("rad", fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)

    if has_cmd:
        ref_x, ref_y = _command_to_ideal_trajectory(mocap_ts, cmd_ts, cmd_x, cmd_y, cmd_yaw)
        axs[1, 1].plot(ref_x, ref_y, "--", label="ref (ideal)", color="tab:orange", linewidth=2)
        axs[1, 1].plot(base_pos_x, base_pos_y, label="actual (mocap)", color="tab:blue", linewidth=1.5)
        axs[1, 1].scatter([base_pos_x[0]], [base_pos_y[0]], color="tab:green", s=60, marker="o", label="start", zorder=5)
        axs[1, 1].scatter([base_pos_x[-1]], [base_pos_y[-1]], color="tab:red", s=60, marker="x", label="end", zorder=5)
    else:
        axs[1, 1].plot(base_pos_x, base_pos_y, label="actual (mocap)", color="tab:blue", linewidth=1.5)
    axs[1, 1].set_title("XY trajectory (Initial Base Frame)", fontsize=14)
    axs[1, 1].set_xlabel("X (m)", fontsize=12)
    axs[1, 1].set_ylabel("Y (m)", fontsize=12)
    axs[1, 1].set_aspect("equal", adjustable="datalim")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Mocap displacement plot saved to {save_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def _plot_velocity_tracking(
    t: np.ndarray,
    base_vx: np.ndarray,
    base_vy: np.ndarray,
    base_yaw: np.ndarray,
    cmd_ts: np.ndarray | None,
    cmd_x: np.ndarray | None,
    cmd_y: np.ndarray | None,
    cmd_yaw: np.ndarray | None,
    save_path: str | None,
    title: str = "Mocap velocity (base frame)",
    show: bool = False,
    t0_mocap: float | None = None,  # noqa: ARG001  kept for API compatibility
    mocap_ts: np.ndarray | None = None,
    velocity_for_rmse: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    tracking_summary: dict[str, Any] | None = None,
    metric_window: dict[str, Any] | None = None,
) -> None:
    """Plot velocity with optional command curves and RMSE.

    *base_vx* / *base_vy* / *base_yaw* are drawn as the mocap curves. If *velocity_for_rmse* is
    set to *(vx_m, vy_m, yaw_m)* (e.g. outlier-masked + smoothed), RMSE uses those series with
    ``nanmean`` so masked samples are excluded; plots stay continuous. If *velocity_for_rmse* is
    None, RMSE uses the same arrays as the plot.
    """
    import matplotlib.pyplot as plt

    tracking_plot_payload = None
    if tracking_summary is not None and tracking_summary.get("available", False):
        tracking_plot_payload = tracking_summary.get("plot_payload")

    has_step_tracking = bool(tracking_plot_payload)
    lin_frame_label = ""
    fall_step = None
    nominal_step_hz = float("nan")
    plot_vx = np.asarray(base_vx, dtype=np.float64)
    plot_vy = np.asarray(base_vy, dtype=np.float64)
    plot_yaw = np.asarray(base_yaw, dtype=np.float64)
    t_plot = np.asarray(t, dtype=np.float64)
    if has_step_tracking:
        plot_vx = np.asarray(tracking_plot_payload["lin_actual_xy"], dtype=np.float64)[:, 0]
        plot_vy = np.asarray(tracking_plot_payload["lin_actual_xy"], dtype=np.float64)[:, 1]
        plot_yaw = np.asarray(tracking_plot_payload["ang_actual_z"], dtype=np.float64).reshape(-1)
        t_plot = np.asarray(tracking_plot_payload["step_axis"], dtype=np.float64).reshape(-1)
        cmd_x_i = np.asarray(tracking_plot_payload["lin_command_xy"], dtype=np.float64)[:, 0]
        cmd_y_i = np.asarray(tracking_plot_payload["lin_command_xy"], dtype=np.float64)[:, 1]
        cmd_yaw_i = np.asarray(tracking_plot_payload["ang_command_z"], dtype=np.float64).reshape(-1)
        lin_frame_label = str(tracking_plot_payload.get("lin_frame") or "")
        fall_step = tracking_plot_payload.get("fall_step")
        nominal_step_hz = float(tracking_plot_payload.get("step_hz") or np.nan)

    has_cmd = (
        cmd_ts is not None and cmd_x is not None and cmd_y is not None and cmd_yaw is not None
        and len(cmd_ts) >= 2
    )
    rmse_window_note = ""
    rmse_title_note = ""
    if has_step_tracking:
        has_cmd = len(cmd_x_i) == len(plot_vx) and len(cmd_y_i) == len(plot_vy) and len(cmd_yaw_i) == len(plot_yaw)
        # These RMSEs span the FULL post-trim sequence, including any post-fall steps. The
        # exported `lin_vel_tracking_err` stops at `fall_step` and the tracking return
        # zeroes reward after it, so the figure carries three windows; `rmse_window_note` /
        # `rmse_title_note` below and the labelled fall marker state which is which.
        rmse_x = float(np.sqrt(np.nanmean(np.square(plot_vx - cmd_x_i)))) if has_cmd else 0.0
        rmse_y = float(np.sqrt(np.nanmean(np.square(plot_vy - cmd_y_i)))) if has_cmd else 0.0
        rmse_yaw = float(np.sqrt(np.nanmean(np.square(plot_yaw - cmd_yaw_i)))) if has_cmd else 0.0
        rmse_linear = float(np.sqrt(np.nanmean(np.square(plot_vx - cmd_x_i) + np.square(plot_vy - cmd_y_i)))) if has_cmd else 0.0
        if fall_step is not None:
            rmse_window_note = (
                f" — over ALL {len(plot_vx)} plotted steps, fall at step {int(fall_step)} INCLUDED "
                "(the exported lin/ang_vel_tracking_err and the tracking return stop at the fall; "
                "these RMSEs do not)"
            )
            rmse_title_note = ", incl. post-fall"
    elif has_cmd:
        cmd_ts_a = np.asarray(cmd_ts, dtype=np.float64).reshape(-1)
        t0_cmd = float(cmd_ts_a[0])
        if mocap_ts is not None and len(mocap_ts) == len(t):
            t0_mocap_val = float(mocap_ts[0])
            if (t0_mocap_val > 1e6) == (t0_cmd > 1e6):
                t0_common = min(t0_mocap_val, t0_cmd)
                t_plot = (mocap_ts - t0_common).astype(np.float64)
                cmd_t_rel, _cx, _cy, _cyaw = _command_elapsed_sorted(
                    cmd_ts, cmd_x, cmd_y, cmd_yaw, t0_reference=t0_common
                )
            else:
                cmd_t_rel, _cx, _cy, _cyaw = _command_elapsed_sorted(cmd_ts, cmd_x, cmd_y, cmd_yaw)
        else:
            cmd_t_rel, _cx, _cy, _cyaw = _command_elapsed_sorted(cmd_ts, cmd_x, cmd_y, cmd_yaw)
        if len(cmd_t_rel) < 2:
            has_cmd = False
        else:
            cmd_x_i = np.interp(t_plot, cmd_t_rel, _cx)
            cmd_y_i = np.interp(t_plot, cmd_t_rel, _cy)
            cmd_yaw_i = np.interp(t_plot, cmd_t_rel, _cyaw)
            vx_e, vy_e, yaw_e = (
                velocity_for_rmse
                if velocity_for_rmse is not None
                else (plot_vx, plot_vy, plot_yaw)
            )
            err_x = vx_e - cmd_x_i
            err_y = vy_e - cmd_y_i
            err_yaw = yaw_e - cmd_yaw_i
            rmse_x = float(np.sqrt(np.nanmean(err_x**2)))
            rmse_y = float(np.sqrt(np.nanmean(err_y**2)))
            rmse_yaw = float(np.sqrt(np.nanmean(err_yaw**2)))
            rmse_linear = float(np.sqrt(np.nanmean(err_x**2 + err_y**2)))
    if not has_cmd:
        rmse_x = rmse_y = rmse_yaw = rmse_linear = 0.0

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    if metric_window is not None:
        # Stamp the metric window onto the figure; red when no head trim was applied.
        untrimmed = int(metric_window["trim_head_applied"]) == 0
        fig.text(
            0.5,
            0.945,
            format_metric_window(metric_window),
            ha="center",
            va="top",
            fontsize=11,
            color="tab:red" if untrimmed else "0.25",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "tab:red" if untrimmed else "0.8"},
        )
    if has_cmd:
        fig.text(
            0.01, 0.99, f"Linear Vel RMSE (xy): {rmse_linear:.4f} m/s{rmse_window_note}",
            ha="left", va="top", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )
    if tracking_summary is not None:
        lines: list[str] = ["Tracking Returns"]
        if tracking_summary.get("available", False):
            lines.extend(
                [
                    f"lin: {float(tracking_summary['lin_vel_tracking_return']):.4f}",
                    f"ang: {float(tracking_summary['ang_vel_tracking_return']):.4f}",
                    f"total: {float(tracking_summary['tracking_total_return']):.4f}",
                ]
            )
        else:
            reason = str(tracking_summary.get("reason") or "unavailable")
            lines.append(reason)
        fig.text(
            0.99, 0.99, "\n".join(lines),
            ha="right", va="top", fontsize=11,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.8"),
        )
    if tracking_summary is not None and tracking_summary.get("fall_criteria_disagree", False):
        # Banner both fall verdicts on the figure when the two criteria disagree.
        step6_step = tracking_summary.get("fall_step_step6")
        collector_step = tracking_summary.get("collector_fall_step")
        fig.text(
            0.5,
            0.905,
            "FALL CRITERIA DISAGREE — step 6 (tilt > "
            f"{_tilt_deg_from_r22(STEP6_FALL_R22_MAX):.2f}°, first "
            f"{STEP6_FALL_WARMUP_FRACTION:.0%} skipped): "
            + (f"fell at step {int(step6_step)}" if step6_step is not None else "no fall")
            + f"  |  collector (tilt > {_tilt_deg_from_r22(COLLECTOR_FALL_R22_MAX):.2f}°, from step 0): "
            + (f"fell at step {int(collector_step)}" if collector_step is not None else "no fall"),
            ha="center",
            va="top",
            fontsize=10,
            color="tab:red",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "tab:red"},
        )

    x_label = "Control Step" if has_step_tracking else "Time (s)"
    lin_title_suffix = f" [{lin_frame_label}]" if lin_frame_label else ""

    def _rmse_title_suffix(value: float) -> str:
        """Subplot-title RMSE, with the window note appended when a fall was detected."""
        return f" (RMSE={value:.4f}{rmse_title_note})" if has_cmd else ""

    axs[0, 0].plot(t_plot, plot_vx, label="actual_vx", color="tab:blue")
    if has_cmd:
        axs[0, 0].plot(t_plot, cmd_x_i, "--", label="cmd_x", color="tab:orange")
    axs[0, 0].set_title(f"Linear Velocity X{lin_title_suffix}" + _rmse_title_suffix(rmse_x), fontsize=14)
    axs[0, 0].set_ylabel("m/s", fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)

    axs[0, 1].plot(t_plot, plot_vy, label="actual_vy", color="tab:green")
    if has_cmd:
        axs[0, 1].plot(t_plot, cmd_y_i, "--", label="cmd_y", color="tab:red")
    axs[0, 1].set_title(f"Linear Velocity Y{lin_title_suffix}" + _rmse_title_suffix(rmse_y), fontsize=14)
    axs[0, 1].set_ylabel("m/s", fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)

    axs[1, 0].plot(t_plot, plot_yaw, label="actual_yaw", color="tab:blue")
    if has_cmd:
        axs[1, 0].plot(t_plot, cmd_yaw_i, "--", label="cmd_yaw", color="tab:orange")
    axs[1, 0].set_title(f"Angular Velocity Yaw" + _rmse_title_suffix(rmse_yaw), fontsize=14)
    axs[1, 0].set_xlabel(x_label, fontsize=12)
    axs[1, 0].set_ylabel("rad/s", fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)

    if fall_step is not None:
        # Fall marker, labelled with the window split: the RMSEs in the subplot titles span
        # the whole series, the exported tracking error and return stop at this step.
        for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
            ax.axvline(
                float(fall_step),
                color="tab:red",
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
                label=f"fall (step {int(fall_step)}) — exported tracking metrics end here; RMSE above does not",
            )
            ax.legend(fontsize=9)

    axs[1, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Mocap velocity plot saved to {save_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def _plot_reconstruction(
    recon_ts: np.ndarray | None,
    actual_target: np.ndarray | None,
    predicted_target: np.ndarray | None,
    save_path: str | None,
    title: str = "Prediction obs loss",
    show: bool = False,
    planner_first_step_ts: np.ndarray | None = None,
    planner_first_step_actual: np.ndarray | None = None,
    planner_first_step_predicted: np.ndarray | None = None,
    compact_term_order: tuple[str, ...] | list[str] | None = None,
    compact_term_dims: dict[str, int] | None = None,
    idm_ts: np.ndarray | None = None,
    idm_actual_chunk: np.ndarray | None = None,
    idm_predicted_chunk: np.ndarray | None = None,
    fdm_ts: np.ndarray | None = None,
    fdm_actual_chunk: np.ndarray | None = None,
    fdm_predicted_chunk: np.ndarray | None = None,
) -> None:
    """Plot prediction MSE over time using K-step alignment.

    At logged index ``i`` we aggregate all valid predictions that target this step:
    ``predicted[t, k]`` where ``t + k == i``.
    """
    import matplotlib.pyplot as plt

    has_idm = idm_ts is not None and idm_actual_chunk is not None and idm_predicted_chunk is not None and len(idm_ts) > 0
    has_fdm = fdm_ts is not None and fdm_actual_chunk is not None and fdm_predicted_chunk is not None and len(fdm_ts) > 0
    if has_idm or has_fdm:
        panels: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
        if has_idm:
            idm_stats = compute_idm_action_mse_stats(idm_actual_chunk, idm_predicted_chunk)
            panels.append(
                (
                    "IDM",
                    "tab:cyan",
                    np.asarray(idm_ts, dtype=np.float64).reshape(-1),
                    np.asarray(idm_stats["step_mse"], dtype=np.float64),
                    np.asarray(idm_stats["per_horizon_mse"], dtype=np.float64),
                )
            )
        if has_fdm:
            fdm_stats = compute_fdm_obs_mse_stats(fdm_actual_chunk, fdm_predicted_chunk)
            panels.append(
                (
                    "FDM",
                    "tab:green",
                    np.asarray(fdm_ts, dtype=np.float64).reshape(-1),
                    np.asarray(fdm_stats["step_mse"], dtype=np.float64),
                    np.asarray(fdm_stats["per_horizon_mse"], dtype=np.float64),
                )
            )

        fig, axs = plt.subplots(
            2,
            len(panels),
            figsize=(13 * len(panels), 7.5),
            gridspec_kw={"height_ratios": [1.2, 1.0]},
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")
        if len(panels) == 1:
            axs = np.asarray(axs).reshape(2, 1)
        for col, (label, color, ts_block, step_mse, per_horizon) in enumerate(panels):
            t_block = ts_block - ts_block[0] if len(ts_block) > 0 else ts_block
            axs[0, col].plot(t_block[: len(step_mse)], step_mse, color=color)
            axs[0, col].set_title(
                f"Overall {label} Loss (Mean MSE={float(np.mean(per_horizon)) if per_horizon.size > 0 else float('nan'):.6f})",
                fontsize=12,
            )
            axs[0, col].set_xlabel("Time (s)", fontsize=11)
            axs[0, col].set_ylabel("MSE", fontsize=11)
            axs[0, col].grid(True, alpha=0.3)

            bar_x = np.arange(len(per_horizon))
            bar_colors = ["tab:orange" if idx == 0 else "tab:blue" for idx in range(len(per_horizon))]
            axs[1, col].bar(bar_x, per_horizon, color=bar_colors, alpha=0.85, width=0.7)
            axs[1, col].set_title(f"{label} Loss Per Horizon Step", fontsize=12)
            axs[1, col].set_xlabel("Horizon Step", fontsize=11)
            axs[1, col].set_ylabel("MSE", fontsize=11)
            axs[1, col].set_xticks(bar_x, [f"step {idx + 1}" for idx in range(len(per_horizon))], rotation=0)
            axs[1, col].grid(True, alpha=0.3, axis="y")
            if len(bar_x) > 0:
                axs[1, col].set_xlim(-0.6, len(bar_x) - 0.4)
            for idx, value in enumerate(per_horizon):
                if np.isnan(value):
                    continue
                axs[1, col].text(idx, value, f"{value:.6f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.4)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Mocap teacher reconstruction plot saved to {save_path}")
        if show:
            plt.show(block=True)
        else:
            plt.close(fig)

        planner_save_path = None
        if save_path:
            planner_save_path = str(Path(save_path).with_name("mocap_planner_first_step_error.png"))
        _plot_planner_first_step_error(
            planner_first_step_ts,
            planner_first_step_actual,
            planner_first_step_predicted,
            compact_term_order=compact_term_order,
            compact_term_dims=compact_term_dims,
            save_path=planner_save_path,
            show=show,
        )
        return

    if recon_ts is None or actual_target is None or predicted_target is None or len(recon_ts) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "Reconstruction data unavailable", ha="center", va="center", fontsize=14)
        ax.axis("off")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show(block=True)
        else:
            plt.close(fig)
        return

    t = recon_ts - recon_ts[0]
    actual = np.asarray(actual_target, dtype=np.float64)
    predicted = np.asarray(predicted_target, dtype=np.float64)
    N = actual.shape[0]
    if actual.ndim == 1:
        actual = actual.reshape(N, -1)
    if predicted.ndim == 1:
        predicted = predicted.reshape(1, 1, -1)
    elif predicted.ndim == 2:
        predicted = predicted.reshape(predicted.shape[0], 1, -1)
    N = min(N, int(predicted.shape[0]))
    actual = actual[:N]
    predicted = predicted[:N]
    _, K, _ = predicted.shape

    # K-step per-time loss: at step i, average all valid (t, k) pairs with t + k = i.
    recon_loss = np.full(N, np.nan, dtype=np.float64)
    contrib_counts = np.zeros(N, dtype=np.int64)
    for i in range(N):
        t_start = max(0, i - K + 1)
        terms: list[float] = []
        for t_idx in range(t_start, i + 1):
            k = i - t_idx
            if 0 <= k < K:
                err = (predicted[t_idx, k, :] - actual[i, :]) ** 2
                terms.append(float(np.mean(err)))
        if terms:
            recon_loss[i] = float(np.mean(terms))
            contrib_counts[i] = int(len(terms))
    valid = contrib_counts > 0
    if np.any(valid):
        recon_mse = float(np.sum(recon_loss[valid] * contrib_counts[valid]) / np.sum(contrib_counts[valid]))
    else:
        recon_mse = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax.plot(t[:N], np.where(valid, recon_loss, np.nan), color="tab:cyan", label=f"K-step MSE (K={K})")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title(f"Mean MSE = {recon_mse:.6f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Mocap reconstruction plot saved to {save_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def _normalize_matrix(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    out = np.asarray(array, dtype=np.float64)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(1, -1)
    else:
        out = out.reshape(int(np.prod(out.shape[:-1], dtype=np.int64)), out.shape[-1])
    return out


def _normalize_chunk_tensor(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    out = np.asarray(array, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(1, 1, -1)
    elif out.ndim == 2:
        out = out.reshape(out.shape[0], 1, out.shape[1])
    else:
        out = out.reshape(int(np.prod(out.shape[:-2], dtype=np.int64)), out.shape[-2], out.shape[-1])
    return out


def load_mocap_unified_metric_blocks(path: str | Path) -> dict[str, object]:
    """Load optional planner+IDM metric blocks from a unified mocap log."""
    with np.load(path) as data:
        compact_term_order: tuple[str, ...] | None = None
        if "compact_term_order" in data:
            compact_term_order = tuple(str(v) for v in np.asarray(data["compact_term_order"]).reshape(-1).tolist())

        compact_term_dims: dict[str, int] | None = None
        if "compact_term_dims" in data:
            dims_raw = data["compact_term_dims"]
            dims_str = str(np.asarray(dims_raw).reshape(-1)[0])
            try:
                compact_term_dims = {str(k): int(v) for k, v in json.loads(dims_str).items()}
            except Exception:
                compact_term_dims = None

        def _maybe_array(key: str, *, reshape_ts: bool = False) -> np.ndarray | None:
            if key not in data:
                return None
            arr = np.asarray(data[key], dtype=np.float64)
            if reshape_ts:
                arr = arr.reshape(-1)
            return arr

        def _maybe_int(key: str) -> int | None:
            if key not in data:
                return None
            try:
                return int(np.asarray(data[key]).reshape(-1)[0])
            except Exception:
                return None

        def _maybe_float(key: str) -> float | None:
            if key not in data:
                return None
            try:
                return float(np.asarray(data[key]).reshape(-1)[0])
            except Exception:
                return None

        return {
            "compact_term_order": compact_term_order,
            "compact_term_dims": compact_term_dims,
            "tracking_reward_checkpoint_path": _load_optional_string(data, "tracking_reward_checkpoint_path"),
            "plot_trim_head": _maybe_int("plot_trim_head"),
            "plot_trim_tail": _maybe_int("plot_trim_tail"),
            "tracking_step_hz": _maybe_float("tracking_step_hz"),
            "planner_first_step_timestamps": _maybe_array("planner_first_step_timestamps", reshape_ts=True),
            "planner_first_step_actual_obs": _maybe_array("planner_first_step_actual_obs"),
            "planner_first_step_predicted_obs": _maybe_array("planner_first_step_predicted_obs"),
            "idm_timestamps": _maybe_array("idm_timestamps", reshape_ts=True),
            "idm_actual_action_chunk": _maybe_array("idm_actual_action_chunk"),
            "idm_predicted_action_chunk": _maybe_array("idm_predicted_action_chunk"),
            "fdm_timestamps": _maybe_array("fdm_timestamps", reshape_ts=True),
            "fdm_actual_obs_chunk": _maybe_array("fdm_actual_obs_chunk"),
            "fdm_predicted_obs_chunk": _maybe_array("fdm_predicted_obs_chunk"),
        }


def compute_planner_first_step_mse_stats(
    actual_obs: np.ndarray | None,
    predicted_obs: np.ndarray | None,
    *,
    compact_term_order: tuple[str, ...] | list[str] | None = None,
    compact_term_dims: dict[str, int] | None = None,
) -> dict[str, object]:
    """Compute overall and per-term MSE for planner first-step observation alignment."""
    actual = _normalize_matrix(actual_obs)
    predicted = _normalize_matrix(predicted_obs)
    if actual is None or predicted is None:
        return {
            "overall_mse": float("nan"),
            "step_mse": np.array([], dtype=np.float64),
            "per_term_mse": {},
            "per_term_step_mse": np.empty((0, 0), dtype=np.float64),
        }

    n = min(int(actual.shape[0]), int(predicted.shape[0]))
    if n <= 0:
        return {
            "overall_mse": float("nan"),
            "step_mse": np.array([], dtype=np.float64),
            "per_term_mse": {},
            "per_term_step_mse": np.empty((0, 0), dtype=np.float64),
        }

    actual = actual[:n]
    predicted = predicted[:n]
    diff_sq = np.square(predicted - actual)
    step_mse = np.mean(diff_sq, axis=1)
    overall_mse = float(np.mean(diff_sq))

    order = tuple(str(term) for term in compact_term_order) if compact_term_order is not None else tuple()
    dims = {str(key): int(value) for key, value in (compact_term_dims or {}).items()}
    per_term_mse: dict[str, float] = {}
    per_term_step = np.empty((n, 0), dtype=np.float64)
    if order and dims:
        term_slices = build_compact_term_slices(order, dims)
        per_term_step = np.full((n, len(order)), np.nan, dtype=np.float64)
        for idx, term in enumerate(order):
            term_sq = diff_sq[:, term_slices[term]]
            per_term_step[:, idx] = np.mean(term_sq, axis=1)
            per_term_mse[term] = float(np.mean(term_sq))

    return {
        "overall_mse": overall_mse,
        "step_mse": step_mse,
        "per_term_mse": per_term_mse,
        "per_term_step_mse": per_term_step,
    }


def compute_idm_action_mse_stats(
    actual_action_chunk: np.ndarray | None,
    predicted_action_chunk: np.ndarray | None,
) -> dict[str, object]:
    """Compute overall and per-horizon MSE for IDM inverse teacher alignment."""
    actual = _normalize_chunk_tensor(actual_action_chunk)
    predicted = _normalize_chunk_tensor(predicted_action_chunk)
    if actual is None or predicted is None:
        return {
            "overall_mse": float("nan"),
            "step_mse": np.array([], dtype=np.float64),
            "per_horizon_mse": np.array([], dtype=np.float64),
            "per_horizon_step_mse": np.empty((0, 0), dtype=np.float64),
        }

    n = min(int(actual.shape[0]), int(predicted.shape[0]))
    k = min(int(actual.shape[1]), int(predicted.shape[1]))
    if n <= 0 or k <= 0:
        return {
            "overall_mse": float("nan"),
            "step_mse": np.array([], dtype=np.float64),
            "per_horizon_mse": np.array([], dtype=np.float64),
            "per_horizon_step_mse": np.empty((0, 0), dtype=np.float64),
        }

    actual = actual[:n, :k]
    predicted = predicted[:n, :k]
    diff_sq = np.square(predicted - actual)
    step_mse = np.mean(diff_sq, axis=(1, 2))
    per_horizon_step_mse = np.mean(diff_sq, axis=2)
    per_horizon_mse = np.mean(diff_sq, axis=(0, 2))
    return {
        "overall_mse": float(np.mean(diff_sq)),
        "step_mse": step_mse,
        "per_horizon_mse": per_horizon_mse,
        "per_horizon_step_mse": per_horizon_step_mse,
    }


def compute_fdm_obs_mse_stats(
    actual_obs_chunk: np.ndarray | None,
    predicted_obs_chunk: np.ndarray | None,
) -> dict[str, object]:
    """Compute overall and per-horizon MSE for FDM forward teacher alignment."""
    actual = _normalize_chunk_tensor(actual_obs_chunk)
    predicted = _normalize_chunk_tensor(predicted_obs_chunk)
    if actual is None or predicted is None:
        return {
            "overall_mse": float("nan"),
            "step_mse": np.array([], dtype=np.float64),
            "per_horizon_mse": np.array([], dtype=np.float64),
            "per_horizon_step_mse": np.empty((0, 0), dtype=np.float64),
        }

    n = min(int(actual.shape[0]), int(predicted.shape[0]))
    k = min(int(actual.shape[1]), int(predicted.shape[1]))
    if n <= 0 or k <= 0:
        return {
            "overall_mse": float("nan"),
            "step_mse": np.array([], dtype=np.float64),
            "per_horizon_mse": np.array([], dtype=np.float64),
            "per_horizon_step_mse": np.empty((0, 0), dtype=np.float64),
        }

    actual = actual[:n, :k]
    predicted = predicted[:n, :k]
    diff_sq = np.square(predicted - actual)
    step_mse = np.mean(diff_sq, axis=(1, 2))
    per_horizon_step_mse = np.mean(diff_sq, axis=2)
    per_horizon_mse = np.mean(diff_sq, axis=(0, 2))
    return {
        "overall_mse": float(np.mean(diff_sq)),
        "step_mse": step_mse,
        "per_horizon_mse": per_horizon_mse,
        "per_horizon_step_mse": per_horizon_step_mse,
    }


def _plot_planner_first_step_error(
    planner_ts: np.ndarray | None,
    actual_obs: np.ndarray | None,
    predicted_obs: np.ndarray | None,
    *,
    compact_term_order: tuple[str, ...] | list[str] | None,
    compact_term_dims: dict[str, int] | None,
    save_path: str | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    stats = compute_planner_first_step_mse_stats(
        actual_obs,
        predicted_obs,
        compact_term_order=compact_term_order,
        compact_term_dims=compact_term_dims,
    )
    step_mse = np.asarray(stats["step_mse"], dtype=np.float64)
    if planner_ts is None or len(step_mse) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "Planner first-step error unavailable", ha="center", va="center", fontsize=14)
        ax.axis("off")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Mocap planner first-step plot saved to {save_path}")
        if show:
            plt.show(block=True)
        else:
            plt.close(fig)
        return

    t = np.asarray(planner_ts, dtype=np.float64).reshape(-1)
    t = t[: len(step_mse)] - float(t[0])
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    fig.suptitle("Planner First-Step Observation Error", fontsize=14, fontweight="bold")
    ax.plot(t, step_mse, color="tab:red")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title(f"Mean MSE = {float(stats['overall_mse']):.6f}", fontsize=12)
    ax.grid(True, alpha=0.3)

    lines = [f"overall: {float(stats['overall_mse']):.6f}"]
    for term, value in stats["per_term_mse"].items():
        lines.append(f"{term}: {value:.6f}")
    ax.text(
        1.01,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"),
    )
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Mocap planner first-step plot saved to {save_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def plot_mocap_raw_history(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    save_path: str | None = None,
    title: str = "Mocap raw data",
    show: bool = True,
) -> None:
    """
    Plot mocap raw data: position (x,y,z) and quaternion (x,y,z,w) vs time.

    Parameters
    ----------
    timestamps : (N,) array
        Time in seconds for each sample.
    positions : (N, 3) array
        Position [x, y, z] per sample.
    quaternions : (N, 4) array
        Quaternion [x, y, z, w] per sample.
    save_path : str | None
        If set, save figure to this path before showing.
    title : str
        Figure title.
    show : bool
        If True, call plt.show(block=True); else plt.close(fig).
    """
    import matplotlib.pyplot as plt

    n = len(timestamps)
    if n == 0:
        logger.warning("plot_mocap_raw_history: no samples to plot")
        return

    t = timestamps - timestamps[0]  # relative time in seconds

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle(title, fontsize=12)

    # Row 0: position x, y, z
    ax0 = axes[0]
    ax0.set_ylabel("Position (m)")
    ax0.plot(t, positions[:, 0], label="x", alpha=0.9)
    ax0.plot(t, positions[:, 1], label="y", alpha=0.9)
    ax0.plot(t, positions[:, 2], label="z", alpha=0.9)
    ax0.legend(loc="upper right", ncol=3)
    ax0.grid(True, alpha=0.3)

    # Row 1: quaternion x, y, z, w
    ax1 = axes[1]
    ax1.set_ylabel("Quaternion (xyzw)")
    ax1.set_xlabel("Time (s)")
    ax1.plot(t, quaternions[:, 0], label="qx", alpha=0.9)
    ax1.plot(t, quaternions[:, 1], label="qy", alpha=0.9)
    ax1.plot(t, quaternions[:, 2], label="qz", alpha=0.9)
    ax1.plot(t, quaternions[:, 3], label="qw", alpha=0.9)
    ax1.legend(loc="upper right", ncol=4)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Mocap raw plot saved to {path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Unified log: mocap raw + command + reconstruction (save/load and plot)
# --------------------------------------------------------------------------- #


def save_mocap_unified_log(
    timestamps: np.ndarray | None,
    positions: np.ndarray | None,
    quaternions: np.ndarray | None,
    command_timestamps: np.ndarray | None,
    command_x: np.ndarray | None,
    command_y: np.ndarray | None,
    command_yaw: np.ndarray | None,
    recon_timestamps: np.ndarray | None,
    actual_target: np.ndarray | None,
    predicted_target: np.ndarray | None,
    *,
    compact_term_order: tuple[str, ...] | list[str] | None = None,
    compact_term_dims: dict[str, int] | None = None,
    planner_first_step_timestamps: np.ndarray | None = None,
    planner_first_step_actual_obs: np.ndarray | None = None,
    planner_first_step_predicted_obs: np.ndarray | None = None,
    idm_timestamps: np.ndarray | None = None,
    idm_actual_action_chunk: np.ndarray | None = None,
    idm_predicted_action_chunk: np.ndarray | None = None,
    fdm_timestamps: np.ndarray | None = None,
    fdm_actual_obs_chunk: np.ndarray | None = None,
    fdm_predicted_obs_chunk: np.ndarray | None = None,
    tracking_reward_checkpoint_path: str | None = None,
    plot_trim_head: int | None = None,
    plot_trim_tail: int | None = None,
    tracking_step_hz: float | None = None,
    path: str | Path,
) -> None:
    """Save unified log: mocap raw + command + reconstruction. Mocap can be None/empty; omit command/recon if None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if timestamps is None or len(timestamps) == 0:
        timestamps = np.array([], dtype=np.float64)
        positions = np.array([]).reshape(0, 3)
        quaternions = np.array([]).reshape(0, 4)
    else:
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        quaternions = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    out = dict(
        timestamps=timestamps,
        positions=positions,
        quaternions=quaternions,
    )
    if command_timestamps is not None and command_x is not None and command_y is not None and command_yaw is not None:
        out["command_timestamps"] = command_timestamps
        out["command_x"] = command_x
        out["command_y"] = command_y
        out["command_yaw"] = command_yaw
    if recon_timestamps is not None and actual_target is not None and predicted_target is not None:
        out["recon_timestamps"] = recon_timestamps
        out["actual_target"] = actual_target
        out["predicted_target"] = predicted_target
    if compact_term_order:
        out["compact_term_order"] = np.asarray(tuple(compact_term_order))
    if compact_term_dims:
        out["compact_term_dims"] = np.asarray(json.dumps(compact_term_dims))
    if (
        planner_first_step_timestamps is not None
        and planner_first_step_actual_obs is not None
        and planner_first_step_predicted_obs is not None
    ):
        out["planner_first_step_timestamps"] = np.asarray(planner_first_step_timestamps, dtype=np.float64).reshape(-1)
        out["planner_first_step_actual_obs"] = np.asarray(planner_first_step_actual_obs, dtype=np.float64)
        out["planner_first_step_predicted_obs"] = np.asarray(planner_first_step_predicted_obs, dtype=np.float64)
    if idm_timestamps is not None and idm_actual_action_chunk is not None and idm_predicted_action_chunk is not None:
        out["idm_timestamps"] = np.asarray(idm_timestamps, dtype=np.float64).reshape(-1)
        out["idm_actual_action_chunk"] = np.asarray(idm_actual_action_chunk, dtype=np.float64)
        out["idm_predicted_action_chunk"] = np.asarray(idm_predicted_action_chunk, dtype=np.float64)
    if fdm_timestamps is not None and fdm_actual_obs_chunk is not None and fdm_predicted_obs_chunk is not None:
        out["fdm_timestamps"] = np.asarray(fdm_timestamps, dtype=np.float64).reshape(-1)
        out["fdm_actual_obs_chunk"] = np.asarray(fdm_actual_obs_chunk, dtype=np.float64)
        out["fdm_predicted_obs_chunk"] = np.asarray(fdm_predicted_obs_chunk, dtype=np.float64)
    if tracking_reward_checkpoint_path:
        out["tracking_reward_checkpoint_path"] = np.asarray(str(tracking_reward_checkpoint_path))
    if plot_trim_head is not None:
        out["plot_trim_head"] = np.asarray(int(plot_trim_head), dtype=np.int64)
    if plot_trim_tail is not None:
        out["plot_trim_tail"] = np.asarray(int(plot_trim_tail), dtype=np.int64)
    if tracking_step_hz is not None:
        out["tracking_step_hz"] = np.asarray(float(tracking_step_hz), dtype=np.float64)
    np.savez_compressed(path, **out)
    logger.info(f"Mocap unified log saved to {path}")


def load_mocap_unified_log(
    path: str | Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None,
    np.ndarray | None, np.ndarray | None, np.ndarray | None,
]:
    """
    Load unified .npz. Returns (ts, pos, quat, cmd_ts, cmd_x, cmd_y, cmd_yaw, recon_ts, actual, predicted).
    Missing command/recon return None.
    """
    data = np.load(path)
    ts = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    pos = np.asarray(data["positions"], dtype=np.float64).reshape(-1, 3)
    quat = np.asarray(data["quaternions"], dtype=np.float64).reshape(-1, 4)
    cmd_ts = data["command_timestamps"] if "command_timestamps" in data else None
    cmd_x = data["command_x"] if "command_x" in data else None
    cmd_y = data["command_y"] if "command_y" in data else None
    cmd_yaw = data["command_yaw"] if "command_yaw" in data else None
    recon_ts = data["recon_timestamps"] if "recon_timestamps" in data else None
    actual = data["actual_target"] if "actual_target" in data else None
    predicted = data["predicted_target"] if "predicted_target" in data else None
    if cmd_ts is not None:
        cmd_ts = np.asarray(cmd_ts, dtype=np.float64).reshape(-1)
        cmd_x = np.asarray(cmd_x, dtype=np.float64).reshape(-1)
        cmd_y = np.asarray(cmd_y, dtype=np.float64).reshape(-1)
        cmd_yaw = np.asarray(cmd_yaw, dtype=np.float64).reshape(-1)
    if recon_ts is not None:
        recon_ts = np.asarray(recon_ts, dtype=np.float64).reshape(-1)
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
    return ts, pos, quat, cmd_ts, cmd_x, cmd_y, cmd_yaw, recon_ts, actual, predicted


# Legacy: raw-only log (no command/recon)
def save_mocap_raw_log(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    path: str | Path,
) -> None:
    """Save mocap raw data to .npz. Keys: timestamps (N,), positions (N,3), quaternions (N,4) xyzw."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=timestamps,
        positions=positions,
        quaternions=quaternions,
    )
    logger.info(f"Mocap raw data log saved to {path}")


def load_mocap_raw_log(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mocap raw data from .npz saved by save_mocap_raw_log or by plot_mocap_raw_on_exit.

    Returns
    -------
    timestamps : (N,) float64
    positions : (N, 3) float64
    quaternions : (N, 4) float64, xyzw
    """
    data = np.load(path)
    ts = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    pos = np.asarray(data["positions"], dtype=np.float64).reshape(-1, 3)
    quat = np.asarray(data["quaternions"], dtype=np.float64).reshape(-1, 4)
    return ts, pos, quat


def plot_from_mocap_raw_log(
    log_path: str | Path,
    save_dir: str | Path | None = None,
    show: bool = True,
    mocap_plot_hz: float = 50.0,
    checkpoint_path: str | Path | None = None,
) -> None:
    """
    Load mocap log (.npz): either unified (mocap_unified.npz) or legacy (mocap_raw_data.npz).
    If mocap_plot_hz > 0, resample mocap to that rate before computing velocity and plotting.
    Compute displacement/velocity, then plot raw, displacement (+ command + RMSE), velocity (+ command + RMSE + tracking return), reconstruction.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Mocap log not found: {log_path}")

    with np.load(log_path) as z:
        is_unified = "command_timestamps" in z.keys()
    if is_unified:
        ts, pos, quat, cmd_ts, cmd_x, cmd_y, cmd_yaw, recon_ts, actual_target, predicted_target = load_mocap_unified_log(log_path)
        metric_blocks = load_mocap_unified_metric_blocks(log_path)
    else:
        ts, pos, quat = load_mocap_raw_log(log_path)
        cmd_ts = cmd_x = cmd_y = cmd_yaw = None
        recon_ts = actual_target = predicted_target = None
        metric_blocks = {}

    plot_trim_head = int(metric_blocks.get("plot_trim_head") or PLOT_TRIM_STEPS)
    plot_trim_tail = int(metric_blocks.get("plot_trim_tail") or PLOT_TRIM_STEPS)
    tracking_step_hz = metric_blocks.get("tracking_step_hz")
    if tracking_step_hz is None:
        tracking_step_hz = mocap_plot_hz if mocap_plot_hz and mocap_plot_hz > 0 else 50.0

    if len(ts) == 0:
        logger.warning("Mocap log is empty, nothing to plot.")
        return

    if mocap_plot_hz > 0:
        ts, pos, quat = resample_mocap_to_hz(ts, pos, quat, mocap_plot_hz)

    metric_window = describe_metric_window(len(ts), plot_trim_head, plot_trim_tail, rate_hz=float(tracking_step_hz))
    for line in metric_window_log_lines(metric_window):
        logger.warning(line)

    tracking_summary = _compute_velocity_tracking_summary(
        ts,
        pos,
        quat,
        cmd_ts,
        cmd_x,
        cmd_y,
        cmd_yaw,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else metric_blocks.get("tracking_reward_checkpoint_path"),
        trim_head=plot_trim_head,
        trim_tail=plot_trim_tail,
        step_hz=float(tracking_step_hz),
    )
    for line in fall_criteria_log_lines(tracking_summary):
        logger.warning(line)

    # Trim using the same saved plot settings as the exit-time plots.
    if len(ts) > plot_trim_head + plot_trim_tail:
        ts, pos, quat = _trim_initial_final_steps(len(ts), plot_trim_head, ts, pos, quat, trim_tail=plot_trim_tail)
    if recon_ts is not None and len(recon_ts) > plot_trim_head + plot_trim_tail:
        recon_ts, actual_target, predicted_target = _trim_initial_final_steps(
            len(recon_ts), plot_trim_head, recon_ts, actual_target, predicted_target, trim_tail=plot_trim_tail
        )
    planner_ts = metric_blocks.get("planner_first_step_timestamps")
    planner_actual = metric_blocks.get("planner_first_step_actual_obs")
    planner_predicted = metric_blocks.get("planner_first_step_predicted_obs")
    if planner_ts is not None and len(planner_ts) > plot_trim_head + plot_trim_tail:
        planner_ts, planner_actual, planner_predicted = _trim_initial_final_steps(
            len(planner_ts),
            plot_trim_head,
            planner_ts,
            planner_actual,
            planner_predicted,
            trim_tail=plot_trim_tail,
        )
    idm_ts = metric_blocks.get("idm_timestamps")
    idm_actual_chunk = metric_blocks.get("idm_actual_action_chunk")
    idm_predicted_chunk = metric_blocks.get("idm_predicted_action_chunk")
    if idm_ts is not None and len(idm_ts) > plot_trim_head + plot_trim_tail:
        idm_ts, idm_actual_chunk, idm_predicted_chunk = _trim_initial_final_steps(
            len(idm_ts),
            plot_trim_head,
            idm_ts,
            idm_actual_chunk,
            idm_predicted_chunk,
            trim_tail=plot_trim_tail,
        )
    fdm_ts = metric_blocks.get("fdm_timestamps")
    fdm_actual_chunk = metric_blocks.get("fdm_actual_obs_chunk")
    fdm_predicted_chunk = metric_blocks.get("fdm_predicted_obs_chunk")
    if fdm_ts is not None and len(fdm_ts) > plot_trim_head + plot_trim_tail:
        fdm_ts, fdm_actual_chunk, fdm_predicted_chunk = _trim_initial_final_steps(
            len(fdm_ts),
            plot_trim_head,
            fdm_ts,
            fdm_actual_chunk,
            fdm_predicted_chunk,
            trim_tail=plot_trim_tail,
        )

    out_dir = Path(save_dir) if save_dir is not None else log_path.parent
    raw_path = str(out_dir / "mocap_raw.png")
    disp_path = str(out_dir / "mocap_displacement.png")
    vel_path = str(out_dir / "mocap_velocity.png")
    recon_path = str(out_dir / "mocap_reconstruction.png")

    t, base_pos_x, base_pos_y, base_angle, base_vx, base_vy, base_yaw = (
        compute_velocity_and_displacement_from_raw(ts, pos, quat)
    )
    plot_mocap_raw_history(ts, pos, quat, save_path=raw_path, show=show)
    _plot_displacement_tracking(
        t, base_pos_x, base_pos_y, base_angle,
        cmd_ts, cmd_x, cmd_y, cmd_yaw, ts, quat,
        save_path=disp_path, show=show,
    )
    _plot_velocity_tracking(
        t, base_vx, base_vy, base_yaw,
        cmd_ts, cmd_x, cmd_y, cmd_yaw,
        save_path=vel_path, show=show,
        t0_mocap=float(ts[0]) if len(ts) > 0 else None,
        mocap_ts=ts,
        tracking_summary=tracking_summary,
        metric_window=metric_window,
    )
    _plot_reconstruction(
        recon_ts,
        actual_target,
        predicted_target,
        save_path=recon_path,
        show=show,
        planner_first_step_ts=planner_ts,
        planner_first_step_actual=planner_actual,
        planner_first_step_predicted=planner_predicted,
        compact_term_order=metric_blocks.get("compact_term_order"),
        compact_term_dims=metric_blocks.get("compact_term_dims"),
        idm_ts=idm_ts,
        idm_actual_chunk=idm_actual_chunk,
        idm_predicted_chunk=idm_predicted_chunk,
        fdm_ts=fdm_ts,
        fdm_actual_chunk=fdm_actual_chunk,
        fdm_predicted_chunk=fdm_predicted_chunk,
    )


def _compute_velocity_tracking_summary(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    cmd_ts: np.ndarray | None,
    cmd_x: np.ndarray | None,
    cmd_y: np.ndarray | None,
    cmd_yaw: np.ndarray | None,
    *,
    checkpoint_path: str | None,
    trim_head: int = 0,
    trim_tail: int = 0,
    step_hz: float = 50.0,
) -> dict[str, Any] | None:
    if cmd_ts is None or cmd_x is None or cmd_y is None or cmd_yaw is None or len(cmd_ts) < 2:
        return None
    if not checkpoint_path:
        return {"available": False, "reason": "missing checkpoint"}
    try:
        from holosoma_inference.utils.compute_mocap_velocity_metrics import (
            ANG_VEL_TRACKING_RETURN_KEY,
            LIN_VEL_TRACKING_RETURN_KEY,
            TRACKING_TOTAL_RETURN_KEY,
            _compute_tracking_returns,
            load_tracking_reward_spec_from_checkpoint,
        )

        tracking_reward_spec = load_tracking_reward_spec_from_checkpoint(checkpoint_path)
        # Tilt-based fall detection (terrain-invariant): R[2,2] = 1-2*(qx²+qy²) < 0.5 ≈ 60° tilt
        #
        # NOT THE SAME CRITERION AS THE COLLECTOR'S. See STEP6_FALL_R22_MAX (top of this
        # module): step 6 fires at 60 deg of tilt and ignores the first 5% of the run, while
        # collection and the dual-mode safety guard fire at ~45.6 deg
        # (`collect_fall_projected_gravity_z_max` / `dual_mode_projected_gravity_z_min`, both
        # -0.7, and projected_gravity_z == -R[2,2] exactly). A rollout tilted between those
        # two angles is "fallen" to the collector and "still upright" here.
        quat_arr = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)  # xyzw
        r22 = 1.0 - 2.0 * (quat_arr[:, 0] ** 2 + quat_arr[:, 1] ** 2)
        skip = max(1, int(len(r22) * STEP6_FALL_WARMUP_FRACTION))
        fall_candidates = np.where(r22[skip:] < STEP6_FALL_R22_MAX)[0]
        fall_step = int(skip + int(fall_candidates[0])) if fall_candidates.size > 0 else None
        # REPORT ONLY: the collector's criterion evaluated on this same rollout. `fall_step`
        # above remains the only thing that truncates the exported metrics.
        collector_fall_step = _first_index_below(r22, COLLECTOR_FALL_R22_MAX)
        returns = _compute_tracking_returns(
            np.asarray(timestamps, dtype=np.float64),
            np.asarray(positions, dtype=np.float64),
            np.asarray(quaternions, dtype=np.float64),
            np.asarray(cmd_ts, dtype=np.float64),
            np.asarray(cmd_x, dtype=np.float64),
            np.asarray(cmd_y, dtype=np.float64),
            np.asarray(cmd_yaw, dtype=np.float64),
            tracking_reward_spec,
            trim_head=int(trim_head),
            trim_tail=int(trim_tail),
            fall_step=fall_step,
            step_hz=float(step_hz),
        )
        return {
            "available": True,
            "lin_vel_tracking_return": float(returns[LIN_VEL_TRACKING_RETURN_KEY]),
            "ang_vel_tracking_return": float(returns[ANG_VEL_TRACKING_RETURN_KEY]),
            "tracking_total_return": float(returns[TRACKING_TOTAL_RETURN_KEY]),
            "lin_vel_tracking_err": float(returns.get("lin_vel_tracking_err", float("nan"))),
            "ang_vel_tracking_err": float(returns.get("ang_vel_tracking_err", float("nan"))),
            "fell": fall_step is not None,
            # Report-only companions to `fell`: which step each criterion fired on
            # (untrimmed indices, so the two are directly comparable) and whether they
            # disagree. Consumed by fall_criteria_log_lines() and the figure's banner; never
            # by any metric.
            "fall_step_step6": fall_step,
            "collector_fall_step": collector_fall_step,
            "collector_fell": collector_fall_step is not None,
            "fall_criteria_disagree": (fall_step is None) != (collector_fall_step is None)
            or (fall_step is not None and collector_fall_step is not None and fall_step != collector_fall_step),
            "checkpoint_path": str(checkpoint_path),
            "plot_payload": {
                "step_axis": np.asarray(returns["_tracking_plot_step_axis"], dtype=np.float64),
                "time_axis_s": np.asarray(returns["_tracking_plot_time_axis_s"], dtype=np.float64),
                "lin_actual_xy": np.asarray(returns["_tracking_plot_lin_actual_xy"], dtype=np.float64),
                "ang_actual_z": np.asarray(returns["_tracking_plot_ang_actual_z"], dtype=np.float64),
                "lin_command_xy": np.asarray(returns["_tracking_plot_lin_command_xy"], dtype=np.float64),
                "ang_command_z": np.asarray(returns["_tracking_plot_ang_command_z"], dtype=np.float64),
                "lin_frame": str(returns["_tracking_plot_lin_frame"]),
                "nominal_dt": float(returns["_tracking_plot_nominal_dt"]),
                "step_hz": float(returns["_tracking_plot_step_hz"]),
                "fall_step": returns["_tracking_plot_fall_step"],
                "command_steps_total": int(returns["_tracking_plot_command_steps_total"]),
                "command_steps_after_trim": int(returns["_tracking_plot_command_steps_after_trim"]),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Tracking return summary unavailable for velocity plot: {exc}")
        return {"available": False, "reason": str(exc)}
