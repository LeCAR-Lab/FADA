#!/usr/bin/env python
"""Load mocap_unified.npz files and report tracking-return + planner/IDM metrics.

Diagnostic/report implementation only -- no standalone CLI entry point is shipped for
this module (see tools/check_entrypoints.py). `compute_velocity_errors_from_mocap_unified`
and `load_tracking_reward_spec_from_checkpoint` are used directly by other modules
(e.g. `plot_mocap_raw.py`); the table-printing helpers below remain importable too.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from holosoma_inference.utils.math.quat import quat_rotate_inverse, wxyz_to_xyzw, xyzw_to_wxyz
from holosoma_inference.utils.plot_inference_log import (
    _k_step_recon_loss_over_time,
    _moving_average,
    _remove_outliers,
)
from holosoma_inference.utils.plot_mocap_raw import (
    PLOT_TRIM_STEPS,
    _command_elapsed_sorted,
    _ensure_quat_continuous_wxyz,
    _quat_delta_to_ang_vel,
    _trim_initial_final_steps,
    _yaw_from_quat_xyzw,
    compute_fdm_obs_mse_stats,
    compute_idm_action_mse_stats,
    compute_planner_first_step_mse_stats,
    describe_metric_window,
    load_mocap_unified_log,
    load_mocap_unified_metric_blocks,
    resample_mocap_to_hz,
)


LIN_VEL_TRACKING_RETURN_KEY = "lin_vel_tracking_return"
ANG_VEL_TRACKING_RETURN_KEY = "ang_vel_tracking_return"
TRACKING_TOTAL_RETURN_KEY = "tracking_total_return"
DEFAULT_TRACKING_STEP_HZ = 50.0

_TRACKING_CHANNEL_SPECS: dict[str, dict[str, Any]] = {
    "linear": {
        "metric_key": LIN_VEL_TRACKING_RETURN_KEY,
        "candidate_names": ("tracking_lin_vel", "track_lin_vel_xy"),
        "recognized_funcs": {
            "holosoma.managers.reward.terms.locomotion:tracking_lin_vel": {
                "kernel": "tracking_sigma",
                "kernel_param": "tracking_sigma",
                "frame": "body_xy",
                "error_formula": "exp(-err / tracking_sigma)",
            },
            "holosoma.managers.reward.terms.locomotion_unitree:track_lin_vel_xy_yaw_frame_exp": {
                "kernel": "std_sq",
                "kernel_param": "std",
                "frame": "yaw_frame_xy",
                "error_formula": "exp(-err / std^2)",
            },
        },
    },
    "angular": {
        "metric_key": ANG_VEL_TRACKING_RETURN_KEY,
        "candidate_names": ("tracking_ang_vel", "track_ang_vel_z"),
        "recognized_funcs": {
            "holosoma.managers.reward.terms.locomotion:tracking_ang_vel": {
                "kernel": "tracking_sigma",
                "kernel_param": "tracking_sigma",
                "frame": "body_yaw_rate",
                "error_formula": "exp(-err / tracking_sigma)",
            },
            "holosoma.managers.reward.terms.locomotion_unitree:track_ang_vel_z_exp": {
                "kernel": "std_sq",
                "kernel_param": "std",
                "frame": "body_yaw_rate",
                "error_formula": "exp(-err / std^2)",
            },
        },
    },
}

def _get_container_value(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _reward_terms_mapping(reward_cfg: Any) -> dict[str, Any]:
    terms = _get_container_value(reward_cfg, "terms")
    if not isinstance(terms, dict):
        raise ValueError("Checkpoint reward config is missing a dict-like 'terms' field.")
    return terms


def _normalize_term_cfg(term_cfg: Any) -> dict[str, Any]:
    if isinstance(term_cfg, dict):
        params = term_cfg.get("params") or {}
        tags = term_cfg.get("tags") or []
        return {
            "func": term_cfg.get("func"),
            "params": params if isinstance(params, dict) else {},
            "weight": term_cfg.get("weight"),
            "tags": list(tags) if isinstance(tags, (list, tuple)) else [],
        }
    params = _get_container_value(term_cfg, "params") or {}
    tags = _get_container_value(term_cfg, "tags") or []
    return {
        "func": _get_container_value(term_cfg, "func"),
        "params": params if isinstance(params, dict) else {},
        "weight": _get_container_value(term_cfg, "weight"),
        "tags": list(tags) if isinstance(tags, (list, tuple)) else [],
    }


def _term_spec_from_cfg(term_name: str, normalized: dict[str, Any], func_spec: dict[str, Any]) -> dict[str, Any]:
    params = dict(normalized.get("params") or {})
    kernel_param_name = str(func_spec["kernel_param"])
    if kernel_param_name not in params:
        raise ValueError(
            f"Term '{term_name}' is missing parameter {kernel_param_name!r}."
        )
    weight = float(normalized.get("weight"))
    kernel_param_value = float(params[kernel_param_name])
    if func_spec["kernel"] == "tracking_sigma":
        error_denominator = kernel_param_value
    else:
        error_denominator = kernel_param_value**2
    return {
        "checkpoint_term_name": term_name,
        "func": str(normalized.get("func") or ""),
        "weight": weight,
        "params": params,
        "frame": func_spec["frame"],
        "kernel": func_spec["kernel"],
        "kernel_param_name": kernel_param_name,
        "kernel_param_value": kernel_param_value,
        "error_denominator": error_denominator,
        "error_formula": func_spec["error_formula"],
    }


def _load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        # Availability probe only -- the actual read goes through the shared
        # restricted-unpickler helper below, imported lazily as torch is (this module
        # is also used in plotting-only contexts). See
        # holosoma/utils/safe_torch_load.py.
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "Tracking return metrics require torch to read the checkpoint payload."
        ) from exc

    from holosoma.utils.safe_torch_load import load_checkpoint as safe_load_checkpoint

    checkpoint_path = Path(checkpoint_path).resolve()
    payload = safe_load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(payload)}")
    return payload


@lru_cache(maxsize=32)
def _load_tracking_reward_spec_cached(checkpoint_path_str: str) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path_str).resolve()
    payload = _load_checkpoint_payload(checkpoint_path)
    experiment_config = payload.get("experiment_config")
    if experiment_config is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing 'experiment_config'; cannot derive tracking returns."
        )
    reward_cfg = _get_container_value(experiment_config, "reward")
    if reward_cfg is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing 'experiment_config.reward'; cannot derive tracking returns."
        )
    reward_terms = _reward_terms_mapping(reward_cfg)
    only_positive_rewards = _get_container_value(reward_cfg, "only_positive_rewards")

    spec: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "source": "experiment_config.reward",
        "only_positive_rewards": (
            bool(only_positive_rewards) if only_positive_rewards is not None else None
        ),
        "terms": {},
        "tracking_total_return_note": (
            "tracking_total_return includes only weighted linear and angular tracking terms, "
            "not the full training reward."
        ),
        "weight_dt_semantics": "per-step term return = raw_tracking_reward * weight * dt",
    }

    for channel, channel_spec in _TRACKING_CHANNEL_SPECS.items():
        candidate_names = channel_spec["candidate_names"]
        recognized_funcs = channel_spec["recognized_funcs"]
        matched_name: str | None = None
        matched_cfg: dict[str, Any] | None = None
        for term_name in candidate_names:
            raw = reward_terms.get(term_name)
            if raw is None:
                continue
            normalized = _normalize_term_cfg(raw)
            func_name = str(normalized.get("func") or "")
            if func_name not in recognized_funcs:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} term '{term_name}' uses unsupported tracking func "
                    f"{func_name!r}."
                )
            matched_name = str(term_name)
            matched_cfg = normalized
            break
        if matched_cfg is None:
            fallback_matches: list[tuple[str, dict[str, Any]]] = []
            for term_name, raw in reward_terms.items():
                normalized = _normalize_term_cfg(raw)
                func_name = str(normalized.get("func") or "")
                if func_name in recognized_funcs:
                    fallback_matches.append((str(term_name), normalized))
            if len(fallback_matches) == 1:
                matched_name, matched_cfg = fallback_matches[0]
            elif not fallback_matches:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} reward config is missing the {channel} tracking term. "
                    f"Tried names {candidate_names}."
                )
            else:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} has ambiguous {channel} tracking terms: "
                    f"{[name for name, _ in fallback_matches]}"
                )

        assert matched_name is not None
        func_name = str(matched_cfg.get("func") or "")
        func_spec = recognized_funcs[func_name]
        term_spec = _term_spec_from_cfg(matched_name, matched_cfg, func_spec)
        spec["terms"][channel_spec["metric_key"]] = {
            "channel": channel,
            "metric_key": channel_spec["metric_key"],
            **term_spec,
        }

    return spec


def load_tracking_reward_spec_from_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    return copy.deepcopy(_load_tracking_reward_spec_cached(str(Path(checkpoint_path).resolve())))


def _yaw_only_quat_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    yaw = _yaw_from_quat_xyzw(np.asarray(quat_xyzw, dtype=np.float64).reshape(-1, 4))
    out = np.zeros((len(yaw), 4), dtype=np.float64)
    out[:, 0] = np.cos(yaw / 2.0)
    out[:, 3] = np.sin(yaw / 2.0)
    return out


def _tracking_step_dt(step_hz: float | None) -> float:
    hz = float(step_hz) if step_hz is not None and float(step_hz) > 0.0 else DEFAULT_TRACKING_STEP_HZ
    return 1.0 / hz


def _relative_time_axes(source_ts: np.ndarray, target_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_ts, dtype=np.float64).reshape(-1)
    target = np.asarray(target_ts, dtype=np.float64).reshape(-1)
    if len(source) == 0 or len(target) == 0:
        return source, target
    t0_source = float(source[0])
    t0_target = float(target[0])
    if (t0_source > 1e6) == (t0_target > 1e6):
        t0 = min(t0_source, t0_target)
        return source - t0, target - t0
    return source - t0_source, target - t0_target


def _prepare_interp_series(source_ts: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_ts, dtype=np.float64).reshape(-1)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        valid = np.isfinite(source) & np.isfinite(arr)
    else:
        valid = np.isfinite(source) & np.all(np.isfinite(arr), axis=1)
    source = source[valid]
    arr = arr[valid]
    if len(source) == 0:
        return source, arr
    order = np.argsort(source, kind="stable")
    source = source[order]
    arr = arr[order]
    keep = np.ones(len(source), dtype=bool)
    if len(source) > 1:
        keep[1:] = np.diff(source) > 1e-12
    return source[keep], arr[keep]


def _interpolate_series_to_target_times(
    source_ts: np.ndarray,
    values: np.ndarray,
    target_ts: np.ndarray,
) -> np.ndarray:
    source_rel, target_rel = _relative_time_axes(source_ts, target_ts)
    source_rel, arr = _prepare_interp_series(source_rel, values)
    n_target = len(np.asarray(target_ts, dtype=np.float64).reshape(-1))
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim == 1:
        out = np.full((n_target,), np.nan, dtype=np.float64)
    else:
        out = np.full((n_target, values_arr.shape[1]), np.nan, dtype=np.float64)
    if len(source_rel) < 2:
        return out
    if values_arr.ndim == 1:
        return np.interp(target_rel, source_rel, arr, left=np.nan, right=np.nan)
    for dim in range(arr.shape[1]):
        out[:, dim] = np.interp(target_rel, source_rel, arr[:, dim], left=np.nan, right=np.nan)
    return out


def _interpolate_quaternions_to_target_times(
    source_ts: np.ndarray,
    quaternions_xyzw: np.ndarray,
    target_ts: np.ndarray,
) -> np.ndarray:
    quats_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64).reshape(-1, 4)
    source_rel, target_rel = _relative_time_axes(source_ts, target_ts)
    source_rel, quats_xyzw = _prepare_interp_series(source_rel, quats_xyzw)
    out = np.full((len(np.asarray(target_ts, dtype=np.float64).reshape(-1)), 4), np.nan, dtype=np.float64)
    if len(source_rel) < 2:
        return out
    quats_wxyz = _ensure_quat_continuous_wxyz(xyzw_to_wxyz(quats_xyzw))
    quats_xyzw = wxyz_to_xyzw(quats_wxyz)
    for dim in range(4):
        out[:, dim] = np.interp(target_rel, source_rel, quats_xyzw[:, dim], left=np.nan, right=np.nan)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    valid = np.isfinite(out).all(axis=1) & (norms[:, 0] > 1e-12)
    out[valid] = out[valid] / norms[valid]
    out[~valid] = np.nan
    return out


def interpolate_mocap_pose_to_command_steps(
    mocap_ts: np.ndarray,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    command_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _interpolate_series_to_target_times(mocap_ts, positions, command_timestamps),
        _interpolate_quaternions_to_target_times(mocap_ts, quaternions_xyzw, command_timestamps),
    )


def step_aligned_height_series_from_mocap(
    mocap_ts: np.ndarray,
    positions: np.ndarray,
    command_timestamps: np.ndarray | None,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if (
        command_timestamps is None
        or len(np.asarray(command_timestamps, dtype=np.float64).reshape(-1)) < 2
        or len(np.asarray(mocap_ts, dtype=np.float64).reshape(-1)) < 2
    ):
        return np.asarray(pos[:, 2], dtype=np.float64)
    z_interp = _interpolate_series_to_target_times(
        np.asarray(mocap_ts, dtype=np.float64),
        np.asarray(pos[:, 2], dtype=np.float64),
        np.asarray(command_timestamps, dtype=np.float64),
    )
    if np.any(np.isfinite(z_interp)):
        return np.asarray(z_interp, dtype=np.float64)
    return np.asarray(pos[:, 2], dtype=np.float64)


def _map_step_index_after_trim(
    step_index: int | None,
    *,
    original_n: int,
    trim_head: int,
    trim_tail: int,
) -> int | None:
    if step_index is None:
        return None
    idx = int(step_index)
    if not (0 <= idx < original_n):
        return None
    head = max(int(trim_head), 0)
    tail = max(int(trim_tail), 0)
    if head + tail <= 0 or original_n <= head + tail:
        return idx
    start = head
    end = original_n - tail if tail > 0 else original_n
    if idx < start:
        return 0
    if idx >= end:
        return None
    return idx - start


def _compute_tracking_velocity_series(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> dict[str, np.ndarray]:
    n = int(len(timestamps))
    body_lin_xy = np.zeros((n, 2), dtype=np.float64)
    yaw_lin_xy = np.zeros((n, 2), dtype=np.float64)
    body_ang_z = np.zeros((n,), dtype=np.float64)
    step_dt = np.zeros((n,), dtype=np.float64)
    if n <= 1:
        return {
            "body_lin_xy": body_lin_xy,
            "yaw_lin_xy": yaw_lin_xy,
            "body_ang_z": body_ang_z,
            "step_dt": step_dt,
        }

    quat_wxyz = xyzw_to_wxyz(np.asarray(quaternions_xyzw, dtype=np.float64).reshape(n, 4))
    quat_wxyz = _ensure_quat_continuous_wxyz(quat_wxyz)
    yaw_quat_wxyz = _yaw_only_quat_wxyz(quaternions_xyzw)

    for idx in range(1, n):
        dt = float(timestamps[idx] - timestamps[idx - 1])
        if dt <= 1e-9:
            continue
        step_dt[idx] = dt
        lin_vel_world = (positions[idx] - positions[idx - 1]) / dt
        ang_vel_world = _quat_delta_to_ang_vel(quat_wxyz[idx - 1], quat_wxyz[idx], dt)
        body_lin = quat_rotate_inverse(quat_wxyz[idx : idx + 1], lin_vel_world.reshape(1, 3))[0]
        yaw_lin = quat_rotate_inverse(yaw_quat_wxyz[idx : idx + 1], lin_vel_world.reshape(1, 3))[0]
        body_ang = quat_rotate_inverse(quat_wxyz[idx : idx + 1], ang_vel_world.reshape(1, 3))[0]
        body_lin_xy[idx] = body_lin[:2]
        yaw_lin_xy[idx] = yaw_lin[:2]
        body_ang_z[idx] = body_ang[2]
    return {
        "body_lin_xy": body_lin_xy,
        "yaw_lin_xy": yaw_lin_xy,
        "body_ang_z": body_ang_z,
        "step_dt": step_dt,
    }


def _tracking_term_return(
    actual: np.ndarray,
    command: np.ndarray,
    step_dt: np.ndarray,
    *,
    term_spec: dict[str, Any],
    fall_step: int | None,
) -> tuple[float, np.ndarray]:
    if actual.ndim == 1:
        error = np.square(command - actual)
        finite_mask = np.isfinite(actual) & np.isfinite(command)
    else:
        error = np.sum(np.square(command - actual), axis=1)
        finite_mask = np.all(np.isfinite(actual), axis=1) & np.all(np.isfinite(command), axis=1)
    valid = finite_mask & np.isfinite(step_dt) & (step_dt >= 0.0)
    had_overlap = bool(np.any(valid))
    raw_reward = np.full(error.shape, np.nan, dtype=np.float64)
    if had_overlap:
        raw_reward[valid] = np.exp(-error[valid] / float(term_spec["error_denominator"]))
    if fall_step is not None:
        raw_reward[int(fall_step) :] = 0.0
    scaled = np.zeros(error.shape, dtype=np.float64)
    valid_scaled = valid & np.isfinite(raw_reward)
    scaled[valid_scaled] = raw_reward[valid_scaled] * float(term_spec["weight"]) * step_dt[valid_scaled]
    if not had_overlap:
        return float("nan"), scaled
    return float(np.sum(scaled)), scaled


def _compute_tracking_returns(
    metric_timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    cmd_ts: np.ndarray,
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    cmd_yaw: np.ndarray,
    tracking_reward_spec: dict[str, Any],
    *,
    trim_head: int,
    trim_tail: int,
    fall_step: int | None,
    step_hz: float | None,
) -> dict[str, Any]:
    cmd_ts_full = np.asarray(cmd_ts, dtype=np.float64).reshape(-1)
    cmd_x_full = np.asarray(cmd_x, dtype=np.float64).reshape(-1)
    cmd_y_full = np.asarray(cmd_y, dtype=np.float64).reshape(-1)
    cmd_yaw_full = np.asarray(cmd_yaw, dtype=np.float64).reshape(-1)
    pose_pos, pose_quat = interpolate_mocap_pose_to_command_steps(
        np.asarray(metric_timestamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(quaternions_xyzw, dtype=np.float64),
        cmd_ts_full,
    )
    nominal_dt = _tracking_step_dt(step_hz)
    synthetic_step_ts = np.arange(len(cmd_ts_full), dtype=np.float64) * nominal_dt
    velocity_series = _compute_tracking_velocity_series(synthetic_step_ts, pose_pos, pose_quat)
    body_lin_xy = velocity_series["body_lin_xy"]
    yaw_lin_xy = velocity_series["yaw_lin_xy"]
    body_ang_z = velocity_series["body_ang_z"]
    step_dt = velocity_series["step_dt"]

    n_steps = len(cmd_ts_full)
    if trim_head + trim_tail > 0 and n_steps > trim_head + trim_tail:
        (
            body_lin_xy,
            yaw_lin_xy,
            body_ang_z,
            step_dt,
            cmd_x_full,
            cmd_y_full,
            cmd_yaw_full,
        ) = _trim_initial_final_steps(
            n_steps,
            trim_head,
            body_lin_xy,
            yaw_lin_xy,
            body_ang_z,
            step_dt,
            cmd_x_full,
            cmd_y_full,
            cmd_yaw_full,
            trim_tail=trim_tail,
        )
    trimmed_fall_step = _map_step_index_after_trim(
        fall_step,
        original_n=n_steps,
        trim_head=trim_head,
        trim_tail=trim_tail,
    )

    lin_spec = tracking_reward_spec["terms"][LIN_VEL_TRACKING_RETURN_KEY]
    ang_spec = tracking_reward_spec["terms"][ANG_VEL_TRACKING_RETURN_KEY]

    lin_actual = body_lin_xy if str(lin_spec["frame"]) == "body_xy" else yaw_lin_xy
    lin_command = np.column_stack([cmd_x_full, cmd_y_full]).astype(np.float64, copy=False)
    plot_step_axis = np.arange(len(cmd_x_full), dtype=np.float64)
    plot_time_axis_s = plot_step_axis * nominal_dt
    if "components" in lin_spec:
        lin_return = 0.0
        lin_scaled = np.zeros((len(cmd_x_full),), dtype=np.float64)
        had_valid_component = False
        for component_spec in lin_spec["components"]:
            axis = int(component_spec["axis"])
            component_return, component_scaled = _tracking_term_return(
                lin_actual[:, axis],
                lin_command[:, axis],
                step_dt,
                term_spec=component_spec,
                fall_step=trimmed_fall_step,
            )
            if not np.isnan(component_return):
                had_valid_component = True
                lin_return += float(component_return)
            lin_scaled += np.nan_to_num(component_scaled, nan=0.0)
        if not had_valid_component:
            lin_return = float("nan")
    else:
        lin_return, lin_scaled = _tracking_term_return(
            lin_actual,
            lin_command,
            step_dt,
            term_spec=lin_spec,
            fall_step=trimmed_fall_step,
        )
    ang_return, ang_scaled = _tracking_term_return(
        body_ang_z,
        np.asarray(cmd_yaw_full, dtype=np.float64),
        step_dt,
        term_spec=ang_spec,
        fall_step=trimmed_fall_step,
    )
    tracking_total_return = (
        float("nan")
        if np.isnan(lin_return) or np.isnan(ang_return)
        else float(lin_return + ang_return)
    )
    _eff_steps = trimmed_fall_step if trimmed_fall_step is not None else int(len(cmd_x_full))
    tracking_total_per_step = (
        float(tracking_total_return) / _eff_steps
        if _eff_steps > 0 and not np.isnan(tracking_total_return)
        else float("nan")
    )
    # Direct absolute tracking error in physical units: per-step RMSE of ||cmd-actual||.
    # lin_err = sqrt( mean_t( (cmd_x-act_x)^2 + (cmd_y-act_y)^2 ) )         [m/s]
    # ang_err = sqrt( mean_t( (cmd_yaw_rate - body_yaw_rate)^2 ) )          [rad/s]
    # Computed in the same body/yaw frame the reward uses; post-trim; excludes
    # steps at/after fall_step.  This is the RMSE counterpart to the reward
    # kernel exp(-err^2 / denom), so reward-based and rmse-based deltas align.
    lin_actual_arr = np.asarray(lin_actual, dtype=np.float64)
    lin_command_arr = np.asarray(lin_command, dtype=np.float64)
    ang_actual_arr = np.asarray(body_ang_z, dtype=np.float64).reshape(-1)
    ang_command_arr = np.asarray(cmd_yaw_full, dtype=np.float64).reshape(-1)
    lin_sq_per_step = np.sum(
        np.square(lin_command_arr - lin_actual_arr), axis=1
    )  # ||v_err||^2 (m^2/s^2)
    ang_sq_per_step = np.square(ang_command_arr - ang_actual_arr)  # rad^2/s^2
    def _rmse_pre_fall(sq_arr: np.ndarray) -> float:
        end = trimmed_fall_step if trimmed_fall_step is not None else len(sq_arr)
        sub = sq_arr[:end]
        finite = sub[np.isfinite(sub)]
        if finite.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(finite)))
    lin_err_mean = _rmse_pre_fall(lin_sq_per_step)
    ang_err_mean = _rmse_pre_fall(ang_sq_per_step)
    return {
        LIN_VEL_TRACKING_RETURN_KEY: lin_return,
        ANG_VEL_TRACKING_RETURN_KEY: ang_return,
        TRACKING_TOTAL_RETURN_KEY: tracking_total_return,
        "tracking_total_per_step": tracking_total_per_step,
        "lin_vel_tracking_err": lin_err_mean,
        "ang_vel_tracking_err": ang_err_mean,
        "_tracking_return_step_dt": step_dt,
        "_tracking_return_nominal_dt": nominal_dt,
        "_tracking_return_lin_scaled": lin_scaled,
        "_tracking_return_ang_scaled": ang_scaled,
        "_tracking_plot_step_axis": plot_step_axis,
        "_tracking_plot_time_axis_s": plot_time_axis_s,
        "_tracking_plot_lin_actual_xy": np.asarray(lin_actual, dtype=np.float64),
        "_tracking_plot_ang_actual_z": np.asarray(body_ang_z, dtype=np.float64),
        "_tracking_plot_lin_command_xy": np.asarray(lin_command, dtype=np.float64),
        "_tracking_plot_ang_command_z": np.asarray(cmd_yaw_full, dtype=np.float64),
        "_tracking_plot_lin_frame": str(lin_spec["frame"]),
        "_tracking_plot_nominal_dt": nominal_dt,
        "_tracking_plot_step_hz": (1.0 / nominal_dt) if nominal_dt > 0.0 else float("nan"),
        "_tracking_plot_fall_step": trimmed_fall_step,
        "_tracking_plot_command_steps_total": int(n_steps),
        "_tracking_plot_command_steps_after_trim": int(len(cmd_x_full)),
    }


def _interpolate_command_to_mocap_times(
    mocap_ts: np.ndarray,
    cmd_ts: np.ndarray,
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    cmd_yaw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate command to mocap timestamps.

    Outside the command history overlap window we return ``NaN`` instead of edge-
    extrapolating the first/last command value. That avoids inflating tracking
    error when mocap starts earlier than command recording or lingers slightly
    after the final command sample.
    """
    cmd_ts_a = np.asarray(cmd_ts, dtype=np.float64).reshape(-1)
    t0_cmd = float(cmd_ts_a[0])
    t0_mocap_val = float(mocap_ts[0])
    if (t0_mocap_val > 1e6) == (t0_cmd > 1e6):
        t0_common = min(t0_mocap_val, t0_cmd)
        t_plot = (mocap_ts - t0_common).astype(np.float64)
        cmd_t_rel, cx, cy, cyaw = _command_elapsed_sorted(
            cmd_ts, cmd_x, cmd_y, cmd_yaw, t0_reference=t0_common
        )
    else:
        t_plot = (mocap_ts - t0_mocap_val).astype(np.float64)
        cmd_t_rel, cx, cy, cyaw = _command_elapsed_sorted(cmd_ts, cmd_x, cmd_y, cmd_yaw)
    if len(cmd_t_rel) < 2:
        n = len(mocap_ts)
        nan = np.full(n, np.nan, dtype=np.float64)
        return t_plot, nan.copy(), nan.copy(), nan.copy()
    cmd_x_i = np.interp(t_plot, cmd_t_rel, cx, left=np.nan, right=np.nan)
    cmd_y_i = np.interp(t_plot, cmd_t_rel, cy, left=np.nan, right=np.nan)
    cmd_yaw_i = np.interp(t_plot, cmd_t_rel, cyaw, left=np.nan, right=np.nan)
    return t_plot, cmd_x_i, cmd_y_i, cmd_yaw_i


def _compute_obs_prediction_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    trim_head: int,
    trim_tail: int,
) -> float:
    """K-step observation prediction MSE (same as plot_inference_log)."""
    recon_loss, recon_counts = _k_step_recon_loss_over_time(actual, predicted)
    n = len(recon_loss)
    head = max(trim_head, 0)
    tail = max(trim_tail, 0)
    if head + tail > 0 and n > head + tail:
        end = n - tail if tail > 0 else n
        recon_loss = recon_loss[head:end]
        recon_counts = recon_counts[head:end]
    valid = recon_counts > 0
    if not np.any(valid):
        return float("nan")
    return float(np.sum(recon_loss[valid] * recon_counts[valid]) / np.sum(recon_counts[valid]))


def _trim_optional_time_series(
    timestamps: np.ndarray | None,
    *arrays: np.ndarray | None,
    trim_head: int,
    trim_tail: int,
) -> tuple[np.ndarray | None, ...]:
    if trim_head <= 0 and trim_tail <= 0:
        return (timestamps, *arrays)
    if timestamps is not None:
        n = len(np.asarray(timestamps).reshape(-1))
    else:
        first = next((arr for arr in arrays if arr is not None), None)
        if first is None:
            return (timestamps, *arrays)
        n = int(np.asarray(first).shape[0])
    if n <= trim_head + trim_tail:
        return (timestamps, *arrays)

    normalized: list[np.ndarray] = []
    for item in (timestamps, *arrays):
        if item is None:
            normalized.append(np.array([], dtype=np.float64))
        else:
            normalized.append(np.asarray(item))
    trimmed = _trim_initial_final_steps(
        n,
        trim_head,
        *normalized,
        trim_tail=trim_tail,
    )
    out: list[np.ndarray | None] = []
    for original, trimmed_item in zip((timestamps, *arrays), trimmed, strict=False):
        out.append(None if original is None else trimmed_item)
    return tuple(out)


def _empty_metrics() -> dict[str, Any]:
    return {
        LIN_VEL_TRACKING_RETURN_KEY: float("nan"),
        ANG_VEL_TRACKING_RETURN_KEY: float("nan"),
        TRACKING_TOTAL_RETURN_KEY: float("nan"),
        "obs_pred_mse": float("nan"),
        "idm_action_mse": float("nan"),
        "idm_action_mse_per_horizon": np.array([], dtype=np.float64),
        "fdm_obs_mse": float("nan"),
        "fdm_obs_mse_per_horizon": np.array([], dtype=np.float64),
        "planner_first_step_obs_mse": float("nan"),
        "planner_first_step_obs_mse_per_term": {},
    }


def compute_velocity_errors_from_mocap_unified(
    npz_path: str | Path,
    mocap_plot_hz: float = 50.0,
    trim_steps: int | None = None,
    trim_head: int | None = None,
    trim_tail: int | None = None,
    smooth_window: int = 1,
    max_linear_vel: float = 1.0,
    max_angular_vel: float = 2.0,
    filter_outliers: bool = True,
    checkpoint_path: str | Path | None = None,
    fall_step: int | None = None,
) -> dict[str, Any]:
    """Compute tracking-return, planner first-step, and IDM metrics from one unified log."""
    del smooth_window, max_linear_vel, max_angular_vel, filter_outliers
    npz_path = Path(npz_path)
    metrics = _empty_metrics()

    if trim_head is None and trim_tail is None:
        _sym = trim_steps if trim_steps is not None else PLOT_TRIM_STEPS
        trim_head = _sym
        trim_tail = _sym
    else:
        trim_head = trim_head if trim_head is not None else 0
        trim_tail = trim_tail if trim_tail is not None else 0

    ts, pos, quat, cmd_ts, cmd_x, cmd_y, cmd_yaw, _recon_ts, actual_target, predicted_target = load_mocap_unified_log(
        npz_path
    )
    metric_blocks = load_mocap_unified_metric_blocks(npz_path)
    # Provenance, always: every metric below is aggregated over this window, and with the
    # released defaults (trim_head = trim_tail = PLOT_TRIM_STEPS = 0) that window starts at
    # the first recorded step -- i.e. it includes the gantry transient. Recording it in the
    # returned dict means no caller can present these numbers as "clean" by omission.
    metrics["metric_window"] = describe_metric_window(
        len(ts),
        int(trim_head),
        int(trim_tail),
        rate_hz=mocap_plot_hz if mocap_plot_hz and mocap_plot_hz > 0 else DEFAULT_TRACKING_STEP_HZ,
    )
    if len(ts) > 0 and cmd_ts is not None and len(cmd_ts) >= 2:
        if checkpoint_path is None:
            raise ValueError(
                f"checkpoint_path is required to compute tracking returns for {npz_path}."
            )
        tracking_reward_spec = load_tracking_reward_spec_from_checkpoint(checkpoint_path)
        tracking_returns = _compute_tracking_returns(
            ts,
            pos,
            quat,
            cmd_ts,
            cmd_x,
            cmd_y,
            cmd_yaw,
            tracking_reward_spec,
            trim_head=int(trim_head),
            trim_tail=int(trim_tail),
            fall_step=fall_step,
            step_hz=mocap_plot_hz if mocap_plot_hz and mocap_plot_hz > 0 else DEFAULT_TRACKING_STEP_HZ,
        )
        metrics[LIN_VEL_TRACKING_RETURN_KEY] = float(
            tracking_returns[LIN_VEL_TRACKING_RETURN_KEY]
        )
        metrics[ANG_VEL_TRACKING_RETURN_KEY] = float(
            tracking_returns[ANG_VEL_TRACKING_RETURN_KEY]
        )
        metrics[TRACKING_TOTAL_RETURN_KEY] = float(
            tracking_returns[TRACKING_TOTAL_RETURN_KEY]
        )
        metrics["tracking_total_per_step"] = float(
            tracking_returns["tracking_total_per_step"]
        )
        metrics["lin_vel_tracking_err"] = float(
            tracking_returns["lin_vel_tracking_err"]
        )
        metrics["ang_vel_tracking_err"] = float(
            tracking_returns["ang_vel_tracking_err"]
        )

    if actual_target is not None and predicted_target is not None:
        metrics["obs_pred_mse"] = _compute_obs_prediction_error(actual_target, predicted_target, trim_head, trim_tail)

    planner_ts, planner_actual, planner_predicted = _trim_optional_time_series(
        metric_blocks.get("planner_first_step_timestamps"),
        metric_blocks.get("planner_first_step_actual_obs"),
        metric_blocks.get("planner_first_step_predicted_obs"),
        trim_head=trim_head,
        trim_tail=trim_tail,
    )
    idm_stats = compute_idm_action_mse_stats(
        *(_trim_optional_time_series(
            metric_blocks.get("idm_timestamps"),
            metric_blocks.get("idm_actual_action_chunk"),
            metric_blocks.get("idm_predicted_action_chunk"),
            trim_head=trim_head,
            trim_tail=trim_tail,
        )[1:])
    )
    if np.asarray(idm_stats["step_mse"]).size > 0:
        metrics["idm_action_mse"] = float(idm_stats["overall_mse"])
        metrics["idm_action_mse_per_horizon"] = np.asarray(idm_stats["per_horizon_mse"], dtype=np.float64)
    fdm_stats = compute_fdm_obs_mse_stats(
        *(_trim_optional_time_series(
            metric_blocks.get("fdm_timestamps"),
            metric_blocks.get("fdm_actual_obs_chunk"),
            metric_blocks.get("fdm_predicted_obs_chunk"),
            trim_head=trim_head,
            trim_tail=trim_tail,
        )[1:])
    )
    if np.asarray(fdm_stats["step_mse"]).size > 0:
        metrics["fdm_obs_mse"] = float(fdm_stats["overall_mse"])
        metrics["fdm_obs_mse_per_horizon"] = np.asarray(fdm_stats["per_horizon_mse"], dtype=np.float64)

    planner_stats = compute_planner_first_step_mse_stats(
        planner_actual,
        planner_predicted,
        compact_term_order=metric_blocks.get("compact_term_order"),
        compact_term_dims=metric_blocks.get("compact_term_dims"),
    )
    if np.asarray(planner_stats["step_mse"]).size > 0:
        metrics["planner_first_step_obs_mse"] = float(planner_stats["overall_mse"])
        metrics["planner_first_step_obs_mse_per_term"] = {
            str(key): float(value) for key, value in planner_stats["per_term_mse"].items()
        }

    return metrics


def _compute_group_metrics(
    npz_paths: list[str],
    mocap_hz: float,
    trim_steps: int | None,
    smooth_window: int,
    max_linear_vel: float = 1.0,
    max_angular_vel: float = 2.0,
    filter_outliers: bool = True,
    trim_head: int | None = None,
    trim_tail: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Compute metrics for a list of unified logs."""
    results: list[tuple[str, dict[str, Any]]] = []
    for p in npz_paths:
        path = Path(p)
        if not path.exists():
            print(f"Skip (not found): {path}")
            results.append((path.name, _empty_metrics()))
            continue
        label = path.parent.parent.name if path.parent.parent.name else path.parent.name or path.stem
        metrics = compute_velocity_errors_from_mocap_unified(
            path,
            mocap_plot_hz=mocap_hz,
            trim_steps=trim_steps,
            smooth_window=smooth_window,
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            filter_outliers=filter_outliers,
            trim_head=trim_head,
            trim_tail=trim_tail,
            checkpoint_path=checkpoint_path,
        )
        results.append((label, metrics))
    return results


def _fmt(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}" if not np.isnan(value) else "N/A"


def _scalar_stats(metrics: list[tuple[str, dict[str, Any]]], key: str) -> tuple[float, float]:
    values = [float(item[1].get(key, float("nan"))) for item in metrics if not np.isnan(float(item[1].get(key, float("nan"))))]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def _mean_vector_metric(metrics: list[tuple[str, dict[str, Any]]], key: str) -> np.ndarray:
    arrays = []
    max_len = 0
    for _, item in metrics:
        arr = np.asarray(item.get(key, np.array([])), dtype=np.float64).reshape(-1)
        if arr.size <= 0:
            continue
        arrays.append(arr)
        max_len = max(max_len, arr.size)
    if not arrays or max_len <= 0:
        return np.array([], dtype=np.float64)
    stacked = np.full((len(arrays), max_len), np.nan, dtype=np.float64)
    for idx, arr in enumerate(arrays):
        stacked[idx, : arr.size] = arr
    return np.nanmean(stacked, axis=0)


def _mean_dict_metric(metrics: list[tuple[str, dict[str, Any]]], key: str) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _, item in metrics:
        values = item.get(key, {})
        if not isinstance(values, dict):
            continue
        for sub_key, value in values.items():
            if np.isnan(float(value)):
                continue
            sums[str(sub_key)] = sums.get(str(sub_key), 0.0) + float(value)
            counts[str(sub_key)] = counts.get(str(sub_key), 0) + 1
    return {
        sub_key: sums[sub_key] / counts[sub_key]
        for sub_key in sorted(sums)
        if counts.get(sub_key, 0) > 0
    }


def _print_idm_horizon_summary(metrics: list[tuple[str, dict[str, Any]]], header: str) -> None:
    mean_vec = _mean_vector_metric(metrics, "idm_action_mse_per_horizon")
    if mean_vec.size <= 0:
        return
    print(f"\n{header}")
    for idx, value in enumerate(mean_vec):
        print(f"  step {idx + 1:<2}: {_fmt(float(value), 6)}")


def _print_fdm_horizon_summary(metrics: list[tuple[str, dict[str, Any]]], header: str) -> None:
    mean_vec = _mean_vector_metric(metrics, "fdm_obs_mse_per_horizon")
    if mean_vec.size <= 0:
        return
    print(f"\n{header}")
    for idx, value in enumerate(mean_vec):
        print(f"  step {idx + 1:<2}: {_fmt(float(value), 6)}")


def _print_planner_term_summary(metrics: list[tuple[str, dict[str, Any]]], header: str) -> None:
    mean_terms = _mean_dict_metric(metrics, "planner_first_step_obs_mse_per_term")
    if not mean_terms:
        return
    print(f"\n{header}")
    for term, value in mean_terms.items():
        print(f"  {term:<18} {_fmt(float(value), 6)}")


def _print_single_table(results: list[tuple[str, dict[str, Any]]], csv: bool = False):
    """Print metrics for a single group."""
    if not results:
        return
    if csv:
        print(
            "run,lin_vel_tracking_return,ang_vel_tracking_return,tracking_total_return,obs_pred_error_mse,idm_action_mse,fdm_obs_mse,planner_first_step_obs_mse"
        )
        for label, metrics in results:
            print(
                ",".join(
                    [
                        label,
                        _fmt(float(metrics[LIN_VEL_TRACKING_RETURN_KEY]), 6).replace("N/A", ""),
                        _fmt(float(metrics[ANG_VEL_TRACKING_RETURN_KEY]), 6).replace("N/A", ""),
                        _fmt(float(metrics[TRACKING_TOTAL_RETURN_KEY]), 6).replace("N/A", ""),
                        _fmt(float(metrics["obs_pred_mse"]), 6).replace("N/A", ""),
                        _fmt(float(metrics["idm_action_mse"]), 6).replace("N/A", ""),
                        _fmt(float(metrics["fdm_obs_mse"]), 6).replace("N/A", ""),
                        _fmt(float(metrics["planner_first_step_obs_mse"]), 6).replace("N/A", ""),
                    ]
                )
            )
        return

    col_w = 13
    header = (
        f"{'Run':<36} {'lin_track':>{col_w}} {'ang_track':>{col_w}} {'track_total':>{col_w}} "
        f"{'obs_pred_err':>{col_w}} {'idm_mse':>{col_w}} {'fdm_mse':>{col_w}} {'planner1_mse':>{col_w}}"
    )
    sep_len = 36 + 7 * (col_w + 1)
    print(header)
    print("-" * sep_len)
    for label, metrics in results:
        print(
            f"{label:<36} "
            f"{_fmt(float(metrics[LIN_VEL_TRACKING_RETURN_KEY])):>{col_w}} "
            f"{_fmt(float(metrics[ANG_VEL_TRACKING_RETURN_KEY])):>{col_w}} "
            f"{_fmt(float(metrics[TRACKING_TOTAL_RETURN_KEY])):>{col_w}} "
            f"{_fmt(float(metrics['obs_pred_mse']), 6):>{col_w}} "
            f"{_fmt(float(metrics['idm_action_mse']), 6):>{col_w}} "
            f"{_fmt(float(metrics['fdm_obs_mse']), 6):>{col_w}} "
            f"{_fmt(float(metrics['planner_first_step_obs_mse']), 6):>{col_w}}"
        )

    stat_keys = (
        LIN_VEL_TRACKING_RETURN_KEY,
        ANG_VEL_TRACKING_RETURN_KEY,
        TRACKING_TOTAL_RETURN_KEY,
        "obs_pred_mse",
        "idm_action_mse",
        "fdm_obs_mse",
        "planner_first_step_obs_mse",
    )
    means_stds = [_scalar_stats(results, key) for key in stat_keys]
    mean_std_strs = [
        f"{means_stds[0][0]:.4f}±{means_stds[0][1]:.4f}",
        f"{means_stds[1][0]:.4f}±{means_stds[1][1]:.4f}",
        f"{means_stds[2][0]:.4f}±{means_stds[2][1]:.4f}",
        f"{means_stds[3][0]:.6f}±{means_stds[3][1]:.6f}",
        f"{means_stds[4][0]:.6f}±{means_stds[4][1]:.6f}",
        f"{means_stds[5][0]:.6f}±{means_stds[5][1]:.6f}",
        f"{means_stds[6][0]:.6f}±{means_stds[6][1]:.6f}",
    ]
    print("-" * sep_len)
    print(
        f"{'Mean ± Std':<36} "
        f"{mean_std_strs[0]:>{col_w}} "
        f"{mean_std_strs[1]:>{col_w}} "
        f"{mean_std_strs[2]:>{col_w}} "
        f"{mean_std_strs[3]:>{col_w}} "
        f"{mean_std_strs[4]:>{col_w}} "
        f"{mean_std_strs[5]:>{col_w}}"
        f" {mean_std_strs[6]:>{col_w}}"
    )
    print(
        "\n(tracking returns are weighted per-term returns; track_total = lin + ang only; obs_pred_err/idm_mse/fdm_mse/planner1_mse: MSE)"
    )
    _print_idm_horizon_summary(results, "[IDM Per-Horizon MSE]")
    _print_fdm_horizon_summary(results, "[FDM Per-Horizon MSE]")
    _print_planner_term_summary(results, "[Planner First-Step Per-Term MSE]")


def _delta_str(
    pre_val: float,
    post_val: float,
    precision: int = 4,
    *,
    higher_is_better: bool = False,
) -> str:
    if np.isnan(pre_val) or np.isnan(post_val):
        return "N/A"
    delta = post_val - pre_val
    pct = (delta / abs(pre_val) * 100) if abs(pre_val) > 1e-9 else float("nan")
    if delta == 0:
        arrow = "flat"
    else:
        improved = delta > 0 if higher_is_better else delta < 0
        arrow = "better" if improved else "worse"
    pct_str = f"{abs(pct):.1f}%" if not np.isnan(pct) else "N/A"
    return f"{delta:+.{precision}f} ({arrow} {pct_str})"


def print_comparison_table(
    pre_metrics: list[tuple[str, dict[str, Any]]],
    post_metrics: list[tuple[str, dict[str, Any]]],
):
    """Print a pre/post finetune comparison table with planner+IDM metrics."""
    col_w = 14
    header = (
        f"  {'Run':<36} {'lin_track':>{col_w}} {'ang_track':>{col_w}} {'track_total':>{col_w}} "
        f"{'obs_pred_err':>{col_w}} {'idm_mse':>{col_w}} {'fdm_mse':>{col_w}} {'planner1_mse':>{col_w}}"
    )
    sep = "=" * (38 + 7 * (col_w + 1))

    def _print_group(title: str, group: list[tuple[str, dict[str, Any]]]) -> dict[str, tuple[float, float]]:
        print(title)
        for name, metrics in group:
            print(
                f"    {name:<36} "
                f"{_fmt(float(metrics[LIN_VEL_TRACKING_RETURN_KEY])):>{col_w}} "
                f"{_fmt(float(metrics[ANG_VEL_TRACKING_RETURN_KEY])):>{col_w}} "
                f"{_fmt(float(metrics[TRACKING_TOTAL_RETURN_KEY])):>{col_w}} "
                f"{_fmt(float(metrics['obs_pred_mse']), 6):>{col_w}} "
                f"{_fmt(float(metrics['idm_action_mse']), 6):>{col_w}} "
                f"{_fmt(float(metrics['fdm_obs_mse']), 6):>{col_w}} "
                f"{_fmt(float(metrics['planner_first_step_obs_mse']), 6):>{col_w}}"
            )
        summary = {
            key: _scalar_stats(group, key)
            for key in (
                LIN_VEL_TRACKING_RETURN_KEY,
                ANG_VEL_TRACKING_RETURN_KEY,
                TRACKING_TOTAL_RETURN_KEY,
                "obs_pred_mse",
                "idm_action_mse",
                "fdm_obs_mse",
                "planner_first_step_obs_mse",
            )
        }
        mean_std_strs = [
            f"{summary[LIN_VEL_TRACKING_RETURN_KEY][0]:.4f}±{summary[LIN_VEL_TRACKING_RETURN_KEY][1]:.4f}",
            f"{summary[ANG_VEL_TRACKING_RETURN_KEY][0]:.4f}±{summary[ANG_VEL_TRACKING_RETURN_KEY][1]:.4f}",
            f"{summary[TRACKING_TOTAL_RETURN_KEY][0]:.4f}±{summary[TRACKING_TOTAL_RETURN_KEY][1]:.4f}",
            f"{summary['obs_pred_mse'][0]:.6f}±{summary['obs_pred_mse'][1]:.6f}",
            f"{summary['idm_action_mse'][0]:.6f}±{summary['idm_action_mse'][1]:.6f}",
            f"{summary['fdm_obs_mse'][0]:.6f}±{summary['fdm_obs_mse'][1]:.6f}",
            f"{summary['planner_first_step_obs_mse'][0]:.6f}±{summary['planner_first_step_obs_mse'][1]:.6f}",
        ]
        print(
            f"    {'Mean ± Std':<36} "
            f"{mean_std_strs[0]:>{col_w}} "
            f"{mean_std_strs[1]:>{col_w}} "
            f"{mean_std_strs[2]:>{col_w}} "
            f"{mean_std_strs[3]:>{col_w}} "
            f"{mean_std_strs[4]:>{col_w}} "
            f"{mean_std_strs[5]:>{col_w}} "
            f"{mean_std_strs[6]:>{col_w}}"
        )
        return summary

    print(f"\n{sep}")
    print(header)
    print(sep)
    pre_summary = _print_group("  [Pre-Finetune]", pre_metrics)
    print(
        f"  {'-' * 36} {'-' * col_w} {'-' * col_w} {'-' * col_w} {'-' * col_w} {'-' * col_w} {'-' * col_w} {'-' * col_w}"
    )
    post_summary = _print_group("  [Post-Finetune]", post_metrics)
    print(sep)
    print(
        f"  {'Delta (Post - Pre)':<36} "
        f"{_delta_str(pre_summary[LIN_VEL_TRACKING_RETURN_KEY][0], post_summary[LIN_VEL_TRACKING_RETURN_KEY][0], higher_is_better=True):>{col_w}} "
        f"{_delta_str(pre_summary[ANG_VEL_TRACKING_RETURN_KEY][0], post_summary[ANG_VEL_TRACKING_RETURN_KEY][0], higher_is_better=True):>{col_w}} "
        f"{_delta_str(pre_summary[TRACKING_TOTAL_RETURN_KEY][0], post_summary[TRACKING_TOTAL_RETURN_KEY][0], higher_is_better=True):>{col_w}} "
        f"{_delta_str(pre_summary['obs_pred_mse'][0], post_summary['obs_pred_mse'][0], 6):>{col_w}} "
        f"{_delta_str(pre_summary['idm_action_mse'][0], post_summary['idm_action_mse'][0], 6):>{col_w}} "
        f"{_delta_str(pre_summary['fdm_obs_mse'][0], post_summary['fdm_obs_mse'][0], 6):>{col_w}} "
        f"{_delta_str(pre_summary['planner_first_step_obs_mse'][0], post_summary['planner_first_step_obs_mse'][0], 6):>{col_w}}"
    )
    print(sep)
    print("  (tracking returns: higher = better; track_total = weighted lin + ang tracking only; MSE metrics: lower = better)")

    pre_idm = _mean_vector_metric(pre_metrics, "idm_action_mse_per_horizon")
    post_idm = _mean_vector_metric(post_metrics, "idm_action_mse_per_horizon")
    if pre_idm.size > 0 or post_idm.size > 0:
        max_len = max(pre_idm.size, post_idm.size)
        print("\n  [IDM Per-Horizon MSE]")
        for idx in range(max_len):
            pre_val = float(pre_idm[idx]) if idx < pre_idm.size else float("nan")
            post_val = float(post_idm[idx]) if idx < post_idm.size else float("nan")
            print(
                f"    step {idx + 1:<2} "
                f"pre={_fmt(pre_val, 6):>12} "
                f"post={_fmt(post_val, 6):>12} "
                f"delta={_delta_str(pre_val, post_val, 6)}"
            )

    pre_fdm = _mean_vector_metric(pre_metrics, "fdm_obs_mse_per_horizon")
    post_fdm = _mean_vector_metric(post_metrics, "fdm_obs_mse_per_horizon")
    if pre_fdm.size > 0 or post_fdm.size > 0:
        max_len = max(pre_fdm.size, post_fdm.size)
        print("\n  [FDM Per-Horizon MSE]")
        for idx in range(max_len):
            pre_val = float(pre_fdm[idx]) if idx < pre_fdm.size else float("nan")
            post_val = float(post_fdm[idx]) if idx < post_fdm.size else float("nan")
            print(
                f"    step {idx + 1:<2} "
                f"pre={_fmt(pre_val, 6):>12} "
                f"post={_fmt(post_val, 6):>12} "
                f"delta={_delta_str(pre_val, post_val, 6)}"
            )

    pre_terms = _mean_dict_metric(pre_metrics, "planner_first_step_obs_mse_per_term")
    post_terms = _mean_dict_metric(post_metrics, "planner_first_step_obs_mse_per_term")
    if pre_terms or post_terms:
        print("\n  [Planner First-Step Per-Term MSE]")
        for term in sorted(set(pre_terms) | set(post_terms)):
            pre_val = float(pre_terms.get(term, float("nan")))
            post_val = float(post_terms.get(term, float("nan")))
            print(
                f"    {term:<18} "
                f"pre={_fmt(pre_val, 6):>12} "
                f"post={_fmt(post_val, 6):>12} "
                f"delta={_delta_str(pre_val, post_val, 6)}"
            )
    print()


