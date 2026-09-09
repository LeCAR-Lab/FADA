#!/usr/bin/env python
"""Plot inference logs (single trajectory) with tracking errors and optional reconstruction loss.

Implementation only -- no standalone CLI entry point is shipped for this module (see
tools/check_entrypoints.py). `_k_step_recon_loss_over_time`, `_moving_average`, and
`_remove_outliers` are imported directly by `compute_mocap_velocity_metrics.py`;
`plot_inference_log(...)` remains importable too.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _flatten(arr):
    arr = np.asarray(arr)
    return arr.reshape(-1)


def _k_step_recon_loss_over_time(
    actual_raw: np.ndarray,
    predicted_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual_raw, dtype=np.float64)
    predicted = np.asarray(predicted_raw, dtype=np.float64)

    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    else:
        actual = actual.reshape(actual.shape[0], -1)

    if predicted.ndim == 1:
        predicted = predicted.reshape(1, 1, -1)
    elif predicted.ndim == 2:
        predicted = predicted.reshape(predicted.shape[0], 1, -1)

    n = min(int(actual.shape[0]), int(predicted.shape[0]))
    actual = actual[:n]
    predicted = predicted[:n]
    _, k_horizon, _ = predicted.shape

    loss = np.full(n, np.nan, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t_start = max(0, i - k_horizon + 1)
        terms: list[float] = []
        for t_idx in range(t_start, i + 1):
            k = i - t_idx
            if 0 <= k < k_horizon:
                err = (predicted[t_idx, k, :] - actual[i, :]) ** 2
                terms.append(float(np.mean(err)))
        if terms:
            loss[i] = float(np.mean(terms))
            counts[i] = int(len(terms))
    return loss, counts


def _remove_outliers(data: np.ndarray, max_value: float) -> np.ndarray:
    """
    Remove outliers from data by setting them to NaN.
    
    Args:
        data: Input array
        max_value: Maximum reasonable value. Values with absolute value > max_value are considered outliers.
    
    Returns:
        Array with outliers replaced by NaN
    """
    result = data.copy()
    outlier_mask = np.abs(data) > max_value
    result[outlier_mask] = np.nan
    return result


def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average smoothing to data, handling NaN values.
    
    Args:
        data: Input array (may contain NaN values)
        window_size: Size of the moving average window (must be odd)
    
    Returns:
        Smoothed array with same shape as input
    """
    if window_size <= 1:
        return data
    
    # Ensure window_size is odd for symmetric smoothing
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    smoothed = np.full_like(data, np.nan)
    
    # Apply moving average with NaN-aware computation
    for i in range(len(data)):
        # Define the window boundaries (symmetric around current point)
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        
        # Extract the window
        window = data[start:end]
        
        # Only compute average if there are non-NaN values
        if np.any(~np.isnan(window)):
            smoothed[i] = np.nanmean(window)
        else:
            # Preserve NaN if all values in window are NaN
            smoothed[i] = np.nan
    
    return smoothed


def plot_inference_log(
    log_path: str,
    title: str | None = None,
    start_step: int = 0,
    trim_tail: int = 0,
    smooth_window: int = 0,
):
    data = np.load(log_path)
    name = Path(log_path).stem
    title = title or f"Inference Results: {name}"

    base_vx = _flatten(data.get("base_vel_x", np.array([])))
    base_vy = _flatten(data.get("base_vel_y", np.array([])))
    base_yaw = _flatten(data.get("base_vel_yaw", np.array([])))
    cmd_x = _flatten(data.get("command_x", np.array([])))
    cmd_y = _flatten(data.get("command_y", np.array([])))
    cmd_yaw = _flatten(data.get("command_yaw", np.array([])))

    total_steps = min(len(base_vx), len(cmd_x))
    if total_steps == 0:
        raise ValueError("No velocity/command data found in log.")

    start = max(0, int(start_step))
    end = total_steps - max(0, int(trim_tail))
    if end < start:
        raise ValueError(f"Invalid slice: start={start} >= end={end}. "
                         f"Total steps={total_steps}, trim_tail={trim_tail}.")

    base_vx = base_vx[start:end]
    base_vy = base_vy[start:end]
    base_yaw = base_yaw[start:end]
    cmd_x = cmd_x[start:end]
    cmd_y = cmd_y[start:end]
    cmd_yaw = cmd_yaw[start:end]

    # Plot raw velocities; RMSE uses outlier-masked (+ optional smooth) series only.
    max_linear_vel = 1.0
    max_angular_vel = 2.0
    vx_rmse = _remove_outliers(np.asarray(base_vx, dtype=np.float64).copy(), max_linear_vel)
    vy_rmse = _remove_outliers(np.asarray(base_vy, dtype=np.float64).copy(), max_linear_vel)
    yaw_rmse = _remove_outliers(np.asarray(base_yaw, dtype=np.float64).copy(), max_angular_vel)
    if smooth_window > 0:
        vx_rmse = _moving_average(vx_rmse, smooth_window)
        vy_rmse = _moving_average(vy_rmse, smooth_window)
        yaw_rmse = _moving_average(yaw_rmse, smooth_window)

    steps = len(base_vx)

    err_x = vx_rmse - cmd_x
    err_y = vy_rmse - cmd_y
    err_yaw = yaw_rmse - cmd_yaw
    # Use nanmean to handle potential NaN values
    rmse_x = float(np.sqrt(np.nanmean(err_x**2)))
    rmse_y = float(np.sqrt(np.nanmean(err_y**2)))
    rmse_yaw = float(np.sqrt(np.nanmean(err_yaw**2)))
    rmse_linear = float(np.sqrt(np.nanmean(err_x**2 + err_y**2)))

    recon_loss = None
    recon_counts = None
    if "actual_target" in data and "predicted_target" in data:
        actual_raw = np.asarray(data["actual_target"])
        pred_raw = np.asarray(data["predicted_target"])
        recon_full, recon_counts_full = _k_step_recon_loss_over_time(actual_raw, pred_raw)
        steps_recon = len(recon_full)
        recon_start = min(start, steps_recon)
        recon_end = max(recon_start, min(end, steps_recon))
        recon_loss = recon_full[recon_start:recon_end]
        recon_counts = recon_counts_full[recon_start:recon_end]
    elif "reconstruction_loss" in data:
        recon_raw = _flatten(data["reconstruction_loss"])
        recon_steps = len(recon_raw)
        recon_start = min(start, recon_steps)
        recon_end = max(recon_start, min(end, recon_steps))
        recon_loss = recon_raw[recon_start:recon_end]
    recon_available = recon_loss is not None and len(recon_loss) > 0
    if recon_available and recon_counts is not None and np.any(recon_counts > 0):
        valid = recon_counts > 0
        recon_mse = float(np.sum(recon_loss[valid] * recon_counts[valid]) / np.sum(recon_counts[valid]))
    else:
        recon_mse = float(np.mean(recon_loss)) if recon_available else None

    t = np.arange(steps)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.01,
        0.99,
        f"Linear Vel RMSE (xy): {rmse_linear:.4f} m/s",
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    axs[0, 0].plot(t, base_vx, label="base_vx", color="tab:blue")
    axs[0, 0].plot(t, cmd_x, "--", label="cmd_x", color="tab:orange")
    axs[0, 0].set_title(f"Linear Velocity X (RMSE={rmse_x:.4f})", fontsize=14)
    axs[0, 0].set_ylabel("m/s", fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)

    axs[0, 1].plot(t, base_vy, label="base_vy", color="tab:green")
    axs[0, 1].plot(t, cmd_y, "--", label="cmd_y", color="tab:red")
    axs[0, 1].set_title(f"Linear Velocity Y (RMSE={rmse_y:.4f})", fontsize=14)
    axs[0, 1].set_ylabel("m/s", fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)

    axs[1, 0].plot(t, base_yaw, label="base_yaw", color="tab:blue")
    axs[1, 0].plot(t, cmd_yaw, "--", label="cmd_yaw", color="tab:orange")
    axs[1, 0].set_title(f"Angular Velocity Yaw (RMSE={rmse_yaw:.4f})", fontsize=14)
    axs[1, 0].set_xlabel("Step", fontsize=12)
    axs[1, 0].set_ylabel("rad/s", fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)

    if recon_available:
        axs[1, 1].plot(np.arange(len(recon_loss)), recon_loss, color="tab:cyan")
        axs[1, 1].set_title(f"Prediction Obs Loss (MSE={recon_mse:.6f})", fontsize=14)
        axs[1, 1].set_xlabel("Step", fontsize=12)
        axs[1, 1].set_ylabel("MSE", fontsize=12)
    else:
        axs[1, 1].axis("off")
        axs[1, 1].text(
            0.5,
            0.5,
            "Reconstruction loss unavailable",
            ha="center",
            va="center",
            fontsize=13,
            transform=axs[1, 1].transAxes,
        )
    axs[1, 1].grid(True, alpha=0.3) if recon_available else None

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    log_path_obj = Path(log_path)
    save_path = log_path_obj.with_name(log_path_obj.stem + "_inference_plot.png")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved inference plot to {save_path}")
    plt.close(fig)
