#!/usr/bin/env python
"""Plot inference logs (single trajectory) with integrated position/angle curves and optional reconstruction loss.

Implementation only -- no standalone CLI entry point is shipped for this module (see
tools/check_entrypoints.py). Call `plot_inference_log_position(...)` directly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from holosoma_inference.utils.math.quat import quat_apply, quat_rotate_inverse, xyzw_to_wxyz


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


def plot_inference_log_position(log_path: str, title: str | None = None, start_step: int = 0, trim_tail: int = 0, dt: float = 0.02):
    """
    Plot inference log with integrated position/angle curves.
    
    Parameters
    ----------
    log_path : str
        Path to inference state_log.npz
    title : str | None
        Figure title
    start_step : int
        Skip the first N steps when plotting
    trim_tail : int
        Drop the last N steps when plotting
    dt : float
        Time step in seconds for integration (default: 0.02 for 50 Hz)
    """
    data = np.load(log_path)
    name = Path(log_path).stem
    title = title or f"Inference Results (Position): {name}"

    # Get ground truth position/angle from log (required)
    base_pos_x = _flatten(data.get("base_pos_x", np.array([])))
    base_pos_y = _flatten(data.get("base_pos_y", np.array([])))
    base_angle = _flatten(data.get("base_yaw", np.array([])))
    
    # Get quaternion data for coordinate transformation (need current orientation to convert base frame velocities)
    base_quat_x = _flatten(data.get("base_quat_x", np.array([])))
    base_quat_y = _flatten(data.get("base_quat_y", np.array([])))
    base_quat_z = _flatten(data.get("base_quat_z", np.array([])))
    base_quat_w = _flatten(data.get("base_quat_w", np.array([])))
    
    # Get initial pose for coordinate transformation
    initial_base_pos_x = _flatten(data.get("initial_base_pos_x", np.array([])))
    initial_base_pos_y = _flatten(data.get("initial_base_pos_y", np.array([])))
    initial_base_pos_z = _flatten(data.get("initial_base_pos_z", np.array([])))
    initial_base_quat_x = _flatten(data.get("initial_base_quat_x", np.array([])))
    initial_base_quat_y = _flatten(data.get("initial_base_quat_y", np.array([])))
    initial_base_quat_z = _flatten(data.get("initial_base_quat_z", np.array([])))
    initial_base_quat_w = _flatten(data.get("initial_base_quat_w", np.array([])))
    
    # Get command velocities for integration (in base frame)
    cmd_x = _flatten(data.get("command_x", np.array([])))
    cmd_y = _flatten(data.get("command_y", np.array([])))
    cmd_yaw = _flatten(data.get("command_yaw", np.array([])))
    
    # Check if we have ground truth position data
    if len(base_pos_x) == 0 or len(base_pos_y) == 0 or len(base_angle) == 0:
        raise ValueError(
            "Ground truth position data (base_pos_x, base_pos_y, base_yaw) not found in log. "
            "Please ensure inference was run with ZMQ/mocap enabled to record ground truth poses."
        )
    
    # Check if we have quaternion data (required for coordinate transformation)
    has_quat = len(base_quat_x) > 0
    has_initial_pos = len(initial_base_pos_x) > 0
    has_initial_quat = len(initial_base_quat_x) > 0
    if not has_quat:
        raise ValueError(
            "Quaternion data (base_quat_*) not found in log. Please ensure you're using a recent version "
            "that records base_quat_* fields."
        )
    if not has_initial_pos:
        raise ValueError(
            "Initial position data (initial_base_pos_*) not found in log. Please ensure you're using a recent version "
            "that records initial_base_pos_* fields."
        )
    if not has_initial_quat:
        raise ValueError(
            "Initial quaternion data (initial_base_quat_*) not found in log. Please ensure you're using a recent version "
            "that records initial_base_quat_* fields."
        )
    
    total_steps = min(len(base_pos_x), len(cmd_x))
    if total_steps == 0:
        raise ValueError("No position/command data found in log.")
    
    start = max(0, int(start_step))
    end = total_steps - max(0, int(trim_tail))
    if end < start:
        raise ValueError(f"Invalid slice: start={start} >= end={end}. "
                         f"Total steps={total_steps}, trim_tail={trim_tail}.")
    
    base_pos_x = base_pos_x[start:end]
    base_pos_y = base_pos_y[start:end]
    base_angle = base_angle[start:end]
    base_quat_x = base_quat_x[start:end]
    base_quat_y = base_quat_y[start:end]
    base_quat_z = base_quat_z[start:end]
    base_quat_w = base_quat_w[start:end]
    cmd_x = cmd_x[start:end]
    cmd_y = cmd_y[start:end]
    cmd_yaw = cmd_yaw[start:end]
    
    steps = len(base_pos_x)
    
    # Get initial pose (first value, should be constant)
    initial_pos = np.array([initial_base_pos_x[0], initial_base_pos_y[0], initial_base_pos_z[0]])
    initial_quat_xyzw = np.array([initial_base_quat_x[0], initial_base_quat_y[0], initial_base_quat_z[0], initial_base_quat_w[0]])
    initial_quat_wxyz = xyzw_to_wxyz(initial_quat_xyzw.reshape(1, 4)).reshape(-1)
    
    # Convert world frame positions to initial base frame
    # Step 1: Get positions relative to initial position in world frame
    base_pos_world = np.stack([base_pos_x, base_pos_y, np.zeros_like(base_pos_x)], axis=1)  # [steps, 3]
    base_pos_world = base_pos_world - initial_pos.reshape(1, 3)  # Relative to initial position
    
    # Step 2: Transform to initial base frame
    initial_quat_wxyz_repeated = np.tile(initial_quat_wxyz.reshape(1, 4), (steps, 1))  # [steps, 4]
    base_pos_initial_base = quat_rotate_inverse(initial_quat_wxyz_repeated, base_pos_world)  # [steps, 3]
    
    base_pos_x_initial_base = base_pos_initial_base[:, 0]
    base_pos_y_initial_base = base_pos_initial_base[:, 1]
    
    # Handle angle wrap-around: unwrap angles to ensure continuity across ±π boundary
    base_angle_unwrapped = np.unwrap(base_angle)
    # Convert yaw from world frame to initial base frame
    # For yaw, we need to compute relative yaw from initial orientation
    base_angle_initial_base = base_angle_unwrapped - base_angle_unwrapped[0]
    
    # Convert command velocities from current base frame to initial base frame
    # Step 1: Convert current base frame velocities to world frame
    base_quat_xyzw_all = np.stack([base_quat_x, base_quat_y, base_quat_z, base_quat_w], axis=1)  # [steps, 4]
    base_quat_wxyz_all = xyzw_to_wxyz(base_quat_xyzw_all.reshape(steps, 4))  # [steps, 4]
    
    cmd_vel_base = np.stack([cmd_x, cmd_y, np.zeros_like(cmd_x)], axis=1)  # [steps, 3]
    cmd_vel_world = quat_apply(base_quat_wxyz_all, cmd_vel_base)  # [steps, 3]
    
    # Step 2: Convert world frame velocities to initial base frame
    cmd_vel_initial_base = quat_rotate_inverse(initial_quat_wxyz_repeated, cmd_vel_world)  # [steps, 3]
    
    # Integrate initial base frame velocities to get initial base frame positions
    cmd_pos_x_initial_base = np.cumsum(cmd_vel_initial_base[:, 0] * dt)
    cmd_pos_y_initial_base = np.cumsum(cmd_vel_initial_base[:, 1] * dt)
    
    # For yaw, command is already angular velocity, integrate directly
    cmd_angle_initial_base = np.cumsum(cmd_yaw * dt)
    
    # Unwrap command angle to ensure continuity (in case of any wrap-around issues)
    cmd_angle_initial_base = np.unwrap(cmd_angle_initial_base)
    
    # All values are now in initial base frame
    base_pos_x = base_pos_x_initial_base
    base_pos_y = base_pos_y_initial_base
    base_angle = base_angle_initial_base
    cmd_pos_x = cmd_pos_x_initial_base
    cmd_pos_y = cmd_pos_y_initial_base
    cmd_angle = cmd_angle_initial_base
    
    # Calculate tracking errors
    err_x = base_pos_x - cmd_pos_x
    err_y = base_pos_y - cmd_pos_y
    err_yaw = base_angle - cmd_angle
    
    # Calculate RMSE
    rmse_x = float(np.sqrt(np.mean(err_x**2)))
    rmse_y = float(np.sqrt(np.mean(err_y**2)))
    rmse_yaw = float(np.sqrt(np.mean(err_yaw**2)))
    rmse_linear = float(np.sqrt(np.mean(err_x**2 + err_y**2)))

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
        f"Linear Pos RMSE (xy): {rmse_linear:.4f} m",
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    axs[0, 0].plot(t, base_pos_x, label="base_pos_x (initial base frame)", color="tab:blue")
    axs[0, 0].plot(t, cmd_pos_x, "--", label="cmd_pos_x (initial base frame)", color="tab:orange")
    axs[0, 0].set_title(f"Position X (Initial Base Frame) (RMSE={rmse_x:.4f})", fontsize=14)
    axs[0, 0].set_ylabel("m", fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)

    axs[0, 1].plot(t, base_pos_y, label="base_pos_y (initial base frame)", color="tab:green")
    axs[0, 1].plot(t, cmd_pos_y, "--", label="cmd_pos_y (initial base frame)", color="tab:red")
    axs[0, 1].set_title(f"Position Y (Initial Base Frame) (RMSE={rmse_y:.4f})", fontsize=14)
    axs[0, 1].set_ylabel("m", fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)

    axs[1, 0].plot(t, base_angle, label="base_angle (initial base frame)", color="tab:blue")
    axs[1, 0].plot(t, cmd_angle, "--", label="cmd_angle (initial base frame)", color="tab:orange")
    axs[1, 0].set_title(f"Angle Yaw (Initial Base Frame) (RMSE={rmse_yaw:.4f})", fontsize=14)
    axs[1, 0].set_xlabel("Step", fontsize=12)
    axs[1, 0].set_ylabel("rad", fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)

    if recon_available:
        axs[1, 1].plot(np.arange(len(recon_loss)), recon_loss, color="tab:cyan")
        axs[1, 1].set_title(f"Prediction Obs Loss (MSE={recon_mse:.6f})", fontsize=14)
        axs[1, 1].set_xlabel("Step", fontsize=12)
        axs[1, 1].set_ylabel("MSE", fontsize=12)
        axs[1, 1].grid(True, alpha=0.3)
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

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    log_path_obj = Path(log_path)
    save_path = log_path_obj.with_name(log_path_obj.stem + "_position_plot.png")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved position plot to {save_path}")
    plt.close(fig)
