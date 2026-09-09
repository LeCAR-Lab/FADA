"""
Analyze mocap_unified.npz to find causes of velocity spikes (vx, vy, yaw separately).

Diagnostic implementation only -- no standalone CLI entry point is shipped for this
module (see tools/check_entrypoints.py). Call `analyze(npz_path)` directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from holosoma_inference.utils.plot_mocap_raw import (
    load_mocap_unified_log,
    compute_velocity_and_displacement_from_raw,
)
from holosoma_inference.utils.math.quat import xyzw_to_wxyz


def _yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Extract yaw (rad) from quaternion (N, 4) in xyzw format."""
    qx, qy, qz, qw = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def analyze(npz_path: str | Path) -> None:
    path = Path(npz_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    ts, pos, quat, cmd_ts, cmd_x, cmd_y, cmd_yaw, recon_ts, actual, predicted = load_mocap_unified_log(path)
    n = len(ts)
    if n == 0:
        print("No mocap data in file.")
        return

    t, base_pos_x, base_pos_y, base_angle, base_vx, base_vy, base_yaw = (
        compute_velocity_and_displacement_from_raw(ts, pos, quat)
    )
    quat_wxyz = xyzw_to_wxyz(quat.reshape(n, 4))
    yaw_world = _yaw_from_quat_xyzw(quat.reshape(n, 4))

    dt = np.diff(ts)
    dt_at_i = np.concatenate([[np.nan], dt])  # dt_at_i[i] = ts[i]-ts[i-1]

    # World-frame velocity (before base transform) for each step
    vel_world_x = np.zeros(n)
    vel_world_y = np.zeros(n)
    vel_world_z = np.zeros(n)
    for i in range(1, n):
        if dt_at_i[i] > 1e-9:
            vel_world_x[i] = (pos[i, 0] - pos[i - 1, 0]) / dt_at_i[i]
            vel_world_y[i] = (pos[i, 1] - pos[i - 1, 1]) / dt_at_i[i]
            vel_world_z[i] = (pos[i, 2] - pos[i - 1, 2]) / dt_at_i[i]

    # Spike thresholds (separate for vx, vy, yaw)
    thresh_vx = 1.0   # m/s, typical walking ~0.5
    thresh_vy = 1.0   # m/s
    thresh_yaw = 2.0  # rad/s
    spike_vx = np.where(np.abs(base_vx) > thresh_vx)[0]
    spike_vy = np.where(np.abs(base_vy) > thresh_vy)[0]
    spike_yaw = np.where(np.abs(base_yaw) > thresh_yaw)[0]
    # Union of all spike indices for detailed table
    all_spike_idx = sorted(set(spike_vx.tolist() + spike_vy.tolist() + spike_yaw.tolist()))
    all_spike_idx = [i for i in all_spike_idx if i > 0]

    print("=" * 70)
    print(f"Mocap unified log: {path.name}")
    print("=" * 70)
    print(f"N = {n},  span = {ts[-1] - ts[0]:.3f} s,  rate ≈ {n / (ts[-1] - ts[0]):.1f} Hz")
    print()

    # --- 1. Spike counts by component (vx, vy, yaw) ---
    print("--- 1. Spike counts (by component) ---")
    print(f"  |base_vx| > {thresh_vx} m/s:  {len(spike_vx)}  indices  (first 30: {spike_vx[:30].tolist()})")
    print(f"  |base_vy| > {thresh_vy} m/s:  {len(spike_vy)}  indices  (first 30: {spike_vy[:30].tolist()})")
    print(f"  |base_yaw| > {thresh_yaw} rad/s: {len(spike_yaw)}  indices  (first 30: {spike_yaw[:30].tolist()})")
    print(f"  Combined unique spike indices (i>0): {len(all_spike_idx)}")
    print()

    # --- 2. Velocity stats (vx vs vy vs yaw) ---
    print("--- 2. Velocity statistics ---")
    print(f"  base_vx:  min={base_vx.min():.4f}, max={base_vx.max():.4f}, mean={base_vx.mean():.4f}, std={np.nanstd(base_vx):.4f} m/s")
    print(f"  base_vy:  min={base_vy.min():.4f}, max={base_vy.max():.4f}, mean={base_vy.mean():.4f}, std={np.nanstd(base_vy):.4f} m/s")
    print(f"  base_yaw: min={base_yaw.min():.4f}, max={base_yaw.max():.4f}, mean={base_yaw.mean():.4f}, std={np.nanstd(base_yaw):.4f} rad/s")
    print()

    # --- 3. Per-spike detailed cause (vx and vy separately) ---
    print("--- 3. Cause of each spike (vx / vy / yaw) ---")
    print("  At each spike index i we show: dt, world-frame delta_pos and vel_world,")
    print("  then base-frame vx, vy, yaw. Cause: which of world X/Y/Z jump or rotation dominates.")
    print()

    for i in all_spike_idx[:50]:  # first 50 spikes
        d = dt_at_i[i]
        if d <= 1e-9:
            continue
        # World position change and world velocity
        dp = pos[i] - pos[i - 1]
        vw_x = dp[0] / d
        vw_y = dp[1] / d
        vw_z = dp[2] / d
        # Base-frame velocity at i (already computed)
        vx = base_vx[i]
        vy = base_vy[i]
        yaw_vel = base_yaw[i]
        # Yaw change (rad) over this step
        dyaw = yaw_world[i] - yaw_world[i - 1]
        # Normalize to [-pi,pi] for display
        if dyaw > np.pi:
            dyaw -= 2 * np.pi
        if dyaw < -np.pi:
            dyaw += 2 * np.pi

        is_vx = i in spike_vx
        is_vy = i in spike_vy
        is_ya = i in spike_yaw
        comp = []
        if is_vx:
            comp.append("vx")
        if is_vy:
            comp.append("vy")
        if is_ya:
            comp.append("yaw")

        # Decide cause: compare |vel_world| components and |dyaw|
        cause = []
        if abs(vw_x) > thresh_vx:
            cause.append("world_X_jump")
        if abs(vw_y) > thresh_vy:
            cause.append("world_Y_jump")
        if abs(vw_z) > 0.5:  # 0.5 m/s vertical is already large
            cause.append("world_Z_jump")
        if abs(dyaw) > 0.15:  # ~8.6 deg in one step
            cause.append("yaw_change")
        if not cause:
            cause.append("small_deltas_rotation_proj")  # base spike from rotation projecting world vel

        print(f"  i={i:5d}  t={t[i]:.2f}s  dt={d:.4f}s  comp=[{','.join(comp):6s}]  cause=[{','.join(cause)}]")
        print(f"         pos[i-1] = ({pos[i-1,0]:.4f}, {pos[i-1,1]:.4f}, {pos[i-1,2]:.4f})  ->  pos[i] = ({pos[i,0]:.4f}, {pos[i,1]:.4f}, {pos[i,2]:.4f})")
        print(f"         delta_pos_world (dx,dy,dz) = ({dp[0]:.5f}, {dp[1]:.5f}, {dp[2]:.5f}) m")
        print(f"         vel_world (vx_w, vy_w, vz_w) = ({vw_x:.4f}, {vw_y:.4f}, {vw_z:.4f}) m/s")
        print(f"         base (vx, vy, yaw_vel) = ({vx:.4f}, {vy:.4f}, {yaw_vel:.4f})")
        print(f"         yaw[i-1]={yaw_world[i-1]:.4f}  yaw[i]={yaw_world[i]:.4f}  dyaw={dyaw:.4f} rad")
        print()

    if len(all_spike_idx) > 50:
        print(f"  ... and {len(all_spike_idx) - 50} more spike indices (truncated).")
    print()

    # --- 4. Summary: which cause is most common ---
    print("--- 4. Summary of causes ---")
    causes_vx = []
    causes_vy = []
    causes_yaw = []
    for i in all_spike_idx:
        if i == 0:
            continue
        d = dt_at_i[i]
        if d <= 1e-9:
            continue
        dp = pos[i] - pos[i - 1]
        vw_x = dp[0] / d
        vw_y = dp[1] / d
        vw_z = dp[2] / d
        dyaw = yaw_world[i] - yaw_world[i - 1]
        if dyaw > np.pi:
            dyaw -= 2 * np.pi
        if dyaw < -np.pi:
            dyaw += 2 * np.pi

        c = []
        if abs(vw_x) > thresh_vx:
            c.append("world_X")
        if abs(vw_y) > thresh_vy:
            c.append("world_Y")
        if abs(vw_z) > 0.5:
            c.append("world_Z")
        if abs(dyaw) > 0.15:
            c.append("yaw")
        if not c:
            c.append("rotation_proj")

        if i in spike_vx:
            causes_vx.append(c)
        if i in spike_vy:
            causes_vy.append(c)
        if i in spike_yaw:
            causes_yaw.append(c)

    def count_causes(causes):
        from collections import Counter
        flat = [x for sub in causes for x in sub]
        return Counter(flat)

    cx = count_causes(causes_vx)
    cy = count_causes(causes_vy)
    ca = count_causes(causes_yaw)
    print("  For vx spikes, cause tags:", dict(cx))
    print("  For vy spikes, cause tags:", dict(cy))
    print("  For yaw spikes, cause tags:", dict(ca))
    print()
    print("  Interpretation:")
    if cx.get("world_X", 0) or cy.get("world_X", 0):
        print("  - world_X: raw mocap X position jumped -> linear velocity spike in world (and base) X.")
    if cx.get("world_Y", 0) or cy.get("world_Y", 0):
        print("  - world_Y: raw mocap Y position jumped -> linear velocity spike in world (and base) Y.")
    if cx.get("world_Z", 0) or cy.get("world_Z", 0):
        print("  - world_Z: raw mocap Z (height) jumped -> projects to base vx/vy after rotation.")
    if ca.get("yaw", 0):
        print("  - yaw: large yaw change in one step -> angular velocity spike (or orientation glitch).")
    if cx.get("rotation_proj", 0) or cy.get("rotation_proj", 0):
        print("  - rotation_proj: world velocity moderate but rotation projects it into large base vx/vy.")
    print()
