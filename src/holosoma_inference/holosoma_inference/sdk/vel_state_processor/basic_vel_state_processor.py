import time
from typing import Iterable

import numpy as np
from loguru import logger

from holosoma_inference.utils.math.quat import (
    quat_inverse,
    quat_mul,
    quat_rotate_inverse,
    xyzw_to_wxyz,
)


class BasicVelStateProcessor:
    """Convert streamed poses into base-frame linear/angular velocities."""

    def __init__(
        self,
        target_name: str,
        orientation_order: str = "xyzw",
        record_mocap_history: bool = False,
        mocap_plot_hz: float = 0.0,
        plot_trim_head: int = 0,
        plot_trim_tail: int = 0,
        smooth_window: int = 0,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 2.0,
    ):
        self.target_name = target_name
        self.orientation_order = orientation_order
        self.record_mocap_history = record_mocap_history
        self.mocap_plot_hz = mocap_plot_hz
        self.plot_trim_head = plot_trim_head
        self.plot_trim_tail = plot_trim_tail
        self.smooth_window = smooth_window
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self._recording_enabled = True

        self._last_pos: np.ndarray | None = None
        self._last_quat: np.ndarray | None = None
        self._last_ts: float | None = None

        # Latest velocity cache: [lin_vel(3), ang_vel(3)] in base frame
        self._latest_vel = np.zeros((1, 6), dtype=np.float64)

        # Latest pose cache: [pos(3), quat_xyzw(4)] in world frame
        self._latest_pose = np.zeros((1, 7), dtype=np.float64)

        # Optional: raw mocap history for plotting on exit [(ts, pos(3,), quat_xyzw(4,)), ...]
        self._mocap_history: list[tuple[float, np.ndarray, np.ndarray]] = []
        # Command at policy steps [(ts, cmd_x, cmd_y, cmd_yaw), ...]
        self._command_history: list[tuple[float, float, float, float]] = []
        # Reconstruction (actual, predicted) at policy steps [(ts, actual, predicted), ...]
        self._recon_history: list[tuple[float, np.ndarray, np.ndarray]] = []
        # Planner first-step observation alignment [(ts, actual_obs, predicted_obs), ...]
        self._planner_first_step_history: list[tuple[float, np.ndarray, np.ndarray]] = []
        # IDM inverse teacher alignment [(ts, actual_action_chunk, predicted_action_chunk), ...]
        self._idm_inverse_history: list[tuple[float, np.ndarray, np.ndarray]] = []
        # FDM forward teacher alignment [(ts, actual_obs_chunk, predicted_obs_chunk), ...]
        self._fdm_forward_history: list[tuple[float, np.ndarray, np.ndarray]] = []
        # Compact observation layout for per-term metrics/plotting.
        self._compact_term_order: tuple[str, ...] = tuple()
        self._compact_term_dims: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_command(self, timestamp: float, cmd_x: float, cmd_y: float, cmd_yaw: float) -> None:
        """Record one command sample (called each policy step when record_mocap_history)."""
        if not self.record_mocap_history or not self._recording_enabled:
            return
        self._command_history.append((float(timestamp), float(cmd_x), float(cmd_y), float(cmd_yaw)))

    def record_reconstruction(self, timestamp: float, actual: np.ndarray, predicted: np.ndarray) -> None:
        """Record one dynamics reconstruction sample. actual: (dim,); predicted: (K, dim) for K-step or (dim,) for 1-step."""
        if not self.record_mocap_history or not self._recording_enabled:
            return
        a = np.asarray(actual, dtype=np.float64).reshape(-1)
        p = np.asarray(predicted, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        self._recon_history.append((float(timestamp), a.copy(), p.copy()))

    def set_compact_obs_layout(
        self,
        term_order: tuple[str, ...] | list[str],
        term_dims: dict[str, int],
    ) -> None:
        """Set compact observation layout metadata used for planner metric slicing."""
        self._compact_term_order = tuple(str(term) for term in term_order)
        self._compact_term_dims = {str(key): int(value) for key, value in term_dims.items()}

    def record_planner_first_step_obs(
        self,
        timestamp: float,
        actual_obs: np.ndarray,
        predicted_obs: np.ndarray,
    ) -> None:
        """Record one planner first-step observation alignment sample."""
        if not self.record_mocap_history or not self._recording_enabled:
            return
        actual = np.asarray(actual_obs, dtype=np.float64)
        predicted = np.asarray(predicted_obs, dtype=np.float64)
        if actual.ndim == 1:
            actual = actual.reshape(1, -1)
        if predicted.ndim == 1:
            predicted = predicted.reshape(1, -1)
        self._planner_first_step_history.append((float(timestamp), actual.copy(), predicted.copy()))

    def record_idm_inverse(
        self,
        timestamp: float,
        actual_action_chunk: np.ndarray,
        predicted_action_chunk: np.ndarray,
    ) -> None:
        """Record one IDM inverse teacher window alignment sample."""
        if not self.record_mocap_history or not self._recording_enabled:
            return
        actual = np.asarray(actual_action_chunk, dtype=np.float64)
        predicted = np.asarray(predicted_action_chunk, dtype=np.float64)
        if actual.ndim == 1:
            actual = actual.reshape(1, -1)
        if predicted.ndim == 1:
            predicted = predicted.reshape(1, -1)
        self._idm_inverse_history.append((float(timestamp), actual.copy(), predicted.copy()))

    def record_fdm_forward(
        self,
        timestamp: float,
        actual_obs_chunk: np.ndarray,
        predicted_obs_chunk: np.ndarray,
    ) -> None:
        """Record one FDM forward teacher window alignment sample."""
        if not self.record_mocap_history or not self._recording_enabled:
            return
        actual = np.asarray(actual_obs_chunk, dtype=np.float64)
        predicted = np.asarray(predicted_obs_chunk, dtype=np.float64)
        if actual.ndim == 1:
            actual = actual.reshape(1, -1)
        if predicted.ndim == 1:
            predicted = predicted.reshape(1, -1)
        self._fdm_forward_history.append((float(timestamp), actual.copy(), predicted.copy()))

    def get_command_history(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return (timestamps, cmd_x, cmd_y, cmd_yaw) or (None,)*4 if empty."""
        if not self._command_history:
            return (None, None, None, None)
        ts = np.array([h[0] for h in self._command_history], dtype=np.float64)
        cmd_x = np.array([h[1] for h in self._command_history], dtype=np.float64)
        cmd_y = np.array([h[2] for h in self._command_history], dtype=np.float64)
        cmd_yaw = np.array([h[3] for h in self._command_history], dtype=np.float64)
        return (ts, cmd_x, cmd_y, cmd_yaw)

    def get_recon_history(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return (timestamps, actual_target (N,dim), predicted_target (N,K,dim)) or (None, None, None). K from model (K>=1)."""
        if not self._recon_history:
            return (None, None, None)
        ts = np.array([h[0] for h in self._recon_history], dtype=np.float64)
        actual = np.array([h[1] for h in self._recon_history], dtype=np.float64)
        pred_list = [h[2] for h in self._recon_history]
        K = max(p.shape[0] for p in pred_list)
        dim = pred_list[0].shape[1]
        predicted = np.zeros((len(pred_list), K, dim), dtype=np.float64)
        for i, p in enumerate(pred_list):
            k = p.shape[0]
            predicted[i, :k] = p
            if k < K:
                predicted[i, k:] = p[-1:]  # repeat last row for padding
        return (ts, actual, predicted)

    def get_planner_first_step_history(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return (timestamps, actual_obs (N,dim), predicted_obs (N,dim)) or (None, None, None)."""
        if not self._planner_first_step_history:
            return (None, None, None)
        ts = np.array([h[0] for h in self._planner_first_step_history], dtype=np.float64)
        actual = np.concatenate([h[1] for h in self._planner_first_step_history], axis=0).astype(np.float64, copy=False)
        predicted = np.concatenate([h[2] for h in self._planner_first_step_history], axis=0).astype(
            np.float64, copy=False
        )
        return (ts, actual, predicted)

    def get_idm_inverse_history(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return (timestamps, actual_action_chunk (N,K,A), predicted_action_chunk (N,K,A)) or (None, None, None)."""
        if not self._idm_inverse_history:
            return (None, None, None)
        return self._flatten_chunk_history(self._idm_inverse_history)

    def get_fdm_forward_history(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return (timestamps, actual_obs_chunk (N,K,O), predicted_obs_chunk (N,K,O)) or (None, None, None)."""
        if not self._fdm_forward_history:
            return (None, None, None)
        return self._flatten_chunk_history(self._fdm_forward_history)

    def get_vel_state(self) -> np.ndarray | None:
        """Return the most recent base-frame velocity estimate."""
        return self._latest_vel

    def get_last_mocap_timestamp(self) -> float | None:
        """Timestamp of the most recent mocap sample (e.g. sim_time in MuJoCo). Used to record command in same time base."""
        return self._last_ts

    def get_pose_state(self) -> np.ndarray | None:
        """Return the most recent base pose (position and quaternion in xyzw format)."""
        return self._latest_pose

    def reset(self) -> None:
        """Clear cached history; call this when the simulator/robot resets."""
        self._last_pos = None
        self._last_quat = None
        self._last_ts = None
        self._latest_vel = np.zeros((1, 6), dtype=np.float64)
        self._latest_pose = np.zeros((1, 7), dtype=np.float64)
        if self.record_mocap_history:
            self._mocap_history.clear()
            self._command_history.clear()
            self._recon_history.clear()
            self._planner_first_step_history.clear()
            self._idm_inverse_history.clear()
            self._fdm_forward_history.clear()

    def reset_session_history(self) -> None:
        """Clear per-session recorded history while keeping latest pose/velocity caches."""
        self._mocap_history.clear()
        self._command_history.clear()
        self._recon_history.clear()
        self._planner_first_step_history.clear()
        self._idm_inverse_history.clear()
        self._fdm_forward_history.clear()

    def set_recording_enabled(self, enabled: bool) -> None:
        """Enable or disable session recording without affecting live pose cache."""
        self._recording_enabled = bool(enabled)

    def get_mocap_history(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return (timestamps, positions (N,3), quaternions_xyzw (N,4)) or (None, None, None) if empty."""
        if not self._mocap_history:
            return (None, None, None)
        ts = np.array([h[0] for h in self._mocap_history], dtype=np.float64)
        pos = np.array([h[1] for h in self._mocap_history], dtype=np.float64)
        quat = np.array([h[2] for h in self._mocap_history], dtype=np.float64)
        return (ts, pos, quat)

    def plot_mocap_raw_on_exit(
        self,
        save_path: str | None = None,
        tracking_reward_checkpoint_path: str | None = None,
    ) -> None:
        """
        On exit: save unified log (mocap raw + command + reconstruction); batch-compute
        displacement and velocity; plot raw, displacement, velocity, reconstruction.
        If no mocap data was received, still saves command+recon to npz and logs a warning.
        """
        try:
            self._plot_mocap_raw_on_exit_impl(
                save_path,
                tracking_reward_checkpoint_path=tracking_reward_checkpoint_path,
            )
        except Exception as e:
            logger.exception(f"Exit plot/log failed: {e}")

    def _plot_mocap_raw_on_exit_impl(
        self,
        save_path: str | None = None,
        *,
        tracking_reward_checkpoint_path: str | None = None,
    ) -> None:
        from pathlib import Path

        from holosoma_inference.utils.plot_inference_log import (
            _moving_average,
            _remove_outliers,
        )
        from holosoma_inference.utils.plot_mocap_raw import (
            _trim_initial_final_steps,
            _compute_velocity_tracking_summary,
            compute_velocity_and_displacement_from_raw,
            plot_mocap_raw_history,
            resample_mocap_to_hz,
            save_mocap_unified_log,
            _plot_displacement_tracking,
            _plot_reconstruction,
            _plot_velocity_tracking,
            describe_metric_window,
            fall_criteria_log_lines,
            metric_window_log_lines,
        )

        ts, pos, quat = self.get_mocap_history()
        if ts is not None and len(ts) > 0 and getattr(self, "mocap_plot_hz", 0) > 0:
            ts, pos, quat = resample_mocap_to_hz(ts, pos, quat, self.mocap_plot_hz)
        cmd_ts, cmd_x, cmd_y, cmd_yaw = self.get_command_history()
        recon_ts, actual_target, predicted_target = self.get_recon_history()
        planner_ts, planner_actual_obs, planner_predicted_obs = self.get_planner_first_step_history()
        idm_ts, idm_actual_chunk, idm_predicted_chunk = self.get_idm_inverse_history()
        fdm_ts, fdm_actual_chunk, fdm_predicted_chunk = self.get_fdm_forward_history()

        # Save full log first
        save_dir = Path(save_path).parent if save_path else None
        raw_path = save_path
        disp_path = str(save_dir / "mocap_displacement.png") if save_dir else None
        vel_path = str(save_dir / "mocap_velocity.png") if save_dir else None
        recon_path = str(save_dir / "mocap_reconstruction.png") if save_dir else None
        log_path = str(save_dir / "mocap_unified.npz") if save_dir else None

        if log_path:
            save_mocap_unified_log(
                ts,
                pos,
                quat,
                cmd_ts,
                cmd_x,
                cmd_y,
                cmd_yaw,
                recon_ts,
                actual_target,
                predicted_target,
                compact_term_order=self._compact_term_order if self._compact_term_order else None,
                compact_term_dims=self._compact_term_dims if self._compact_term_dims else None,
                planner_first_step_timestamps=planner_ts,
                planner_first_step_actual_obs=planner_actual_obs,
                planner_first_step_predicted_obs=planner_predicted_obs,
                idm_timestamps=idm_ts,
                idm_actual_action_chunk=idm_actual_chunk,
                idm_predicted_action_chunk=idm_predicted_chunk,
                fdm_timestamps=fdm_ts,
                fdm_actual_obs_chunk=fdm_actual_chunk,
                fdm_predicted_obs_chunk=fdm_predicted_chunk,
                tracking_reward_checkpoint_path=tracking_reward_checkpoint_path,
                plot_trim_head=self.plot_trim_head,
                plot_trim_tail=self.plot_trim_tail,
                tracking_step_hz=self.mocap_plot_hz if getattr(self, "mocap_plot_hz", 0) > 0 else 50.0,
                path=log_path,
            )

        head, tail = self.plot_trim_head, self.plot_trim_tail
        # State the window every metric below is computed over, before computing it. With
        # head=tail=0 that window is the whole run, gantry transient included.
        metric_window = describe_metric_window(
            len(ts) if ts is not None else 0,
            head,
            tail,
            rate_hz=self.mocap_plot_hz if getattr(self, "mocap_plot_hz", 0) > 0 else 50.0,
        )
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
            checkpoint_path=tracking_reward_checkpoint_path,
            trim_head=head,
            trim_tail=tail,
            step_hz=self.mocap_plot_hz if getattr(self, "mocap_plot_hz", 0) > 0 else 50.0,
        )
        # Both fall verdicts: the collector's criterion and step 6's disagree between 45.573
        # and 60.000 degrees of tilt. Reporting only.
        for line in fall_criteria_log_lines(tracking_summary):
            logger.warning(line)
        if ts is not None and head + tail > 0 and len(ts) > head + tail:
            ts, pos, quat = _trim_initial_final_steps(len(ts), head, ts, pos, quat, trim_tail=tail)
        if recon_ts is not None and head + tail > 0 and len(recon_ts) > head + tail:
            recon_ts, actual_target, predicted_target = _trim_initial_final_steps(
                len(recon_ts), head, recon_ts, actual_target, predicted_target, trim_tail=tail,
            )
        if planner_ts is not None and head + tail > 0 and len(planner_ts) > head + tail:
            planner_ts, planner_actual_obs, planner_predicted_obs = _trim_initial_final_steps(
                len(planner_ts),
                head,
                planner_ts,
                planner_actual_obs,
                planner_predicted_obs,
                trim_tail=tail,
            )
        if idm_ts is not None and head + tail > 0 and len(idm_ts) > head + tail:
            idm_ts, idm_actual_chunk, idm_predicted_chunk = _trim_initial_final_steps(
                len(idm_ts),
                head,
                idm_ts,
                idm_actual_chunk,
                idm_predicted_chunk,
                trim_tail=tail,
            )
        if fdm_ts is not None and head + tail > 0 and len(fdm_ts) > head + tail:
            fdm_ts, fdm_actual_chunk, fdm_predicted_chunk = _trim_initial_final_steps(
                len(fdm_ts),
                head,
                fdm_ts,
                fdm_actual_chunk,
                fdm_predicted_chunk,
                trim_tail=tail,
            )

        if ts is None or len(ts) == 0:
            logger.warning(
                "No mocap data received; only command/recon saved. "
                "Check ZMQ (task.vel_state_zmq_url) and that mocap is publishing."
            )
            return

        t, base_pos_x, base_pos_y, base_angle, base_vx, base_vy, base_yaw = (
            compute_velocity_and_displacement_from_raw(ts, pos, quat)
        )
        max_lv = self.max_linear_vel
        max_av = self.max_angular_vel
        sw = self.smooth_window
        # Plot raw mocap velocities; RMSE matches metrics (masked outliers + optional smooth).
        vx_rmse = _remove_outliers(np.asarray(base_vx, dtype=np.float64).copy(), max_lv)
        vy_rmse = _remove_outliers(np.asarray(base_vy, dtype=np.float64).copy(), max_lv)
        yaw_rmse = _remove_outliers(np.asarray(base_yaw, dtype=np.float64).copy(), max_av)
        if sw > 1:
            vx_rmse = _moving_average(vx_rmse, sw)
            vy_rmse = _moving_average(vy_rmse, sw)
            yaw_rmse = _moving_average(yaw_rmse, sw)

        plot_mocap_raw_history(ts, pos, quat, save_path=raw_path, show=True)
        _plot_displacement_tracking(
            t, base_pos_x, base_pos_y, base_angle,
            cmd_ts, cmd_x, cmd_y, cmd_yaw,
            ts, quat,
            save_path=disp_path, show=True,
        )
        _plot_velocity_tracking(
            t, base_vx, base_vy, base_yaw,
            cmd_ts, cmd_x, cmd_y, cmd_yaw,
            save_path=vel_path, show=True,
            t0_mocap=float(ts[0]) if len(ts) > 0 else None,
            mocap_ts=ts,
            velocity_for_rmse=(vx_rmse, vy_rmse, yaw_rmse),
            tracking_summary=tracking_summary,
            metric_window=metric_window,
        )
        _plot_reconstruction(
            recon_ts, actual_target, predicted_target,
            save_path=recon_path, show=True,
            planner_first_step_ts=planner_ts,
            planner_first_step_actual=planner_actual_obs,
            planner_first_step_predicted=planner_predicted_obs,
            compact_term_order=self._compact_term_order if self._compact_term_order else None,
            compact_term_dims=self._compact_term_dims if self._compact_term_dims else None,
            idm_ts=idm_ts,
            idm_actual_chunk=idm_actual_chunk,
            idm_predicted_chunk=idm_predicted_chunk,
            fdm_ts=fdm_ts,
            fdm_actual_chunk=fdm_actual_chunk,
            fdm_predicted_chunk=fdm_predicted_chunk,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _to_wxyz(self, quat: Iterable[float]) -> np.ndarray:
        quat_np = np.asarray(quat, dtype=np.float64).reshape(-1)
        if quat_np.shape[0] != 4:
            raise ValueError(f"Expected quaternion of length 4, got shape {quat_np.shape}")

        if self.orientation_order.lower() == "wxyz":
            return quat_np
        if self.orientation_order.lower() == "xyzw":
            return xyzw_to_wxyz(quat_np.reshape(1, 4)).reshape(-1)
        raise ValueError(f"Unsupported orientation_order '{self.orientation_order}'")

    def _flatten_chunk_history(
        self,
        history: list[tuple[float, np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        timestamps: list[np.ndarray] = []
        actual_chunks: list[np.ndarray] = []
        predicted_chunks: list[np.ndarray] = []

        for timestamp, actual, predicted in history:
            actual_arr = np.asarray(actual, dtype=np.float64)
            predicted_arr = np.asarray(predicted, dtype=np.float64)

            if actual_arr.ndim == 1:
                actual_arr = actual_arr.reshape(1, 1, -1)
            elif actual_arr.ndim == 2:
                actual_arr = actual_arr.reshape(1, actual_arr.shape[0], actual_arr.shape[1])

            if predicted_arr.ndim == 1:
                predicted_arr = predicted_arr.reshape(1, 1, -1)
            elif predicted_arr.ndim == 2:
                predicted_arr = predicted_arr.reshape(1, predicted_arr.shape[0], predicted_arr.shape[1])

            n = min(int(actual_arr.shape[0]), int(predicted_arr.shape[0]))
            if n <= 0:
                continue

            actual_chunks.append(actual_arr[:n])
            predicted_chunks.append(predicted_arr[:n])
            timestamps.append(np.full((n,), float(timestamp), dtype=np.float64))

        if not actual_chunks or not predicted_chunks:
            return (None, None, None)

        ts = np.concatenate(timestamps, axis=0)
        actual = np.concatenate(actual_chunks, axis=0)
        predicted = np.concatenate(predicted_chunks, axis=0)
        return (ts, actual, predicted)

    def _quat_delta_to_ang_vel(self, prev_q: np.ndarray, curr_q: np.ndarray, dt: float) -> np.ndarray:
        """Compute world-frame angular velocity from two quaternions."""
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

    def update(self, position: Iterable[float], orientation: Iterable[float], timestamp: float | None = None) -> None:
        """Update velocity estimate using a new pose sample."""
        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        quat_wxyz = self._to_wxyz(orientation)
        ts = float(timestamp) if timestamp is not None else time.time()

        if pos.shape[0] != 3:
            raise ValueError(f"Expected position length 3, got shape {pos.shape}")

        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

        if self.record_mocap_history:
            # Only record raw data; no velocity computation during run.
            self._latest_pose = np.concatenate([pos.reshape(1, 3), quat_xyzw.reshape(1, 4)], axis=1)
            if self._recording_enabled:
                # Detect time reset (e.g. stale Phase 1 messages followed by
                # Phase 2 messages after sim-time reset in sync stepping).
                # Discard earlier entries when a large backward time jump occurs.
                if self._mocap_history and ts < self._mocap_history[-1][0] - 0.1:
                    logger.debug(
                        f"Mocap time reset detected: {self._mocap_history[-1][0]:.4f} → {ts:.4f}; "
                        f"clearing {len(self._mocap_history)} stale entries"
                    )
                    self._mocap_history.clear()
                self._mocap_history.append((ts, pos.copy(), quat_xyzw.copy()))
            self._last_ts = ts  # so get_last_mocap_timestamp() returns current sample time (e.g. sim_time in MuJoCo)
            return

        if self._last_ts is None:
            # First sample initializes history; keep velocities at zero.
            self._last_pos = pos
            self._last_quat = quat_wxyz
            self._last_ts = ts
            self._latest_pose = np.concatenate([pos.reshape(1, 3), quat_xyzw.reshape(1, 4)], axis=1)
            return

        dt = ts - self._last_ts
        if dt < 0:
            logger.warning(f"Time delta is negative: {dt}. Resetting velocity processor.")
            self.reset()
            self._last_pos = pos
            self._last_quat = quat_wxyz
            self._last_ts = ts
            return
        if dt <= 1e-6:
            return

        lin_vel_world = (pos - self._last_pos) / dt
        ang_vel_world = self._quat_delta_to_ang_vel(self._last_quat, quat_wxyz, dt)

        base_lin_vel = quat_rotate_inverse(quat_wxyz.reshape(1, 4), lin_vel_world.reshape(1, 3))
        base_ang_vel = quat_rotate_inverse(quat_wxyz.reshape(1, 4), ang_vel_world.reshape(1, 3))
        self._latest_vel = np.concatenate([base_lin_vel, base_ang_vel], axis=1)
        self._latest_pose = np.concatenate([pos.reshape(1, 3), quat_xyzw.reshape(1, 4)], axis=1)

        self._last_pos = pos
        self._last_quat = quat_wxyz
        self._last_ts = ts
