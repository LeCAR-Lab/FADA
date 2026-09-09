from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from loguru import logger

try:
    import zmq
except Exception:  # noqa: BLE001
    zmq = None

if TYPE_CHECKING:
    from holosoma.config_types.simulator import PredictionTwinVizConfig
    from holosoma.simulator.mujoco.mujoco import MuJoCo


class PredictionTwinViz:
    """Render transformer chunk/obs prediction trajectories."""

    def __init__(self, simulator: MuJoCo, config: PredictionTwinVizConfig) -> None:
        self.simulator = simulator
        self.config = config
        self.enabled = bool(config.enabled)
        self._render_mode = self._resolve_render_mode(getattr(config, "render_mode", "skeleton"))
        self._ctx: Any | None = None
        self._sub_socket: Any | None = None
        self._latest_packet: dict[str, Any] | None = None
        self._warned_bad_packet = False
        self._warned_obs_shape = False
        self._warned_missing_feet = False

        self._body_ids: list[int] = []
        self._edge_indices: list[tuple[int, int]] = []
        self._root_body_index = 0
        self._root_body_id = 0
        self._foot_body_indices: list[int] = []
        self._foot_body_names: list[str] = []
        self._actuator_ids: list[int] = []
        self._kp = np.array([], dtype=np.float64)
        self._kd = np.array([], dtype=np.float64)
        self._torque_limits = np.array([], dtype=np.float64)

        if not self.enabled:
            return
        if zmq is None:
            logger.warning("prediction_twin_viz enabled but pyzmq is unavailable; disabling.")
            self.enabled = False
            return

        self._build_robot_topology()
        self._build_dof_pd_maps()
        self._setup_subscriber()

    def _resolve_render_mode(self, raw_mode: Any) -> str:
        normalized = str(raw_mode).strip().lower().replace("-", "_")
        if normalized in {"skeleton", "full_skeleton"}:
            return "skeleton"
        if normalized in {"foot_com", "foot_com_traj", "foot_com_trajectory", "trajectory"}:
            return "foot_com"
        logger.warning(
            f"prediction_twin_viz: unsupported render mode '{raw_mode}', falling back to 'skeleton'."
        )
        return "skeleton"

    def _build_robot_topology(self) -> None:
        model = self.simulator.root_model
        if model is None:
            self.enabled = False
            return

        prefix = getattr(self.simulator.scene_manager, "robot_prefix", "robot_")
        body_ids = [body_id for body_id in range(1, model.nbody) if model.body(body_id).name.startswith(prefix)]
        if not body_ids:
            # Fallback: treat all non-world bodies as robot bodies.
            body_ids = list(range(1, model.nbody))

        id_to_local = {body_id: i for i, body_id in enumerate(body_ids)}
        edge_indices: list[tuple[int, int]] = []
        root_local = 0
        for body_id in body_ids:
            parent_id = int(model.body_parentid[body_id])
            child_local = id_to_local[body_id]
            if parent_id in id_to_local:
                edge_indices.append((id_to_local[parent_id], child_local))
            else:
                root_local = child_local

        self._body_ids = body_ids
        self._edge_indices = edge_indices
        self._root_body_index = root_local
        self._root_body_id = int(body_ids[root_local])
        self._resolve_foot_body_local_indices(id_to_local)

    def _resolve_foot_body_local_indices(self, id_to_local: dict[int, int]) -> None:
        model = self.simulator.root_model
        if model is None:
            return

        foot_name_pattern = str(getattr(self.simulator.robot_config, "foot_body_name", "")).strip()
        matched: list[tuple[int, str]] = []
        for body_id in self._body_ids:
            clean_name = self.simulator._get_clean_name(model.body(body_id).name)
            if foot_name_pattern and foot_name_pattern in clean_name:
                matched.append((body_id, clean_name))

        if not matched:
            for body_id in self._body_ids:
                clean_name = self.simulator._get_clean_name(model.body(body_id).name)
                if "foot" in clean_name.lower():
                    matched.append((body_id, clean_name))

        if matched:
            # De-duplicate while preserving deterministic ordering.
            deduped: list[tuple[int, str]] = []
            seen_ids: set[int] = set()
            for body_id, clean_name in matched:
                if body_id in seen_ids:
                    continue
                seen_ids.add(body_id)
                deduped.append((body_id, clean_name))
            matched = deduped

        target_num_feet = int(getattr(self.simulator.robot_config, "num_feet", 0))
        if target_num_feet > 0 and len(matched) > target_num_feet:
            matched = matched[:target_num_feet]

        self._foot_body_indices = [id_to_local[body_id] for body_id, _ in matched if body_id in id_to_local]
        self._foot_body_names = [name for _, name in matched]
        if self._render_mode == "foot_com" and not self._foot_body_indices:
            logger.warning(
                "prediction_twin_viz: render_mode='foot_com' but no foot bodies were resolved. "
                "Rendering will fall back to skeleton mode."
            )

    def _kp_kd_from_control_config(self) -> tuple[list[float], list[float]]:
        """Build kp/kd lists from robot_config.control.stiffness/damping (holosoma config)."""
        stiffness_dict = self.simulator.robot_config.control.stiffness
        damping_dict = self.simulator.robot_config.control.damping
        kp_list: list[float] = []
        kd_list: list[float] = []
        for dof_name in self.simulator.dof_names:
            matches = [p for p in stiffness_dict if p in dof_name]
            if len(matches) != 1:
                raise ValueError(
                    f"prediction_twin_viz: expected exactly 1 stiffness pattern for DOF '{dof_name}', "
                    f"got {len(matches)}: {matches}"
                )
            pattern = matches[0]
            kp_list.append(float(stiffness_dict[pattern]))
            kd_list.append(float(damping_dict[pattern]))
        return kp_list, kd_list

    def _build_dof_pd_maps(self) -> None:
        model = self.simulator.root_model
        if model is None:
            self.enabled = False
            return

        actuator_ids: list[int] = []
        for dof_name in self.simulator.dof_names:
            actuator_name = self.simulator._get_prefixed_name(dof_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id == -1:
                raise ValueError(
                    f"prediction_twin_viz: actuator for DOF '{dof_name}' "
                    f"(MuJoCo name '{actuator_name}') was not found."
                )
            actuator_ids.append(int(actuator_id))
        self._actuator_ids = actuator_ids
        # Holosoma RobotConfig uses control.stiffness / control.damping (dict);
        # holosoma_inference RobotConfig uses motor_kp / motor_kd (lists). Support both.
        if hasattr(self.simulator.robot_config, "motor_kp") and hasattr(
            self.simulator.robot_config, "motor_kd"
        ):
            kp_src = self.simulator.robot_config.motor_kp
            kd_src = self.simulator.robot_config.motor_kd
            if kp_src is not None and kd_src is not None:
                self._kp = np.asarray(kp_src, dtype=np.float64)
                self._kd = np.asarray(kd_src, dtype=np.float64)
            else:
                kp_list, kd_list = self._kp_kd_from_control_config()
                self._kp = np.asarray(kp_list, dtype=np.float64)
                self._kd = np.asarray(kd_list, dtype=np.float64)
        else:
            kp_list, kd_list = self._kp_kd_from_control_config()
            self._kp = np.asarray(kp_list, dtype=np.float64)
            self._kd = np.asarray(kd_list, dtype=np.float64)
        self._torque_limits = np.asarray(self.simulator.robot_config.dof_effort_limit_list, dtype=np.float64)

        dof_count = int(self.simulator.num_dof)
        if self._kp.size != dof_count or self._kd.size != dof_count:
            raise ValueError(
                "prediction_twin_viz: KP/KD dim mismatch with num_dof: "
                f"kp={self._kp.size} kd={self._kd.size} num_dof={dof_count}"
            )
        if self._torque_limits.size != dof_count:
            raise ValueError(
                "prediction_twin_viz: torque limit dim mismatch with num_dof: "
                f"tau={self._torque_limits.size} num_dof={dof_count}"
            )

    def _setup_subscriber(self) -> None:
        assert zmq is not None
        try:
            self._ctx = zmq.Context.instance()
            self._sub_socket = self._ctx.socket(zmq.SUB)
            self._sub_socket.setsockopt(zmq.LINGER, 0)
            self._sub_socket.setsockopt(zmq.RCVHWM, 2)
            self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._sub_socket.bind(str(self.config.sub_url))
            logger.info(f"prediction_twin_viz subscriber bound: {self.config.sub_url}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"prediction_twin_viz failed to bind subscriber {self.config.sub_url}: {exc}")
            self.enabled = False
            self._sub_socket = None

    def close(self) -> None:
        if self._sub_socket is not None:
            try:
                self._sub_socket.close()
            except Exception:  # noqa: BLE001
                pass
            self._sub_socket = None

    def _poll_latest_packet(self) -> None:
        if not self.enabled or self._sub_socket is None or zmq is None:
            return
        while True:
            try:
                message = self._sub_socket.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as exc:  # noqa: BLE001
                if not self._warned_bad_packet:
                    self._warned_bad_packet = True
                    logger.warning(f"prediction_twin_viz recv failed; suppressing repeated warnings: {exc}")
                break
            packet = self._parse_packet(message)
            if packet is not None:
                self._latest_packet = packet

    def _parse_packet(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return None
        if message.get("schema") != "holosoma.transformer_chunk_twin.v1":
            return None

        try:
            control_dt = float(message["control_dt_sec"])
            if control_dt <= 0.0:
                return None
            q_target_chunk_abs = np.asarray(message["q_target_chunk_abs"], dtype=np.float32)
            if q_target_chunk_abs.ndim == 3:
                if q_target_chunk_abs.shape[0] != 1:
                    return None
                q_target_chunk_abs = q_target_chunk_abs[0]
            if q_target_chunk_abs.ndim != 2:
                return None
            if int(q_target_chunk_abs.shape[1]) != int(self.simulator.num_dof):
                return None
            max_frames = max(1, int(self.config.max_horizon_frames))
            q_target_chunk_abs = q_target_chunk_abs[:max_frames]

            obs_chunk = message.get("obs_pred_qpos_chunk_abs")
            obs_pred_qpos_chunk_abs = None
            if obs_chunk is not None:
                obs_pred_qpos_chunk_abs = np.asarray(obs_chunk, dtype=np.float32)
                if obs_pred_qpos_chunk_abs.ndim == 3:
                    if obs_pred_qpos_chunk_abs.shape[0] != 1:
                        obs_pred_qpos_chunk_abs = None
                    else:
                        obs_pred_qpos_chunk_abs = obs_pred_qpos_chunk_abs[0]
                if obs_pred_qpos_chunk_abs is not None:
                    if (
                        obs_pred_qpos_chunk_abs.ndim != 2
                        or int(obs_pred_qpos_chunk_abs.shape[1]) != int(self.simulator.num_dof)
                    ):
                        if not self._warned_obs_shape:
                            self._warned_obs_shape = True
                            logger.warning(
                                "prediction_twin_viz received invalid obs_pred_qpos_chunk_abs shape; "
                                "obs twin rendering will be skipped for this packet."
                            )
                        obs_pred_qpos_chunk_abs = None
                    else:
                        obs_pred_qpos_chunk_abs = obs_pred_qpos_chunk_abs[:max_frames]

            return {
                "timestamp_sec": float(message.get("timestamp_sec", time.time())),
                "control_dt_sec": control_dt,
                "q_target_chunk_abs": q_target_chunk_abs,
                "obs_pred_qpos_chunk_abs": obs_pred_qpos_chunk_abs,
                "meta": message.get("meta", {}),
            }
        except Exception:  # noqa: BLE001
            return None

    def _is_packet_fresh(self, packet: dict[str, Any]) -> bool:
        age = float(time.time() - float(packet["timestamp_sec"]))
        return age <= float(self.config.max_packet_age_sec)

    def _copy_data_state(self, src: mujoco.MjData, dst: mujoco.MjData) -> None:
        dst.qpos[:] = src.qpos
        dst.qvel[:] = src.qvel
        if dst.act is not None and src.act is not None and dst.act.shape == src.act.shape:
            dst.act[:] = src.act
        dst.ctrl[:] = src.ctrl

    def _snapshot_body_positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.xpos[self._body_ids], dtype=np.float32).copy()

    def _snapshot_robot_com(self, data: mujoco.MjData) -> np.ndarray:
        if 0 <= self._root_body_id < data.subtree_com.shape[0]:
            return np.asarray(data.subtree_com[self._root_body_id], dtype=np.float32).copy()
        return np.asarray(data.xpos[self._body_ids[self._root_body_index]], dtype=np.float32).copy()

    def _compute_pd_torques(self, data: mujoco.MjData, q_target_abs: np.ndarray) -> np.ndarray:
        q_actual = np.asarray(data.qpos[self.simulator.dof_qpos_addrs], dtype=np.float64)
        dq_actual = np.asarray(data.qvel[self.simulator.dof_qvel_addrs], dtype=np.float64)
        q_target = np.asarray(q_target_abs, dtype=np.float64)
        torques = self._kp * (q_target - q_actual) - self._kd * dq_actual
        return np.clip(torques, -self._torque_limits, self._torque_limits)

    def _rollout_chunk_positions(
        self,
        base_data: mujoco.MjData,
        q_target_chunk_abs: np.ndarray,
        control_dt_sec: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        model = self.simulator.root_model
        assert model is not None
        temp_data = mujoco.MjData(model)
        self._copy_data_state(base_data, temp_data)
        mujoco.mj_forward(model, temp_data)

        hold_steps = max(1, int(round(float(control_dt_sec) / float(self.simulator.sim_dt))))
        frames: list[np.ndarray] = []
        com_frames: list[np.ndarray] = []
        for q_target in q_target_chunk_abs:
            for _ in range(hold_steps):
                torques = self._compute_pd_torques(temp_data, q_target)
                temp_data.ctrl[self._actuator_ids] = torques
                mujoco.mj_step(model, temp_data)
            frames.append(self._snapshot_body_positions(temp_data))
            com_frames.append(self._snapshot_robot_com(temp_data))
        return frames, com_frames

    def _rollout_obs_positions(
        self,
        base_data: mujoco.MjData,
        obs_pred_qpos_chunk_abs: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        model = self.simulator.root_model
        assert model is not None
        temp_data = mujoco.MjData(model)
        base_qpos = np.asarray(base_data.qpos, dtype=np.float64).copy()
        base_qvel = np.asarray(base_data.qvel, dtype=np.float64).copy()

        frames: list[np.ndarray] = []
        com_frames: list[np.ndarray] = []
        for q_target in obs_pred_qpos_chunk_abs:
            temp_data.qpos[:] = base_qpos
            temp_data.qvel[:] = base_qvel
            temp_data.qpos[self.simulator.dof_qpos_addrs] = np.asarray(q_target, dtype=np.float64)
            temp_data.qvel[self.simulator.dof_qvel_addrs] = 0.0
            mujoco.mj_forward(model, temp_data)
            frames.append(self._snapshot_body_positions(temp_data))
            com_frames.append(self._snapshot_robot_com(temp_data))
        return frames, com_frames

    def _draw_skeleton_frames(self, frames: list[np.ndarray], base_rgb: np.ndarray) -> None:
        if not frames:
            return
        num_frames = len(frames)
        for frame_idx, positions in enumerate(frames):
            if num_frames > 1:
                fade = 1.0 - 0.55 * (frame_idx / (num_frames - 1))
            else:
                fade = 1.0
            color = np.clip(base_rgb * fade + 0.12, 0.0, 1.0)

            for parent_idx, child_idx in self._edge_indices:
                self.simulator.draw_line(
                    positions[parent_idx],
                    positions[child_idx],
                    color,
                    env_id=0,
                )

            self.simulator.draw_sphere(
                positions[self._root_body_index],
                radius=0.012,
                color=color,
                env_id=0,
            )

    def _draw_foot_com_trajectories(
        self,
        frames: list[np.ndarray],
        com_frames: list[np.ndarray],
        base_rgb: np.ndarray,
    ) -> None:
        if not frames:
            return
        if not self._foot_body_indices:
            if not self._warned_missing_feet:
                self._warned_missing_feet = True
                logger.warning(
                    "prediction_twin_viz: no foot body indices resolved; "
                    "falling back to skeleton rendering for this twin."
                )
            self._draw_skeleton_frames(frames, base_rgb)
            return

        num_frames = len(frames)
        if num_frames >= 2:
            for foot_idx in self._foot_body_indices:
                for frame_idx in range(num_frames - 1):
                    fade = 1.0 - 0.55 * (frame_idx / (num_frames - 2)) if num_frames > 2 else 1.0
                    color = np.clip(base_rgb * fade + 0.12, 0.0, 1.0)
                    self.simulator.draw_line(
                        frames[frame_idx][foot_idx],
                        frames[frame_idx + 1][foot_idx],
                        color,
                        env_id=0,
                    )

        if not com_frames:
            return
        if len(com_frames) >= 2:
            num_com_frames = len(com_frames)
            for frame_idx in range(num_com_frames - 1):
                fade = 1.0 - 0.55 * (frame_idx / (num_com_frames - 2)) if num_com_frames > 2 else 1.0
                color = np.clip(base_rgb * fade + 0.12, 0.0, 1.0)
                self.simulator.draw_line(
                    com_frames[frame_idx],
                    com_frames[frame_idx + 1],
                    color,
                    env_id=0,
                )

    def _draw_frames(self, frames: list[np.ndarray], com_frames: list[np.ndarray], base_rgb: np.ndarray) -> None:
        if self._render_mode == "foot_com":
            self._draw_foot_com_trajectories(frames, com_frames, base_rgb)
            return
        self._draw_skeleton_frames(frames, base_rgb)

    def draw(self) -> None:
        if not self.enabled:
            return
        if self.simulator.root_data is None or self.simulator.root_model is None:
            return
        self._poll_latest_packet()
        packet = self._latest_packet
        if packet is None:
            return
        if not self._is_packet_fresh(packet):
            return

        # Color semantics:
        # - action rollout (executing predicted action): red
        # - obs prediction rollout: blue
        action_rollout_color = np.array([0.95, 0.20, 0.20], dtype=np.float32)
        obs_prediction_color = np.array([0.20, 0.50, 0.95], dtype=np.float32)

        chunk_frames, chunk_com_frames = self._rollout_chunk_positions(
            self.simulator.root_data,
            packet["q_target_chunk_abs"],
            float(packet["control_dt_sec"]),
        )
        self._draw_frames(chunk_frames, chunk_com_frames, action_rollout_color)

        if bool(self.config.show_obs_prediction):
            obs_chunk = packet.get("obs_pred_qpos_chunk_abs")
            if obs_chunk is not None:
                obs_frames, obs_com_frames = self._rollout_obs_positions(self.simulator.root_data, obs_chunk)
                self._draw_frames(obs_frames, obs_com_frames, obs_prediction_color)
