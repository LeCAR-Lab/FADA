from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from holosoma.fada.common.current_command import (
    DEFAULT_COMMAND_PROFILE,
    build_current_command_np,
    command_components_for_profile,
    normalize_command_profile,
)
from loguru import logger

from holosoma_inference.utils.chunk_twin_publisher import (
    ChunkTwinPublisher,
    build_chunk_twin_packet,
    build_compact_term_slices,
    build_q_target_chunk_abs,
    model_name_from_path,
    recover_obs_pred_dof_pos_abs,
)

from .base import TRANSFORMER_FINETUNE_DYNAMICS_TERMS, BasePolicy
from .locomotion import LocomotionPolicy_Deploy

_SUPPORTED_COMPACT_TERMS = TRANSFORMER_FINETUNE_DYNAMICS_TERMS


def _as_2d_float32(array: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out


class LocomotionPolicy_FADA(LocomotionPolicy_Deploy):
    """Transformer/Planner+IDM ONNX deploy policy with history buffers, compact-obs
    preprocessing, and inverse-dynamics error logging.

    Serves both dispatch entries (policy_mode="transformer" and "fada"). Supports a
    plain compact-obs transformer ONNX (no teacher_future_obs/idm_teacher_actions
    I/O) and a full Planner+IDM ONNX: the teacher I/O handling below is a no-op
    whenever those names are absent from the loaded ONNX graph.

    NOTE: ``_metadata_prefix`` is "transformer" -- it is the literal key prefix
    (``transformer_dims``/``transformer_runtime``/``transformer_preprocess``) baked
    into exported ONNX metadata by ``holosoma.fada.planner_idm.eval_checkpoint``.
    The two names are one contract; changing one requires changing the exporter.
    """

    _metadata_prefix = "transformer"
    _policy_label = "Transformer"

    def __init__(self, config):
        self._idm_teacher_input_name = "teacher_future_obs"
        self._idm_teacher_output_name = "idm_teacher_actions"
        self._fdm_teacher_input_name = "teacher_future_actions"
        self._fdm_teacher_output_name = "fdm_teacher_future_obs"
        self._idm_teacher_supported = False
        self._fdm_teacher_supported = False
        self.idm_use_current_command_for_history = False
        self.fdm_enabled = False
        self._last_policy_input_for_teacher: dict[str, np.ndarray] | None = None
        self._idm_inverse_running_sq_sum = 0.0
        self._idm_inverse_running_count = 0
        self._idm_inverse_running_sq_sum_per_horizon = np.zeros((0,), dtype=np.float64)
        self._idm_inverse_running_count_per_horizon = np.zeros((0,), dtype=np.int64)
        self._idm_inverse_pending_windows: list[dict[str, Any]] = []
        self._planner_first_step_running_sq_sum = 0.0
        self._planner_first_step_running_count = 0
        self._planner_first_step_running_sq_sum_per_term = np.zeros((0,), dtype=np.float64)
        self._planner_first_step_running_count_per_term = np.zeros((0,), dtype=np.int64)
        self._pending_planner_first_step_mask: np.ndarray | None = None
        self._compact_term_slices: dict[str, slice] = {}
        self._fdm_forward_running_sq_sum = 0.0
        self._fdm_forward_running_count = 0
        self._fdm_forward_running_sq_sum_per_horizon = np.zeros((0,), dtype=np.float64)
        self._fdm_forward_running_count_per_horizon = np.zeros((0,), dtype=np.int64)

        self.obs_dim = 0
        self.act_dim = 0
        self.cmd_dim = 0
        self.history_len = 0
        self.pred_horizon = 0
        self.predict_future_obs = False
        self._command_input_name = "current_command"
        self.command_profile = DEFAULT_COMMAND_PROFILE

        self.compact_obs_add_noise = False
        self.compact_term_order: tuple[str, ...] = tuple(_SUPPORTED_COMPACT_TERMS)
        self.compact_term_scale: dict[str, float] = {k: 1.0 for k in _SUPPORTED_COMPACT_TERMS}
        self.compact_term_noise: dict[str, float] = {k: 0.0 for k in _SUPPORTED_COMPACT_TERMS}
        self.io_norm_fused = False

        self._history_obs = np.zeros((1, 1, 1), dtype=np.float32)
        self._history_act = np.zeros((1, 1, 1), dtype=np.float32)
        self._history_valid_mask = np.zeros((1, 1), dtype=np.bool_)
        self._prev_action = np.zeros((1, 1), dtype=np.float32)
        self._noise_rng = np.random.default_rng()
        self._compact_term_dims: dict[str, int] = {}

        self._chunk_twin_publish_obs = False
        self._chunk_twin_publish_every_n_steps = 1
        self._chunk_twin_max_horizon_frames = 10
        self._chunk_twin_step_counter = 0
        self._chunk_twin_warned_obs_missing = False
        self._chunk_twin_warned_obs_parse = False
        self._chunk_twin_publisher = ChunkTwinPublisher(enabled=False, pub_url="tcp://127.0.0.1:6001")

        super().__init__(config)

        seed = getattr(self.config.task, "seed", None)
        if seed is not None:
            self._noise_rng = np.random.default_rng(int(seed))
        self._setup_chunk_twin_publisher()

    def _setup_chunk_twin_publisher(self) -> None:
        task_cfg = getattr(self.config, "task", None)
        enabled = bool(getattr(task_cfg, "chunk_twin_publish", False))
        pub_url = str(getattr(task_cfg, "chunk_twin_pub_url", "tcp://127.0.0.1:6001"))
        self._chunk_twin_publish_obs = bool(getattr(task_cfg, "chunk_twin_publish_obs", False))
        self._chunk_twin_publish_every_n_steps = max(1, int(getattr(task_cfg, "chunk_twin_publish_every_n_steps", 1)))
        self._chunk_twin_max_horizon_frames = max(1, int(getattr(task_cfg, "chunk_twin_max_horizon_frames", 10)))
        self._chunk_twin_publisher = ChunkTwinPublisher(enabled=enabled, pub_url=pub_url)

    def _capture_policy_state(self) -> dict:
        state = super()._capture_policy_state()
        state["command_input_name"] = str(getattr(self, "_command_input_name", "current_command"))
        state["command_profile"] = str(getattr(self, "command_profile", DEFAULT_COMMAND_PROFILE))
        state.update(
            {
                "idm_teacher_supported": bool(getattr(self, "_idm_teacher_supported", False)),
                "fdm_teacher_supported": bool(getattr(self, "_fdm_teacher_supported", False)),
                "idm_use_current_command_for_history": bool(
                    getattr(self, "idm_use_current_command_for_history", False)
                ),
                "fdm_enabled": bool(getattr(self, "fdm_enabled", False)),
                "policy_output_names": list(getattr(self, "_policy_output_names", [])),
            }
        )
        return state

    def _restore_policy_state(self, state: dict):
        super()._restore_policy_state(state)
        self._command_input_name = str(state.get("command_input_name", "current_command"))
        self.command_profile = normalize_command_profile(str(state.get("command_profile", DEFAULT_COMMAND_PROFILE)))
        self._idm_teacher_supported = bool(state.get("idm_teacher_supported", False))
        self._fdm_teacher_supported = bool(state.get("fdm_teacher_supported", False))
        self.idm_use_current_command_for_history = bool(
            state.get("idm_use_current_command_for_history", False)
        )
        self.fdm_enabled = bool(state.get("fdm_enabled", False))
        if "policy_output_names" in state:
            self._policy_output_names = list(state["policy_output_names"])
        if hasattr(self, "pred_horizon") and hasattr(self, "compact_term_order"):
            self._reset_recon_k_step_log()

    def _teacher_future_obs_placeholder(self, batch_size: int) -> dict[str, np.ndarray]:
        """Planner+IDM ONNX always wires teacher_future_obs into the graph; ORT requires it every run.

        Zeros are correct for policy-only inference: the planner heads do not use this tensor (see
        _PlannerIDMOnnxWrapper). The IDM head output is only consumed when computing teacher metrics.
        """
        if self._idm_teacher_input_name not in getattr(self, "onnx_input_names", []):
            return {}
        return {
            self._idm_teacher_input_name: np.zeros(
                (batch_size, self.pred_horizon, self.obs_dim), dtype=np.float32
            )
        }

    def _teacher_future_actions_placeholder(self, batch_size: int) -> dict[str, np.ndarray]:
        if self._fdm_teacher_input_name not in getattr(self, "onnx_input_names", []):
            return {}
        return {
            self._fdm_teacher_input_name: np.zeros(
                (batch_size, self.pred_horizon, self.act_dim), dtype=np.float32
            )
        }

    def _validate_io_contract(self, input_names: list[str], output_names: list[str]) -> None:
        teacher_output_name = getattr(self, "_idm_teacher_output_name", "idm_teacher_actions")
        teacher_input_name = getattr(self, "_idm_teacher_input_name", "teacher_future_obs")
        fdm_teacher_output_name = getattr(self, "_fdm_teacher_output_name", "fdm_teacher_future_obs")
        fdm_teacher_input_name = getattr(self, "_fdm_teacher_input_name", "teacher_future_actions")
        required_inputs = {"history_obs", "history_act", "history_valid_mask"}
        missing_inputs = required_inputs - set(input_names)
        if missing_inputs:
            raise ValueError(
                f"ONNX model missing required Planner+IDM inputs: {missing_inputs}. Found inputs: {input_names}."
            )
        if "current_command" not in input_names:
            raise ValueError(
                "ONNX model missing required Planner+IDM command input. "
                f"Expected 'current_command', found inputs: {input_names}."
            )
        self._command_input_name = "current_command"
        if teacher_output_name in output_names and teacher_input_name not in input_names:
            raise ValueError(
                f"ONNX model exports '{teacher_output_name}' but is missing input "
                f"'{teacher_input_name}'. Found inputs: {input_names}."
            )
        if fdm_teacher_output_name in output_names and fdm_teacher_input_name not in input_names:
            raise ValueError(
                f"ONNX model exports '{fdm_teacher_output_name}' but is missing input "
                f"'{fdm_teacher_input_name}'. Found inputs: {input_names}."
            )
        if "actions" not in output_names:
            raise ValueError(
                f"ONNX model missing required Planner+IDM output 'actions'. Found outputs: {output_names}."
            )

    def _metadata_key(self, suffix: str) -> str:
        return f"{self._metadata_prefix}_{suffix}"

    def _parse_onnx_metadata(self, model_path: str) -> dict[str, Any]:
        onnx_model = onnx.load(model_path)
        metadata: dict[str, Any] = {}
        for prop in onnx_model.metadata_props:
            try:
                metadata[prop.key] = json.loads(prop.value)
            except Exception:
                metadata[prop.key] = prop.value
        return metadata

    def _load_compact_preprocess_from_metadata(self, metadata: dict[str, Any]) -> None:
        dims = metadata.get(self._metadata_key("dims"))
        runtime = metadata.get(self._metadata_key("runtime"))
        preprocess = metadata.get(self._metadata_key("preprocess"))
        missing = [
            name
            for name, value in (
                (self._metadata_key("dims"), dims),
                (self._metadata_key("runtime"), runtime),
                (self._metadata_key("preprocess"), preprocess),
            )
            if not isinstance(value, dict)
        ]
        if missing:
            raise ValueError(
                f"{self._policy_label} metadata missing required fields: {missing}. "
                "Use ONNX exported by `python -m holosoma.fada.planner_idm.eval_checkpoint "
                "--checkpoint <ckpt>` (ONNX export is automatic; see README FADA section step 3)."
            )

        self.obs_dim = int(dims["obs_dim"])
        self.act_dim = int(dims["act_dim"])
        self.cmd_dim = int(dims["cmd_dim"])
        self.history_len = int(dims["history_len"])
        self.pred_horizon = int(dims["pred_horizon"])
        self.predict_future_obs = bool(runtime.get("predict_future_obs", False))
        self.command_profile = normalize_command_profile(str(runtime.get("command_profile", DEFAULT_COMMAND_PROFILE)))
        self.compact_obs_add_noise = bool(runtime.get("compact_obs_add_noise", False))
        self.io_norm_fused = bool(runtime.get("io_norm_fused", False))

        term_order_raw = preprocess.get("term_order", runtime.get("compact_obs_term_order", list(_SUPPORTED_COMPACT_TERMS)))
        term_order = tuple(str(v) for v in term_order_raw)
        unsupported = [name for name in term_order if name not in _SUPPORTED_COMPACT_TERMS]
        if unsupported:
            raise ValueError(f"Unsupported compact_obs terms in metadata: {unsupported}")
        self.compact_term_order = term_order

        term_scale_raw = preprocess.get("term_scale")
        term_noise_raw = preprocess.get("term_noise")
        if not isinstance(term_scale_raw, dict):
            raise ValueError(
                f"{self._policy_label} metadata field {self._metadata_key('preprocess')}.term_scale must be a dict."
            )
        if not isinstance(term_noise_raw, dict):
            raise ValueError(
                f"{self._policy_label} metadata field {self._metadata_key('preprocess')}.term_noise must be a dict."
            )
        missing_scale = [name for name in self.compact_term_order if name not in term_scale_raw]
        missing_noise = [name for name in self.compact_term_order if name not in term_noise_raw]
        if missing_scale:
            raise ValueError(f"Transformer metadata term_scale missing terms: {missing_scale}")
        if missing_noise:
            raise ValueError(f"Transformer metadata term_noise missing terms: {missing_noise}")

        self.compact_term_scale = {name: float(term_scale_raw[name]) for name in self.compact_term_order}
        self.compact_term_noise = {name: float(term_noise_raw[name]) for name in self.compact_term_order}

        term_dims: dict[str, int] = {}
        for name in self.compact_term_order:
            if name in self.obs_dims:
                term_dims[name] = int(self.obs_dims[name])
            elif name == "dof_pos" or name == "dof_vel":
                term_dims[name] = int(self.num_dofs)
            elif name == "base_ang_vel" or name == "projected_gravity":
                term_dims[name] = 3
            else:
                raise ValueError(f"Cannot resolve compact term dim for '{name}'")

        total_dim = int(sum(term_dims.values()))
        if total_dim != int(self.obs_dim):
            raise ValueError(
                f"{self._policy_label} compact term dims do not match obs_dim: "
                f"sum={total_dim}, obs_dim={self.obs_dim}, term_dims={term_dims}, term_order={self.compact_term_order}"
            )
        self._compact_term_dims = term_dims

        runtime = metadata.get("transformer_runtime")
        if isinstance(runtime, dict):
            self.idm_use_current_command_for_history = bool(
                runtime.get("idm_use_current_command_for_history", False)
            )
            self.fdm_enabled = bool(runtime.get("fdm_enabled", False))
        self._compact_term_slices = build_compact_term_slices(self.compact_term_order, self._compact_term_dims)

    def _reset_transformer_state(self) -> None:
        if self.history_len <= 0 or self.obs_dim <= 0 or self.act_dim <= 0 or self.cmd_dim <= 0:
            return
        self._history_obs = np.zeros((1, self.history_len, self.obs_dim), dtype=np.float32)
        self._history_act = np.zeros((1, self.history_len, self.act_dim), dtype=np.float32)
        self._history_valid_mask = np.zeros((1, self.history_len), dtype=np.bool_)
        self._prev_action = np.zeros((1, self.act_dim), dtype=np.float32)

    def _warn_if_observation_prescale_enabled(self) -> None:
        # Transformer compact obs scaling/noise is sourced from ONNX metadata, so the
        # inference observation config scales must stay identity.
        non_identity: dict[str, float] = {}
        for key in _SUPPORTED_COMPACT_TERMS:
            scale = float(self.obs_scales.get(key, 1.0))
            if abs(scale - 1.0) > 1e-6:
                non_identity[key] = scale
        if non_identity:
            logger.warning(
                "Transformer inference ignores observation.obs_scales for compact terms "
                "(raw terms are used before metadata scaling). Non-identity config scales: "
                f"{non_identity}"
            )

    def _sync_compact_obs_layout_to_vel_processor(self) -> None:
        proc = getattr(getattr(self, "interface", None), "vel_state_processor", None)
        if proc is None or not hasattr(proc, "set_compact_obs_layout"):
            return
        proc.set_compact_obs_layout(self.compact_term_order, self._compact_term_dims)

    def _reset_recon_k_step_log(self):
        super()._reset_recon_k_step_log()
        self._last_policy_input_for_teacher = None
        self._idm_inverse_running_sq_sum = 0.0
        self._idm_inverse_running_count = 0
        self._idm_inverse_running_sq_sum_per_horizon = np.zeros((int(self.pred_horizon),), dtype=np.float64)
        self._idm_inverse_running_count_per_horizon = np.zeros((int(self.pred_horizon),), dtype=np.int64)
        self._idm_inverse_pending_windows = []
        self._planner_first_step_running_sq_sum = 0.0
        self._planner_first_step_running_count = 0
        self._planner_first_step_running_sq_sum_per_term = np.zeros((len(self.compact_term_order),), dtype=np.float64)
        self._planner_first_step_running_count_per_term = np.zeros((len(self.compact_term_order),), dtype=np.int64)
        self._pending_planner_first_step_mask = None
        self._fdm_forward_running_sq_sum = 0.0
        self._fdm_forward_running_count = 0
        self._fdm_forward_running_sq_sum_per_horizon = np.zeros((int(self.pred_horizon),), dtype=np.float64)
        self._fdm_forward_running_count_per_horizon = np.zeros((int(self.pred_horizon),), dtype=np.int64)
        self._pending_dynamics_prediction = None

    def _make_teacher_window(self, policy_input: dict[str, np.ndarray]) -> dict[str, Any] | None:
        history_valid = np.asarray(policy_input.get("history_valid_mask"), dtype=bool)
        if history_valid.ndim == 1:
            history_valid = history_valid[None, :]
        if history_valid.ndim != 2:
            return None
        eligible_mask = np.all(history_valid, axis=1)
        if not np.any(eligible_mask):
            return None
        return {
            "policy_input": {k: np.array(v, copy=True) for k, v in policy_input.items()},
            "eligible_mask": eligible_mask.astype(bool, copy=True),
            "future_obs": [],
            "future_actions": [],
        }

    def _planner_running_term_mse(self) -> np.ndarray:
        if self._planner_first_step_running_count_per_term.size == 0:
            return np.zeros((0,), dtype=np.float32)
        running = np.zeros_like(self._planner_first_step_running_sq_sum_per_term, dtype=np.float64)
        valid = self._planner_first_step_running_count_per_term > 0
        running[valid] = (
            self._planner_first_step_running_sq_sum_per_term[valid]
            / self._planner_first_step_running_count_per_term[valid]
        )
        return running.astype(np.float32, copy=False)

    def _term_step_mse(self, diff_sq: np.ndarray) -> np.ndarray:
        if not self.compact_term_order:
            return np.zeros((0,), dtype=np.float32)
        per_term = np.zeros((len(self.compact_term_order),), dtype=np.float64)
        for idx, term_name in enumerate(self.compact_term_order):
            term_sq = diff_sq[:, self._compact_term_slices[term_name]]
            per_term[idx] = float(np.mean(term_sq))
        return per_term.astype(np.float32, copy=False)

    def _log_planner_first_step_alignment(self, actual_next_obs: np.ndarray) -> None:
        self._current_step_planner_first_step = None
        if self._pending_dynamics_prediction is None:
            return

        predicted_chunk = np.asarray(self._pending_dynamics_prediction, dtype=np.float32)
        if predicted_chunk.ndim == 1:
            predicted_chunk = predicted_chunk.reshape(1, -1)
        elif predicted_chunk.ndim == 3:
            predicted_chunk = predicted_chunk[0]
        if predicted_chunk.ndim != 2 or predicted_chunk.shape[0] <= 0:
            return

        actual = np.asarray(actual_next_obs, dtype=np.float32)
        if actual.ndim == 1:
            actual = actual.reshape(1, -1)
        elif actual.ndim != 2:
            actual = actual.reshape(actual.shape[0], -1)

        predicted_first = predicted_chunk[0].reshape(1, -1)
        eligible_mask = self._pending_planner_first_step_mask
        if eligible_mask is None or eligible_mask.size != actual.shape[0]:
            eligible_mask = np.ones((actual.shape[0],), dtype=bool)
        if predicted_first.shape[0] != actual.shape[0]:
            predicted_first = np.repeat(predicted_first, actual.shape[0], axis=0)
        if not np.any(eligible_mask):
            return

        actual_target = actual[eligible_mask]
        predicted_target = predicted_first[eligible_mask]
        diff_sq = np.square(predicted_target - actual_target)
        mse = float(np.mean(diff_sq))
        self._planner_first_step_running_sq_sum += float(np.sum(diff_sq))
        self._planner_first_step_running_count += int(diff_sq.size)
        running_mse = (
            self._planner_first_step_running_sq_sum / float(self._planner_first_step_running_count)
            if self._planner_first_step_running_count > 0
            else mse
        )

        term_step_mse = self._term_step_mse(diff_sq)
        if term_step_mse.size > 0:
            for idx, term_name in enumerate(self.compact_term_order):
                term_sq = diff_sq[:, self._compact_term_slices[term_name]]
                self._planner_first_step_running_sq_sum_per_term[idx] += float(np.sum(term_sq))
                self._planner_first_step_running_count_per_term[idx] += int(term_sq.size)

        if self.state_logger is not None:
            self.state_logger.log_states(
                {
                    "planner_first_step_obs_mse": np.array([running_mse], dtype=np.float32),
                    "planner_first_step_obs_mse_step": np.array([mse], dtype=np.float32),
                    "planner_first_step_obs_mse_per_term": self._planner_running_term_mse().copy(),
                    "planner_first_step_obs_mse_per_term_step": term_step_mse.copy(),
                }
            )
        self._current_step_planner_first_step = (actual_target.copy(), predicted_target.copy())

    def _log_pending_dynamics_targets(self, obs_for_rl):
        self._current_step_recon = None
        self._current_step_idm_inverse = None
        self._current_step_fdm_forward = None
        if obs_for_rl is None or "dynamics_obs" not in obs_for_rl:
            self._pending_dynamics_prediction = None
            return

        actual_next_obs = np.asarray(obs_for_rl["dynamics_obs"], dtype=np.float32)
        self._log_planner_first_step_alignment(actual_next_obs)

        if (
            (not self._idm_teacher_supported and not self._fdm_teacher_supported)
            or self._last_policy_input_for_teacher is None
            or "history_act" not in obs_for_rl
        ):
            self._pending_dynamics_prediction = None
            return

        current_history_valid = obs_for_rl.get("history_valid_mask")
        if current_history_valid is not None:
            current_history_valid = np.asarray(current_history_valid, dtype=bool)
            if current_history_valid.ndim == 1:
                current_history_valid = current_history_valid[None, :]
            if current_history_valid.ndim != 2 or not np.all(current_history_valid):
                self._idm_inverse_pending_windows = []
                self._pending_dynamics_prediction = None
                return

        history_act = np.asarray(obs_for_rl["history_act"], dtype=np.float32)
        if history_act.ndim != 3 or history_act.shape[1] <= 0:
            self._pending_dynamics_prediction = None
            return
        actual_action = history_act[:, -1, :].astype(np.float32, copy=False)

        new_window = self._make_teacher_window(self._last_policy_input_for_teacher)
        if new_window is not None:
            self._idm_inverse_pending_windows.append(new_window)

        remaining_windows: list[dict[str, Any]] = []
        for window in self._idm_inverse_pending_windows:
            current_mask = np.asarray(window["eligible_mask"], dtype=bool)
            if not np.any(current_mask):
                continue

            window["future_obs"].append(actual_next_obs.astype(np.float32, copy=True))
            window["future_actions"].append(actual_action.astype(np.float32, copy=True))

            if len(window["future_obs"]) < int(self.pred_horizon):
                remaining_windows.append(window)
                continue

            teacher_future_obs = np.stack(window["future_obs"], axis=1).astype(np.float32, copy=False)
            actual_action_chunk = np.stack(window["future_actions"], axis=1).astype(np.float32, copy=False)
            input_feed = {k: v.copy() for k, v in window["policy_input"].items()}
            teacher_outputs: dict[str, np.ndarray] = {}
            if self._idm_teacher_supported:
                input_feed[self._idm_teacher_input_name] = teacher_future_obs
                teacher_outputs[self._idm_teacher_output_name] = self.onnx_policy_session.run(
                    [self._idm_teacher_output_name], input_feed
                )[0]
            if self._fdm_teacher_supported:
                input_feed[self._fdm_teacher_input_name] = actual_action_chunk
                teacher_outputs[self._fdm_teacher_output_name] = self.onnx_policy_session.run(
                    [self._fdm_teacher_output_name], input_feed
                )[0]

            if self._idm_teacher_output_name in teacher_outputs:
                teacher_actions = np.asarray(teacher_outputs[self._idm_teacher_output_name], dtype=np.float32)
                if teacher_actions.ndim == 2:
                    teacher_actions = teacher_actions[:, None, :]
                if teacher_actions.ndim == 3:
                    teacher_pred = teacher_actions[current_mask]
                    actual_target = actual_action_chunk[current_mask]
                    diff_sq = np.square(teacher_pred - actual_target)
                    mse = float(np.mean(diff_sq))
                    per_horizon_step_mse = np.mean(diff_sq, axis=(0, 2)).astype(np.float32, copy=False)
                    self._idm_inverse_running_sq_sum += float(np.sum(diff_sq))
                    self._idm_inverse_running_count += int(diff_sq.size)
                    self._idm_inverse_running_sq_sum_per_horizon += np.sum(diff_sq, axis=(0, 2))
                    self._idm_inverse_running_count_per_horizon += np.full(
                        (int(self.pred_horizon),),
                        int(diff_sq.shape[0] * diff_sq.shape[2]),
                        dtype=np.int64,
                    )

                    running_mse = (
                        self._idm_inverse_running_sq_sum / float(self._idm_inverse_running_count)
                        if self._idm_inverse_running_count > 0
                        else mse
                    )
                    running_per_horizon = np.zeros_like(self._idm_inverse_running_sq_sum_per_horizon, dtype=np.float64)
                    valid_h = self._idm_inverse_running_count_per_horizon > 0
                    running_per_horizon[valid_h] = (
                        self._idm_inverse_running_sq_sum_per_horizon[valid_h]
                        / self._idm_inverse_running_count_per_horizon[valid_h]
                    )

                    if self.state_logger is not None:
                        self.state_logger.log_states(
                            {
                                "idm_inverse_action_mse": np.array([running_mse], dtype=np.float32),
                                "idm_inverse_action_mse_step": np.array([mse], dtype=np.float32),
                                "idm_inverse_action_mse_per_horizon": running_per_horizon.astype(np.float32, copy=False),
                                "idm_inverse_action_mse_per_horizon_step": per_horizon_step_mse.copy(),
                                "idm_inverse_window_horizon": np.array([self.pred_horizon], dtype=np.int64),
                            }
                        )
                    self._current_step_idm_inverse = (actual_target.copy(), teacher_pred.copy())

            if self._fdm_teacher_output_name in teacher_outputs:
                teacher_future_pred = np.asarray(teacher_outputs[self._fdm_teacher_output_name], dtype=np.float32)
                if teacher_future_pred.ndim == 2:
                    teacher_future_pred = teacher_future_pred[:, None, :]
                if teacher_future_pred.ndim == 3:
                    fdm_pred = teacher_future_pred[current_mask]
                    actual_obs_target = teacher_future_obs[current_mask]
                    fdm_diff_sq = np.square(fdm_pred - actual_obs_target)
                    fdm_mse = float(np.mean(fdm_diff_sq))
                    fdm_per_horizon_step_mse = np.mean(fdm_diff_sq, axis=(0, 2)).astype(np.float32, copy=False)
                    self._fdm_forward_running_sq_sum += float(np.sum(fdm_diff_sq))
                    self._fdm_forward_running_count += int(fdm_diff_sq.size)
                    self._fdm_forward_running_sq_sum_per_horizon += np.sum(fdm_diff_sq, axis=(0, 2))
                    self._fdm_forward_running_count_per_horizon += np.full(
                        (int(self.pred_horizon),),
                        int(fdm_diff_sq.shape[0] * fdm_diff_sq.shape[2]),
                        dtype=np.int64,
                    )
                    fdm_running_mse = (
                        self._fdm_forward_running_sq_sum / float(self._fdm_forward_running_count)
                        if self._fdm_forward_running_count > 0
                        else fdm_mse
                    )
                    fdm_running_per_horizon = np.zeros_like(self._fdm_forward_running_sq_sum_per_horizon, dtype=np.float64)
                    valid_h = self._fdm_forward_running_count_per_horizon > 0
                    fdm_running_per_horizon[valid_h] = (
                        self._fdm_forward_running_sq_sum_per_horizon[valid_h]
                        / self._fdm_forward_running_count_per_horizon[valid_h]
                    )
                    if self.state_logger is not None:
                        self.state_logger.log_states(
                            {
                                "fdm_forward_obs_mse": np.array([fdm_running_mse], dtype=np.float32),
                                "fdm_forward_obs_mse_step": np.array([fdm_mse], dtype=np.float32),
                                "fdm_forward_obs_mse_per_horizon": fdm_running_per_horizon.astype(np.float32, copy=False),
                                "fdm_forward_obs_mse_per_horizon_step": fdm_per_horizon_step_mse.copy(),
                                "fdm_forward_window_horizon": np.array([self.pred_horizon], dtype=np.int64),
                            }
                        )
                    self._current_step_fdm_forward = (actual_obs_target.copy(), fdm_pred.copy())

        self._idm_inverse_pending_windows = remaining_windows
        self._pending_dynamics_prediction = None

    def _flush_pending_dynamics_log(self):
        self._pending_dynamics_prediction = None
        self._current_step_planner_first_step = None
        self._current_step_idm_inverse = None
        self._current_step_fdm_forward = None

    def _create_onnx_session(self, model_path: str) -> onnxruntime.InferenceSession:
        """Create an ONNX Runtime session using the shared provider/thread helper."""
        return super()._create_onnx_session(model_path)

    def _warmup_onnx_session(self) -> None:
        """Warm up both the policy path and the teacher IDM path."""
        import time as _time

        warmup_feed: dict[str, np.ndarray] = {
            "history_obs": np.zeros((1, self.history_len, self.obs_dim), dtype=np.float32),
            "history_act": np.zeros((1, self.history_len, self.act_dim), dtype=np.float32),
            "history_valid_mask": np.ones((1, self.history_len), dtype=np.bool_),
        }
        warmup_feed[self._command_input_name] = np.zeros((1, self.cmd_dim), dtype=np.float32)
        warmup_feed.update(self._teacher_future_obs_placeholder(1))
        warmup_feed.update(self._teacher_future_actions_placeholder(1))

        warmup_repeats = 3
        t0 = _time.perf_counter()
        for _ in range(warmup_repeats):
            self.onnx_policy_session.run(self._policy_output_names, warmup_feed)
        dt_policy = (_time.perf_counter() - t0) * 1000.0

        dt_teacher = 0.0
        dt_fdm = 0.0
        if self._idm_teacher_supported:
            teacher_feed = {**warmup_feed}
            teacher_feed.update(self._teacher_future_obs_placeholder(1))
            t1 = _time.perf_counter()
            for _ in range(warmup_repeats):
                self.onnx_policy_session.run([self._idm_teacher_output_name], teacher_feed)
            dt_teacher = (_time.perf_counter() - t1) * 1000.0
        if self._fdm_teacher_supported:
            teacher_feed = {**warmup_feed}
            teacher_feed.update(self._teacher_future_actions_placeholder(1))
            t2 = _time.perf_counter()
            for _ in range(warmup_repeats):
                self.onnx_policy_session.run([self._fdm_teacher_output_name], teacher_feed)
            dt_fdm = (_time.perf_counter() - t2) * 1000.0

        logger.info(
            f"ONNX warmup done ({warmup_repeats}x): "
            f"policy={dt_policy:.1f}ms, idm_teacher={dt_teacher:.1f}ms, fdm_teacher={dt_fdm:.1f}ms"
        )

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self._reset_transformer_state()
        self._chunk_twin_step_counter = 0
        self._chunk_twin_warned_obs_missing = False
        self._chunk_twin_warned_obs_parse = False
        self._reset_recon_k_step_log()

    def setup_policy(self, model_path):
        self.onnx_policy_session = self._create_onnx_session(model_path)
        # NOTE: model_loaded signal is written AFTER _warmup_onnx_session() below,
        # because FP32 TRT first-run warmup can take 10-15s (or longer if building from ONNX).
        input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        output_names = [out.name for out in self.onnx_policy_session.get_outputs()]
        self.onnx_input_names = input_names
        self.onnx_output_names = output_names
        self._validate_io_contract(input_names, output_names)
        self._idm_teacher_supported = (
            self._idm_teacher_input_name in input_names and self._idm_teacher_output_name in output_names
        )
        self._fdm_teacher_supported = (
            self._fdm_teacher_input_name in input_names and self._fdm_teacher_output_name in output_names
        )

        metadata = self._parse_onnx_metadata(model_path)
        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None
        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        self._load_compact_preprocess_from_metadata(metadata)
        self._warn_if_observation_prescale_enabled()
        self._reset_transformer_state()
        self._sync_compact_obs_layout_to_vel_processor()

        self._policy_output_names = [
            n for n in output_names if n not in {self._idm_teacher_output_name, self._fdm_teacher_output_name}
        ]

        logger.info(
            "Loaded Planner+IDM Transformer ONNX: "
            f"obs_dim={self.obs_dim} act_dim={self.act_dim} cmd_dim={self.cmd_dim} "
            f"history_len={self.history_len} pred_horizon={self.pred_horizon} "
            f"command_input_name={self._command_input_name} "
            f"compact_obs_add_noise={self.compact_obs_add_noise} "
            f"io_norm_fused={self.io_norm_fused} "
            f"idm_use_current_command_for_history={self.idm_use_current_command_for_history} "
            f"idm_teacher_metric={'enabled' if self._idm_teacher_supported else 'disabled'} "
            f"fdm_enabled={self.fdm_enabled} "
            f"fdm_teacher_metric={'enabled' if self._fdm_teacher_supported else 'disabled'}"
        )

        def policy_act(obs_dict):
            current_command = obs_dict.get("current_command")
            if current_command is None:
                raise KeyError("Missing current_command in planner-idm policy input.")
            input_feed = {
                "history_obs": obs_dict["history_obs"].astype(np.float32, copy=False),
                "history_act": obs_dict["history_act"].astype(np.float32, copy=False),
                "history_valid_mask": obs_dict["history_valid_mask"].astype(bool, copy=False),
            }
            input_feed[self._command_input_name] = np.asarray(current_command, dtype=np.float32)
            batch_size = int(np.asarray(obs_dict["history_obs"]).shape[0])
            input_feed.update(self._teacher_future_obs_placeholder(batch_size))
            input_feed.update(self._teacher_future_actions_placeholder(batch_size))
            self._last_policy_input_for_teacher = {k: np.array(v, copy=True) for k, v in input_feed.items()}
            outputs = self.onnx_policy_session.run(self._policy_output_names, input_feed)
            outputs_by_name = {name: outputs[i] for i, name in enumerate(self._policy_output_names)}
            self.dynamics_prediction = outputs_by_name.get("pred_future_obs", outputs_by_name.get("pred_next_obs"))
            history_valid = np.asarray(input_feed["history_valid_mask"], dtype=bool)
            if history_valid.ndim == 1:
                history_valid = history_valid[None, :]
            self._pending_planner_first_step_mask = np.all(history_valid, axis=1)
            return outputs_by_name["actions"]

        self.policy = policy_act
        self._warmup_onnx_session()
        self._write_model_loaded_signal()

    def _collect_compact_obs(self, robot_state_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Read raw terms directly; do not apply observation-config scaling here.
        obs_buf = BasePolicy.get_current_obs_buffer_dict(self, robot_state_data)
        chunks: list[np.ndarray] = []
        raw_chunks: list[np.ndarray] = []
        for term_name in self.compact_term_order:
            if term_name not in obs_buf:
                raise KeyError(f"Missing compact obs term '{term_name}' in current observation buffer.")
            value = _as_2d_float32(obs_buf[term_name])
            # Keep a copy of raw (unscaled, no extra noise) compact terms for offline trajectory finetuning.
            raw_value = value.astype(np.float32, copy=True)
            if self.compact_obs_add_noise:
                noise_scale = float(self.compact_term_noise.get(term_name, 0.0))
                if noise_scale > 0.0:
                    noise = self._noise_rng.uniform(-noise_scale, noise_scale, size=value.shape).astype(np.float32)
                    value = value + noise
            value = value * float(self.compact_term_scale.get(term_name, 1.0))
            chunks.append(value.astype(np.float32, copy=False))
            raw_chunks.append(raw_value)
        compact_obs = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
        raw_compact_obs = np.concatenate(raw_chunks, axis=1).astype(np.float32, copy=False)
        if compact_obs.shape[1] != self.obs_dim:
            raise ValueError(f"{self._policy_label} obs_dim mismatch: got {compact_obs.shape[1]}, expected {self.obs_dim}")
        if raw_compact_obs.shape[1] != self.obs_dim:
            raise ValueError(
                f"{self._policy_label} raw obs_dim mismatch: got {raw_compact_obs.shape[1]}, expected {self.obs_dim}"
            )
        return raw_compact_obs, compact_obs

    def _collect_current_command(self) -> np.ndarray:
        current_command = build_current_command_np(
            command_lin_vel=self.lin_vel_command,
            command_ang_vel=self.ang_vel_command,
            sin_phase=self._get_obs_sin_phase(),
            cos_phase=self._get_obs_cos_phase(),
            profile=self.command_profile,
        )
        if current_command.shape != (1, self.cmd_dim):
            raise ValueError(
                f"current_command shape mismatch: got {current_command.shape}, expected {(1, self.cmd_dim)}"
            )
        return current_command

    def prepare_obs_for_rl(self, robot_state_data):
        raw_obs_curr, obs_curr = self._collect_compact_obs(robot_state_data)
        current_command = self._collect_current_command()

        self._history_obs = np.roll(self._history_obs, shift=-1, axis=1)
        self._history_act = np.roll(self._history_act, shift=-1, axis=1)
        self._history_valid_mask = np.roll(self._history_valid_mask, shift=-1, axis=1)

        self._history_obs[:, -1, :] = obs_curr
        self._history_act[:, -1, :] = self._prev_action
        self._history_valid_mask[:, -1] = True

        # Populate collection payload explicitly for transformer runs.
        self.obs_buf_dict = {
            "raw_dynamics_obs": raw_obs_curr.astype(np.float32, copy=False),
            "dynamics_obs": obs_curr.astype(np.float32, copy=False),
            "current_command": current_command.astype(np.float32, copy=False),
            "history_obs": self._history_obs.astype(np.float32, copy=False),
            "history_act": self._history_act.astype(np.float32, copy=False),
            "history_valid_mask": self._history_valid_mask.astype(np.bool_, copy=False),
        }

        return {
            "history_obs": self._history_obs.astype(np.float32, copy=False),
            "history_act": self._history_act.astype(np.float32, copy=False),
            "current_command": current_command.astype(np.float32, copy=False),
            "history_valid_mask": self._history_valid_mask.astype(np.bool_, copy=False),
            "dynamics_obs": obs_curr.astype(np.float32, copy=False),
        }

    def _init_data_collector(self):
        """Initialize data collector with transformer-specific trajectory keys."""
        try:
            holosoma_path = Path(__file__).parent.parent.parent.parent / "holosoma" / "holosoma"
            if str(holosoma_path) not in sys.path:
                sys.path.insert(0, str(holosoma_path.parent.parent))

            from holosoma.utils.data_collector import DataCollector
            robot_type = getattr(self.robot_config, "robot_type", "unknown")
            simulator_type = "mujoco" if self.config.task.interface == "lo" else "real"
            output_dir = Path(self._resolve_data_collection_output_dir())

            obs_dict = {
                "raw_dynamics_obs": list(self.compact_term_order),
                "dynamics_obs": list(self.compact_term_order),
                "current_command": list(command_components_for_profile(self.command_profile)),
                "history_obs": ["history_obs"],
                "history_act": ["history_act"],
                "history_valid_mask": ["history_valid_mask"],
            }
            skip_obs_keys_cfg = tuple(
                getattr(self.config.task, "data_collection_skip_obs_keys", ("actor_obs", "critic_obs"))
            )
            skip_obs_keys = tuple(k for k in skip_obs_keys_cfg if k not in obs_dict)

            planned_steps, planned_steps_source = self._resolve_planned_collection_steps()

            self.data_collector = DataCollector(
                output_dir=str(output_dir),
                dataset_name=self.config.task.data_collection_dataset_name,
                compress=self.config.task.data_collection_compress,
                batch_size=10,
                robot_type=robot_type,
                simulator=simulator_type,
                policy_checkpoint=(
                    str(self.config.task.model_path) if isinstance(self.config.task.model_path, str) else None
                ),
                num_envs=1,
                obs_dict=obs_dict,
                skip_obs_keys=skip_obs_keys,
                # Same completion provenance as the base collector, so an interrupted
                # collection is distinguishable from a complete one on this path too.
                planned_steps=planned_steps,
                planned_steps_source=planned_steps_source,
            )
            self.logger.info(
                f"Data collection enabled: {output_dir}/{self.config.task.data_collection_dataset_name}.h5"
            )
        except Exception as e:
            # This method only runs when --task.collect-data was passed (see the guarded call
            # site in LocomotionPolicy_Deploy.__init__), so a failure here means the H5
            # dataset cannot be produced. Raise rather than leaving data_collector=None, so
            # the process exits non-zero.
            self.data_collector = None
            raise RuntimeError(
                f"--task.collect-data was requested but data collector initialization failed: {e}"
            ) from e

    def rl_inference(self, robot_state_data, obs=None):
        if obs is None:
            obs = self.prepare_obs_for_rl(robot_state_data)
        actions_chunk = np.asarray(self.policy(obs), dtype=np.float32)

        pred_obs_array = None
        if getattr(self, "dynamics_prediction", None) is not None:
            pred_obs_array = np.asarray(self.dynamics_prediction, dtype=np.float32)

        self._maybe_publish_chunk_twin(actions_chunk, pred_obs_array)

        if actions_chunk.ndim == 3:
            raw_action = actions_chunk[:, 0, :]
        elif actions_chunk.ndim == 2:
            raw_action = actions_chunk
        else:
            raise ValueError(f"Unexpected {self._policy_label} actions shape: {actions_chunk.shape}")
        raw_action = np.clip(raw_action, -100.0, 100.0).astype(np.float32, copy=False)

        pred_obs = None
        if pred_obs_array is not None:
            if pred_obs_array.ndim == 3:
                # Keep full prediction horizon for reconstruction/K-step metrics.
                pred_obs = pred_obs_array[0]
            elif pred_obs_array.ndim == 2:
                pred_obs = pred_obs_array
            elif pred_obs_array.ndim == 1:
                pred_obs = pred_obs_array

        self.last_policy_action = raw_action.copy()
        self.scaled_policy_action = raw_action * self.policy_action_scale
        self._prev_action = raw_action.copy()
        return self.scaled_policy_action, raw_action, pred_obs

    def _maybe_publish_chunk_twin(self, actions_chunk: np.ndarray, pred_obs_array: np.ndarray | None) -> None:
        self._chunk_twin_step_counter += 1
        if (self._chunk_twin_step_counter - 1) % self._chunk_twin_publish_every_n_steps != 0:
            return
        if not getattr(self._chunk_twin_publisher, "enabled", False):
            return

        try:
            q_target_chunk_abs = build_q_target_chunk_abs(
                actions_chunk_raw=actions_chunk,
                policy_action_scale=float(self.policy_action_scale),
                default_dof_angles=self.default_dof_angles,
                num_dofs=int(self.num_dofs),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to build chunk twin q_target from actions_chunk: {exc}")
            return

        max_frames = int(self._chunk_twin_max_horizon_frames)
        if max_frames > 0:
            q_target_chunk_abs = q_target_chunk_abs[:max_frames]

        obs_pred_qpos_chunk_abs = None
        if self._chunk_twin_publish_obs:
            if pred_obs_array is None:
                if not self._chunk_twin_warned_obs_missing:
                    self._chunk_twin_warned_obs_missing = True
                    logger.warning(
                        "chunk_twin_publish_obs=True but pred_next_obs is unavailable from current ONNX output. "
                        "Obs twin will be disabled for current packets."
                    )
            else:
                try:
                    obs_pred_qpos_chunk_abs = recover_obs_pred_dof_pos_abs(
                        pred_next_obs=pred_obs_array,
                        term_order=self.compact_term_order,
                        term_scale=self.compact_term_scale,
                        term_dims=self._compact_term_dims,
                        default_dof_angles=self.default_dof_angles,
                        num_dofs=int(self.num_dofs),
                        max_horizon_frames=max_frames,
                    )
                    if obs_pred_qpos_chunk_abs is None and not self._chunk_twin_warned_obs_parse:
                        self._chunk_twin_warned_obs_parse = True
                        logger.warning(
                            "pred_next_obs is available but compact term 'dof_pos' is missing; "
                            "obs prediction twin will be skipped."
                        )
                except Exception as exc:  # noqa: BLE001
                    if not self._chunk_twin_warned_obs_parse:
                        self._chunk_twin_warned_obs_parse = True
                        logger.warning(
                            f"Failed to parse obs-pred dof_pos for chunk twin (skipping obs twin): {exc}"
                        )

        packet = build_chunk_twin_packet(
            q_target_chunk_abs=q_target_chunk_abs,
            obs_pred_qpos_chunk_abs=obs_pred_qpos_chunk_abs,
            pred_horizon=int(self.pred_horizon),
            control_dt_sec=1.0 / float(self.rl_rate),
            model_name=model_name_from_path(getattr(self, "active_model_path", None)),
        )
        self._chunk_twin_publisher.publish(packet)
