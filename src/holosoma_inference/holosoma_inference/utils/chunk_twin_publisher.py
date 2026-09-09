from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping

import numpy as np
import zmq
from loguru import logger

_SCHEMA = "holosoma.transformer_chunk_twin.v1"


def _as_horizon_matrix(array: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == 3:
        if out.shape[0] != 1:
            raise ValueError(f"Expected single-env chunk tensor, got shape {out.shape}")
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"Expected [H, D] matrix, got shape {out.shape}")
    return out.astype(np.float32, copy=False)


def _pad_front_to_num_dofs(values: np.ndarray, num_dofs: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected [H, D] matrix, got shape {values.shape}")
    dim = int(values.shape[1])
    if dim == num_dofs:
        return values.astype(np.float32, copy=False)
    if dim > num_dofs:
        raise ValueError(f"Tensor dim {dim} exceeds num_dofs {num_dofs}")
    pad = np.zeros((int(values.shape[0]), num_dofs - dim), dtype=np.float32)
    return np.concatenate([pad, values], axis=1).astype(np.float32, copy=False)


def build_q_target_chunk_abs(
    *,
    actions_chunk_raw: np.ndarray | list[float] | list[list[float]],
    policy_action_scale: float,
    default_dof_angles: np.ndarray,
    num_dofs: int,
) -> np.ndarray:
    chunk = _as_horizon_matrix(actions_chunk_raw)
    scaled = chunk * float(policy_action_scale)
    scaled_padded = _pad_front_to_num_dofs(scaled, int(num_dofs))
    default_angles = np.asarray(default_dof_angles, dtype=np.float32).reshape(1, -1)
    if int(default_angles.shape[1]) != int(num_dofs):
        raise ValueError(
            f"default_dof_angles dim mismatch: got {default_angles.shape[1]}, expected {num_dofs}"
        )
    return (scaled_padded + default_angles).astype(np.float32, copy=False)


def build_compact_term_slices(term_order: tuple[str, ...], term_dims: Mapping[str, int]) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    cursor = 0
    for term in term_order:
        if term not in term_dims:
            raise KeyError(f"Missing compact term dim for '{term}'")
        width = int(term_dims[term])
        if width <= 0:
            raise ValueError(f"Invalid compact term dim for '{term}': {width}")
        slices[term] = slice(cursor, cursor + width)
        cursor += width
    return slices


def recover_obs_pred_dof_pos_abs(
    *,
    pred_next_obs: np.ndarray | None,
    term_order: tuple[str, ...],
    term_scale: Mapping[str, float],
    term_dims: Mapping[str, int],
    default_dof_angles: np.ndarray,
    num_dofs: int,
    max_horizon_frames: int,
) -> np.ndarray | None:
    if pred_next_obs is None:
        return None
    obs_matrix = _as_horizon_matrix(pred_next_obs)
    if max_horizon_frames > 0:
        obs_matrix = obs_matrix[: int(max_horizon_frames)]

    term_slices = build_compact_term_slices(term_order, term_dims)
    dof_slice = term_slices.get("dof_pos")
    if dof_slice is None:
        return None

    dof_scale = float(term_scale.get("dof_pos", 1.0))
    if abs(dof_scale) < 1e-8:
        raise ValueError("compact term scale for 'dof_pos' is zero, cannot invert scaling.")

    dof_pos_scaled = obs_matrix[:, dof_slice].astype(np.float32, copy=False)
    dof_pos_unscaled = dof_pos_scaled / dof_scale
    dof_pos_padded = _pad_front_to_num_dofs(dof_pos_unscaled, int(num_dofs))

    default_angles = np.asarray(default_dof_angles, dtype=np.float32).reshape(1, -1)
    if int(default_angles.shape[1]) != int(num_dofs):
        raise ValueError(
            f"default_dof_angles dim mismatch: got {default_angles.shape[1]}, expected {num_dofs}"
        )
    return (dof_pos_padded + default_angles).astype(np.float32, copy=False)


def build_chunk_twin_packet(
    *,
    q_target_chunk_abs: np.ndarray,
    obs_pred_qpos_chunk_abs: np.ndarray | None,
    pred_horizon: int,
    control_dt_sec: float,
    model_name: str,
) -> dict:
    chunk = _as_horizon_matrix(q_target_chunk_abs)
    obs_chunk = None if obs_pred_qpos_chunk_abs is None else _as_horizon_matrix(obs_pred_qpos_chunk_abs)
    return {
        "schema": _SCHEMA,
        "timestamp_sec": float(time.time()),
        "control_dt_sec": float(control_dt_sec),
        "q_target_chunk_abs": chunk.astype(np.float32, copy=False),
        "obs_pred_qpos_chunk_abs": None if obs_chunk is None else obs_chunk.astype(np.float32, copy=False),
        "meta": {
            "pred_horizon": int(pred_horizon),
            "model_name": str(model_name),
            "has_obs_pred": bool(obs_chunk is not None),
        },
    }


class ChunkTwinPublisher:
    """Best-effort ZMQ publisher for transformer chunk twin packets."""

    def __init__(self, *, enabled: bool, pub_url: str) -> None:
        self.enabled = bool(enabled)
        self.pub_url = str(pub_url)
        self._ctx: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._send_failed_logged = False
        if not self.enabled:
            return
        try:
            self._ctx = zmq.Context.instance()
            self._socket = self._ctx.socket(zmq.PUB)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.SNDHWM, 2)
            self._socket.connect(self.pub_url)
            logger.info(f"Chunk twin publisher connected: {self.pub_url}")
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            self._socket = None
            logger.warning(f"Failed to initialize chunk twin publisher ({self.pub_url}): {exc}")

    def publish(self, packet: dict) -> bool:
        if not self.enabled or self._socket is None:
            return False
        try:
            self._socket.send_pyobj(packet, zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False
        except Exception as exc:  # noqa: BLE001
            if not self._send_failed_logged:
                self._send_failed_logged = True
                logger.warning(f"Chunk twin publish failed; disabling future warnings: {exc}")
            return False

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:  # noqa: BLE001
                pass
            self._socket = None

    def __del__(self) -> None:
        self.close()


def model_name_from_path(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).name

