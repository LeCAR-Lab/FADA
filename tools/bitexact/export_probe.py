"""Bit-exactness probe: ONNX export.

Builds a PlannerIDMPolicy from the shared `build_probe_config()` and drives the
checkpoint-round-trip + ONNX-export path that `eval_checkpoint.py` runs:

- `_build_model_from_checkpoint(payload, device)`: rebuilds the model from a payload
  shaped like a real checkpoint (cfg dict + obs/act/cmd_dim + model_state_dict). Its
  field set is kept aligned with the payload built by
  `test_planner_idm_arch_checkpoint_rebuild_preserves_architecture`
  (`src/holosoma/tests/fada/test_planner_idm_model.py`).
- `_export_planner_idm_onnx(...)`: the shipped ONNX export wrapper
  (`_PlannerIDMOnnxWrapper`) and its export/validation/metadata-writing logic, called
  rather than reimplemented.

The export goes to a temporary file whose bytes are hashed with sha256. onnxruntime
then runs one inference pass over fixed inputs (generated under
`torch.manual_seed(PROBE_SEED)`) and every output tensor is flattened into a
`list[float]` and recorded.

It emits no hash of the whole checkpoint payload: constant-folding a config field
changes the key set of the `cfg` dict inside that payload. Only the ONNX bytes'
sha256 and the flattened inference outputs are compared. See `compare.py`'s
docstring.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from holosoma.fada.planner_idm.eval_checkpoint import (
    _build_model_from_checkpoint,
    _export_planner_idm_onnx,
)
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.utils.safe_torch_import import torch

from tools.bitexact.harness import (
    PROBE_ACT_DIM,
    PROBE_CMD_DIM,
    PROBE_OBS_DIM,
    PROBE_SEED,
    _seed_everything,
    build_probe_config,
)


def _build_probe_checkpoint_payload() -> dict[str, Any]:
    """Build the smallest payload shaped like a real checkpoint.

    `_build_model_from_checkpoint` reads only `payload["cfg"]` (a dict, key by key:
    some via `.get()` with a default, others via `[]`),
    `payload["obs_dim"|"act_dim"|"cmd_dim"]`, and `payload["model_state_dict"]`. This
    payload carries no `planner_state_dict`/`idm_state_dict`, so it takes the
    `model.load_compatible_state_dict` branch rather than `load_split_state_dict`.
    The cfg keys with no `.get()` default, and so required here, are
    planner_d_model/nhead/num_layers/dim_feedforward/dropout,
    planner_use_learned_positional_encoding, idm_d_model/nhead/dim_feedforward/dropout,
    and idm_use_learned_positional_encoding.
    """
    cfg = build_probe_config()
    model = PlannerIDMPolicy(
        obs_dim=PROBE_OBS_DIM,
        act_dim=PROBE_ACT_DIM,
        cmd_dim=PROBE_CMD_DIM,
        history_len=cfg.history_len,
        pred_horizon=cfg.pred_horizon,
        planner_d_model=cfg.planner_d_model,
        planner_nhead=cfg.planner_nhead,
        planner_num_layers=cfg.planner_num_layers,
        planner_dim_feedforward=cfg.planner_dim_feedforward,
        planner_dropout=cfg.planner_dropout,
        planner_use_action_history=cfg.planner_use_action_history,
        planner_predict_delta=cfg.planner_predict_delta,
        idm_d_model=cfg.idm_d_model,
        idm_nhead=cfg.idm_nhead,
        idm_encoder_num_layers=cfg.idm_encoder_num_layers,
        idm_decoder_num_layers=cfg.idm_decoder_num_layers,
        idm_dim_feedforward=cfg.idm_dim_feedforward,
        idm_dropout=cfg.idm_dropout,
        idm_use_current_command_for_history=cfg.idm_use_current_command_for_history,
    )
    train_cfg = {
        "policy_arch": "planner_idm",
        "history_len": cfg.history_len,
        "pred_horizon": cfg.pred_horizon,
        "planner_model_type": "transformer",
        "planner_d_model": cfg.planner_d_model,
        "planner_nhead": cfg.planner_nhead,
        "planner_num_layers": cfg.planner_num_layers,
        "planner_dim_feedforward": cfg.planner_dim_feedforward,
        "planner_dropout": cfg.planner_dropout,
        "planner_use_learned_positional_encoding": False,
        "planner_use_action_history": cfg.planner_use_action_history,
        "idm_model_type": "transformer",
        "idm_encoder_model_type": "transformer",
        "idm_decoder_model_type": "transformer",
        "idm_use_current_command_for_history": cfg.idm_use_current_command_for_history,
        "idm_d_model": cfg.idm_d_model,
        "idm_nhead": cfg.idm_nhead,
        "idm_encoder_num_layers": cfg.idm_encoder_num_layers,
        "idm_decoder_num_layers": cfg.idm_decoder_num_layers,
        "idm_dim_feedforward": cfg.idm_dim_feedforward,
        "idm_dropout": cfg.idm_dropout,
        "idm_use_learned_positional_encoding": False,
        "idm_use_teacher_forcing": cfg.idm_use_teacher_forcing,
        "idm_teacher_forcing_ratio": cfg.idm_teacher_forcing_ratio,
        "fdm_enabled": False,
        "planner_predict_delta": cfg.planner_predict_delta,
        "fdm_predict_delta": False,
        "command_profile": cfg.command_profile,
    }
    return {
        "cfg": train_cfg,
        "obs_dim": PROBE_OBS_DIM,
        "act_dim": PROBE_ACT_DIM,
        "cmd_dim": PROBE_CMD_DIM,
        "model_state_dict": model.state_dict(),
        "experiment_config": {},
    }


def run_export_probe(out_path: Path) -> dict[str, Any]:
    _seed_everything(PROBE_SEED)
    cfg = build_probe_config()
    payload = _build_probe_checkpoint_payload()

    model, dims = _build_model_from_checkpoint(payload, device=torch.device("cpu"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / "planner_idm_policy.onnx"
        _export_planner_idm_onnx(
            model=model,
            onnx_output_path=onnx_path,
            dims=dims,
            io_normalization=False,
            obs_norm_stats=None,
            action_norm_stats=None,
            command_norm_stats=None,
            compact_obs_term_scale=cfg.compact_obs_term_scale,
            compact_obs_term_noise=cfg.compact_obs_term_noise,
            payload=payload,
        )
        onnx_bytes = onnx_path.read_bytes()
        onnx_sha256 = hashlib.sha256(onnx_bytes).hexdigest()

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        gen = torch.Generator().manual_seed(PROBE_SEED)
        history_obs = torch.randn((1, cfg.history_len, PROBE_OBS_DIM), generator=gen)
        history_act = torch.randn((1, cfg.history_len, PROBE_ACT_DIM), generator=gen)
        current_command = torch.randn((1, PROBE_CMD_DIM), generator=gen)
        history_valid_mask = torch.ones((1, cfg.history_len), dtype=torch.bool)
        teacher_future_obs = torch.randn((1, cfg.pred_horizon, PROBE_OBS_DIM), generator=gen)

        ort_inputs = {
            "history_obs": history_obs.numpy(),
            "history_act": history_act.numpy(),
            "current_command": current_command.numpy(),
            "history_valid_mask": history_valid_mask.numpy(),
            "teacher_future_obs": teacher_future_obs.numpy(),
        }
        ort_outputs = session.run(None, ort_inputs)

    outputs = [np.asarray(o, dtype=np.float64).reshape(-1).tolist() for o in ort_outputs]

    result: dict[str, Any] = {"onnx_sha256": onnx_sha256, "outputs": outputs}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
