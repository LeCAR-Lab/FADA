"""The exported graph must honour the dynamic batch axis it declares.

`_export_planner_idm_onnx` passes `dynamic_axes={... {0: "batch"} ...}` for every input
and output. `torch.onnx.export` traces at batch 1; with `nhead=1` the multi-head
attention reshapes constant-fold against that traced batch, so the exported graph loads,
runs at batch 1, and fails at batch 2 with a reshape error.

Asserts nhead 1 is rejected at export time and nhead 2 still exports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from holosoma.fada.planner_idm.eval_checkpoint import _export_planner_idm_onnx
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.utils.safe_torch_import import torch

_OBS_DIM = 6
_ACT_DIM = 3
_CMD_DIM = 3
_HISTORY_LEN = 4
_PRED_HORIZON = 2
_COMPACT_TERMS = ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")


def _dims() -> dict[str, object]:
    return {
        "obs_dim": _OBS_DIM,
        "act_dim": _ACT_DIM,
        "cmd_dim": _CMD_DIM,
        "history_len": _HISTORY_LEN,
        "pred_horizon": _PRED_HORIZON,
        "planner_use_action_history": False,
    }


def _export(tmp_path: Path, *, nhead: int) -> Path:
    torch.manual_seed(0)
    model = PlannerIDMPolicy(
        obs_dim=_OBS_DIM,
        act_dim=_ACT_DIM,
        cmd_dim=_CMD_DIM,
        history_len=_HISTORY_LEN,
        pred_horizon=_PRED_HORIZON,
        planner_d_model=8,
        planner_nhead=nhead,
        planner_num_layers=1,
        planner_dim_feedforward=16,
        planner_dropout=0.0,
        idm_d_model=8,
        idm_nhead=nhead,
        idm_encoder_num_layers=1,
        idm_decoder_num_layers=1,
        idm_dim_feedforward=16,
        idm_dropout=0.0,
    ).eval()
    out = tmp_path / f"policy_nhead{nhead}.onnx"
    return _export_planner_idm_onnx(
        model=model,
        onnx_output_path=out,
        dims=_dims(),
        io_normalization=False,
        obs_norm_stats=None,
        action_norm_stats=None,
        command_norm_stats=None,
        compact_obs_term_scale=dict.fromkeys(_COMPACT_TERMS, 1.0),
        compact_obs_term_noise=dict.fromkeys(_COMPACT_TERMS, 0.0),
        payload={"experiment_config": {}},
    )


def test_a_multi_head_export_still_succeeds(tmp_path: Path) -> None:
    """The postcheck accepts an nhead >= 2 export, the configuration the released students use."""
    pytest.importorskip("onnxruntime")
    path = _export(tmp_path, nhead=2)
    assert path.is_file()


def test_a_single_head_export_is_rejected_rather_than_published(tmp_path: Path) -> None:
    """`_export_planner_idm_onnx` raises instead of returning a path when nhead == 1."""
    pytest.importorskip("onnxruntime")
    with pytest.raises(RuntimeError) as excinfo:
        _export(tmp_path, nhead=1)

    message = str(excinfo.value)
    assert "batch 2" in message, message
    assert "dynamic batch" in message, "the error must say what is actually wrong: " + message
    assert "nhead" in message, "the error must name the cause the operator can act on: " + message


def test_the_rejected_graph_really_does_run_at_batch_one(tmp_path: Path) -> None:
    """An nhead=1 graph loads and runs at batch 1; only batch 2 fails."""
    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")

    # `_export_planner_idm_onnx` writes and saves the graph before running its postcheck,
    # so the file is on disk even when the postcheck raises.
    with pytest.raises(RuntimeError):
        _export(tmp_path, nhead=1)
    path = tmp_path / "policy_nhead1.onnx"
    assert path.is_file(), "the graph must have been written before the postcheck ran"

    session = ort.InferenceSession(str(path))  # the graph parses and loads

    def feed(n: int) -> dict[str, object]:
        return {
            "history_obs": np.zeros((n, _HISTORY_LEN, _OBS_DIM), dtype=np.float32),
            "history_act": np.zeros((n, _HISTORY_LEN, _ACT_DIM), dtype=np.float32),
            "current_command": np.zeros((n, _CMD_DIM), dtype=np.float32),
            "history_valid_mask": np.ones((n, _HISTORY_LEN), dtype=bool),
            "teacher_future_obs": np.zeros((n, _PRED_HORIZON, _OBS_DIM), dtype=np.float32),
        }

    session.run(None, feed(1))  # runs at the traced batch size
    # onnxruntime raises an internal pybind exception class here, so match on the message.
    with pytest.raises(Exception, match="Reshape"):
        session.run(None, feed(2))
