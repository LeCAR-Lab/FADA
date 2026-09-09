"""Two silent-coercion defects in the H5 reading path.

Both share a shape: a value that cannot be read was turned into a plausible one instead
of being refused, so the run continued and reported success.

* A `--command-key` that no episode carries fell back to an all-zero command array. That
  is correct when nothing reads the command channel, and it is silent corruption when
  the checkpoint conditions the IDM on it -- a typo trained a command-conditioned IDM on
  zeros with no error, no warning, and a normal-looking `summary.json`.
* `session_index` was read with an unconditional `int()`. A record claiming `0.9` and an
  episode tagged `0.1` both became `0`, so they matched and the completeness reporter
  announced a clean reconciliation between two values that agree about nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from holosoma.fada.common.lora_utils import (
    _as_session_index,
    _session_slot_index,
    describe_session_reconciliation,
    invalid_session_index_records,
    load_trajectories_from_h5,
)
from holosoma.fada.planner_idm.finetune_idm_lora import _FinetuneDefaults
from holosoma.fada.planner_idm.finetune_idm_lora import main as finetune_idm_main
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.utils.safe_torch_import import torch

_TIMESTEPS = 20
_OBS_DIM = 4
_ACT_DIM = 2
_CMD_DIM = 3


def _write_dataset(path: Path, *, command_key: str) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        episodes = handle.create_group("episodes")
        episode = episodes.create_group("episode_0000")
        episode.create_dataset(
            "dynamics_obs",
            data=np.arange(_TIMESTEPS * _OBS_DIM, dtype=np.float32).reshape(_TIMESTEPS, 1, _OBS_DIM),
        )
        episode.create_dataset(
            "actions",
            data=np.arange(_TIMESTEPS * _ACT_DIM, dtype=np.float32).reshape(_TIMESTEPS, 1, _ACT_DIM) + 1.0,
        )
        episode.create_dataset(
            command_key,
            data=np.arange(_TIMESTEPS * _CMD_DIM, dtype=np.float32).reshape(_TIMESTEPS, 1, _CMD_DIM) + 10.0,
        )
        episode.create_dataset("dones", data=np.zeros((_TIMESTEPS, 1, 1), dtype=np.bool_))


def _load(path: Path, *, command_key: str, require_command_key: bool):
    return load_trajectories_from_h5(
        [path],
        obs_key="dynamics_obs",
        command_key=command_key,
        obs_dim=_OBS_DIM,
        act_dim=_ACT_DIM,
        cmd_dim=_CMD_DIM,
        raw_obs_preprocess=None,
        pred_horizon_k=1,
        require_command_key=require_command_key,
    )


# ---------------------------------------------------------------------------
# A mistyped --command-key
# ---------------------------------------------------------------------------


def test_a_missing_command_key_still_yields_zeros_when_nothing_reads_the_command(tmp_path: Path) -> None:
    """The documented fallback, and the reason it exists: with
    `idm_use_current_command_for_history=False` (the shipped default)
    `_resolve_idm_current_command` returns None, so these zeros are never read.
    """
    dataset = tmp_path / "dataset.h5"
    _write_dataset(dataset, command_key="current_command")

    trajectories, _stats = _load(dataset, command_key="typo_command", require_command_key=False)

    assert trajectories, "the fallback must still load the episode"
    commands = np.asarray(trajectories[0].current_command)
    assert commands.shape == (_TIMESTEPS, _CMD_DIM)
    assert not commands.any(), "the documented fallback is an all-zero command"


def test_a_missing_command_key_is_fatal_when_the_command_is_actually_used(tmp_path: Path) -> None:
    """This is the case that had no symptom: a command-conditioned IDM trained on zeros."""
    dataset = tmp_path / "dataset.h5"
    _write_dataset(dataset, command_key="current_command")

    with pytest.raises(KeyError) as excinfo:
        _load(dataset, command_key="typo_command", require_command_key=True)

    message = str(excinfo.value)
    assert "typo_command" in message, message
    assert "current_command" in message, "the error must list the keys the episode does carry"
    assert "idm_use_current_command_for_history" in message, message


def test_the_correct_command_key_is_unaffected_by_the_requirement(tmp_path: Path) -> None:
    """Narrowness: requiring the key must not change a run that has it."""
    dataset = tmp_path / "dataset.h5"
    _write_dataset(dataset, command_key="current_command")

    trajectories, _stats = _load(dataset, command_key="current_command", require_command_key=True)

    commands = np.asarray(trajectories[0].current_command)
    assert commands.any(), "the real command array must be loaded, not zeros"


def _build_checkpoint(path: Path, *, command_conditioned: bool) -> None:
    """A tiny real PlannerIDM checkpoint whose cfg drives `resolve_finetune_recipe`.

    The module is built with the same flag the cfg carries, because `main()` rebuilds the
    model from the cfg before it loads any data -- a mismatch would fail on the state dict
    instead of on the thing under test.
    """
    model = PlannerIDMPolicy(
        obs_dim=_OBS_DIM,
        act_dim=_ACT_DIM,
        cmd_dim=_CMD_DIM,
        history_len=3,
        pred_horizon=2,
        planner_d_model=8,
        planner_nhead=2,
        planner_num_layers=1,
        planner_dim_feedforward=16,
        planner_dropout=0.0,
        planner_use_action_history=True,
        idm_d_model=8,
        idm_nhead=2,
        idm_encoder_num_layers=1,
        idm_decoder_num_layers=1,
        idm_dim_feedforward=16,
        idm_dropout=0.0,
        idm_use_current_command_for_history=command_conditioned,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "planner_state_dict": model.planner_state_dict(),
            "idm_state_dict": model.idm_state_dict(),
            "cfg": {
                "history_len": 3,
                "pred_horizon": 2,
                "planner_d_model": 8,
                "planner_nhead": 2,
                "planner_num_layers": 1,
                "planner_dim_feedforward": 16,
                "planner_dropout": 0.0,
                "planner_use_learned_positional_encoding": False,
                "planner_use_action_history": True,
                "idm_d_model": 8,
                "idm_nhead": 2,
                "idm_encoder_num_layers": 1,
                "idm_decoder_num_layers": 1,
                "idm_dim_feedforward": 16,
                "idm_dropout": 0.0,
                "idm_use_learned_positional_encoding": False,
                "idm_use_current_command_for_history": command_conditioned,
                "fdm_enabled": False,
            },
            "obs_dim": _OBS_DIM,
            "act_dim": _ACT_DIM,
            "cmd_dim": _CMD_DIM,
            "compact_obs": {
                "term_order": ["base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
                "term_scale": dict.fromkeys(
                    ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 1.0
                ),
                "term_noise": dict.fromkeys(
                    ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 0.0
                ),
                "add_noise": False,
            },
        },
        path,
    )


def _run_finetune_main(
    *,
    checkpoint: Path,
    dataset: Path,
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    for key, value in {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 2,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": False,
        "save_checkpoints": False,
        "save_adapter_only": True,
        "show_progress": False,
    }.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune_idm_lora",
            "--checkpoint",
            str(checkpoint),
            "--target-datasets",
            str(dataset),
            "--run-name",
            "command_key_wiring",
            "--output-dir",
            str(output_dir),
        ],
    )
    return finetune_idm_main()


def test_the_finetune_entry_point_refuses_a_missing_key_when_the_command_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring, exercised rather than read.

    A substring search over the source for `require_command_key=bool(resolved_...)`
    would stay green if the production call were changed to `require_command_key=False`,
    while a command-conditioned finetune trained on all-zero commands. So drive `main()`
    with a command-conditioned checkpoint and a dataset that does not carry the key, and
    require the refusal.
    """
    dataset = tmp_path / "dataset.h5"
    _write_dataset(dataset, command_key="mistyped_command")
    checkpoint = tmp_path / "planner_idm.pt"
    _build_checkpoint(checkpoint, command_conditioned=True)

    with pytest.raises(KeyError) as excinfo:
        _run_finetune_main(
            checkpoint=checkpoint,
            dataset=dataset,
            output_dir=tmp_path / "out",
            monkeypatch=monkeypatch,
        )

    message = str(excinfo.value)
    assert "current_command" in message, message
    assert "idm_use_current_command_for_history" in message, message


def test_the_finetune_entry_point_still_allows_the_fallback_when_the_command_is_unused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half, and the reason the requirement has to be *derived*.

    Pinning `require_command_key=True` unconditionally would pass the test above and break
    every run of the shipped default (`idm_use_current_command_for_history=False`), where
    the zeros are never read. Both tests together fix the value to the checkpoint's.
    """
    dataset = tmp_path / "dataset.h5"
    _write_dataset(dataset, command_key="mistyped_command")
    checkpoint = tmp_path / "planner_idm.pt"
    _build_checkpoint(checkpoint, command_conditioned=False)

    output_dir = tmp_path / "out"
    provenance = _run_finetune_main(
        checkpoint=checkpoint,
        dataset=dataset,
        output_dir=output_dir,
        monkeypatch=monkeypatch,
    )

    assert isinstance(provenance, dict), "the run must have completed, not been refused"
    summaries = list(output_dir.glob("*/summary.json"))
    assert len(summaries) == 1, summaries
    summary = json.loads(summaries[0].read_text())
    assert summary["idm_use_current_command_for_history"] is False, (
        "the run that was allowed to fall back must be the one that does not read the command"
    )


# ---------------------------------------------------------------------------
# Session indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        0.9,  # truncated to 0 by int(), which is how 0.9 and 0.1 "reconciled"
        0.1,
        np.float64(2.5),
        True,  # bool is an int subclass: int(True) == 1
        False,
        np.bool_(True),
        -1,
        -3.0,
        "0.9",
        "not-an-index",
        b"1.5",
        None,
        object(),
    ],
)
def test_values_that_are_not_session_indices_are_rejected(value: object) -> None:
    assert _as_session_index(value) is None, f"{value!r} was coerced into an index"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (3, 3), (np.int64(7), 7), (1.0, 1), (np.float32(2.0), 2), ("4", 4), (b"5", 5), (" 6 ", 6)],
)
def test_values_that_are_session_indices_are_still_accepted(value: object, expected: int) -> None:
    """Narrowness: h5py hands back numpy scalars and float64 routinely, and `1.0` is an
    integer written as a float. Rejecting those would break every real dataset."""
    assert _as_session_index(value) == expected


def test_a_fractional_record_and_a_fractional_tag_no_longer_reconcile() -> None:
    """The demonstrated case, end to end.

    `session_index: 0.9` and an episode tagged `0.1` both became `0` under `int()`, so
    the record's claim matched the tag count and the reporter printed a clean
    reconciliation. Now neither is an index: the tag is unattributed and the record is
    reported as unreadable.
    """
    sessions = [{"session_index": 0.9, "episodes": 1}]
    assert invalid_session_index_records(sessions) == [(0, 0.9)]

    row = {
        "dataset": "/tmp/dataset.h5",
        "sessions": sessions,
        "file_episodes": 1,
        # What `_read_collection_provenance` now produces for an episode tagged 0.1:
        # unparseable, therefore absent from episodes_by_session.
        "episodes_by_session": {},
        "untagged_episodes": 0,
        "unparseable_session_tags": 1,
    }
    lines = "\n".join(describe_session_reconciliation(row))
    assert "not a session index" in lines, lines
    assert "0.9" in lines, lines


def test_a_record_with_no_index_still_falls_back_to_its_position() -> None:
    """Narrowness again: "no index recorded" is the documented case, not an error."""
    assert invalid_session_index_records([{"episodes": 2}]) == []
    assert _session_slot_index({"episodes": 2}, 3) == 3
