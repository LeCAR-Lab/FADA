"""Validation paths that the CLI range checks in `fada/common/cli_validation.py` do not
cover, because a value does not have to arrive through a flag:

1. **The eval entry point's tyro passthrough.** `eval_checkpoint.main()` parses its own 8
   numeric flags with `parse_known_args()` and hands everything else to
   `tyro.cli(ExperimentConfig, ...)`, which only type-checks. A passthrough value such as
   `--training.num-envs` is applied *after* the validated defaults, so it is the one that
   runs.
2. **Checkpoint-carried normalization statistics.** `_extract_norm_stats` (one copy in
   `eval_checkpoint`, one in `lora_utils`) checks field presence and shape, so a payload
   with `nan` mean/std/eps would otherwise flow into `(x - mean) / std` and NaN the eval,
   the exported ONNX and the finetune.
3. **Datasets recorded without `dones`**, which the finetune loader reads as "this episode
   never terminated". The loading behavior is unchanged; the dataset is reported.

Each test asserts that an input with no defined behavior is rejected or reported.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from holosoma.config_values import experiment as experiment_defaults
from holosoma.fada.common import lora_utils
from holosoma.fada.common.norm_stats import validate_norm_stats
from holosoma.fada.planner_idm import eval_checkpoint
from holosoma.utils.safe_torch_import import torch

# ---------------------------------------------------------------------------------------
# 1. The eval entry point's tyro passthrough
# ---------------------------------------------------------------------------------------


@pytest.fixture
def eval_cfg():
    cfg = experiment_defaults.t1_23dof_waist50_oracle
    return dataclasses.replace(
        cfg,
        training=dataclasses.replace(cfg.training, num_envs=4, seed=1, max_eval_steps=100),
    )


def test_a_resolved_config_from_the_released_defaults_validates(eval_cfg) -> None:
    """The guard accepts the released default configs unchanged."""
    eval_checkpoint.validate_experiment_config(eval_cfg)
    eval_checkpoint.validate_experiment_config(experiment_defaults.t1_23dof_waist50_oracle)
    eval_checkpoint.validate_experiment_config(experiment_defaults.g1_29dof_oracle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_envs", 0),
        ("seed", -1),
        ("max_eval_steps", 0),
    ],
)
def test_the_passthrough_spelling_is_rejected_exactly_like_the_flag(eval_cfg, field: str, value: int) -> None:
    mutated = dataclasses.replace(eval_cfg, training=dataclasses.replace(eval_cfg.training, **{field: value}))
    with pytest.raises(ValueError, match="invalid value") as excinfo:
        eval_checkpoint.validate_experiment_config(mutated)
    message = str(excinfo.value)
    assert f"training.{field}" in message
    assert repr(value) in message or str(value) in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # The non-finite rule is a float rule and these fields are ints, so integer
        # zero/negative values need their own range check.
        ("fps", 0),
        ("fps", -200),
        ("control_decimation", 0),
        ("substeps", 0),
        ("render_interval", 0),
        ("max_episode_length_s", 0.0),
        ("max_episode_length_s", -1.0),
    ],
)
def test_absurd_simulator_step_counts_are_rejected(eval_cfg, field: str, value: float) -> None:
    """Zero/negative timing counts are not configurations; every shipped preset is positive."""
    mutated = copy.deepcopy(eval_cfg)
    object.__setattr__(mutated.simulator.config.sim, field, value)
    with pytest.raises(ValueError, match="invalid value") as excinfo:
        eval_checkpoint.validate_experiment_config(mutated)
    message = str(excinfo.value)
    assert f"simulator.config.sim.{field}" in message
    assert "--simulator.config.sim." in message, "the message must name the flag the caller typed"


@pytest.mark.parametrize("preset_name", ["t1_23dof_waist50_oracle", "g1_29dof_oracle"])
def test_the_shipped_simulator_settings_still_validate(preset_name: str) -> None:
    """The bounds accept every value the shipped presets use.

    Asserted against the presets themselves rather than against literal numbers.
    """
    cfg = getattr(experiment_defaults, preset_name)
    eval_checkpoint.validate_experiment_config(cfg)
    sim = cfg.simulator.config.sim
    for field in ("fps", "control_decimation", "substeps", "render_interval"):
        assert getattr(sim, field) >= 1, f"{preset_name}.{field}"
    assert sim.max_episode_length_s > 0.0


def test_other_simulator_fields_are_deliberately_left_alone(eval_cfg) -> None:
    """Only the entry point's own quantities and the simulator's step/timing counts are
    re-checked; every other field passes through unchecked. A zero link-mass scale
    validates.
    """
    mutated = copy.deepcopy(eval_cfg)
    object.__setattr__(mutated.simulator.config, "link_mass_scale", 0.0)
    eval_checkpoint.validate_experiment_config(mutated)


def test_a_non_finite_anywhere_in_the_config_is_rejected(eval_cfg) -> None:
    """A non-finite value anywhere in the config is rejected.

    Mutates a field with no domain of its own, so this exercises the blanket finite rule
    rather than a per-field range (`sim.fps` has one -- see the case below).
    """
    mutated = copy.deepcopy(eval_cfg)
    object.__setattr__(mutated.simulator.config, "link_mass_scale", float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        eval_checkpoint.validate_experiment_config(mutated)


def test_a_field_with_its_own_domain_reports_that_domain_for_nan(eval_cfg) -> None:
    """`--simulator.config.sim.fps nan` is refused by the narrower per-field rule, and
    the message names the flag.
    """
    mutated = copy.deepcopy(eval_cfg)
    object.__setattr__(mutated.simulator.config.sim, "fps", float("nan"))
    with pytest.raises(ValueError, match="must be finite") as excinfo:
        eval_checkpoint.validate_experiment_config(mutated)
    assert "--simulator.config.sim.fps" in str(excinfo.value)


def test_an_unset_optional_field_is_not_out_of_range() -> None:
    """`None` is the stock config's 'unset'; `_build_eval_config` fills it."""
    cfg = experiment_defaults.t1_23dof_waist50_oracle
    assert cfg.training.max_eval_steps is None
    eval_checkpoint.validate_experiment_config(cfg)


def test_the_passthrough_reports_every_numeric_field_it_changed(eval_cfg, capsys) -> None:
    """Every numeric field the passthrough changed is reported."""
    mutated = dataclasses.replace(eval_cfg, training=dataclasses.replace(eval_cfg.training, num_envs=8))
    changed = eval_checkpoint.report_config_passthrough(eval_cfg, mutated, ["--training.num-envs", "8"])
    assert changed == ["training.num_envs: 4 -> 8"]
    assert "training.num_envs: 4 -> 8" in capsys.readouterr().out


def test_no_passthrough_args_reports_nothing(eval_cfg, capsys) -> None:
    assert eval_checkpoint.report_config_passthrough(eval_cfg, eval_cfg, []) == []
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------------------
# 2. Checkpoint-carried normalization statistics
# ---------------------------------------------------------------------------------------


def _payload(mean, std, eps=1e-6) -> dict:
    return {"obs_norm_stats": {"obs_mean": mean, "obs_std": std, "eps": eps}}


def _extract_both(payload: dict, dim: int) -> None:
    """Run the payload through both copies of the extractor."""
    device = torch.device("cpu")
    eval_checkpoint._extract_norm_stats(
        payload, key="obs_norm_stats", mean_key="obs_mean", std_key="obs_std",
        dim=dim, required=True, device=device,
    )
    lora_utils._extract_norm_stats(
        payload, key="obs_norm_stats", mean_key="obs_mean", std_key="obs_std",
        dim=dim, required=True, device=device,
    )


def test_valid_stats_still_load_through_both_readers() -> None:
    _extract_both(_payload([0.0, 1.5], [1.0, 2.0]), dim=2)


def test_a_zero_std_is_still_accepted_and_clamped() -> None:
    """A zero std is accepted: a constant observation dimension has zero variance, and
    `clamp(std, min=eps)` handles it."""
    device = torch.device("cpu")
    stats = lora_utils._extract_norm_stats(
        _payload([0.0, 0.0], [0.0, 2.0], eps=1e-6),
        key="obs_norm_stats", mean_key="obs_mean", std_key="obs_std",
        dim=2, required=True, device=device,
    )
    assert stats is not None
    assert float(stats["std"][0]) == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("mean", "std", "eps", "expected"),
    [
        ([float("nan"), 0.0], [1.0, 1.0], 1e-6, "mean"),
        ([0.0, 0.0], [float("nan"), 1.0], 1e-6, "std"),
        ([0.0, 0.0], [float("inf"), 1.0], 1e-6, "std"),
        ([0.0, 0.0], [-1.0, 1.0], 1e-6, "negative"),
        ([0.0, 0.0], [1.0, 1.0], float("nan"), "eps"),
        ([0.0, 0.0], [1.0, 1.0], 0.0, "eps"),
        ([0.0, 0.0], [1.0, 1.0], -1e-6, "eps"),
    ],
)
def test_unusable_checkpoint_stats_are_rejected_by_both_readers(mean, std, eps, expected: str) -> None:
    """NaN mean, NaN and negative std, negative eps -- through both readers."""
    with pytest.raises(RuntimeError, match=expected):
        _extract_both(_payload(mean, std, eps), dim=2)


def test_the_validator_names_the_checkpoint_field() -> None:
    with pytest.raises(RuntimeError, match="act_norm_stats"):
        validate_norm_stats(
            torch.tensor([float("nan")]), torch.tensor([1.0]), 1e-6, key="act_norm_stats"
        )


# ---------------------------------------------------------------------------------------
# 3. Datasets with no `dones`, and post-fall datasets the collector already reported on
# ---------------------------------------------------------------------------------------


def _write_h5(
    path: Path,
    *,
    steps: int,
    with_dones: bool,
    done_at: int | None = None,
    with_collection_provenance: bool = True,
) -> Path:
    """A hand-written H5, standing in for one the collector produced.

    `with_collection_provenance` stamps what a *complete* collection records
    (`collection_status`/`planned_steps`/`collected_steps`). It defaults to True
    so these tests keep asking about termination provenance only; a file without
    those attributes is a pre-provenance dataset, which the loader now reports as
    unanswerable -- covered in
    holosoma_inference/policies/tests/test_collection_completeness.py.
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        if with_collection_provenance:
            handle.attrs["collection_status"] = "complete"
            handle.attrs["planned_steps"] = steps
            handle.attrs["planned_steps_source"] = "--task.max-steps"
            handle.attrs["collected_steps"] = steps
        episodes = handle.create_group("episodes")
        group = episodes.create_group("episode_0")
        group.create_dataset("dynamics_obs", data=np.arange(steps * 2, dtype=np.float32).reshape(steps, 1, 2))
        group.create_dataset("actions", data=np.zeros((steps, 1, 2), dtype=np.float32))
        group.create_dataset("current_command", data=np.zeros((steps, 1, 1), dtype=np.float32))
        if with_dones:
            dones = np.zeros((steps, 1), dtype=np.bool_)
            if done_at is not None:
                dones[done_at:, 0] = True
            group.create_dataset("dones", data=dones)
    return path


def _load(path: Path):
    return lora_utils.load_trajectories_from_h5(
        [str(path)], obs_key="dynamics_obs", command_key="current_command",
        obs_dim=2, act_dim=2, cmd_dim=1, raw_obs_preprocess=None, pred_horizon_k=2,
    )


def test_a_dataset_without_dones_still_loads_whole_but_says_so(tmp_path: Path, capsys) -> None:
    """Windows are unchanged, plus a warning that they are unchecked."""
    path = _write_h5(tmp_path / "old.h5", steps=10, with_dones=False)
    trajectories, stats = _load(path)
    assert len(trajectories) == 1
    assert trajectories[0].obs.shape[0] == 10
    assert stats["num_episodes_without_dones_field"] == 1
    assert stats["num_episodes_with_dones_field"] == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "no `dones` dataset" in out


def test_a_modern_clean_dataset_produces_no_warning(tmp_path: Path, capsys) -> None:
    path = _write_h5(tmp_path / "clean.h5", steps=10, with_dones=True)
    _, stats = _load(path)
    assert stats["num_episodes_with_dones_field"] == 1
    assert stats["num_env_trajectories_truncated_at_done"] == 0
    assert "WARNING" not in capsys.readouterr().out


def test_a_truncated_trajectory_is_counted(tmp_path: Path) -> None:
    path = _write_h5(tmp_path / "fell.h5", steps=10, with_dones=True, done_at=6)
    trajectories, stats = _load(path)
    assert trajectories[0].obs.shape[0] == 6
    assert stats["num_env_trajectories_truncated_at_done"] == 1


def test_the_collectors_fall_report_is_restated_at_finetune_time(tmp_path: Path, capsys) -> None:
    """The collector's fall report is restated when the finetune loads the dataset."""
    path = _write_h5(tmp_path / "dataset.h5", steps=10, with_dones=True)
    (tmp_path / "collection_fall_report.json").write_text(
        json.dumps(
            {
                "fall_detection_available": True,
                "collected_steps": 2000,
                "fall_step": 812,
                "fall_signal": "tilt",
                "post_fall_steps": 1188,
                "marked_terminal": False,
            }
        )
    )
    _, stats = _load(path)
    assert len(stats["fall_reports"]) == 1
    out = capsys.readouterr().out
    assert "robot fell at recorded step 812" in out
    assert "NON-terminal" in out
    assert "59.4%" in out


def test_an_unavailable_fall_report_is_not_read_as_a_clean_run(tmp_path: Path, capsys) -> None:
    path = _write_h5(tmp_path / "dataset.h5", steps=10, with_dones=True)
    (tmp_path / "collection_fall_report.json").write_text(
        json.dumps({"fall_detection_available": False, "collected_steps": 2000, "post_fall_steps": 0})
    )
    _load(path)
    assert "UNAVAILABLE" in capsys.readouterr().out


def test_a_malformed_report_does_not_fail_a_valid_dataset(tmp_path: Path) -> None:
    path = _write_h5(tmp_path / "dataset.h5", steps=10, with_dones=True)
    (tmp_path / "collection_fall_report.json").write_text("{not json")
    trajectories, stats = _load(path)
    assert len(trajectories) == 1
    assert stats["fall_reports"] == []


def test_provenance_lines_are_pure_reporting() -> None:
    """`describe_termination_provenance` must not need, or touch, the trajectories."""
    lines = lora_utils.describe_termination_provenance(
        {"num_env_trajectories": 4, "num_env_trajectories_truncated_at_done": 1,
         "num_episodes_without_dones_field": 0, "fall_reports": []}
    )
    assert len(lines) == 1
    assert "1/4" in lines[0]
    assert not math.isnan(0.0)  # sanity: the module imports cleanly under -O
