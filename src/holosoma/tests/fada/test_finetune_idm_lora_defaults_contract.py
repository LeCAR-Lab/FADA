"""Default-contract tests for the finetune entrypoint's release defaults.

These tests are separate from `test_finetune_idm_lora.py`'s integration tests and do
NOT monkeypatch `_FinetuneDefaults` (or anything else); that file's small-scale
execution tests override `_FinetuneDefaults` fields to keep synthetic runs fast. This
file asserts the literal release values directly against `_FinetuneDefaults` and
against `resolve_finetune_recipe()` / `resolve_finetune_artifact_policy()` called
with no overrides, so the assertions hold regardless of what any other test's fixture
overrides.

The values below are the release recipe: `_FinetuneDefaults` and the
`FinetuneRecipe`/`FinetuneArtifactPolicy` hashes derived from it are the source of
truth if this file drifts, not the other way around.
"""

from __future__ import annotations

import dataclasses

import pytest

from holosoma.fada.planner_idm.finetune_idm_lora import (
    FinetuneArtifactPolicy,
    FinetuneRecipe,
    _FinetuneDefaults,
    resolve_finetune_artifact_policy,
    resolve_finetune_recipe,
)


def test_finetune_defaults_optimizer_hparams_match_release_recipe() -> None:
    assert _FinetuneDefaults.lr == pytest.approx(1e-4)
    assert _FinetuneDefaults.weight_decay == pytest.approx(1e-3)
    assert _FinetuneDefaults.grad_clip == pytest.approx(0.5)


def test_finetune_defaults_lora_config_matches_release_recipe() -> None:
    assert _FinetuneDefaults.finetune_method == "lora"
    assert _FinetuneDefaults.lora_r == 8
    assert _FinetuneDefaults.lora_alpha == pytest.approx(16.0)
    assert _FinetuneDefaults.lora_dropout == pytest.approx(0.05)
    assert _FinetuneDefaults.lora_target_scope == "encoder_decoder_qkv"
    assert _FinetuneDefaults.idm_action_loss_horizon_gamma == pytest.approx(0.0)


def test_finetune_defaults_data_composition_and_duration_match_release_recipe() -> None:
    """Pins the data-composition and training-duration defaults: batch_size=4096,
    max_train_steps=100 (finetuning is step-driven only),
    trim_head_steps=trim_tail_steps=0.

    These four are covered by `recipe_hash`. There is no train/val split to go with
    them (see `test_finetune_defaults_have_no_train_val_split_surface`).
    """
    assert _FinetuneDefaults.batch_size == 4096
    assert _FinetuneDefaults.max_train_steps == 100
    assert _FinetuneDefaults.trim_head_steps == 0
    assert _FinetuneDefaults.trim_tail_steps == 0


def test_finetune_defaults_seed_and_horizon() -> None:
    assert _FinetuneDefaults.seed == 42
    assert _FinetuneDefaults.pred_horizon_k is None


def test_finetune_defaults_have_no_train_val_split_surface() -> None:
    """The four train/val split fields are absent from `_FinetuneDefaults`."""
    for removed in ("train_val_split", "target_dataset_count", "val_every_steps", "val_batches"):
        assert not hasattr(_FinetuneDefaults, removed), f"_FinetuneDefaults.{removed} must stay removed"


def test_finetune_defaults_data_selection_and_loss_semantics() -> None:
    """The five defaults that decide *which arrays are read out of the target
    H5 files* and *what loss is optimized*.

    `obs_key="auto"` resolves per episode to `raw_dynamics_obs` (term-scaled through
    `RawObsPreprocess`) or `dynamics_obs` (already scaled), so its value decides
    whether observations are rescaled at all.
    """
    assert _FinetuneDefaults.obs_key == "auto"
    assert _FinetuneDefaults.command_key == "current_command"
    assert _FinetuneDefaults.idm_use_current_command_for_history is None
    assert _FinetuneDefaults.idm_loss_target == "idm_planner_fdm_obs"
    assert _FinetuneDefaults.idm_anchor_lambda == pytest.approx(1.0)


def test_every_finetune_default_is_hashed_or_explicitly_excluded() -> None:
    """Every public field of `_FinetuneDefaults` must be covered by `FinetuneRecipe`
    (behavioral / data-selection / loss-semantics -> `recipe_hash`), by
    `FinetuneArtifactPolicy` (which files get written -> `artifact_policy_hash`), or
    appear in the explicit exclusion set below. Adding a new default outside those
    three buckets fails here.

    The exclusions are listed field-by-field in `FinetuneRecipe`'s docstring: none of
    them changes a trained weight, a loss value, or which data is selected. `main()`
    reads the NPZ/ratio/budget group at exactly one place -- the
    `data_composition_provenance` dict, which is recorded and never consulted -- so no
    ratio or budget reaches a batch on the release target-H5-only path.
    """
    excluded = {
        # logging / infrastructure
        "device",
        "show_progress",
        "use_wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_name",
        "wandb_mode",
        "wandb_tags",
        # where an artifact lands / when intermediate ones are dumped
        "onnx_output_path",
        "save_step_milestones",
        # assigned in main() and never read (inert on the single-stage, step-driven path)
        "finetune_stage",
        # mixed-source NPZ knobs: recorded into provenance, never consulted
        "npz_optimal",
        "npz_suboptimal",
        "npz_online",
        "buffer_step_budget",
        "target_step_budget",
        "idm_suboptimal_ratio",
        "idm_suboptimal_expert_ratio",
        "idm_online_trajectory_ratio",
        "idm_mixed_offline_ratio",
    }
    recipe_fields = {f.name for f in dataclasses.fields(FinetuneRecipe)}
    policy_fields = {f.name for f in dataclasses.fields(FinetuneArtifactPolicy)}
    covered = recipe_fields | policy_fields | excluded

    default_fields = {
        name
        for name in vars(_FinetuneDefaults)
        if not name.startswith("_") and not callable(getattr(_FinetuneDefaults, name))
    }
    # `_FinetuneDefaults` mixes annotated (`x: int | None = None`) and plain class
    # attributes; the annotated-only ones don't show up in `vars()` values, so union
    # the annotations in too.
    default_fields |= {name for name in getattr(_FinetuneDefaults, "__annotations__", {})}

    uncovered = sorted(default_fields - covered)
    assert not uncovered, (
        "these _FinetuneDefaults fields are neither in FinetuneRecipe (recipe_hash) nor "
        "FinetuneArtifactPolicy (artifact_policy_hash) nor the explicit exclusion set, so a "
        f"regression to them would be invisible to tools/bitexact/run.sh: {uncovered}"
    )
    # The exclusion set must not name fields that no longer exist.
    stale_exclusions = sorted(excluded - default_fields)
    assert not stale_exclusions, f"exclusion set names fields that no longer exist: {stale_exclusions}"


def test_finetune_defaults_artifact_policy() -> None:
    """Which artifacts a default (no-override) release run writes: `export_onnx` is
    the only one enabled; `save_checkpoints`, `export_onnx_every_epoch` and
    `save_adapter_only` default off.
    """
    assert _FinetuneDefaults.save_checkpoints is False
    assert _FinetuneDefaults.export_onnx is True
    assert _FinetuneDefaults.export_onnx_every_epoch is False
    assert _FinetuneDefaults.save_adapter_only is False


def test_resolve_finetune_recipe_with_no_overrides_matches_release_defaults() -> None:
    """`resolve_finetune_recipe(train_cfg=None)` is what `tools/bitexact/sft_probe.py`
    calls at import time to build `_SFT_RECIPE`, and what `main()` resolves to for any
    checkpoint whose `cfg` does not carry its own lr/weight_decay/grad_clip.

    Asserts on the resolved `FinetuneRecipe` object, not on the raw
    `_FinetuneDefaults` class attributes, so the resolution function itself is
    covered.
    """
    recipe = resolve_finetune_recipe(train_cfg=None)

    assert recipe.lr == pytest.approx(1e-4)
    assert recipe.lr_source == "cli"
    assert recipe.weight_decay == pytest.approx(1e-3)
    assert recipe.weight_decay_source == "cli"
    assert recipe.grad_clip == pytest.approx(0.5)
    assert recipe.grad_clip_source == "cli"
    assert recipe.finetune_method == "lora"
    assert recipe.lora_r == 8
    assert recipe.lora_alpha == pytest.approx(16.0)
    assert recipe.lora_dropout == pytest.approx(0.05)
    assert recipe.lora_target_scope == "encoder_decoder_qkv"
    assert recipe.idm_action_loss_horizon_gamma == pytest.approx(0.0)
    assert recipe.batch_size == 4096
    assert recipe.max_train_steps == 100
    assert recipe.trim_head_steps == 0
    assert recipe.trim_tail_steps == 0
    assert recipe.seed == 42
    assert recipe.pred_horizon_k is None
    # `train_val_split`/`target_dataset_count`/`val_every_steps`/`val_batches` are not
    # recipe fields -- there is no split for them to configure.
    assert not hasattr(recipe, "train_val_split")
    assert not hasattr(recipe, "target_dataset_count")
    assert not hasattr(recipe, "val_every_steps")
    assert not hasattr(recipe, "val_batches")
    # Data-selection / loss-semantics fields. `obs_key`/`command_key`/
    # `idm_loss_target`/`idm_anchor_lambda` are verbatim mirrors of `_FinetuneDefaults`;
    # `idm_use_current_command_for_history` is resolved through tiers (D -> checkpoint
    # cfg -> False), so with no checkpoint cfg it lands on the `fallback` tier.
    assert recipe.obs_key == "auto"
    assert recipe.command_key == "current_command"
    assert recipe.idm_loss_target == "idm_planner_fdm_obs"
    assert recipe.idm_anchor_lambda == pytest.approx(1.0)
    assert recipe.idm_use_current_command_for_history is False
    assert recipe.idm_use_current_command_for_history_source == "fallback"


def test_resolve_finetune_recipe_takes_idm_command_history_from_checkpoint_cfg() -> None:
    """`idm_use_current_command_for_history` is the one of the five data-selection
    recipe fields `main()` resolves against the checkpoint cfg (the
    `_FinetuneDefaults` value is `None`): a checkpoint carrying `True` wins, and the
    source field reports `"checkpoint"`.

    The resolved value is what is written into the checkpoint `cfg` payload and the
    exported ONNX metadata.
    """
    recipe = resolve_finetune_recipe(train_cfg={"idm_use_current_command_for_history": True})

    assert recipe.idm_use_current_command_for_history is True
    assert recipe.idm_use_current_command_for_history_source == "checkpoint"


def test_resolve_finetune_artifact_policy_with_no_overrides_matches_release_defaults() -> None:
    policy = resolve_finetune_artifact_policy()

    assert policy.save_checkpoints is False
    assert policy.export_onnx is True
    assert policy.export_onnx_every_epoch is False
    assert policy.save_adapter_only is False
