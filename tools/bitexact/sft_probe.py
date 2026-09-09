"""Bit-exactness probe: the SFT (target-domain IDM LoRA finetune) forward/backward step.

Builds a PlannerIDMPolicy from the shared `build_probe_config()`, attaches the
adapter through `finetune_idm_lora.py`'s production LoRA injection path
(`configure_idm_finetune_method`, `finetune_method="lora"`,
`lora_target_scope="encoder_decoder_qkv"` -- the values `_FinetuneDefaults` ships
in `holosoma/fada/planner_idm/finetune_idm_lora.py`), then calls the production
`_run_epoch(..., loss_target="idm")` for N steps on batches from a synthetic
`TrajectoryWindowDataset`, recording the per-step loss and the sha256 of the LoRA
adapter `state_dict`.

Calling `_run_epoch` rather than reimplementing the forward/loss computation puts
the shipped code in the probe: the `predict_idm_actions` call, `weighted_horizon_mse`'s
`horizon_gamma` weighting and the `grad_clip` handling.

It emits no hash of a whole checkpoint payload -- only the loss sequence (at full
float64 precision) plus the LoRA adapter's sha256. See `compare.py`'s docstring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.fada.common.lora_utils import Trajectory
from holosoma.fada.planner_idm.finetune_idm_lora import (
    FinetuneArtifactPolicy,
    FinetuneRecipe,
    TrajectoryWindowDataset,
    _run_epoch,
    configure_idm_finetune_method,
    extract_lora_adapter_state_dict,
    resolve_finetune_artifact_policy,
    resolve_finetune_recipe,
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

# LoRA config, optimizer hyperparameters and the IDM action-loss horizon weighting
# come from the same resolve_finetune_recipe() that main() calls, not from constants
# copied into this file.
#
# `resolve_finetune_recipe(train_cfg=None)` passes no checkpoint cfg (this probe
# loads no checkpoint), so every field resolves from `_FinetuneDefaults`: the recipe
# the shipped entrypoint uses when no checkpoint override applies. A change to those
# defaults changes `_SFT_RECIPE` and hence the canonical-JSON sha256 recorded as
# run_sft_probe()'s `recipe_hash` field in `sft.json`.
#
# `_SFT_RECIPE` also carries `batch_size`/`max_train_steps`/`trim_head_steps`/
# `trim_tail_steps`/`seed`/`pred_horizon_k`, which mirror `_FinetuneDefaults` but are
# not read by this probe's synthetic loop (it uses `_SFT_BATCH_SIZE`/`_SFT_TRAJ_LEN`
# and loads no H5 data); they move `recipe_hash` only -- see `FinetuneRecipe`'s
# "Mirrored-defaults note" docstring.
#
# `finetune_idm_lora` has no train/val split, so `train_val_split`,
# `target_dataset_count`, `val_every_steps` and `val_batches` are not in that group.
# Adding or removing a field from the group moves `recipe_hash`; `loss_sequence` and
# `adapter_sha256` move only when a number the optimizer sees changes.
_SFT_RECIPE: FinetuneRecipe = resolve_finetune_recipe(train_cfg=None)

# Artifact-production defaults (which files main() writes), hashed separately from
# `_SFT_RECIPE`; they affect no weight or loss value. See
# `FinetuneArtifactPolicy`'s docstring.
_SFT_ARTIFACT_POLICY: FinetuneArtifactPolicy = resolve_finetune_artifact_policy()

# Synthetic-dataset sizing owned by this probe, not part of
# FinetuneRecipe/_SFT_RECIPE and independent of harness.py's episode constants.
# _SFT_NUM_TRAJECTORIES * (_SFT_TRAJ_LEN - pred_horizon) / _SFT_BATCH_SIZE exceeds
# the probe step count (50), so the DataLoader iterator does not wrap mid-run.
_SFT_TRAJ_LEN = 400
_SFT_NUM_TRAJECTORIES = 4
_SFT_BATCH_SIZE = 16


def _recipe_hash(recipe: FinetuneRecipe) -> str:
    canonical = json.dumps(recipe.as_canonical_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _artifact_policy_hash(policy: FinetuneArtifactPolicy) -> str:
    canonical = json.dumps(policy.as_canonical_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _build_probe_model(cfg: Any) -> PlannerIDMPolicy:
    return PlannerIDMPolicy(
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


def _build_synthetic_trajectories(rng: np.random.Generator) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for i in range(_SFT_NUM_TRAJECTORIES):
        n = _SFT_TRAJ_LEN
        trajectories.append(
            Trajectory(
                obs=rng.standard_normal((n, PROBE_OBS_DIM)).astype(np.float32),
                actions=rng.standard_normal((n, PROBE_ACT_DIM)).astype(np.float32),
                current_command=rng.standard_normal((n, PROBE_CMD_DIM)).astype(np.float32),
                source=f"probe_traj_{i}",
            )
        )
    return trajectories


def _adapter_sha256(adapter_state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(adapter_state.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_sft_probe(out_path: Path, steps: int = 50) -> dict[str, Any]:
    _seed_everything(PROBE_SEED)
    cfg = build_probe_config()
    model = _build_probe_model(cfg)

    # model.idm is neither IDMMLPModel nor IDMHybridModel, so
    # configure_idm_finetune_method takes the generic branch and injects LoRA at
    # lora_target_scope verbatim.
    replaced, resolved_method, resolved_scope = configure_idm_finetune_method(
        model,
        finetune_method=_SFT_RECIPE.finetune_method,
        lora_r=_SFT_RECIPE.lora_r,
        lora_alpha=_SFT_RECIPE.lora_alpha,
        lora_dropout=_SFT_RECIPE.lora_dropout,
        lora_target_scope=_SFT_RECIPE.lora_target_scope,
    )
    _method_ok = resolved_method == _SFT_RECIPE.finetune_method
    _scope_ok = resolved_scope == _SFT_RECIPE.lora_target_scope
    if not replaced or not _method_ok or not _scope_ok:
        raise RuntimeError(
            f"LoRA injection did not exercise the expected path: replaced={replaced!r} "
            f"resolved_method={resolved_method!r} resolved_scope={resolved_scope!r}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=_SFT_RECIPE.lr, weight_decay=_SFT_RECIPE.weight_decay)

    trajectories = _build_synthetic_trajectories(np.random.default_rng(PROBE_SEED))
    dataset = TrajectoryWindowDataset(
        trajectories,
        history_len=cfg.history_len,
        pred_horizon_k=cfg.pred_horizon,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=_SFT_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(PROBE_SEED),
    )

    losses: list[float] = []
    loader_iter = iter(loader)
    for _ in range(steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        # Calls `_run_epoch`, which is the inner step runner (there is no
        # epoch-driven outer loop -- see `_FinetuneDefaults`' "Step-driven-only
        # note"), with a single-batch "loader" so each probe step is one optimizer
        # step.
        metrics = _run_epoch(
            model=model,
            loader=[batch],
            optimizer=optimizer,
            device=torch.device("cpu"),
            io_norm_enabled=False,
            obs_stats=None,
            action_stats=None,
            command_stats=None,
            pred_horizon_k=cfg.pred_horizon,
            grad_clip=_SFT_RECIPE.grad_clip,
            show_progress=False,
            epoch_desc="sft_probe",
            loss_target="idm",
            horizon_gamma=_SFT_RECIPE.idm_action_loss_horizon_gamma,
        )
        losses.append(float(metrics["loss"]))

    adapter_state = extract_lora_adapter_state_dict(model)
    if not adapter_state:
        raise RuntimeError("No LoRA adapter parameters found after finetuning; probe missed the LoRA path.")

    result: dict[str, Any] = {
        "loss_sequence": losses,
        "adapter_sha256": _adapter_sha256(adapter_state),
        # Hash of the FinetuneRecipe that main() and this probe both resolve from
        # _FinetuneDefaults. It moves on any change to the LoRA/optimizer/gamma
        # defaults, including changes that leave the loss sequence and adapter
        # weights unchanged.
        "recipe_hash": _recipe_hash(_SFT_RECIPE),
        # Separate hash for the artifact-production defaults (which files
        # get written), independent of the numerical recipe -- see
        # `FinetuneArtifactPolicy`'s docstring and `_SFT_ARTIFACT_POLICY` above.
        "artifact_policy_hash": _artifact_policy_hash(_SFT_ARTIFACT_POLICY),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
