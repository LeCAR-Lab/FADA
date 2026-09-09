from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.fada.common import cli_validation
from holosoma.fada.common.lora_utils import (  # noqa: F401
    LoRALinear,
    LoRAMultiheadAttention,
    RawObsPreprocess,
    Trajectory,
    _clone_state_dict_to_cpu,
    _extract_norm_stats,
    _init_wandb_run,
    _inject_lora_modules,
    _inject_lora_multihead_attention_modules,
    _parse_bool,
    _parse_csv_list,
    _resolve_bool_hparam,
    _resolve_checkpoint_path,
    _resolve_output_root,
    _resolve_training_hparam,
    _safe_wandb_log,
    build_merged_state_dict,
    extract_lora_adapter_state_dict,
    extract_trainable_parameter_state_dict,
    load_trajectories_from_h5,
)
from holosoma.fada.common.utils import _parse_step_milestones
from holosoma.fada.planner_idm.eval_checkpoint import (
    _build_model_from_checkpoint,
    _export_planner_idm_onnx,
)
from holosoma.fada.planner_idm.loss_utils import weighted_horizon_mse
from holosoma.utils.safe_torch_import import F, optim, torch
from holosoma.utils.safe_torch_load import load_checkpoint as safe_load_checkpoint


def _default_run_name(finetune_method: str = "lora") -> str:
    token = "fada_idm_lora" if str(finetune_method) == "lora" else "fada_idm_full"
    return f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{token}"


def _checkpoint_run_dir(ckpt_path: Path) -> Path:
    """Return the DAgger *run directory* implied by ``--checkpoint``.

    Mirrors ``eval_checkpoint._default_output_dir``'s rule so both entry points agree
    on what "the run this checkpoint belongs to" means: the checkpoint's parent, or
    its grandparent when the checkpoint sits in a ``checkpoints/`` subdirectory.
    """
    run_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    if run_dir.name == "checkpoints":
        run_dir = run_dir.parent
    return run_dir


def _default_run_name_from_checkpoint(ckpt_path: Path) -> str:
    """Derive a checkpoint-scoped ``--run-name`` default, or "" if none can be derived.

    Scheme: the source run directory's name with any leading ``YYYYmmdd_HHMMSS_`` stamp
    stripped, plus a ``_sft`` suffix. A fresh timestamp is prefixed onto the result by
    ``_ensure_timestamp_prefixed_run_name`` so repeated finetunes of one checkpoint land
    in distinct run directories instead of overwriting each other.
    """
    token = re.sub(r"^\d{8}_\d{6}_", "", _checkpoint_run_dir(ckpt_path).resolve().name).strip()
    return f"{token}_sft" if token else ""


def _default_output_dir_from_checkpoint(ckpt_path: Path) -> str:
    """Derive an absolute ``--output-dir`` default: ``<checkpoint's run dir>/finetune``.

    Absolute on purpose: ``_resolve_output_root`` resolves a *relative* ``--output-dir``
    against the checkpoint's directory rather than the cwd, so a relative default would
    read ambiguously. Returning an absolute path means the derived default and an
    explicitly typed one behave identically.
    """
    return str(_checkpoint_run_dir(ckpt_path).resolve() / "finetune")


def _trajectory_window_dataset_sources(dataset: Any) -> list[str]:
    """The `Trajectory.source` ids (``<h5 path>:<episode>:env<N>``, stable and unique
    per trajectory) actually reachable through `dataset`.

    Reads the dataset object that was handed to the DataLoader rather than the
    trajectory list `main()` built it from, so the provenance record reflects what
    `main()` really trained on even if something mutated that list afterwards.
    Trajectories that contribute zero windows are skipped -- they are not part of the
    training set in any meaningful sense.
    """
    if dataset is None:
        return []
    sources: list[str] = []
    for traj, count in zip(dataset.trajectories, dataset.sample_counts):
        if int(count) <= 0:
            continue
        sources.append(str(traj.source))
    return sources


def _ensure_timestamp_prefixed_run_name(name: str) -> str:
    token = str(name).strip()
    if token == "":
        token = "fada_idm_lora"
    if re.match(r"^\d{8}_\d{6}_", token):
        return token
    return f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{token}"


def _remap_norm_stats_keys(
    stats: dict[str, torch.Tensor] | None,
    *,
    mean_key: str,
    std_key: str,
) -> dict[str, torch.Tensor] | None:
    if stats is None:
        return None
    if mean_key in stats and std_key in stats:
        return {mean_key: stats[mean_key], std_key: stats[std_key]}
    if "mean" in stats and "std" in stats:
        return {mean_key: stats["mean"], std_key: stats["std"]}
    raise KeyError(
        f"Normalization stats missing expected keys. "
        f"Found keys={sorted(str(key) for key in stats.keys())}, "
        f"expected either ({mean_key}, {std_key}) or ('mean', 'std')."
    )


class TrajectoryWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: list[Trajectory],
        *,
        history_len: int,
        pred_horizon_k: int,
        include_metadata: bool = False,
    ):
        super().__init__()
        self.trajectories = trajectories
        self.history_len = int(history_len)
        self.pred_horizon_k = int(pred_horizon_k)
        self.include_metadata = bool(include_metadata)
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if self.pred_horizon_k <= 0:
            raise ValueError("pred_horizon_k must be positive")

        self.sample_counts = [max(0, int(traj.obs.shape[0]) - self.pred_horizon_k) for traj in trajectories]
        self.cumsum = np.cumsum(np.asarray(self.sample_counts, dtype=np.int64))
        self.total_samples = int(self.cumsum[-1]) if self.cumsum.size > 0 else 0
        if self.total_samples <= 0:
            raise RuntimeError("No valid windows in trajectory dataset. Collect longer trajectories or reduce K.")

    def __len__(self) -> int:
        return self.total_samples

    def _resolve(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.total_samples}")
        traj_idx = int(bisect_right(self.cumsum, index))
        prev = int(self.cumsum[traj_idx - 1]) if traj_idx > 0 else 0
        local_t = int(index - prev)
        return traj_idx, local_t

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj_idx, t = self._resolve(index)
        traj = self.trajectories[traj_idx]
        obs = traj.obs
        actions = traj.actions
        current_command_seq = traj.current_command

        history_obs = np.zeros((self.history_len, obs.shape[1]), dtype=np.float32)
        history_act = np.zeros((self.history_len, actions.shape[1]), dtype=np.float32)

        start = max(0, t - self.history_len + 1)
        hist_indices = np.arange(start, t + 1, dtype=np.int64)
        offset = int(self.history_len - hist_indices.shape[0])

        history_obs[offset:] = obs[hist_indices]
        history_valid = np.zeros((self.history_len,), dtype=np.bool_)
        history_valid[offset:] = True

        prev_indices = hist_indices - 1
        valid_prev = prev_indices >= 0
        if np.any(valid_prev):
            valid_slots = np.flatnonzero(valid_prev)
            history_act[offset + valid_slots] = actions[prev_indices[valid_prev]]

        current_command = current_command_seq[t].astype(np.float32, copy=False)
        future_obs = obs[t + 1 : t + 1 + self.pred_horizon_k]
        target_actions = actions[t : t + self.pred_horizon_k]
        if future_obs.shape[0] != self.pred_horizon_k or target_actions.shape[0] != self.pred_horizon_k:
            raise RuntimeError(
                f"Unexpected target horizon at index {index}: "
                f"future_obs={future_obs.shape[0]} target_actions={target_actions.shape[0]} expected={self.pred_horizon_k}"
            )

        sample = {
            "history_obs": torch.from_numpy(history_obs),
            "history_act": torch.from_numpy(history_act),
            "current_command": torch.from_numpy(current_command),
            "history_valid_mask": torch.from_numpy(history_valid),
            "future_observations": torch.from_numpy(future_obs.astype(np.float32, copy=False)),
            "target_actions": torch.from_numpy(target_actions.astype(np.float32, copy=False)),
        }
        if self.include_metadata:
            sample["trajectory_index"] = torch.tensor(traj_idx, dtype=torch.int64)
            sample["window_index"] = torch.tensor(t, dtype=torch.int64)
            sample["source"] = traj.source
        return sample


def _normalize_idm_if_enabled(
    history_obs: torch.Tensor,
    history_act: torch.Tensor,
    current_command: torch.Tensor,
    future_obs: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    io_norm_enabled: bool,
    obs_stats: dict[str, torch.Tensor] | None,
    action_stats: dict[str, torch.Tensor] | None,
    command_stats: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not io_norm_enabled:
        return history_obs, history_act, current_command, future_obs, target_actions

    if obs_stats is None or action_stats is None or command_stats is None:
        raise RuntimeError("Missing normalization stats for io_normalization=True")

    obs_mean = obs_stats["mean"].view(1, 1, -1)
    obs_std = obs_stats["std"].view(1, 1, -1)
    act_mean = action_stats["mean"].view(1, 1, -1)
    act_std = action_stats["std"].view(1, 1, -1)
    cmd_mean = command_stats["mean"].view(1, -1)
    cmd_std = command_stats["std"].view(1, -1)

    return (
        (history_obs - obs_mean) / obs_std,
        (history_act - act_mean) / act_std,
        (current_command - cmd_mean) / cmd_std,
        (future_obs - obs_mean) / obs_std,
        (target_actions - act_mean) / act_std,
    )


def _normalize_lora_target_scope(scope: str) -> str:
    token = str(scope).strip().lower().replace("-", "_")
    if token in {"encoder_decoder", "full", "all"}:
        return "encoder_decoder"
    if token in {"encoder_decoder_qkv", "full_qkv", "all_qkv", "lora_all"}:
        return "encoder_decoder_qkv"
    if token in {"decoder_only", "decoder", "lora_mlp"}:
        return "decoder_only"
    if token in {"decoder_only_qkv", "decoder_qkv"}:
        return "decoder_only_qkv"
    if token in {"all_linear", "mlp", "mlp_all"}:
        return "all_linear"
    raise ValueError(
        f"Unsupported IDM LoRA target scope: {scope!r}. "
        "Expected one of: encoder_decoder, decoder_only, encoder_decoder_qkv, decoder_only_qkv, all_linear, lora_mlp, lora_all."
    )


def _normalize_finetune_method(method: str) -> str:
    token = str(method).strip().lower().replace("-", "_")
    if token in {"lora", "idm_lora"}:
        return "lora"
    if token in {"full", "full_finetune", "all"}:
        return "full"
    raise ValueError(f"Unsupported finetune method: {method!r}. Expected one of: lora, full.")


def _freeze_all_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _mark_lora_trainables(model) -> None:
    for name, param in model.named_parameters():
        if name.endswith(".A.weight") or name.endswith(".B.weight"):
            param.requires_grad = True


def _name_starts_with_any_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _inject_lora_into_named_prefixes(
    model,
    *,
    linear_prefixes: tuple[str, ...],
    attention_prefixes: tuple[str, ...] = (),
    r: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    _freeze_all_parameters(model)
    replaced = _inject_lora_modules(
        model,
        should_replace=lambda name: _name_starts_with_any_prefix(name, linear_prefixes),
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    if attention_prefixes:
        replaced = _inject_lora_multihead_attention_modules(
            model,
            should_replace=lambda name: _name_starts_with_any_prefix(name, attention_prefixes),
            r=r,
            alpha=alpha,
            dropout=dropout,
            replaced=replaced,
        )
    if not replaced:
        raise RuntimeError(
            "No matching modules were found for LoRA injection "
            f"(linear_prefixes={linear_prefixes}, attention_prefixes={attention_prefixes})."
        )
    _mark_lora_trainables(model)
    return replaced


def _is_idm_backbone_linear(name: str, *, target_scope: str) -> bool:
    normalized_scope = _normalize_lora_target_scope(target_scope)
    if name.startswith("idm.decoder."):
        return True
    if normalized_scope in {"encoder_decoder", "encoder_decoder_qkv"} and name.startswith("idm.history_encoder."):
        return True
    return False


def _is_idm_backbone_attention(name: str, *, target_scope: str) -> bool:
    normalized_scope = _normalize_lora_target_scope(target_scope)
    if normalized_scope not in {"decoder_only_qkv", "encoder_decoder_qkv"}:
        return False
    if name.startswith("idm.decoder."):
        return True
    if normalized_scope == "encoder_decoder_qkv" and name.startswith("idm.history_encoder."):
        return True
    return False


def inject_lora_into_idm_backbone(
    model,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_scope: str = "encoder_decoder",
) -> list[str]:
    normalized_scope = _normalize_lora_target_scope(target_scope)

    _freeze_all_parameters(model)

    replaced = _inject_lora_modules(
        model,
        should_replace=lambda name: _is_idm_backbone_linear(name, target_scope=normalized_scope),
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    if normalized_scope in {"decoder_only_qkv", "encoder_decoder_qkv"}:
        replaced = _inject_lora_multihead_attention_modules(
            model,
            should_replace=lambda name: _is_idm_backbone_attention(name, target_scope=normalized_scope),
            r=r,
            alpha=alpha,
            dropout=dropout,
            replaced=replaced,
        )
    if not replaced:
        raise RuntimeError(
            f"No IDM backbone linear layers matched for LoRA injection (target_scope={normalized_scope})."
        )

    _mark_lora_trainables(model)
    return replaced


def enable_full_idm_finetune(model) -> list[str]:
    _freeze_all_parameters(model)
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if name.startswith("idm.") or name.startswith("idm_decoder.") or name.startswith("shared_history_encoder."):
            param.requires_grad = True
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("No IDM parameters found for full finetuning.")
    return ["idm"]


def configure_idm_finetune_method(
    model,
    *,
    finetune_method: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_scope: str,
) -> tuple[list[str], str, str | None]:
    resolved_method = _normalize_finetune_method(finetune_method)
    if resolved_method == "lora":
        effective_scope = lora_target_scope
        replaced = inject_lora_into_idm_backbone(
            model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_scope=effective_scope,
        )
        return replaced, resolved_method, _normalize_lora_target_scope(effective_scope)
    replaced = enable_full_idm_finetune(model)
    return replaced, resolved_method, None


def _trainable_param_signature(model) -> str:
    """sha256 over every currently-`requires_grad` parameter's name and raw
    bytes, in sorted-name order.

    Purpose: `main()` snapshots this immediately before and immediately after a
    training stage's step-driven `while` loop. If the two signatures are equal after
    a stage that ran `max_train_steps > 0` steps, the optimizer step did not actually
    move any trainable weight -- e.g. a backward()/optimizer.step() call that got
    silently skipped, or a loop body that runs but touches a different (frozen) set
    of parameters than the ones `requires_grad` was set on. This is a distinct
    failure mode from "the loop didn't run at all" (which `total_steps_done !=
    max_train_steps` below already catches): the loop can run every step, log a
    loss, and still leave the shipped weights bit-identical to the input checkpoint,
    which the step-count check alone would never see.
    """
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if not param.requires_grad:
            continue
        h.update(name.encode())
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _optimizer_step_counts(optimizer, params: list) -> tuple[list[int], int]:
    """Read the per-parameter `step` counter AdamW keeps in
    `optimizer.state[param]["step"]`, for exactly the parameter objects that were
    handed to the optimizer.

    Returns `(step_counts, num_params_without_state)`. A parameter only gets a state
    entry once `optimizer.step()` has actually processed it with a non-`None` grad,
    so `num_params_without_state > 0` means some trainable parameter never received a
    real optimizer update. The counter itself is a 0-dim tensor on modern torch and a
    plain float on older ones; both are coerced to `int` here.

    This is what distinguishes "AdamW stepped N times" from "something wrote to the
    parameter tensors". `_trainable_param_signature` above only proves the bits moved
    -- an in-place `param += 1.0` (the exact fault an external review injected in
    place of `optimizer.step()`) satisfies it just as well as real training does.
    """
    step_counts: list[int] = []
    missing = 0
    for param in params:
        state = optimizer.state.get(param)
        if not state or "step" not in state:
            missing += 1
            continue
        raw_step = state["step"]
        try:
            step_counts.append(int(raw_step.item()))  # 0-dim tensor (torch >= 2.x)
        except AttributeError:
            step_counts.append(int(raw_step))  # plain int/float (older torch)
    return step_counts, missing


def _loss_metric_key(loss_target: str) -> str:
    token = str(loss_target).strip().lower()
    if token == "idm":
        return "idm_action_loss"
    raise ValueError(f"Unsupported finetune loss target: {loss_target!r}")


def _run_epoch(
    *,
    model,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    io_norm_enabled: bool,
    obs_stats: dict[str, torch.Tensor] | None,
    action_stats: dict[str, torch.Tensor] | None,
    command_stats: dict[str, torch.Tensor] | None,
    pred_horizon_k: int,
    grad_clip: float,
    show_progress: bool,
    epoch_desc: str,
    max_batches: int | None = None,
    loss_target: str = "idm",
    anchor_lambda: float = 1.0,
    horizon_gamma: float = 1.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    lora_active = any(isinstance(m, (LoRALinear, LoRAMultiheadAttention)) for m in model.modules())
    disable_eval_fastpath = (not is_train) and lora_active and bool(torch.backends.mha.get_fastpath_enabled())
    fastpath_prev = bool(torch.backends.mha.get_fastpath_enabled())
    if disable_eval_fastpath:
        torch.backends.mha.set_fastpath_enabled(False)

    total_loss = 0.0
    total_loss_unweighted = 0.0
    total_planner_cycle_loss = 0.0
    total_action_anchor_loss = 0.0
    total_action_anchor_loss_unweighted = 0.0
    total_steps = 0
    loss_metric_key = _loss_metric_key(loss_target)

    iterator = loader
    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(loader, desc=epoch_desc, leave=False, dynamic_ncols=True)
        except Exception:
            iterator = loader

    try:
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            history_obs = batch["history_obs"].to(device=device, dtype=torch.float32)
            history_act = batch["history_act"].to(device=device, dtype=torch.float32)
            current_command = batch["current_command"].to(device=device, dtype=torch.float32)
            history_valid = batch["history_valid_mask"].to(device=device, dtype=torch.bool)
            future_obs = batch["future_observations"].to(device=device, dtype=torch.float32)
            target_actions = batch["target_actions"].to(device=device, dtype=torch.float32)

            history_obs, history_act, current_command, future_obs, target_actions = _normalize_idm_if_enabled(
                history_obs,
                history_act,
                current_command,
                future_obs,
                target_actions,
                io_norm_enabled=io_norm_enabled,
                obs_stats=obs_stats,
                action_stats=action_stats,
                command_stats=command_stats,
            )

            with torch.set_grad_enabled(is_train):
                idm_current_command = model._resolve_idm_current_command(
                    current_command,
                    idm_use_current_command_for_history=model.idm_use_current_command_for_history,
                )
                pred_actions = model.predict_idm_actions(
                    history_obs,
                    history_act,
                    idm_current_command,
                    future_obs,
                    history_valid_mask=history_valid,
                )
                pred_actions = pred_actions[:, :pred_horizon_k, :]
                # Adapt target when the IDM head outputs a single action step
                # (pred_horizon=1) but the batch carries a multi-step target.
                idm_target = target_actions
                if pred_actions.shape[1] == 1 and idm_target.shape[1] > 1:
                    idm_target = idm_target[:, :1, :]
                loss_uw = F.mse_loss(pred_actions, idm_target[:, :pred_horizon_k, :])
                loss = weighted_horizon_mse(pred_actions, idm_target[:, :pred_horizon_k, :], horizon_gamma)

                if is_train:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip > 0.0:
                        params = [p for p in model.parameters() if p.requires_grad]
                        if params:
                            torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_steps += 1
            total_loss_unweighted += float(loss_uw.detach().cpu().item())
    finally:
        if disable_eval_fastpath:
            torch.backends.mha.set_fastpath_enabled(fastpath_prev)

    if total_steps <= 0:
        result = {"loss": float("nan"), "steps": 0.0}
    else:
        result = {"loss": total_loss / total_steps, "steps": float(total_steps)}
    result[loss_metric_key] = result["loss"]
    if total_steps > 0:
        result["idm_action_loss_unweighted"] = total_loss_unweighted / total_steps
    return result


# ── Buffer dual-target IDM finetune helpers ───────────────────────────────────


class _BufferSampler:
    """Loads buffer NPZ files into ReplayBuffer objects and samples source-aware batches.

    Source semantics mirror the DAgger trainer mixed-mode IDM batch composition:
    - optimal / online / suboptimal_expert  → expert targets (oracle)
    - suboptimal / online_trajectory        → real-causal targets
    """

    def __init__(
        self,
        *,
        optimal_npz: str | None,
        suboptimal_npz: str | None,
        online_npz: str | None,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        history_len: int,
        pred_horizon: int,
        seed: int = 42,
        idm_suboptimal_ratio: float = 0.375,
        suboptimal_expert_ratio: float = 0.5,
        online_trajectory_ratio: float = 0.5,
        mixed_offline_ratio: float = 0.2,
    ) -> None:
        from holosoma.fada.common.dataset import ReplayBuffer

        def _load(npz_path: str | None, label: str) -> ReplayBuffer | None:
            if not npz_path:
                return None
            p = Path(npz_path)
            if not p.exists():
                raise FileNotFoundError(f"Buffer NPZ not found: {npz_path}")
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  [buffer] mmap {label}: {p.name} ({size_mb:.0f} MB)", flush=True)
            buf = ReplayBuffer.from_npz_mmap(
                p,
                obs_dim=obs_dim,
                act_dim=act_dim,
                cmd_dim=cmd_dim,
                history_len=history_len,
                pred_horizon=pred_horizon,
            )
            print(f"  [buffer] building episode index for {label} ({buf.capacity} steps)...", flush=True)
            n_chunks = buf.num_valid_chunks()
            print(f"  [buffer] {label}: {n_chunks} valid chunks", flush=True)
            return buf

        self._IDM_SUBOPTIMAL_RATIO = float(idm_suboptimal_ratio)
        self._SUBOPTIMAL_EXPERT_RATIO = float(suboptimal_expert_ratio)
        self._ONLINE_TRAJECTORY_RATIO = float(online_trajectory_ratio)
        self._MIXED_OFFLINE_RATIO = float(mixed_offline_ratio)
        self._opt = _load(optimal_npz, "optimal")
        self._sub = _load(suboptimal_npz, "suboptimal")
        self._onl = _load(online_npz, "online")
        self._seed = int(seed)
        np.random.seed(self._seed)

    def _batch_counts(self, buffer_bs: int) -> dict[str, int]:
        total_sub = int(round(buffer_bs * self._IDM_SUBOPTIMAL_RATIO))
        total_sub = max(0, min(total_sub, buffer_bs))
        sub_exp = int(round(total_sub * self._SUBOPTIMAL_EXPERT_RATIO))
        sub_exp = max(0, min(sub_exp, total_sub))
        sub_real = total_sub - sub_exp
        rem = buffer_bs - total_sub
        opt_bs = int(round(rem * self._MIXED_OFFLINE_RATIO))
        opt_bs = max(0, min(opt_bs, rem))
        onl_total = rem - opt_bs
        onl_traj = int(round(onl_total * self._ONLINE_TRAJECTORY_RATIO))
        onl_traj = max(0, min(onl_traj, onl_total))
        onl_oracle = onl_total - onl_traj
        return {
            "optimal": opt_bs,
            "suboptimal": sub_real,
            "suboptimal_expert": sub_exp,
            "online": onl_oracle,
            "online_trajectory": onl_traj,
        }

    def sample_source_batches(self, buffer_bs: int) -> dict[str, dict[str, np.ndarray]]:
        if buffer_bs <= 0:
            return {}
        counts = self._batch_counts(buffer_bs)
        batches: dict = {}
        for src, bs, buf in [
            ("optimal", counts["optimal"], self._opt),
            ("suboptimal", counts["suboptimal"], self._sub),
            ("suboptimal_expert", counts["suboptimal_expert"], self._sub),
            ("online", counts["online"], self._onl),
            ("online_trajectory", counts["online_trajectory"], self._onl),
        ]:
            if bs <= 0 or buf is None or buf.num_valid_chunks() <= 0:
                continue
            batches[src] = buf.sample_chunk(bs)
        return batches


class _FinetuneDefaults:
    """Hardcoded values for the flags trimmed from the CLI (76 flags -> 4).

    Trimming the flags was a pure CLI-surface reduction, not a default-value change.
    Overriding one now means editing this class or calling the underlying functions
    directly rather than passing a flag.

    wandb-mode exception: `--wandb-mode` is a real CLI flag for the same
    reason `--device` is one on the training entrypoint -- it is environment/
    infrastructure configuration (does this machine have W&B credentials?), not a
    tunable hyperparameter, and offline release users have no other way to avoid a
    hard crash at step 0. Default stays "online" (unchanged); read from `args`, not
    `_FinetuneDefaults.wandb_mode` (which remains only as the field FADAConfig-style
    call sites expect to exist).

    Trim-steps exception: `--trim-head-steps` / `--trim-tail-steps` are also
    real CLI flags. These describe *what data goes into the batch* (how many H5
    steps are trimmed off each episode), not how training is tuned, and every
    finetune run -- including the small-scale, target-H5-only path this release
    ships -- needs a real way to set them. Defaults are unchanged (still read from
    this class); read the effective value from `args`, not `_FinetuneDefaults`.

    NPZ buffer fields note: the release scope is small-scale finetuning only (~2 minutes of
    target-domain rollouts per the paper); there is no large-scale stage in this
    repo. The NPZ dual-target buffer paths (`--npz-optimal`, `--npz-suboptimal`,
    `--npz-online`), the two step budgets that gated/scaled buffer mixing
    (`--buffer-step-budget`, `--target-step-budget`), and the four NPZ buffer-mix
    ratios (`--idm-suboptimal-ratio`, `--idm-suboptimal-expert-ratio`,
    `--idm-online-trajectory-ratio`, `--idm-mixed-offline-ratio`) were CLI flags
    only because `large_scale_finetune.py` (now removed) needed a way to forward
    them -- the small-scale, target-H5-only path never reads them. They remain
    hardcoded fields on this class purely as provenance recorded into
    `summary.json`/ONNX metadata (values unchanged from their prior CLI defaults);
    read them from `D`, not `args`.

    Buffer-mixing orchestration note: `main()` has no code that constructs a
    buffer sampler and mixes it into the training loader when these fields are
    non-default. Such code would be structurally unreachable -- these fields can
    only ever be their hardcoded defaults from the CLI. The
    `_MixedBatchLoader`/`_PureBufferDataLoader`/`_unify_buffer_source_batches_for_idm`
    machinery it would have driven is absent for the same reason (no callers
    anywhere in the repo, including tests). `_BufferSampler` itself is kept: it is
    still exercised directly by unit tests in `test_finetune_idm_lora.py` (not via
    the CLI), so it is not repo-wide unreachable even though `main()` never
    constructs one.

    Step-driven-only note: finetuning is purely step-driven. The epoch-driven outer training
    loop (and its `epochs`/`idm_epochs`/`fdm_epochs` fields) does not exist:
    `max_train_steps` is the only way this entrypoint trains, so `main()` always
    takes the step-based path implemented for `--save-step-milestones`. There is no
    `save_epoch_milestones`/`_parse_save_epoch_milestones` knob (nor the milestone
    auto-generation it would have driven): it would be reachable only from an epoch
    loop. `export_onnx_every_epoch` is kept -- it gates ONNX export at
    `--save-step-milestones` steps, which is what the step-driven path does.

    No-train/val-split note: there is no train/val split -- `train_val_split`,
    `target_dataset_count`, `val_every_steps` and `val_batches` do not exist as
    fields or flags. The release default was `train_val_split=0.0` (no split, no
    validation loader), so nothing is missing from the shipped path; what is absent
    is an unsound option. A sample-level split of this dataset splits *sliding
    windows* at random, so train and val windows overlap in time almost completely
    (a held-out window's history and future horizon are covered by neighbouring
    training windows), which makes the validation loss -- and any `best_val`
    checkpoint selection built on it -- a memorization score rather than a
    generalization one. Copying such a best-val snapshot back into the model before
    saving and exporting would ship *different weights*, chosen by that corrupted
    signal. With no validation there is no best-vs-last choice left: the run writes
    exactly one checkpoint and one ONNX, both from the final training step.
    """

    obs_key = "auto"
    command_key = "current_command"
    idm_use_current_command_for_history: bool | None = None
    pred_horizon_k: int | None = None
    trim_head_steps = 0
    trim_tail_steps = 0
    batch_size = 4096
    idm_loss_target = "idm_planner_fdm_obs"
    idm_anchor_lambda = 1.0
    idm_action_loss_horizon_gamma = 0.0
    # These three must stay non-None. `None` here sends `_resolve_training_hparam()`
    # straight to its checkpoint-cfg tier (priority is CLI-equivalent value here in
    # `D` -> checkpoint cfg -> fallback), and a DAgger checkpoint carries its own
    # training-time `weight_decay`/`grad_clip` in `cfg`, which would then silently
    # override the finetune values below. Explicit values make `D.lr` etc. win the
    # top priority tier unconditionally. See
    # test_finetune_hparams_prefer_recipe_over_checkpoint_cfg in
    # test_finetune_idm_lora.py, which fails if these regress to None.
    lr: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip: float = 0.5
    lora_r = 8
    lora_alpha = 16.0
    lora_dropout = 0.05
    finetune_method = "lora"
    lora_target_scope = "encoder_decoder_qkv"
    seed = 42
    device = "cuda:0"
    use_wandb = True
    wandb_project = "fada_idm_lora_finetune"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_mode = "online"
    wandb_tags = "transformer,planner_idm,idm_lora,finetune"
    export_onnx = True
    onnx_output_path: str | None = None
    save_adapter_only = False
    save_checkpoints = False
    # Gates ONNX export at each `--save-step-milestones` step. Finetuning is
    # step-driven; there is no epoch loop.
    export_onnx_every_epoch = False
    show_progress = True
    finetune_stage: str | None = None
    # Number of IDM update steps. Override with `--max-train-steps`.
    max_train_steps = 100
    save_step_milestones: str | None = None
    # NPZ dual-target buffer mixing -- not CLI flags; see the docstring's
    # "NPZ buffer fields note".
    npz_optimal: str | None = None
    npz_suboptimal: str | None = None
    npz_online: str | None = None
    buffer_step_budget = 0
    target_step_budget = 0
    idm_suboptimal_ratio = 0.375
    idm_suboptimal_expert_ratio = 0.5
    idm_online_trajectory_ratio = 0.5
    idm_mixed_offline_ratio = 0.2


@dataclasses.dataclass(frozen=True)
class FinetuneRecipe:
    """The behavioral (non-fixture) defaults of the finetune entrypoint: optimizer
    hyperparameters, LoRA config, the IDM action-loss horizon weighting, the
    data-composition / training-duration / RNG knobs that determine what a
    release run actually trains on and for how long, and the
    data-*selection* and loss-*semantics* knobs that decide which arrays get read out
    of the target H5 files and which loss the IDM is optimized against.

    This is the single source of truth `main()` and `tools/bitexact/sft_probe.py`
    both consume. A probe that keeps its own hand-copied `_SFT_LORA_*`/`_SFT_LR`/
    `_SFT_WEIGHT_DECAY`/`_SFT_GRAD_CLIP` constants cannot detect a regression in
    them: a fault injection that breaks `_FinetuneDefaults` (batch_size, gamma,
    dropout, finetune scope, split, max_steps) *and* neuters `main()`'s training
    loop still reports bit-exact `IDENTICAL`, because the probe never reads any of
    those values from the production code, only from its own frozen copies. Routing
    both call sites through this one function means a regression in the shared
    defaults changes what the probe golden-compares against.

    Mirrored-defaults note -- two different senses of "covered" for
    `batch_size`, `max_train_steps`, `trim_head_steps`, `trim_tail_steps`,
    `seed` and `pred_horizon_k`: unlike `lr`/`weight_decay`/`grad_clip`/the
    LoRA fields/`idm_action_loss_horizon_gamma` (which `main()` reads *from the
    resolved `FinetuneRecipe` instance* and which therefore drive its actual
    behavior), these six fields are plain mirrors of `_FinetuneDefaults`' current
    values, carried here purely so `recipe_hash` moves if one of those *defaults*
    regresses. `main()` reads `trim_head_steps`/`trim_tail_steps` from `args`
    (the CLI-overridable value, see `_FinetuneDefaults`' "Trim-steps exception")
    and the other four straight from `D` -- this class does not gate or replace that
    per-run resolution, it only records what the release *default* is so a
    fault-injected `_FinetuneDefaults` change is visible in `recipe_hash` even when
    a given CLI invocation overrides it. See `resolve_finetune_recipe()`'s docstring
    for why none of these six fields get a CLI/checkpoint/fallback "source" tier the
    way `lr`/`weight_decay`/`grad_clip` do. Any change to the set of hashed fields
    moves `recipe_hash`, which is the point of the hash: the release's config
    surface changed.

    Data-selection and loss-semantics note -- five more fields join the hash.
    Without them, mutating `_FinetuneDefaults.obs_key` to a wrong H5 key produces
    ZERO bitexact diffs and a still-green defaults-contract run. These five decide
    *what data is loaded* and *what loss is optimized*, so a silent regression in
    any of them ships a differently-trained model:

    - `obs_key` / `command_key`: the H5 dataset keys `main()` passes straight into
      `load_trajectories_from_h5(...)`. `obs_key="auto"` resolves per episode to
      `raw_dynamics_obs` (which then goes through `RawObsPreprocess` term scaling) or
      `dynamics_obs` (already scaled) -- so a wrong value here does not merely rename
      an array, it changes whether observations are rescaled at all. `main()` reads
      both verbatim from `D`, never from `train_cfg`, so they are mirrored verbatim
      here (same convention as the mirrored-defaults group above).
    - `idm_use_current_command_for_history`: the ONE field of the five that `main()`
      genuinely resolves against the checkpoint cfg, via `_resolve_bool_hparam`
      (`D` -> checkpoint `cfg` -> `False`). It therefore gets a real resolution tier
      and a `_source` companion field here, exactly like `lr`/`weight_decay`/
      `grad_clip`, and `main()` consumes the resolved value off this instance
      rather than calling `_resolve_bool_hparam` a second time itself. It flips
      whether the IDM sees the command in its history tokens (and is written back
      into `payload["cfg"]`, into every saved checkpoint's cfg, and into the exported
      ONNX metadata). That cfg rewrite still happens in `main()` and is unaffected by
      the resolution living here.
    - `idm_loss_target` / `idm_anchor_lambda`: the loss-semantics pair. Both are
      recorded into `summary.json`, the finetuned checkpoint's
      `extra["finetune_idm_lora"]`, and the exported ONNX metadata as the record of
      what was optimized. On the release path `main()` currently *pins* the single
      stage's `loss_target` to the literal `"idm"` and `_run_epoch` accepts but does
      not consume `anchor_lambda` (the planner-cycle/action-anchor loss terms it was
      built for are not part of this release's single-stage IDM finetune), so today
      neither value changes a gradient. They are hashed anyway because they are the
      declared loss contract: if a future change starts reading
      `D.idm_loss_target`/`D.idm_anchor_lambda` (or a regression edits their
      defaults), that must not slip through as an invisible change to what "the
      finetune loss" means. Both are mirrored verbatim from `D`.

    Deliberately excludes -- these cannot affect the trained weights, the loss
    trajectory, or which data is selected, so hashing them would only produce noisy
    golden churn: `wandb_*` and `use_wandb` (logging transport), `device` (which
    accelerator runs the identical math), `show_progress` (tqdm on/off),
    `onnx_output_path` (where a file lands, not what is in it), `finetune_stage`
    (assigned in `main()` and then never read -- inert on the single-stage,
    step-driven path), `save_step_milestones` (when intermediate artifacts are dumped; the
    weights at the end of the run are unchanged), and the mixed-source NPZ knobs
    `npz_optimal`/`npz_suboptimal`/`npz_online`, `buffer_step_budget`/
    `target_step_budget`, `idm_suboptimal_ratio`/`idm_suboptimal_expert_ratio`/
    `idm_online_trajectory_ratio`/`idm_mixed_offline_ratio`. The NPZ group's
    exclusion is checked rather than assumed: the entrypoint does not read them at
    all. `mixed_train_loader` is unconditionally `train_loader` and nothing
    constructs a `_BufferSampler`, so no ratio or budget can reach a batch. If any
    of them ever becomes readable on the plain target-only path, it belongs in this
    hash.

    Also excluded for the different reason that they are not release config at all:
    the pure artifact-production toggles covered by `FinetuneArtifactPolicy`
    (`save_checkpoints`/`export_onnx`/`export_onnx_every_epoch`/`save_adapter_only`
    change what files get written, not the trained weights or loss trajectory, so
    they are hashed separately as `artifact_policy_hash`), and the fixture-scale-only
    knobs owned by the probe itself (`_SFT_BATCH_SIZE`/`_SFT_TRAJ_LEN`/
    `_SFT_NUM_TRAJECTORIES` -- synthetic dataset sizing).
    """

    lr: float
    lr_source: str
    weight_decay: float
    weight_decay_source: str
    grad_clip: float
    grad_clip_source: str
    finetune_method: str
    lora_r: int
    lora_alpha: float
    lora_dropout: float
    lora_target_scope: str | None
    idm_action_loss_horizon_gamma: float
    # Data-composition / training-duration / RNG defaults. Plain mirrors of
    # `_FinetuneDefaults` -- see the class docstring's "Mirrored-defaults note" for
    # why these don't get CLI/checkpoint/fallback source tiers like
    # lr/weight_decay/grad_clip.
    batch_size: int
    max_train_steps: int
    trim_head_steps: int
    trim_tail_steps: int
    seed: int
    pred_horizon_k: int | None
    # Data-selection and loss-semantics defaults -- see the class docstring's
    # "Data-selection and loss-semantics note".
    # `obs_key`/`command_key`/`idm_loss_target`/`idm_anchor_lambda`
    # are verbatim mirrors of `_FinetuneDefaults`;
    # `idm_use_current_command_for_history` is really resolved (D -> checkpoint cfg ->
    # False) and therefore carries a `_source` tier like lr/weight_decay/grad_clip.
    obs_key: str
    command_key: str
    idm_loss_target: str
    idm_anchor_lambda: float
    idm_use_current_command_for_history: bool
    idm_use_current_command_for_history_source: str

    def as_canonical_dict(self) -> dict[str, Any]:
        """JSON-serializable, key-sorted view used for the golden recipe hash."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FinetuneArtifactPolicy:
    """The release defaults that decide *which artifacts get written*
    (checkpoint / ONNX / adapter-only), as opposed to `FinetuneRecipe`'s behavioral
    (weight- and loss-affecting) fields.

    Deliberately hashed separately from `FinetuneRecipe.recipe_hash`: flipping one
    of these does not change a single trained weight or logged loss value -- it only
    changes what `main()` writes to disk at the end of the run -- so folding it into
    the numerical recipe hash would conflate "the model trained differently" with
    "we forgot to export an artifact". Both are real defects, but they are different
    classes of defect and this repo's `main()` precheck (`save_checkpoints`/
    `export_onnx`/`save_adapter_only` must not all be False) already guards the
    "wrote nothing at all" case at runtime; `artifact_policy_hash` guards the
    quieter case where the *content* of what ships silently changes (e.g. ONNX
    export flips off by default) without necessarily tripping that precheck.
    """

    save_checkpoints: bool
    export_onnx: bool
    export_onnx_every_epoch: bool
    save_adapter_only: bool

    def as_canonical_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def resolve_finetune_recipe(
    *,
    train_cfg: dict[str, Any] | None = None,
    defaults: type[_FinetuneDefaults] = _FinetuneDefaults,
) -> FinetuneRecipe:
    """Resolve the finetune recipe from `defaults` (normally `_FinetuneDefaults`,
    itself acting as the "CLI" tier, most flags having been trimmed -- see that
    class's docstring), an optional checkpoint `train_cfg` (the "checkpoint" tier),
    and hardcoded fallbacks -- mirrors `_resolve_training_hparam`'s CLI > checkpoint
    > fallback priority.

    `train_cfg=None`/`{}` (the bitexact probe's use case) makes every hparam
    resolve straight from `defaults`, since `defaults.lr`/`weight_decay`/`grad_clip`
    are non-None and therefore always win the top tier regardless of what a
    checkpoint cfg would have supplied.

    `batch_size`/`max_train_steps`/`trim_head_steps`/`trim_tail_steps`/
    `seed`/`pred_horizon_k` are NOT run through `_resolve_training_hparam`
    here -- `main()` itself never consults `train_cfg` for any of them (they come
    straight from `D`, or from `args` for the two trim fields), so giving them a
    checkpoint-tier here would silently invent a resolution path `main()` doesn't
    actually have. They are copied verbatim from `defaults` purely so a regression
    in the *default* value is visible in `recipe_hash` (see `FinetuneRecipe`'s
    docstring).

    `obs_key`/`command_key`/`idm_loss_target`/`idm_anchor_lambda` follow that
    same verbatim-mirror convention, because `main()` reads all four straight from
    `D` and never consults `train_cfg` for them.
    `idm_use_current_command_for_history` is the exception: `main()` *does* consult
    the checkpoint cfg for it, so it is routed through `_resolve_bool_hparam`
    (`defaults` -> checkpoint `cfg` -> `False`) here with the same
    CLI/checkpoint/fallback tiering `lr`/`weight_decay`/`grad_clip` get, and the
    resolved value + its source are what `main()` consumes -- mirroring the real
    resolution path rather than inventing or dropping one. See `FinetuneRecipe`'s
    "Data-selection and loss-semantics note" for why each of the five is in the hash
    at all.
    """
    cfg = train_cfg or {}
    D = defaults

    resolved_lr, lr_source = _resolve_training_hparam(D.lr, train_cfg=cfg, key="lr", fallback=1e-4)
    resolved_weight_decay, weight_decay_source = _resolve_training_hparam(
        D.weight_decay, train_cfg=cfg, key="weight_decay", fallback=1e-3
    )
    resolved_grad_clip, grad_clip_source = _resolve_training_hparam(
        D.grad_clip, train_cfg=cfg, key="grad_clip", fallback=0.5
    )
    if resolved_lr <= 0.0:
        raise ValueError(f"Resolved learning rate must be positive, got {resolved_lr}")
    if resolved_weight_decay < 0.0:
        raise ValueError(f"Resolved weight decay must be non-negative, got {resolved_weight_decay}")
    if resolved_grad_clip < 0.0:
        raise ValueError(f"Resolved grad clip must be non-negative, got {resolved_grad_clip}")

    # The one field of the data-selection five that `main()` genuinely resolves
    # against the checkpoint cfg -- same tiering as lr/weight_decay/grad_clip above.
    resolved_idm_use_cmd_hist, idm_use_cmd_hist_source = _resolve_bool_hparam(
        D.idm_use_current_command_for_history,
        train_cfg=cfg,
        key="idm_use_current_command_for_history",
        fallback=False,
    )

    resolved_finetune_method = _normalize_finetune_method(D.finetune_method)
    resolved_lora_target_scope = (
        _normalize_lora_target_scope(str(D.lora_target_scope)) if resolved_finetune_method == "lora" else None
    )

    return FinetuneRecipe(
        lr=resolved_lr,
        lr_source=lr_source,
        weight_decay=resolved_weight_decay,
        weight_decay_source=weight_decay_source,
        grad_clip=resolved_grad_clip,
        grad_clip_source=grad_clip_source,
        finetune_method=resolved_finetune_method,
        lora_r=int(D.lora_r),
        lora_alpha=float(D.lora_alpha),
        lora_dropout=float(D.lora_dropout),
        lora_target_scope=resolved_lora_target_scope,
        idm_action_loss_horizon_gamma=float(D.idm_action_loss_horizon_gamma),
        batch_size=int(D.batch_size),
        max_train_steps=int(D.max_train_steps),
        trim_head_steps=int(D.trim_head_steps),
        trim_tail_steps=int(D.trim_tail_steps),
        seed=int(D.seed),
        pred_horizon_k=(int(D.pred_horizon_k) if D.pred_horizon_k is not None else None),
        obs_key=str(D.obs_key),
        command_key=str(D.command_key),
        idm_loss_target=str(D.idm_loss_target),
        idm_anchor_lambda=float(D.idm_anchor_lambda),
        idm_use_current_command_for_history=bool(resolved_idm_use_cmd_hist),
        idm_use_current_command_for_history_source=str(idm_use_cmd_hist_source),
    )


def resolve_finetune_artifact_policy(
    *,
    defaults: type[_FinetuneDefaults] = _FinetuneDefaults,
) -> FinetuneArtifactPolicy:
    """Resolve the artifact-production defaults (see `FinetuneArtifactPolicy`
    docstring for why these are hashed separately from `FinetuneRecipe`). Plain
    mirror of `_FinetuneDefaults` -- no checkpoint tier, matching how `main()` reads
    these fields (`bool(D.save_checkpoints)` etc, never from `train_cfg`).
    """
    D = defaults
    return FinetuneArtifactPolicy(
        save_checkpoints=bool(D.save_checkpoints),
        export_onnx=bool(D.export_onnx),
        export_onnx_every_epoch=bool(D.export_onnx_every_epoch),
        save_adapter_only=bool(D.save_adapter_only),
    )


_BOOL_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "on"})
_BOOL_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n", "off"})


def parse_bool_flag(value: str) -> bool:
    """argparse ``type=`` for explicit ``--flag true`` / ``--flag false`` booleans.

    Same reasoning as the training entrypoint's version: every override flag below uses
    ``default=None`` as an "unset" sentinel so an un-passed flag leaves the
    ``_FinetuneDefaults`` value untouched, which ``store_true``/``store_false`` cannot
    express for a field whose default is already True/False.
    """
    token = str(value).strip().lower()
    if token in _BOOL_TRUE_TOKENS:
        return True
    if token in _BOOL_FALSE_TOKENS:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (true/false), got {value!r}")


# Every `_FinetuneDefaults` field a user may reasonably want to change, as
# (field name, argparse type, choices or None, help). Flag spelling is the field name
# with underscores turned into dashes.
#
# Fields deliberately NOT here:
#   * npz_optimal / npz_suboptimal / npz_online / buffer_step_budget /
#     target_step_budget / idm_suboptimal_ratio / idm_suboptimal_expert_ratio /
#     idm_online_trajectory_ratio / idm_mixed_offline_ratio -- inert fields of the NPZ
#     dual-target buffer-mixing path, which this entrypoint does not have. `main()`
#     does not read them at all, so a flag would give an absent feature the
#     *appearance* of working.
#   * idm_loss_target / idm_anchor_lambda -- also inert on the release path: `main()`
#     pins the single stage's loss target to the literal "idm" and `_run_epoch` accepts
#     but never reads `anchor_lambda`. A flag would parse and change only what is
#     printed into summary.json / ONNX metadata.
#   * finetune_stage -- assigned in `main()` and then never read (inert on the
#     single-stage, step-driven path).
#   * trim_head_steps / trim_tail_steps -- already real flags of their own, read from
#     `args` by `main()`.
_OVERRIDABLE_DEFAULTS: tuple[tuple[str, object, tuple[str, ...] | None, str], ...] = (
    # ── Data selection ────────────────────────────────────────────────────────
    (
        "obs_key",
        str,
        None,
        "H5 observation dataset key. 'auto' resolves per episode to raw_dynamics_obs "
        "(term-scaled through RawObsPreprocess) or dynamics_obs (already scaled).",
    ),
    (
        "command_key",
        str,
        None,
        "H5 command dataset key. An episode that does not carry this key falls back to an "
        "all-zero command -- harmless when the checkpoint has "
        "idm_use_current_command_for_history=False (the shipped default), because nothing "
        "reads the command channel then. When the checkpoint DOES condition the IDM on the "
        "command, a key that no episode carries is an error rather than a silent zero, since "
        "otherwise a typo here trains on an all-zero command with no visible symptom.",
    ),
    ("pred_horizon_k", int, None, "Supervised action-chunk length; defaults to the checkpoint's pred_horizon."),
    # ── Optimization ──────────────────────────────────────────────────────────
    ("batch_size", int, None, "Training batch size."),
    ("max_train_steps", int, None, "Total optimizer steps (finetuning is step-driven only)."),
    ("lr", float, None, "AdamW learning rate."),
    ("weight_decay", float, None, "AdamW weight decay."),
    # Not "<=0 disables" as in the trainer: resolve_finetune_recipe() rejects a negative
    # resolved grad clip, and always has. 0 is the disable value here.
    ("grad_clip", float, None, "Gradient-norm clip; 0 disables clipping."),
    ("idm_action_loss_horizon_gamma", float, None, "Per-horizon discount for the IDM action loss."),
    ("seed", int, None, "RNG seed for numpy/torch."),
    # ── LoRA / finetune method ────────────────────────────────────────────────
    ("finetune_method", str, ("lora", "full"), "LoRA adapters on the IDM backbone, or a full IDM finetune."),
    ("lora_r", int, None, "LoRA rank."),
    ("lora_alpha", float, None, "LoRA alpha."),
    ("lora_dropout", float, None, "LoRA dropout."),
    (
        "lora_target_scope",
        str,
        (
            "encoder_decoder",
            "encoder_decoder_qkv",
            "decoder_only",
            "decoder_only_qkv",
            "all_linear",
        ),
        "Which IDM submodules receive LoRA adapters.",
    ),
    (
        "idm_use_current_command_for_history",
        parse_bool_flag,
        None,
        "Broadcast current_command into IDM history tokens. Unset resolves from the checkpoint cfg.",
    ),
    # ── Artifacts ─────────────────────────────────────────────────────────────
    ("save_checkpoints", parse_bool_flag, None, "Write the merged .pt checkpoint for the final training step."),
    ("export_onnx", parse_bool_flag, None, "Export the finetuned policy to ONNX."),
    ("export_onnx_every_epoch", parse_bool_flag, None, "Also export ONNX at each --save-step-milestones step."),
    ("save_adapter_only", parse_bool_flag, None, "Write a standalone LoRA adapter / trainable-weights file."),
    ("onnx_output_path", str, None, "Explicit ONNX output path (default: <run dir>/planner_idm_policy.onnx)."),
    ("save_step_milestones", str, None, "Comma-separated step numbers at which to dump intermediate artifacts."),
    # ── Infrastructure ────────────────────────────────────────────────────────
    ("device", str, None, "Torch device; falls back to CPU when CUDA is unavailable."),
    ("show_progress", parse_bool_flag, None, "Show tqdm progress bars."),
    ("use_wandb", parse_bool_flag, None, "Enable Weights & Biases logging."),
    ("wandb_project", str, None, "W&B project."),
    ("wandb_entity", str, None, "W&B entity."),
    ("wandb_group", str, None, "W&B group."),
    ("wandb_name", str, None, "W&B run name (also used as the run-directory name when --run-name is omitted)."),
    ("wandb_tags", str, None, "Comma-separated W&B tags."),
)


# Numeric domain for every numeric flag of this entry point, enforced uniformly in
# resolve_finetune_defaults() (see fada/common/cli_validation.py). Covers both the
# generated override flags and the two explicitly declared numeric ones.
_NUMERIC_DOMAINS: dict[str, cli_validation.NumericRange] = {
    # ── Data selection ────────────────────────────────────────────────────────
    "pred_horizon_k": cli_validation.POSITIVE_INT,
    "trim_head_steps": cli_validation.non_negative_int_disabled_by_zero("0 keeps every step of each episode."),
    "trim_tail_steps": cli_validation.non_negative_int_disabled_by_zero("0 keeps every step of each episode."),
    # ── Optimization ──────────────────────────────────────────────────────────
    "batch_size": cli_validation.POSITIVE_INT,
    "max_train_steps": cli_validation.POSITIVE_INT,
    "lr": cli_validation.POSITIVE_FLOAT,
    "weight_decay": cli_validation.NON_NEGATIVE_FLOAT,
    # Unlike the trainer's `--grad-clip` (whose contract really is "<=0 disables clipping"),
    # this entry point resolves grad_clip through `resolve_finetune_recipe`, which has always
    # raised on a negative resolved value -- see the `Resolved grad clip must be non-negative`
    # check there. The domain mirrors that pre-existing behavior so the rejection names the
    # flag up front instead of surfacing after the checkpoint load; `0` still disables
    # clipping in the training loop (`if grad_clip > 0.0:`).
    "grad_clip": cli_validation.non_negative_float_disabled_by_zero("0 disables gradient clipping."),
    "idm_action_loss_horizon_gamma": cli_validation.PROBABILITY,
    "seed": cli_validation.NON_NEGATIVE_INT,
    # ── LoRA ──────────────────────────────────────────────────────────────────
    # rank 0 would build empty adapters: every trainable parameter disappears and the
    # run "succeeds" without changing the model at all.
    "lora_r": cli_validation.POSITIVE_INT,
    "lora_alpha": cli_validation.POSITIVE_FLOAT,
    "lora_dropout": cli_validation.UNIT_HALF_OPEN,
}


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "hyperparameter overrides",
        "Every flag below defaults to the _FinetuneDefaults release value; omitting one leaves "
        "that default untouched. Booleans take an explicit true/false argument.",
    )
    for name, arg_type, choices, help_text in _OVERRIDABLE_DEFAULTS:
        domain = _NUMERIC_DOMAINS.get(name)
        kwargs: dict[str, Any] = {
            "type": arg_type,
            "default": None,
            "dest": name,
            "help": help_text + (domain.help_suffix() if domain is not None else ""),
        }
        if choices is not None:
            kwargs["choices"] = list(choices)
        group.add_argument(f"--{name.replace('_', '-')}", **kwargs)


def resolve_finetune_defaults(args: argparse.Namespace) -> type[_FinetuneDefaults]:
    """Return the `_FinetuneDefaults`-shaped object `main()` should read its config from.

    With no override flags this returns `_FinetuneDefaults` *itself* -- not a copy --
    so a default run reads byte-for-byte the same class attributes it always has, and
    `resolve_finetune_recipe(defaults=...)` produces the identical recipe (and hash)
    that `tools/bitexact/sft_probe.py` compares against. Any flag that was actually
    passed produces a thin subclass carrying only the overridden attributes, which
    every `D.<field>` read in `main()` then picks up.
    """
    overrides = {
        name: getattr(args, name)
        for name, _arg_type, _choices, _help in _OVERRIDABLE_DEFAULTS
        if getattr(args, name, None) is not None
    }
    # Range-check before anything downstream sees the value. `main()`'s own checks run
    # much later (after checkpoint load), cover only four fields, and never reject
    # nan/inf -- without this, a NaN --lr or --idm-action-loss-horizon-gamma trains to
    # NaN weights and still writes checkpoints, ONNX and summary.json.
    cli_validation.validate_numeric_values(
        _NUMERIC_DOMAINS,
        {
            **overrides,
            # Explicitly declared (non-sentinel) numeric flags: always present on the
            # namespace, so they are validated on every invocation, not only when passed.
            "trim_head_steps": getattr(args, "trim_head_steps", None),
            "trim_tail_steps": getattr(args, "trim_tail_steps", None),
        },
    )
    if not overrides:
        return _FinetuneDefaults
    return type("_FinetuneDefaultsWithCLIOverrides", (_FinetuneDefaults,), overrides)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline planner+idm finetuning for IDM checkpoint path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Planner-IDM checkpoint to finetune")
    parser.add_argument(
        "--target-datasets",
        type=str,
        nargs="+",
        required=True,
        dest="target_datasets",
        help="One or more target-domain H5 dataset paths (deployment-domain rollout data).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of this finetune run. Optional: defaults to the source checkpoint's run "
        "directory name with any leading 'YYYYmmdd_HHMMSS_' stamp stripped, plus a '_sft' "
        "suffix (e.g. a student run '.../20250101_120000_t1_loco' gives 't1_loco_sft'). A "
        "fresh timestamp is prefixed onto the run directory either way, so repeated "
        "finetunes of one checkpoint never overwrite each other.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Parent directory for this finetune run. Optional: defaults to the absolute path "
        "'<checkpoint's run directory>/finetune'. The final run directory is "
        "'<output-dir>/<timestamp>_<run-name>'. Note that an explicitly passed *relative* "
        "path is resolved against the checkpoint's directory, not the cwd.",
    )

    # Data-composition flags: which steps make up the batch, not how training is
    # tuned -- see _FinetuneDefaults docstring's "Trim-steps exception".
    # (There are no NPZ dual-target buffer-mixing flags -- see the docstring's "NPZ
    # buffer fields note" -- since this release's method is small-scale
    # (target-H5-only) finetuning and has no large-scale buffer-mixing stage.)
    parser.add_argument(
        "--trim-head-steps",
        type=int,
        default=_FinetuneDefaults.trim_head_steps,
        help="Steps trimmed from the start of every H5 episode before windowing."
        + _NUMERIC_DOMAINS["trim_head_steps"].help_suffix(),
    )
    parser.add_argument(
        "--trim-tail-steps",
        type=int,
        default=_FinetuneDefaults.trim_tail_steps,
        help="Steps trimmed from the end of every H5 episode before windowing. Same caveat as --trim-head-steps."
        + _NUMERIC_DOMAINS["trim_tail_steps"].help_suffix(),
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B run mode. Defaults to 'online' (unchanged). Pass 'disabled' to run without any "
        "W&B account/network access, or 'offline' to log locally without syncing.",
    )
    _add_override_arguments(parser)
    return parser


def main() -> dict[str, Any]:
    """Run the IDM LoRA finetune end to end.

    Returns the training-data provenance record -- the same dict written to
    `<run_dir>/training_data_provenance.json` and embedded in `summary.json`.
    Console entry points discard it; tests assert on it.
    """
    args = _build_arg_parser().parse_args()
    # `D` is `_FinetuneDefaults` itself when no override flag was passed, and a thin
    # subclass carrying only the passed values otherwise -- so every `D.<field>` read
    # below picks the CLI value up, and a default run is bit-identical to before.
    D = resolve_finetune_defaults(args)
    print(
        "[CLI] finetune overrides: "
        + (
            ", ".join(
                f"{name}={getattr(args, name)!r}"
                for name, _t, _c, _h in _OVERRIDABLE_DEFAULTS
                if getattr(args, name, None) is not None
            )
            or "(none -- all release defaults)"
        ),
        flush=True,
    )

    if D.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if int(D.max_train_steps) <= 0:
        raise ValueError("max-train-steps must be positive (finetuning is step-driven only)")
    if int(args.trim_head_steps) < 0 or int(args.trim_tail_steps) < 0:
        raise ValueError(
            f"trim-head-steps/trim-tail-steps must be non-negative, got {args.trim_head_steps}/{args.trim_tail_steps}"
        )
    if str(args.wandb_mode).lower() not in {"online", "offline", "disabled"}:
        raise ValueError("wandb-mode must be one of: online, offline, disabled")
    if not bool(D.save_checkpoints) and not bool(D.export_onnx) and not bool(D.save_adapter_only):
        raise ValueError(
            "At least one of --save-checkpoints, --export-onnx, or --save-adapter-only must be enabled "
            "so the finetune run writes artifacts."
        )

    step_milestones = _parse_step_milestones(D.save_step_milestones)
    max_train_steps = int(D.max_train_steps)
    save_ckpt_early = bool(D.save_checkpoints)
    export_onnx_early = bool(D.export_onnx)
    # The milestone save block further down (see `saved_step_ms`/
    # `export_onnx_each_epoch`) only ever writes a checkpoint or ONNX file when
    # `save_checkpoints` is on, or when `export_onnx` AND `export_onnx_every_epoch`
    # are both on -- `export_onnx=True` alone does not gate milestone writes (only
    # the final-model export at the end of the run). So this precheck must test both
    # flags: testing `export_onnx_early` alone would accept
    # `export_onnx=True, export_onnx_every_epoch=False` as "milestones can be
    # written" and then produce milestone records with `checkpoint=None, onnx=None`
    # -- see the milestone-record assertion below and in test_finetune_idm_lora.py.
    export_onnx_each_epoch_early = export_onnx_early and bool(D.export_onnx_every_epoch)
    if step_milestones is not None:
        if not save_ckpt_early and not export_onnx_each_epoch_early:
            raise ValueError(
                "When --save-step-milestones is set, enable --save-checkpoints and/or "
                "--export-onnx (with --export-onnx-every-epoch true) so milestone artifacts can be written."
            )

    np.random.seed(int(D.seed))
    torch.manual_seed(int(D.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(D.seed))

    requested_device = torch.device(D.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = requested_device

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    print(f"[init] loading checkpoint: {checkpoint_path}", flush=True)
    payload = safe_load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(payload)}")

    train_cfg = payload.get("cfg", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    # Routed through the shared FinetuneRecipe resolver so main() and
    # tools/bitexact/sft_probe.py can't drift apart -- see
    # resolve_finetune_recipe()'s docstring.
    recipe = resolve_finetune_recipe(train_cfg=train_cfg, defaults=D)
    resolved_lr, lr_source = recipe.lr, recipe.lr_source
    resolved_weight_decay, weight_decay_source = recipe.weight_decay, recipe.weight_decay_source
    resolved_grad_clip, grad_clip_source = recipe.grad_clip, recipe.grad_clip_source
    # Resolved inside resolve_finetune_recipe() (`_resolve_bool_hparam(D ->
    # train_cfg -> False)`) so the value that drives `payload["cfg"]`/the ONNX
    # metadata is the same one `recipe_hash` covers -- it cannot be regressed
    # without moving the hash.
    resolved_idm_use_current_command_for_history = recipe.idm_use_current_command_for_history
    idm_use_current_command_for_history_source = recipe.idm_use_current_command_for_history_source
    payload_cfg = copy.deepcopy(train_cfg)
    payload_cfg.pop("use_current_command_for_history", None)
    payload_cfg["idm_use_current_command_for_history"] = bool(resolved_idm_use_current_command_for_history)
    payload["cfg"] = payload_cfg
    train_cfg = payload_cfg

    print("[init] building model from checkpoint...", flush=True)
    model, meta = _build_model_from_checkpoint(payload, device=device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[init] model ready: {n_params:.2f}M params on {device}", flush=True)
    resolved_finetune_method = recipe.finetune_method
    finetuned_model_stem = (
        "model_finetuned_idm_lora" if resolved_finetune_method == "lora" else "model_finetuned_idm_full"
    )
    print(
        "[HParams] "
        f"lr={resolved_lr} ({lr_source}) "
        f"weight_decay={resolved_weight_decay} ({weight_decay_source}) "
        f"grad_clip={resolved_grad_clip} ({grad_clip_source}) "
        f"idm_use_current_command_for_history={resolved_idm_use_current_command_for_history} "
        f"({idm_use_current_command_for_history_source})"
    )

    model_pred_horizon = int(meta["pred_horizon"])
    pred_horizon_k = int(D.pred_horizon_k) if D.pred_horizon_k is not None else model_pred_horizon
    if pred_horizon_k <= 0:
        raise ValueError("pred-horizon-k must be positive")
    if pred_horizon_k > model_pred_horizon:
        raise ValueError(f"pred-horizon-k ({pred_horizon_k}) cannot exceed model pred_horizon ({model_pred_horizon})")

    normalization_cfg = payload.get("normalization")
    io_norm_enabled = bool(normalization_cfg.get("io_enabled", False)) if isinstance(normalization_cfg, dict) else False
    obs_stats = _extract_norm_stats(
        payload,
        key="obs_norm_stats",
        mean_key="obs_mean",
        std_key="obs_std",
        dim=int(meta["obs_dim"]),
        required=io_norm_enabled,
        device=device,
    )
    action_stats = _extract_norm_stats(
        payload,
        key="action_norm_stats",
        mean_key="action_mean",
        std_key="action_std",
        dim=int(meta["act_dim"]),
        required=io_norm_enabled,
        device=device,
    )
    command_stats = _extract_norm_stats(
        payload,
        key="command_norm_stats",
        mean_key="command_mean",
        std_key="command_std",
        dim=int(meta["cmd_dim"]),
        required=io_norm_enabled,
        device=device,
    )

    obs_stats_for_norm = None
    if obs_stats is not None:
        obs_stats_for_norm = {"mean": obs_stats["mean"], "std": obs_stats["std"]}
    action_stats_for_norm = None
    if action_stats is not None:
        action_stats_for_norm = {"mean": action_stats["mean"], "std": action_stats["std"]}
    command_stats_for_norm = None
    if command_stats is not None:
        command_stats_for_norm = {"mean": command_stats["mean"], "std": command_stats["std"]}
    compact_payload = payload.get("compact_obs", {}) if isinstance(payload.get("compact_obs"), dict) else {}
    raw_obs_preprocess: RawObsPreprocess | None = None
    term_order_raw = compact_payload.get("term_order")
    term_scale_raw = compact_payload.get("term_scale")
    if isinstance(term_order_raw, list) and isinstance(term_scale_raw, dict):
        raw_obs_preprocess = RawObsPreprocess(
            term_order=tuple(str(v) for v in term_order_raw),
            term_scale={str(k): float(v) for k, v in term_scale_raw.items()},
            act_dim=int(meta["act_dim"]),
            obs_dim=int(meta["obs_dim"]),
        )

    all_dataset_paths = list(args.target_datasets)
    if not all_dataset_paths:
        raise SystemExit("No dataset paths provided via --target-datasets")
    dataset_paths = [Path(p).expanduser() for p in all_dataset_paths]
    print(f"[data] loading {len(dataset_paths)} H5 dataset file(s)...", flush=True)
    trajectories, load_stats = load_trajectories_from_h5(
        dataset_paths,
        obs_key=str(D.obs_key),
        command_key=str(D.command_key),
        obs_dim=int(meta["obs_dim"]),
        act_dim=int(meta["act_dim"]),
        cmd_dim=int(meta["cmd_dim"]),
        raw_obs_preprocess=raw_obs_preprocess,
        pred_horizon_k=model_pred_horizon,
        trim_head_steps=int(args.trim_head_steps),
        trim_tail_steps=int(args.trim_tail_steps),
        require_command_key=bool(resolved_idm_use_current_command_for_history),
    )
    print(
        f"[data] H5 loading done: {load_stats['num_files']} files, "
        f"{load_stats['num_episodes']} episodes → {len(trajectories)} trajectories "
        f"(skipped short={load_stats['num_skipped_short']} empty={load_stats['num_skipped_trimmed_empty']})",
        flush=True,
    )

    train_dataset = TrajectoryWindowDataset(
        trajectories,
        history_len=int(meta["history_len"]),
        pred_horizon_k=model_pred_horizon,
    )

    # ── Auditable training-data provenance ──────────────────────────────────────
    # There is no train/val split (see `_FinetuneDefaults`' "No-train/val-split
    # note"), so there is no second side to be disjoint from and no disjointness to
    # check -- what is worth recording is which trajectories this run actually
    # trained on. The ids are read
    # back off `train_dataset`, the object the DataLoader really iterates, rather than
    # off the `trajectories` list, so anything that mutates the dataset between
    # construction and use is still visible here. Written to
    # `training_data_provenance.json`, embedded in `summary.json`, and returned from
    # `main()`.
    train_trajectory_sources = sorted(set(_trajectory_window_dataset_sources(train_dataset)))
    train_trajectories = len(train_trajectory_sources)

    training_data_provenance: dict[str, Any] = {
        "seed": int(D.seed),
        "num_loaded_trajectories": len(trajectories),
        "all_trajectory_sources": [str(traj.source) for traj in trajectories],
        "train_trajectory_sources": list(train_trajectory_sources),
        "num_train_trajectories": int(train_trajectories),
        "num_train_windows": len(train_dataset),
    }

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(D.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=bool(device.type == "cuda"),
        drop_last=False,
    )

    # Dual-target NPZ buffer mixing belongs to a large-scale finetune stage this
    # release does not ship; `_FinetuneDefaults.npz_optimal`/`npz_suboptimal`/
    # `npz_online` are always None on the small-scale, target-H5-only release path,
    # and there is no orchestration branch here that would build a buffer sampler
    # from them (see `_FinetuneDefaults`' "Buffer-mixing orchestration note"), so
    # the loader the training loop consumes is always just the target H5 loader.
    #
    # Nothing about data composition is recorded into summary.json or ONNX
    # metadata, deliberately. Any such field would be structurally constant here
    # -- null NPZ paths, zero budgets, buffer mixing off -- and a consumer cannot
    # tell an always-null field from one that happened to be null on this run: it
    # would read as evidence that buffer mixing was configurable and left off,
    # when no code path can turn it on. Silence is the honest report.
    mixed_train_loader = train_loader

    resolved_idm_loss_target = "idm"
    resolved_anchor_lambda = float(D.idm_anchor_lambda)
    resolved_horizon_gamma = recipe.idm_action_loss_horizon_gamma
    resolved_lora_target_scope = recipe.lora_target_scope
    finetune_stage = D.finetune_stage
    # Single step-driven stage: there is no epoch-driven outer loop and no
    # multi-stage/epoch-count machinery -- see `_FinetuneDefaults`'
    # "Step-driven-only note". `main()` always trains this one "idm" stage for
    # `max_train_steps` steps.
    stage_plan: list[dict[str, Any]] = [
        {
            "name": "idm",
            "label": "IDM finetune",
            "loss_target": "idm",
            "setup_fn": lambda: configure_idm_finetune_method(
                model,
                finetune_method=resolved_finetune_method,
                lora_r=recipe.lora_r,
                lora_alpha=recipe.lora_alpha,
                lora_dropout=recipe.lora_dropout,
                lora_target_scope=str(D.lora_target_scope),
            )[0],
        }
    ]

    # --run-name / --output-dir are optional; when omitted both are derived from the
    # resolved --checkpoint (see _default_run_name_from_checkpoint /
    # _default_output_dir_from_checkpoint). _default_run_name() stays as the last-resort
    # fallback for a checkpoint whose run directory yields no usable token.
    if args.run_name:
        run_name_base = str(args.run_name)
    elif D.wandb_name:
        run_name_base = str(D.wandb_name)
    else:
        run_name_base = _default_run_name_from_checkpoint(checkpoint_path) or _default_run_name(
            resolved_finetune_method
        )
    run_name = _ensure_timestamp_prefixed_run_name(run_name_base)
    output_dir_arg = (
        str(args.output_dir)
        if args.output_dir is not None and str(args.output_dir).strip() != ""
        else _default_output_dir_from_checkpoint(checkpoint_path)
    )
    output_root = _resolve_output_root(output_dir_arg, checkpoint_path=checkpoint_path)
    run_dir = output_root / run_name
    epoch_checkpoint_dir = run_dir / "checkpoints"
    epoch_onnx_dir = run_dir / "onnx"
    save_ckpt = bool(D.save_checkpoints)
    export_onnx = bool(D.export_onnx)
    # Gates ONNX export at each `--save-step-milestones` step; there is no epoch
    # loop, so this only ever fires from the step-driven path -- see
    # `_FinetuneDefaults`' "Step-driven-only note".
    export_onnx_each_epoch = export_onnx and bool(D.export_onnx_every_epoch)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Written before training starts so the audit record survives a crash mid-run,
    # and so a reviewer can check what a run trained on without re-deriving it from
    # the H5 files.
    data_provenance_path = run_dir / "training_data_provenance.json"
    training_data_provenance["run_dir"] = str(run_dir)
    data_provenance_path.write_text(json.dumps(training_data_provenance, indent=2), encoding="utf-8")
    if save_ckpt:
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if export_onnx_each_epoch:
        epoch_onnx_dir.mkdir(parents=True, exist_ok=True)

    export_compact_obs_term_scale: dict[str, float] | None = None
    export_compact_obs_term_noise: dict[str, float] | None = None
    export_compact_obs_add_noise = False
    if export_onnx:
        compact_export_payload = payload.get("compact_obs")
        if not isinstance(compact_export_payload, dict):
            raise RuntimeError("Checkpoint missing required compact_obs payload for ONNX export.")
        term_scale_raw = compact_export_payload.get("term_scale")
        term_noise_raw = compact_export_payload.get("term_noise")
        if not isinstance(term_scale_raw, dict) or not isinstance(term_noise_raw, dict):
            raise RuntimeError("Checkpoint compact_obs payload missing term_scale/term_noise for ONNX export.")
        export_compact_obs_term_scale = {str(k): float(v) for k, v in term_scale_raw.items()}
        export_compact_obs_term_noise = {str(k): float(v) for k, v in term_noise_raw.items()}
        export_compact_obs_add_noise = bool(compact_export_payload.get("add_noise", False))
        if export_compact_obs_add_noise:
            raise ValueError(
                "Exporting deploy ONNX with compact observation noise enabled is not allowed. "
                "Use a checkpoint/config with compact_obs.add_noise=False."
            )

    wandb_config = copy.deepcopy(vars(args))
    wandb_config.update(
        {
            "resolved_device": str(device),
            "resolved_checkpoint": str(checkpoint_path),
            "fdm_enabled": False,
            "obs_dim": int(meta["obs_dim"]),
            "act_dim": int(meta["act_dim"]),
            "cmd_dim": int(meta["cmd_dim"]),
            "history_len": int(meta["history_len"]),
            "model_pred_horizon": int(meta["pred_horizon"]),
            "resolved_pred_horizon_k": int(pred_horizon_k),
            "resolved_max_train_steps": int(max_train_steps),
            "resolved_idm_loss_target": str(resolved_idm_loss_target),
            "resolved_anchor_lambda": float(resolved_anchor_lambda),
            "resolved_horizon_gamma": float(resolved_horizon_gamma),
            "resolved_stage_plan": [
                {
                    "name": str(stage["name"]),
                    "loss_target": str(stage["loss_target"]),
                }
                for stage in stage_plan
            ],
            "resolved_lr": float(resolved_lr),
            "resolved_lr_source": str(lr_source),
            "resolved_weight_decay": float(resolved_weight_decay),
            "resolved_weight_decay_source": str(weight_decay_source),
            "resolved_grad_clip": float(resolved_grad_clip),
            "resolved_grad_clip_source": str(grad_clip_source),
            "command_input_name": "current_command",
            "command_input_semantics": "current_command_broadcast_to_history_tokens",
            "resolved_idm_use_current_command_for_history": bool(resolved_idm_use_current_command_for_history),
            "resolved_idm_use_current_command_for_history_source": str(idm_use_current_command_for_history_source),
        }
    )
    wandb_run = _init_wandb_run(
        enabled=bool(D.use_wandb),
        mode=str(args.wandb_mode).lower(),
        project=str(D.wandb_project),
        entity=(str(D.wandb_entity) if D.wandb_entity else None),
        group=(str(D.wandb_group) if D.wandb_group else None),
        name=(str(D.wandb_name) if D.wandb_name else run_name),
        tags=_parse_csv_list(D.wandb_tags),
        run_dir=run_dir,
        config=wandb_config,
    )

    epoch_logs: list[dict[str, float | int | None]] = []
    epoch_artifacts: list[dict[str, Any]] = []
    replaced_modules_by_stage: dict[str, list[str]] = {}
    trainable_params_by_stage: dict[str, int] = {}
    # Per-stage "did trainable weights actually move" result from the
    # post-loop self-check below -- surfaced into summary.json for auditability.
    stage_weights_changed_by_name: dict[str, bool] = {}
    # Per-stage AdamW step counter verified against the loop's own
    # step tally by the post-loop self-check -- surfaced into summary.json too.
    stage_optimizer_steps_by_name: dict[str, int] = {}
    # Trainable parameters AdamW never created state for -- LoRA
    # adapters injected on submodules outside this stage's forward path. Recorded (not
    # raised on); see the self-check's comment for why.
    stage_params_without_optimizer_state_by_name: dict[str, int] = {}
    global_epoch = 0
    for stage in stage_plan:
        stage_name = str(stage["name"])
        stage_loss_target = str(stage["loss_target"])
        replaced_modules_stage = list(stage["setup_fn"]())
        replaced_modules_by_stage[stage_name] = replaced_modules_stage
        stage_trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not stage_trainable_params:
            raise RuntimeError(f"No trainable parameters found for finetune stage '{stage_name}'.")
        trainable_params_by_stage[stage_name] = int(sum(p.numel() for p in stage_trainable_params))
        optimizer = optim.AdamW(stage_trainable_params, lr=resolved_lr, weight_decay=resolved_weight_decay)
        # Pre-training snapshot for the post-loop "did we actually train"
        # self-check below (see `_trainable_param_signature`'s docstring).
        stage_pre_train_signature = _trainable_param_signature(model)

        # ── Step-based IDM training ────────────────────────────────────────
        # Cycles through the DataLoader one full pass at a time.
        # Saves checkpoints / ONNX at each step milestone.
        sorted_step_ms = sorted(step_milestones) if step_milestones else []
        saved_step_ms: set[int] = set()
        total_steps_done = 0
        # Iteration counter + observed-loss finiteness ledger for the
        # in-loop progress invariant below.
        loop_iterations = 0
        observed_train_losses: list[float] = []
        chunk_size = max(1, len(mixed_train_loader))

        try:
            from tqdm.auto import tqdm as _tqdm

            _step_pbar = (
                _tqdm(
                    total=max_train_steps,
                    desc=f"[{stage_name}] training",
                    unit="step",
                    dynamic_ncols=True,
                    leave=True,
                )
                if bool(D.show_progress)
                else None
            )
        except Exception:
            _step_pbar = None

        while total_steps_done < max_train_steps:
            prev_steps = total_steps_done
            remaining = max_train_steps - total_steps_done
            # Cap chunk to the next pending milestone so each milestone is
            # saved at the correct step rather than from a later model.
            _future_ms = [ms for ms in sorted_step_ms if ms > total_steps_done]
            _next_stop = _future_ms[0] if _future_ms else max_train_steps
            batches_this = min(chunk_size, remaining, _next_stop - total_steps_done)
            global_epoch += 1

            train_metrics = _run_epoch(
                model=model,
                loader=mixed_train_loader,
                optimizer=optimizer,
                device=device,
                io_norm_enabled=io_norm_enabled,
                obs_stats=obs_stats_for_norm,
                action_stats=action_stats_for_norm,
                command_stats=command_stats_for_norm,
                pred_horizon_k=pred_horizon_k,
                grad_clip=resolved_grad_clip,
                show_progress=False,
                epoch_desc="",
                max_batches=batches_this,
                loss_target=stage_loss_target,
                anchor_lambda=resolved_anchor_lambda,
                horizon_gamma=resolved_horizon_gamma,
            )
            steps_this = int(train_metrics.get("steps", batches_this))
            total_steps_done += steps_this
            loop_iterations += 1

            # ── In-loop strict-progress invariant ───────────────────────────────
            # The `total_steps_done != max_train_steps` self-check further down sits
            # *after* this `while`, so it is unreachable in exactly the case it most
            # needs to fire: if `_run_epoch` returns `steps=0` (empty/exhausted
            # loader, `max_batches<=0`, a loader that yields nothing), `total_steps_done`
            # never grows, the `while total_steps_done < max_train_steps` condition
            # stays true forever, and the process spins indefinitely instead of
            # failing. Requiring strict progress every iteration turns that hang into
            # an immediate, named crash.
            if steps_this <= 0 or total_steps_done <= prev_steps:
                raise RuntimeError(
                    f"[{stage_name}] training loop made no progress on iteration {loop_iterations}: "
                    f"_run_epoch returned steps={steps_this} (requested max_batches={batches_this}), "
                    f"total_steps_done stalled at {total_steps_done} (was {prev_steps} before this "
                    f"iteration) against max_train_steps={max_train_steps}. Every iteration must "
                    "consume at least one batch; a zero-step iteration would otherwise loop forever "
                    "without ever reaching the post-loop step-count check. Most likely the training "
                    "DataLoader is empty or yields nothing for this batch size."
                )
            step_train_loss = float(train_metrics["loss"])
            if not math.isfinite(step_train_loss):
                raise RuntimeError(
                    f"[{stage_name}] non-finite training loss {step_train_loss!r} at iteration "
                    f"{loop_iterations} (global step {total_steps_done}/{max_train_steps}). "
                    "Refusing to keep training -- every subsequent optimizer step would propagate "
                    "NaN/Inf into the shipped weights, and the post-loop 'weights changed' check "
                    "would still pass because NaN bits differ from the pre-training bits."
                )
            observed_train_losses.append(step_train_loss)

            step_idm_loss_train = (
                float(train_metrics["idm_action_loss"]) if "idm_action_loss" in train_metrics else None
            )
            metric_label = _loss_metric_key(stage_loss_target)
            postfix: dict = {f"train_{metric_label}": f"{float(train_metrics['loss']):.4f}"}
            if _step_pbar is not None:
                _step_pbar.update(steps_this)
                _step_pbar.set_postfix(postfix)
            print(
                f"[Step {total_steps_done:06d}/{max_train_steps}][{stage_name}] "
                f"train_{metric_label}={float(train_metrics['loss']):.6f}",
                flush=True,
            )
            epoch_logs.append(
                {
                    "epoch": global_epoch,
                    "global_step": total_steps_done,
                    "stage": stage_name,
                    "train_loss": float(train_metrics["loss"]),
                    "train_idm_action_loss": step_idm_loss_train,
                }
            )

            if wandb_run is not None:
                _step_wandb_payload: dict[str, Any] = {
                    "epoch": int(global_epoch),
                    "global_step": int(total_steps_done),
                    "stage/name": stage_name,
                    "optimizer/lr": float(optimizer.param_groups[0].get("lr", resolved_lr)),
                    f"train/{stage_loss_target}_loss": float(train_metrics["loss"]),
                }
                if step_idm_loss_train is not None:
                    _step_wandb_payload["train/idm_action_loss"] = step_idm_loss_train
                _step_uw = train_metrics.get("idm_action_loss_unweighted")
                if _step_uw is not None:
                    _step_wandb_payload["train/idm_action_loss_unweighted"] = float(_step_uw)
                _safe_wandb_log(wandb_run, _step_wandb_payload, step=int(total_steps_done))

            # Save step milestone checkpoints / ONNX
            for ms in sorted_step_ms:
                if ms in saved_step_ms:
                    continue
                if not (prev_steps < ms <= total_steps_done):
                    continue
                ms_label = f"step_{ms:06d}"
                ms_ckpt_path: Path | None = None
                ms_onnx_path: Path | None = None
                ms_state: dict[str, Any] | None = None
                ms_payload_snap: dict[str, Any] | None = None
                if save_ckpt or export_onnx_each_epoch:
                    ms_state = _clone_state_dict_to_cpu(build_merged_state_dict(model))
                    ms_payload_snap = copy.deepcopy(payload)
                    ms_payload_snap["model_state_dict"] = ms_state
                    ms_model_cpu, _ = _build_model_from_checkpoint(ms_payload_snap, device=torch.device("cpu"))
                    ms_model_cpu.load_compatible_state_dict(ms_state)
                    ms_payload_snap["planner_state_dict"] = _clone_state_dict_to_cpu(ms_model_cpu.planner_state_dict())
                    ms_payload_snap["idm_state_dict"] = _clone_state_dict_to_cpu(ms_model_cpu.idm_state_dict())
                    ms_cfg_snap = copy.deepcopy(ms_payload_snap.get("cfg") or {})
                    ms_cfg_snap.pop("use_current_command_for_history", None)
                    ms_cfg_snap["idm_use_current_command_for_history"] = bool(
                        resolved_idm_use_current_command_for_history
                    )
                    ms_payload_snap["cfg"] = ms_cfg_snap
                if save_ckpt and ms_state is not None:
                    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    ms_ckpt_path = epoch_checkpoint_dir / f"{ms_label}.pt"
                    torch.save(ms_payload_snap, ms_ckpt_path)
                if export_onnx_each_epoch and ms_payload_snap is not None and ms_state is not None:
                    epoch_onnx_dir.mkdir(parents=True, exist_ok=True)
                    ms_export_model, ms_meta_exp = _build_model_from_checkpoint(ms_payload_snap, device=device)
                    ms_export_model.load_compatible_state_dict(ms_state)
                    ms_export_model.eval()
                    ms_onnx_path = _export_planner_idm_onnx(
                        model=ms_export_model,
                        onnx_output_path=epoch_onnx_dir / f"{ms_label}.onnx",
                        dims=ms_meta_exp,
                        io_normalization=io_norm_enabled,
                        obs_norm_stats=_remap_norm_stats_keys(obs_stats, mean_key="obs_mean", std_key="obs_std"),
                        action_norm_stats=_remap_norm_stats_keys(
                            action_stats, mean_key="action_mean", std_key="action_std"
                        ),
                        command_norm_stats=_remap_norm_stats_keys(
                            command_stats, mean_key="command_mean", std_key="command_std"
                        ),
                        compact_obs_term_scale=export_compact_obs_term_scale,
                        compact_obs_term_noise=export_compact_obs_term_noise,
                        payload=ms_payload_snap,
                    )
                saved_step_ms.add(ms)
                if total_steps_done != ms:
                    print(
                        f"  [WARN] milestone {ms} saved at step {total_steps_done} (off by {total_steps_done - ms})",
                        flush=True,
                    )
                if ms_ckpt_path is None and ms_onnx_path is None:
                    raise RuntimeError(
                        f"Step milestone {ms} produced no checkpoint and no ONNX artifact "
                        "(save_checkpoints/export_onnx_every_epoch misconfigured); the "
                        "front-of-main() precheck should have caught this before training started."
                    )
                epoch_artifacts.append(
                    {
                        "step": ms,
                        "global_step": total_steps_done,
                        "epoch": global_epoch,
                        "stage": stage_name,
                        "checkpoint": (str(ms_ckpt_path) if ms_ckpt_path is not None else None),
                        "onnx": (str(ms_onnx_path) if ms_onnx_path is not None else None),
                        "train_idm_action_loss": step_idm_loss_train,
                    }
                )
                print(
                    f"  [step-milestone {ms}] ckpt={ms_ckpt_path} onnx={ms_onnx_path}",
                    flush=True,
                )
        if _step_pbar is not None:
            _step_pbar.close()

        # ── Hard fail-fast self-checks ──────────────────────────────────────
        # These make "the training loop silently did not train" a crash instead of
        # a quietly-shipped no-op checkpoint. Both conditions are real bugs, not
        # edge cases: `batches_this` is always `min(chunk_size, remaining, ...)`
        # (see above), so `total_steps_done` can never exceed `max_train_steps`,
        # and it can only fall short if the `while` loop body did not execute the
        # expected number of times (e.g. a broken loop condition such as
        # `while False:`). Likewise, real AdamW steps over a nonzero LR and a
        # non-degenerate batch essentially never leave every trainable float32
        # bit unchanged; a match here means the backward/optimizer-step path did
        # not actually touch the parameters `requires_grad` was set on.
        if total_steps_done != max_train_steps:
            raise RuntimeError(
                f"[{stage_name}] training loop completed only {total_steps_done}/{max_train_steps} "
                "requested steps. batches_this is always capped to the steps remaining, so this is "
                "not a normal step-budget stop -- it means the step-driven while-loop in main() did "
                "not execute as intended (e.g. a broken loop condition). Refusing to ship a "
                "checkpoint/ONNX from a run that may not have actually trained."
            )
        stage_post_train_signature = _trainable_param_signature(model)
        stage_weights_changed = stage_post_train_signature != stage_pre_train_signature
        if max_train_steps > 0 and not stage_weights_changed:
            raise RuntimeError(
                f"[{stage_name}] trainable parameters are bit-identical before and after "
                f"{max_train_steps} training steps. The step count matched, but no optimizer step "
                "actually changed any trainable weight -- refusing to ship a checkpoint/ONNX that "
                "did not train."
            )

        # ── "The bits moved" is necessary but nowhere near
        # sufficient ─────────────────────────────────────────────────────────
        # An external review deleted `optimizer.step()`, replaced it with an in-place
        # `param += 1.0`, and the signature comparison above (plus the full `main()`
        # integration test) still passed: any write to a trainable tensor satisfies
        # "post != pre", and NaN/Inf bits differ from the pre-training bits too, so a
        # run that diverged into NaN reads as "successfully trained". The three checks
        # below close that gap by asserting what actually has to be true of a real
        # AdamW run: finite weights out, a real optimizer step counter that matches the
        # number of steps the loop believes it took, and finite losses along the way.
        non_finite_params = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and not bool(torch.isfinite(param.detach()).all())
        ]
        if non_finite_params:
            raise RuntimeError(
                f"[{stage_name}] trainable parameter '{non_finite_params[0]}' contains NaN/Inf after "
                f"{total_steps_done} training steps ({len(non_finite_params)} of "
                f"{len(stage_trainable_params)} trainable tensors are non-finite). Every trainable "
                "parameter must be finite when training ends -- a diverged run still passes the "
                "'weights changed' check above, because NaN bits differ from the pre-training bits. "
                "Refusing to ship a checkpoint/ONNX with non-finite weights."
            )

        observed_step_counts, params_without_optimizer_state = _optimizer_step_counts(optimizer, stage_trainable_params)
        # Note on the "some parameters legitimately have no state" case: LoRA is
        # injected by module *scope* (`lora_target_scope`), and a handful of the
        # injected adapters sit on submodules that this release's single-stage IDM
        # forward (`predict_idm_actions`) never routes through, so they receive no
        # gradient and AdamW never creates state for them. That is a pre-existing
        # property of the scope selection, not a training failure, so it is recorded
        # rather than raised on. What must never happen is *no* trainable parameter
        # having been stepped -- that is the deleted-`optimizer.step()` fault.
        if not observed_step_counts:
            raise RuntimeError(
                f"[{stage_name}] not one of the {len(stage_trainable_params)} trainable parameters "
                f"has AdamW optimizer state after {total_steps_done} steps. AdamW only creates state "
                "for a parameter once optimizer.step() has processed it with a real gradient, so an "
                "entirely empty optimizer state means optimizer.step() never ran -- the trainable "
                "weights must have been written to directly. Refusing to ship a checkpoint/ONNX from "
                "a run whose optimizer never stepped."
            )
        min_step = min(observed_step_counts)
        max_step = max(observed_step_counts)
        if min_step <= 0 or min_step != total_steps_done or max_step != total_steps_done:
            raise RuntimeError(
                f"[{stage_name}] AdamW optimizer step counter does not match the training loop: "
                f"the loop performed {total_steps_done} optimizer steps (one per consumed batch over "
                f"{loop_iterations} loop iterations), but optimizer.state reports step counts in "
                f"[{min_step}, {max_step}] across {len(observed_step_counts)} of "
                f"{len(stage_trainable_params)} trainable parameters. These must be equal and "
                "positive -- a mismatch means the weights changed without AdamW stepping them (e.g. "
                "optimizer.step() removed and replaced by a direct write to the parameter tensors), "
                "which the bit-signature check above cannot distinguish from real training."
            )

        if not observed_train_losses:
            raise RuntimeError(
                f"[{stage_name}] no training loss was observed across {loop_iterations} loop "
                f"iterations / {total_steps_done} steps -- the loop cannot have trained."
            )
        non_finite_losses = [loss for loss in observed_train_losses if not math.isfinite(loss)]
        if non_finite_losses:
            raise RuntimeError(
                f"[{stage_name}] {len(non_finite_losses)} of {len(observed_train_losses)} observed "
                f"training losses were non-finite (first: {non_finite_losses[0]!r}). Refusing to ship "
                "a checkpoint/ONNX from a run whose loss diverged."
            )

        stage_weights_changed_by_name[stage_name] = bool(stage_weights_changed)
        stage_optimizer_steps_by_name[stage_name] = int(total_steps_done)
        stage_params_without_optimizer_state_by_name[stage_name] = int(params_without_optimizer_state)

    # There is exactly one candidate. There is no validation, and so no `best_val`
    # snapshot to copy back into the model here before saving/export: the shipped
    # weights are unambiguously the final training step's -- see
    # `_FinetuneDefaults`' "No-train/val-split note".
    last_merged_state = _clone_state_dict_to_cpu(build_merged_state_dict(model))
    last_model_cpu, _ = _build_model_from_checkpoint(payload, device=torch.device("cpu"))
    last_model_cpu.load_compatible_state_dict(last_merged_state)

    finetune_extra_common = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "dataset_paths": [str(p) for p in dataset_paths],
        "obs_key": str(D.obs_key),
        "command_key": str(D.command_key),
        "trim_head_steps": int(args.trim_head_steps),
        "trim_tail_steps": int(args.trim_tail_steps),
        "pred_horizon_k": pred_horizon_k,
        "max_train_steps": int(max_train_steps),
        "total_steps_done": int(total_steps_done),
        "fdm_enabled": False,
        "idm_loss_target": str(resolved_idm_loss_target),
        "anchor_lambda": float(resolved_anchor_lambda),
        "horizon_gamma": float(resolved_horizon_gamma),
        "stage_plan": [
            {
                "name": str(stage["name"]),
                "loss_target": str(stage["loss_target"]),
            }
            for stage in stage_plan
        ],
        "batch_size": int(D.batch_size),
        "lr": float(resolved_lr),
        "weight_decay": float(resolved_weight_decay),
        "grad_clip": float(resolved_grad_clip),
        "lr_source": str(lr_source),
        "weight_decay_source": str(weight_decay_source),
        "grad_clip_source": str(grad_clip_source),
        "command_input_name": "current_command",
        "command_input_semantics": "current_command_broadcast_to_history_tokens",
        "idm_use_current_command_for_history": bool(resolved_idm_use_current_command_for_history),
        "idm_use_current_command_for_history_source": str(idm_use_current_command_for_history_source),
        "finetune_method": str(resolved_finetune_method),
        "lora_r": int(D.lora_r),
        "lora_alpha": float(D.lora_alpha),
        "lora_dropout": float(D.lora_dropout),
        "lora_target_scope": (str(resolved_lora_target_scope) if resolved_lora_target_scope is not None else None),
        "replaced_modules": [
            module_name
            for stage_name in [str(stage["name"]) for stage in stage_plan]
            for module_name in replaced_modules_by_stage.get(stage_name, [])
        ],
        "replaced_modules_by_stage": copy.deepcopy(replaced_modules_by_stage),
        "trainable_params_by_stage": copy.deepcopy(trainable_params_by_stage),
        "io_norm_enabled": io_norm_enabled,
    }

    def _build_save_payload(
        *,
        merged_model_cpu,
        merged_state_cpu: dict[str, Any],
        selected_epoch: int | None,
    ) -> dict[str, Any]:
        save_payload = copy.deepcopy(payload)
        save_payload["model_state_dict"] = _clone_state_dict_to_cpu(merged_state_cpu)
        save_payload["planner_state_dict"] = _clone_state_dict_to_cpu(merged_model_cpu.planner_state_dict())
        save_payload["idm_state_dict"] = _clone_state_dict_to_cpu(merged_model_cpu.idm_state_dict())
        cfg_payload = save_payload.get("cfg")
        cfg_payload = copy.deepcopy(cfg_payload) if isinstance(cfg_payload, dict) else {}
        cfg_payload.pop("use_current_command_for_history", None)
        cfg_payload["idm_use_current_command_for_history"] = bool(resolved_idm_use_current_command_for_history)
        save_payload["cfg"] = cfg_payload

        existing_extra = save_payload.get("extra")
        extra_payload = existing_extra if isinstance(existing_extra, dict) else {}
        extra_payload = copy.deepcopy(extra_payload)
        finetune_extra = copy.deepcopy(finetune_extra_common)
        # Kept (rather than dropped as a constant) so a consumer reading a checkpoint
        # in isolation can still see *which* training step produced these weights.
        finetune_extra["selected_checkpoint"] = "last"
        finetune_extra["selected_epoch"] = int(selected_epoch) if selected_epoch is not None else None
        extra_payload["finetune_idm_lora"] = finetune_extra
        save_payload["extra"] = extra_payload
        return save_payload

    last_ckpt_path: Path | None = None

    last_payload = _build_save_payload(
        merged_model_cpu=last_model_cpu,
        merged_state_cpu=last_merged_state,
        selected_epoch=int(global_epoch),
    )
    if save_ckpt:
        last_ckpt_path = run_dir / f"{finetuned_model_stem}.pt"
        torch.save(last_payload, last_ckpt_path)

    onnx_last_path: Path | None = None
    if export_onnx:
        compact_payload = last_payload.get("compact_obs")
        if not isinstance(compact_payload, dict):
            raise RuntimeError("Checkpoint missing required compact_obs payload for ONNX export.")
        term_scale_raw = compact_payload.get("term_scale")
        term_noise_raw = compact_payload.get("term_noise")
        if not isinstance(term_scale_raw, dict) or not isinstance(term_noise_raw, dict):
            raise RuntimeError("Checkpoint compact_obs payload missing term_scale/term_noise for ONNX export.")
        compact_obs_term_scale = {str(k): float(v) for k, v in term_scale_raw.items()}
        compact_obs_term_noise = {str(k): float(v) for k, v in term_noise_raw.items()}
        compact_obs_add_noise = bool(compact_payload.get("add_noise", False))
        if compact_obs_add_noise:
            raise ValueError(
                "Exporting deploy ONNX with compact observation noise enabled is not allowed. "
                "Use a checkpoint/config with compact_obs.add_noise=False."
            )

        onnx_last_target = (
            Path(D.onnx_output_path).expanduser() if D.onnx_output_path else run_dir / "planner_idm_policy.onnx"
        )
        export_obs_stats = _remap_norm_stats_keys(obs_stats, mean_key="obs_mean", std_key="obs_std")
        export_action_stats = _remap_norm_stats_keys(
            action_stats,
            mean_key="action_mean",
            std_key="action_std",
        )
        export_command_stats = _remap_norm_stats_keys(
            command_stats,
            mean_key="command_mean",
            std_key="command_std",
        )

        export_last_model, export_last_meta = _build_model_from_checkpoint(last_payload, device=device)
        export_last_model.load_compatible_state_dict(last_merged_state)
        export_last_model.eval()
        onnx_last_path = _export_planner_idm_onnx(
            model=export_last_model,
            onnx_output_path=onnx_last_target,
            dims=export_last_meta,
            io_normalization=io_norm_enabled,
            obs_norm_stats=export_obs_stats,
            action_norm_stats=export_action_stats,
            command_norm_stats=export_command_stats,
            compact_obs_term_scale=compact_obs_term_scale,
            compact_obs_term_noise=compact_obs_term_noise,
            payload=last_payload,
        )

    adapter_path = None
    if bool(D.save_adapter_only):
        flattened_replaced_modules = [
            module_name
            for stage_name in [str(stage["name"]) for stage in stage_plan]
            for module_name in replaced_modules_by_stage.get(stage_name, [])
        ]
        adapter_payload = {
            "checkpoint": str(checkpoint_path),
            "finetune_method": str(resolved_finetune_method),
            "fdm_enabled": False,
            "max_train_steps": int(max_train_steps),
            "total_steps_done": int(total_steps_done),
            "obs_dim": int(meta["obs_dim"]),
            "act_dim": int(meta["act_dim"]),
            "cmd_dim": int(meta["cmd_dim"]),
            "history_len": int(meta["history_len"]),
            "pred_horizon": int(meta["pred_horizon"]),
            "pred_horizon_k": int(pred_horizon_k),
            "replaced_modules": flattened_replaced_modules,
            "replaced_modules_by_stage": copy.deepcopy(replaced_modules_by_stage),
        }
        if resolved_finetune_method == "lora":
            adapter_payload.update(
                {
                    "lora_r": int(D.lora_r),
                    "lora_alpha": float(D.lora_alpha),
                    "lora_dropout": float(D.lora_dropout),
                    "lora_target_scope": str(resolved_lora_target_scope),
                    "adapter_state_dict": extract_lora_adapter_state_dict(model),
                }
            )
            adapter_path = run_dir / "idm_lora_adapters.pt"
        else:
            adapter_payload["trainable_state_dict"] = extract_trainable_parameter_state_dict(model)
            adapter_path = run_dir / "idm_full_trainables.pt"
        torch.save(adapter_payload, adapter_path)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "dataset_paths": [str(p) for p in dataset_paths],
        "load_stats": load_stats,
        "obs_key": str(D.obs_key),
        "command_key": str(D.command_key),
        "trim_head_steps": int(args.trim_head_steps),
        "trim_tail_steps": int(args.trim_tail_steps),
        "device": str(device),
        "io_norm_enabled": io_norm_enabled,
        "fdm_enabled": False,
        "history_len": int(meta["history_len"]),
        "model_pred_horizon": int(meta["pred_horizon"]),
        "pred_horizon_k": int(pred_horizon_k),
        "train_trajectories": int(train_trajectories),
        "train_samples": (len(train_dataset) if train_dataset is not None else 0),
        "replaced_modules": [
            module_name
            for stage_name in [str(stage["name"]) for stage in stage_plan]
            for module_name in replaced_modules_by_stage.get(stage_name, [])
        ],
        "replaced_modules_by_stage": copy.deepcopy(replaced_modules_by_stage),
        "num_trainable_params": (int(sum(trainable_params_by_stage.values())) if trainable_params_by_stage else 0),
        "num_trainable_params_by_stage": copy.deepcopy(trainable_params_by_stage),
        # Result of the post-loop "did trainable weights actually move"
        # self-check (see `_trainable_param_signature`'s docstring). `main()` raises
        # before reaching here if any stage's weights failed to change or the step
        # count fell short, so every key present is necessarily `True` -- these
        # fields exist so a downstream consumer (or a test) can assert on the fact
        # directly instead of only inferring it from the absence of a crash.
        "weights_changed_during_training_by_stage": copy.deepcopy(stage_weights_changed_by_name),
        "weights_changed_during_training": bool(all(stage_weights_changed_by_name.values()))
        if stage_weights_changed_by_name
        else False,
        # AdamW's own per-parameter step counter, verified equal to
        # `total_steps_done` by the post-loop self-check. Recorded so "the optimizer
        # really stepped" is auditable from summary.json, not only inferable from the
        # absence of a crash.
        "optimizer_steps_by_stage": copy.deepcopy(stage_optimizer_steps_by_name),
        "trainable_params_without_optimizer_state_by_stage": copy.deepcopy(
            stage_params_without_optimizer_state_by_name
        ),
        "max_train_steps": int(max_train_steps),
        "total_steps_done": int(total_steps_done),
        "idm_loss_target": str(resolved_idm_loss_target),
        "anchor_lambda": float(resolved_anchor_lambda),
        "horizon_gamma": float(resolved_horizon_gamma),
        "stage_plan": [
            {
                "name": str(stage["name"]),
                "loss_target": str(stage["loss_target"]),
            }
            for stage in stage_plan
        ],
        "lr": float(resolved_lr),
        "weight_decay": float(resolved_weight_decay),
        "grad_clip": float(resolved_grad_clip),
        "lr_source": str(lr_source),
        "weight_decay_source": str(weight_decay_source),
        "grad_clip_source": str(grad_clip_source),
        "command_input_name": "current_command",
        "command_input_semantics": "current_command_broadcast_to_history_tokens",
        "idm_use_current_command_for_history": bool(resolved_idm_use_current_command_for_history),
        "idm_use_current_command_for_history_source": str(idm_use_current_command_for_history_source),
        "finetune_method": str(resolved_finetune_method),
        "lora_target_scope": (str(resolved_lora_target_scope) if resolved_lora_target_scope is not None else None),
        # With no validation there is no best-vs-last choice -- the single
        # checkpoint and the single ONNX below are the final training step's
        # weights. Recorded explicitly so a consumer does not have to infer it
        # from the absence of a `best_*` key.
        "selected_checkpoint": "last",
        "last_checkpoint": (str(last_ckpt_path) if last_ckpt_path is not None else None),
        "merged_checkpoint": (str(last_ckpt_path) if last_ckpt_path is not None else None),
        "last_onnx_output_path": (str(onnx_last_path) if onnx_last_path is not None else None),
        "onnx_output_path": (str(onnx_last_path) if onnx_last_path is not None else None),
        "adapter_checkpoint": (str(adapter_path) if adapter_path is not None else None),
        "epoch_artifacts": epoch_artifacts,
        "epoch_logs": epoch_logs,
        "notes": (
            "Planner is frozen. "
            + (
                "LoRA adapters are injected only into the IDM backbone and supervised with future action MSE."
                if resolved_finetune_method == "lora"
                else "Full finetune unfreezes all IDM parameters and supervises future action MSE."
            )
        ),
    }

    summary["training_data_provenance_path"] = str(data_provenance_path)
    summary["training_data_provenance"] = copy.deepcopy(training_data_provenance)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run is not None:
        _safe_wandb_log(
            wandb_run,
            {
                "final/train_idm_action_loss": (
                    float(epoch_logs[-1]["train_idm_action_loss"])
                    if epoch_logs and epoch_logs[-1].get("train_idm_action_loss") is not None
                    else float("nan")
                ),
                "final/train_idm_planner_fdm_obs_loss": (
                    float(epoch_logs[-1]["train_idm_planner_fdm_obs_loss"])
                    if epoch_logs and epoch_logs[-1].get("train_idm_planner_fdm_obs_loss") is not None
                    else float("nan")
                ),
                "final/train_fdm_obs_loss": (
                    float(epoch_logs[-1]["train_fdm_obs_loss"])
                    if epoch_logs and epoch_logs[-1].get("train_fdm_obs_loss") is not None
                    else float("nan")
                ),
                "final/train_planner_cycle_loss": (
                    float(epoch_logs[-1]["train_planner_cycle_loss"])
                    if epoch_logs and epoch_logs[-1].get("train_planner_cycle_loss") is not None
                    else float("nan")
                ),
                "final/train_action_anchor_loss": (
                    float(epoch_logs[-1]["train_action_anchor_loss"])
                    if epoch_logs and epoch_logs[-1].get("train_action_anchor_loss") is not None
                    else float("nan")
                ),
            },
            step=int(total_steps_done),
        )
        wandb_run.finish()
    # See finetune_obs_lora.py: avoid huge stdout through W&B (BlockingIOError on large writes).
    print(f"Finetune summary written to {summary_path}", flush=True)
    print(f"Training-data provenance written to {data_provenance_path}", flush=True)
    # Returned as well as written, so an in-process caller (notably
    # the end-to-end test) can assert on the exact trajectory set this run trained on
    # without re-reading and re-parsing the JSON. Console entry points ignore it.
    return training_data_provenance


if __name__ == "__main__":
    main()
