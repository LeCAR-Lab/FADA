from __future__ import annotations

from holosoma.utils.data_collector import PLANNED_STEPS_TOLERANCE
import argparse
import copy
import dataclasses
import json
import threading
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.fada.common.backbone import (
    HistoryPolicy,
    MLPPolicy,
    TransformerPolicy,
    infer_backbone_type_from_cfg,
)
from holosoma.fada.common.norm_stats import DEFAULT_NORM_EPS, validate_norm_stats
from holosoma.utils.safe_torch_import import F, nn, optim, torch


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool value: {value}")


def _load_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("h5py is required for trajectory finetuning.") from exc
    return h5py


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resolve_output_root(output_dir_arg: str, *, checkpoint_path: Path) -> Path:
    token = str(output_dir_arg).strip()
    if token == "" or token == "checkpoint_dir":
        return checkpoint_path.parent
    if token.startswith("checkpoint_dir/"):
        suffix = token[len("checkpoint_dir/") :]
        return checkpoint_path.parent / suffix
    if token.startswith("checkpoint_dir\\"):
        suffix = token[len("checkpoint_dir\\") :]
        return checkpoint_path.parent / suffix

    path = Path(token).expanduser()
    if path.is_absolute():
        return path
    # Relative paths default to checkpoint directory for easier co-location.
    return checkpoint_path.parent / path


def _resolve_training_hparam(
    cli_value: float | None,
    *,
    train_cfg: dict[str, Any],
    key: str,
    fallback: float,
) -> tuple[float, str]:
    if cli_value is not None:
        return float(cli_value), "cli"
    if key in train_cfg and train_cfg[key] is not None:
        return float(train_cfg[key]), "checkpoint"
    return float(fallback), "fallback"


def _resolve_bool_hparam(
    cli_value: bool | None,
    *,
    train_cfg: dict[str, Any],
    key: str,
    fallback: bool,
) -> tuple[bool, str]:
    if cli_value is not None:
        return bool(cli_value), "cli"
    if key in train_cfg and train_cfg[key] is not None:
        return bool(train_cfg[key]), "checkpoint"
    return bool(fallback), "fallback"


class _WandbCommGuard:
    """Wraps an initialized wandb Run so a mid-run communication failure inside log()/
    finish() disables further W&B calls for the rest of the process instead of crashing
    the whole training/finetuning job.

    Only `wandb.errors.CommError` (and its subclasses, notably `AuthenticationError`) --
    wandb's exception class for network/auth/service-unavailable failures talking to the
    backend -- is treated as a communication failure and swallowed.

    `wandb.errors.Error` is NOT used here even though CommError derives from it: its other
    direct descendants include `UsageError` (and its subclass `UnsupportedError`), which
    signal a programming/configuration mistake in the calling code.
    `WandbCoreNotAvailableError` is likewise left unswallowed. Every other exception
    propagates.

    Known remaining gaps:

    - A failure inside wandb's background service/sync thread (the separate process/thread
      that actually streams data to the backend) can be recorded only in wandb's own internal
      logs and never raised back into this thread at all -- there is nothing for `log()`/
      `finish()` to catch in that case, guard or not.
    - Offline mode (`wandb_mode="offline"`) still writes run files to local disk; a full disk
      (ENOSPC) there is a local filesystem error, not a `CommError` (no network is even
      involved), so it is not guaranteed to surface as one of the exception types this guard
      swallows or lets through in a predictable way.

    `finish()`'s wait is bounded below: the pinned wandb version (0.22.0) has no
    `Settings.finish_timeout`/`finish_timeout_raises` knob and no `Run.finish(timeout=...)`
    parameter, so `finish()` runs on a background thread with a join timeout.
    """

    # Seconds to wait for `Run.finish()`; its upload-completion wait has no built-in
    # timeout in the pinned wandb version.
    _FINISH_TIMEOUT_SECONDS = 300.0

    def __init__(self, run: Any, wandb_module: Any) -> None:
        self._run = run
        self._wandb_module = wandb_module
        self._disabled = False

    def __getattr__(self, name: str) -> Any:
        # Forwards read-only access (e.g. `.summary`) straight to the real Run.
        return getattr(self._run, name)

    def log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        if self._disabled:
            return
        try:
            if step is None:
                self._run.log(payload)
            else:
                self._run.log(payload, step=step)
        except self._wandb_module.errors.CommError as exc:
            self._disabled = True
            print(
                f"[W&B] log() failed ({exc}); disabling further W&B logging for the rest of "
                "this run (training/finetuning continues).",
                flush=True,
            )

    def finish(self, *args: Any, **kwargs: Any) -> None:
        if self._disabled:
            return

        outcome: dict[str, BaseException] = {}

        def _run_finish() -> None:
            try:
                self._run.finish(*args, **kwargs)
            except BaseException as exc:
                outcome["exc"] = exc

        # Daemon thread with a bounded join(), so a `finish()` call that never returns
        # (e.g. stuck waiting on a stalled upload) cannot hang the process.
        thread = threading.Thread(target=_run_finish, daemon=True)
        thread.start()
        thread.join(timeout=self._FINISH_TIMEOUT_SECONDS)

        if thread.is_alive():
            self._disabled = True
            print(
                f"[W&B] finish() did not complete within {self._FINISH_TIMEOUT_SECONDS:.0f}s "
                "(likely waiting on a slow/stalled upload); continuing without waiting further. "
                "The W&B run summary may be incomplete.",
                flush=True,
            )
            return

        exc = outcome.get("exc")
        if exc is None:
            return
        if isinstance(exc, self._wandb_module.errors.CommError):
            self._disabled = True
            print(f"[W&B] finish() failed ({exc}); the W&B run summary may be incomplete.", flush=True)
            return
        raise exc


def _wandb_credentials_preflight(mode: str) -> str:
    """Local, no-network check for whether W&B credentials are configured.

    `wandb.init()` raises `wandb.errors.UsageError` both for a missing-credentials
    environment ("api_key not configured (no-tty)") and for API misuse (e.g. an invalid
    project name, which raises `UsageError: Invalid project name ... exceeded 128
    characters` even with no credentials at all), so the exception class alone does not
    distinguish them.

    Credentials are therefore checked here, before `wandb.init()` is called, letting that
    call's own try/except narrow to `CommError`; a `UsageError` raised at that point is
    API misuse.

    This mirrors the exact first branch of what `wandb.init()` does internally
    (`wandb.sdk.wandb_login._login()` -> `wandb.sdk.lib.apikey.api_key()`): check the wandb
    singleton settings, the `WANDB_API_KEY` env var, and `.netrc`, all without any network
    call or side effect (this is *not* `wandb.login()`, which can write to `.netrc` and, in
    an interactive tty, prompt).

    Returns:
        "configured": credentials are configured, or `mode` doesn't require them (offline
                      writes local files without auth; disabled doesn't touch wandb at all)
                      -- callers should proceed to `wandb.init()` normally.
        "missing":    `mode == "online"` and no credentials are configured -- the exact
                      condition under which `wandb.init()` would otherwise raise
                      `UsageError: api_key not configured (no-tty)`. Callers should degrade
                      gracefully without even calling `wandb.init()`.
        "unknown":    the check itself could not run (e.g. a future wandb release
                      removed/renamed `wandb.sdk.lib.apikey`, which is not public API).
                      Callers should fall back to also catching UsageError around
                      `wandb.init()`, so a credentials-less environment still degrades
                      gracefully.
    """
    if mode != "online":
        return "configured"
    try:
        from wandb.sdk.lib import apikey as wandb_apikey  # type: ignore
    except ImportError:
        return "unknown"
    try:
        configured = bool(wandb_apikey.api_key())
    except Exception:
        return "unknown"
    return "configured" if configured else "missing"


def _missing_wandb_credentials_message(*, mode: str, project: str) -> str:
    return (
        f"[W&B] wandb.init() skipped (mode='{mode}', project='{project}'): W&B credentials "
        "are not configured (no WANDB_API_KEY, no .netrc entry, and no interactive terminal "
        "to log in from). Continuing without W&B logging. To avoid this, either run "
        "`wandb login` (or set WANDB_API_KEY), or pass --wandb-mode disabled to skip W&B "
        "entirely (or --wandb-mode offline to keep W&B running locally, writing run files "
        "to disk without any network calls)."
    )


def _init_wandb_run(
    *,
    enabled: bool,
    mode: str,
    project: str,
    entity: str | None,
    group: str | None,
    name: str,
    tags: list[str],
    run_dir: Path,
    config: dict[str, Any],
) -> Any | None:
    if not enabled:
        return None
    if str(mode).lower() == "disabled":
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "W&B logging is on (the default) but the `wandb` package is not installed. "
            "Either install it (`pip install wandb`, or re-run scripts/setup_isaacsim.sh, "
            "which pins it), or pass `--wandb-mode disabled` to finetune without W&B."
        ) from exc

    wandb_kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "mode": mode,
        "config": config,
        "dir": str(run_dir),
        "reinit": True,
    }
    if entity:
        wandb_kwargs["entity"] = entity
    if group:
        wandb_kwargs["group"] = group
    if tags:
        wandb_kwargs["tags"] = tags

    # Check credentials before calling wandb.init(), so the try/except below can narrow
    # to CommError only. See _wandb_credentials_preflight.
    preflight = _wandb_credentials_preflight(mode)
    if preflight == "missing":
        print(_missing_wandb_credentials_message(mode=mode, project=project))
        return None

    try:
        run = wandb.init(**wandb_kwargs)
    except wandb.errors.CommError as exc:
        print(
            "[W&B] wandb.init() failed "
            f"(mode='{mode}', project='{project}'): {exc}. "
            "Continuing without W&B logging. To avoid this, either fix your W&B "
            "credentials/network access, or pass --wandb-mode disabled to skip W&B "
            "entirely (or --wandb-mode offline to keep W&B running locally, writing run "
            "files to disk without any network calls)."
        )
        return None
    except wandb.errors.UsageError as exc:
        if preflight != "unknown":
            # The preflight confirmed credentials are present, so this UsageError is API
            # misuse (bad/mutually-exclusive kwargs).
            raise
        # The preflight could not determine credential state, so UsageError is also
        # treated as a possible missing-credentials signal.
        print(
            "[W&B] wandb.init() failed "
            f"(mode='{mode}', project='{project}'): {exc}. "
            "Continuing without W&B logging. To avoid this, either fix your W&B "
            "credentials/network access, or pass --wandb-mode disabled to skip W&B "
            "entirely (or --wandb-mode offline to keep W&B running locally, writing run "
            "files to disk without any network calls)."
        )
        return None
    return _WandbCommGuard(run, wandb)


def _safe_wandb_log(wandb_run: Any | None, payload: dict[str, Any], *, step: int | None = None) -> None:
    if wandb_run is None:
        return
    if step is None:
        wandb_run.log(payload)
    else:
        wandb_run.log(payload, step=step)


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Checkpoint must be a .pt file, got: {path}")
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path is not a file or directory: {path}")

    explicit_candidates = [
        path / "model_final.pt",
    ]
    explicit_candidates.extend(sorted(path.glob("student_iter_*.pt")))
    explicit_candidates.extend(sorted(path.glob("model_*.pt")))

    existing = [p for p in explicit_candidates if p.is_file()]
    if not existing:
        raise FileNotFoundError(
            "No supported Transformer checkpoint found under directory "
            f"{path}. Expected model_final.pt or student_iter_*.pt."
        )

    # Preference order:
    # 1) model_final.pt (if present)
    # 2) highest student_iter_XXXX.pt
    # 3) most recently modified fallback among model_*.pt
    model_final = path / "model_final.pt"
    if model_final.is_file():
        return model_final

    student_ckpts: list[tuple[int, float, Path]] = []
    model_fallback: list[tuple[float, Path]] = []
    for candidate in existing:
        stem = candidate.stem
        if stem.startswith("student_iter_"):
            suffix = stem.split("student_iter_", 1)[1]
            if suffix.isdigit():
                student_ckpts.append((int(suffix), candidate.stat().st_mtime, candidate))
                continue
        model_fallback.append((candidate.stat().st_mtime, candidate))

    if student_ckpts:
        student_ckpts.sort(key=lambda x: (x[0], x[1]))
        return student_ckpts[-1][2]

    model_fallback.sort(key=lambda x: x[0])
    return model_fallback[-1][1]


def _extract_norm_stats(
    payload: dict[str, Any],
    *,
    key: str,
    mean_key: str,
    std_key: str,
    dim: int,
    required: bool,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    raw_stats = payload.get(key)
    if raw_stats is None:
        if required:
            raise RuntimeError(f"io_normalization=True requires checkpoint field '{key}', but it is missing.")
        return None
    if not isinstance(raw_stats, dict):
        raise RuntimeError(f"Invalid {key} payload type: {type(raw_stats)}")
    if mean_key not in raw_stats or std_key not in raw_stats:
        if required:
            raise RuntimeError(
                f"io_normalization=True requires {key}.{mean_key}/{std_key}, but checkpoint is missing them."
            )
        return None

    mean = torch.as_tensor(raw_stats[mean_key], device=device, dtype=torch.float32).flatten()
    std = torch.as_tensor(raw_stats[std_key], device=device, dtype=torch.float32).flatten()
    if mean.numel() != dim or std.numel() != dim:
        raise RuntimeError(
            f"Checkpoint {key} shape mismatch: mean={tuple(mean.shape)}, std={tuple(std.shape)}, expected=({dim},)"
        )
    eps = float(raw_stats.get("eps", DEFAULT_NORM_EPS))
    # Rejects NaN/inf in mean/std/eps as well as bad shapes.
    validate_norm_stats(mean, std, eps, key=key)
    std = torch.clamp(std, min=eps)
    return {"mean": mean, "std": std}


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear, keeping base weights frozen."""

    def __init__(self, linear: nn.Linear, *, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.A = nn.Linear(self.in_features, self.r, bias=False)
        self.B = nn.Linear(self.r, self.out_features, bias=False)
        # Keep adapter weights on the same device/dtype as the replaced backbone layer.
        target_device = linear.weight.device
        target_dtype = linear.weight.dtype
        self.A.to(device=target_device, dtype=target_dtype)
        self.B.to(device=target_device, dtype=target_dtype)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = self.B(self.A(self.dropout(x))) * self.scaling
        return base + delta

    def merged_weight(self) -> torch.Tensor:
        return self.weight.detach() + self.scaling * (self.B.weight.detach() @ self.A.weight.detach())


class LoRAMultiheadAttention(nn.Module):
    """LoRA wrapper for nn.MultiheadAttention packed QKV projection."""

    def __init__(self, mha: nn.MultiheadAttention, *, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        if not bool(getattr(mha, "_qkv_same_embed_dim", True)):
            raise ValueError("LoRAMultiheadAttention only supports same embed dim for q/k/v projections.")

        self.embed_dim = int(mha.embed_dim)
        self.num_heads = int(mha.num_heads)
        self.dropout = float(mha.dropout)
        self.batch_first = bool(mha.batch_first)
        self.add_zero_attn = bool(mha.add_zero_attn)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)

        self.in_proj_weight = nn.Parameter(mha.in_proj_weight.detach().clone(), requires_grad=False)
        if mha.in_proj_bias is not None:
            self.in_proj_bias = nn.Parameter(mha.in_proj_bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("in_proj_bias", None)

        if mha.bias_k is not None:
            self.bias_k = nn.Parameter(mha.bias_k.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias_k", None)
        if mha.bias_v is not None:
            self.bias_v = nn.Parameter(mha.bias_v.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias_v", None)

        self.out_proj = mha.out_proj
        # QKV LoRA is applied as a fused low-rank delta on packed in_proj_weight so it stays
        # compatible with PyTorch's optimized MultiheadAttention forward path.
        self.A = nn.Linear(self.embed_dim, self.r, bias=False)
        self.B = nn.Linear(self.r, 3 * self.embed_dim, bias=False)
        target_device = mha.in_proj_weight.device
        target_dtype = mha.in_proj_weight.dtype
        self.A.to(device=target_device, dtype=target_dtype)
        self.B.to(device=target_device, dtype=target_dtype)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.B.weight)

    def merged_in_proj_weight(self) -> torch.Tensor:
        return self.in_proj_weight.detach() + self.scaling * (self.B.weight.detach() @ self.A.weight.detach())

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(0, 1)
                else:
                    query, key = (tensor.transpose(0, 1) for tensor in (query, key))
                    value = key
            else:
                query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))

        out_proj_weight, out_proj_bias = _resolve_linear_weight_bias(self.out_proj)
        common_kwargs = dict(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight + self.scaling * (self.B.weight @ self.A.weight),
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
        )
        try:
            attn_output, attn_weights = F.multi_head_attention_forward(
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                **common_kwargs,
            )
        except TypeError:
            try:
                attn_output, attn_weights = F.multi_head_attention_forward(
                    average_attn_weights=average_attn_weights,
                    **common_kwargs,
                )
            except TypeError:
                attn_output, attn_weights = F.multi_head_attention_forward(**common_kwargs)

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_weights


def _resolve_linear_weight_bias(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(module, LoRALinear):
        return module.merged_weight(), module.bias
    if isinstance(module, nn.Linear):
        return module.weight, module.bias
    raise TypeError(f"Unsupported projection module type: {type(module)}")


def _inject_lora_modules(
    module: nn.Module,
    *,
    should_replace,
    r: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
    replaced: list[str] | None = None,
) -> list[str]:
    if replaced is None:
        replaced = []

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and should_replace(full_name):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
            continue
        _inject_lora_modules(
            child,
            should_replace=should_replace,
            r=r,
            alpha=alpha,
            dropout=dropout,
            prefix=full_name,
            replaced=replaced,
        )
    return replaced


def _inject_lora_multihead_attention_modules(
    module: nn.Module,
    *,
    should_replace,
    r: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
    replaced: list[str] | None = None,
) -> list[str]:
    if replaced is None:
        replaced = []

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.MultiheadAttention) and should_replace(full_name):
            setattr(module, name, LoRAMultiheadAttention(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(f"{full_name}.in_proj_weight")
            continue
        _inject_lora_multihead_attention_modules(
            child,
            should_replace=should_replace,
            r=r,
            alpha=alpha,
            dropout=dropout,
            prefix=full_name,
            replaced=replaced,
        )
    return replaced


def _normalize_transformer_lora_target_scope(scope: str) -> str:
    token = str(scope).strip().lower().replace("-", "_")
    if token in {"encoder_only", "encoder", "backbone"}:
        return "encoder_only"
    if token in {"encoder_only_qkv", "encoder_qkv", "backbone_qkv"}:
        return "encoder_only_qkv"
    raise ValueError(
        f"Unsupported transformer LoRA target scope: {scope!r}. Expected one of: encoder_only, encoder_only_qkv."
    )


def _is_transformer_backbone_linear(name: str, *, target_scope: str) -> bool:
    _normalize_transformer_lora_target_scope(target_scope)
    return name.startswith("encoder.")


def _is_transformer_backbone_attention(name: str, *, target_scope: str) -> bool:
    normalized_scope = _normalize_transformer_lora_target_scope(target_scope)
    return normalized_scope == "encoder_only_qkv" and name.startswith("encoder.")


def inject_lora_into_transformer_backbone(
    model: TransformerPolicy,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_scope: str = "encoder_only",
) -> list[str]:
    normalized_scope = _normalize_transformer_lora_target_scope(target_scope)
    for param in model.parameters():
        param.requires_grad = False

    replaced = _inject_lora_modules(
        model,
        should_replace=lambda name: _is_transformer_backbone_linear(name, target_scope=normalized_scope),
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    if normalized_scope == "encoder_only_qkv":
        replaced = _inject_lora_multihead_attention_modules(
            model,
            should_replace=lambda name: _is_transformer_backbone_attention(name, target_scope=normalized_scope),
            r=r,
            alpha=alpha,
            dropout=dropout,
            replaced=replaced,
        )
    if not replaced:
        raise RuntimeError("No backbone linear layers matched for LoRA injection.")

    for name, param in model.named_parameters():
        if name.endswith(".A.weight") or name.endswith(".B.weight"):
            param.requires_grad = True
    return replaced


def _normalize_mlp_lora_target_scope(scope: str) -> str:
    token = str(scope).strip().lower().replace("-", "_")
    if token in {"encoder_only", "backbone", "backbone_only"}:
        return "backbone_only"
    raise ValueError(f"Unsupported MLP LoRA target scope: {scope!r}. Expected one of: backbone_only")


def _normalize_finetune_method(*, backbone_type: str, method: str) -> str:
    normalized_backbone = infer_backbone_type_from_cfg({"backbone_type": backbone_type})
    token = str(method).strip().lower().replace("-", "_")
    if token in {"lora", "obs_lora"}:
        return "lora"
    if token in {"full", "full_finetune", "all"}:
        return "full"
    if token in {"last_layer", "last_linear", "mlp_last_layer"}:
        if normalized_backbone != "mlp":
            raise ValueError(
                f"Unsupported finetune method {method!r} for backbone_type={normalized_backbone!r}. "
                "Currently last_layer is only supported for mlp."
            )
        return "last_layer"
    raise ValueError(f"Unsupported finetune method: {method!r}. Expected one of: lora, full, last_layer.")


def inject_lora_into_mlp_backbone(
    model: MLPPolicy,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_scope: str = "backbone_only",
) -> list[str]:
    _normalize_mlp_lora_target_scope(target_scope)
    for param in model.parameters():
        param.requires_grad = False

    replaced = _inject_lora_modules(
        model.backbone,
        should_replace=lambda _name: True,
        r=r,
        alpha=alpha,
        dropout=dropout,
        prefix="backbone",
    )
    if not replaced:
        raise RuntimeError("No MLP backbone linear layers matched for LoRA injection.")

    for name, param in model.named_parameters():
        if name.endswith(".A.weight") or name.endswith(".B.weight"):
            param.requires_grad = True
    return replaced


def enable_mlp_last_layer_finetune(model: MLPPolicy) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False

    linear_modules = [
        (name, module)
        for name, module in model.backbone.named_modules()
        if name != "" and isinstance(module, nn.Linear)
    ]
    if not linear_modules:
        raise RuntimeError("No MLP backbone linear layers found for last-layer finetuning.")

    last_name, _last_module = linear_modules[-1]
    full_name = f"backbone.{last_name}"
    for name, param in model.named_parameters():
        if name.startswith(f"{full_name}."):
            param.requires_grad = True
    return [full_name]


def enable_full_policy_finetune(model: HistoryPolicy) -> list[str]:
    for param in model.parameters():
        param.requires_grad = True
    return ["model"]


def resolve_lora_target_scope(*, backbone_type: str, target_scope: str) -> str:
    normalized_backbone = infer_backbone_type_from_cfg({"backbone_type": backbone_type})
    if normalized_backbone == "mlp":
        return _normalize_mlp_lora_target_scope(target_scope)
    return _normalize_transformer_lora_target_scope(target_scope)


def inject_lora_into_policy_backbone(
    model: HistoryPolicy,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_scope: str,
) -> tuple[list[str], str]:
    backbone_type = str(getattr(model, "backbone_type", "transformer"))
    resolved_scope = resolve_lora_target_scope(backbone_type=backbone_type, target_scope=target_scope)
    if backbone_type == "mlp":
        replaced = inject_lora_into_mlp_backbone(
            model,
            r=r,
            alpha=alpha,
            dropout=dropout,
            target_scope=resolved_scope,
        )
        return replaced, resolved_scope
    replaced = inject_lora_into_transformer_backbone(
        model,
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_scope=resolved_scope,
    )
    return replaced, resolved_scope


def configure_policy_finetune_method(
    model: HistoryPolicy,
    *,
    finetune_method: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_scope: str,
) -> tuple[list[str], str, str | None]:
    backbone_type = str(getattr(model, "backbone_type", "transformer"))
    resolved_method = _normalize_finetune_method(backbone_type=backbone_type, method=finetune_method)
    if resolved_method == "lora":
        replaced, resolved_scope = inject_lora_into_policy_backbone(
            model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_scope=lora_target_scope,
        )
        return replaced, resolved_method, resolved_scope
    if resolved_method == "full":
        replaced = enable_full_policy_finetune(model)
        return replaced, resolved_method, None
    if backbone_type != "mlp":
        raise RuntimeError(
            f"Finetune method '{resolved_method}' is not implemented for backbone_type={backbone_type!r}."
        )
    replaced = enable_mlp_last_layer_finetune(model)
    return replaced, resolved_method, None


def extract_lora_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    adapters: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name.endswith(".A.weight") or name.endswith(".B.weight"):
            adapters[name] = param.detach().cpu().clone()
    return adapters


def extract_trainable_parameter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable[name] = param.detach().cpu()
    return trainable


def build_merged_state_dict(model_with_lora: nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model_with_lora.state_dict()
    merged: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".A.weight") or key.endswith(".B.weight"):
            continue
        merged[key] = value.detach().clone()

    for module_name, module in model_with_lora.named_modules():
        if module_name == "":
            continue
        if isinstance(module, LoRALinear):
            weight_key = f"{module_name}.weight"
            merged[weight_key] = module.merged_weight().detach().clone()
            if module.bias is not None:
                merged[f"{module_name}.bias"] = module.bias.detach().clone()
            continue
        if isinstance(module, LoRAMultiheadAttention):
            merged[f"{module_name}.in_proj_weight"] = module.merged_in_proj_weight().detach().clone()

    return merged


def _clone_state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


@dataclasses.dataclass
class Trajectory:
    obs: np.ndarray
    actions: np.ndarray
    current_command: np.ndarray
    source: str


@dataclasses.dataclass
class RawObsPreprocess:
    term_order: tuple[str, ...]
    term_scale: dict[str, float]
    act_dim: int
    obs_dim: int


_TERM_FIXED_DIMS = {
    "base_ang_vel": 3,
    "base_lin_vel": 3,
    "projected_gravity": 3,
    "command_lin_vel": 2,
    "command_ang_vel": 1,
    "sin_phase": 2,
    "cos_phase": 2,
}


def _term_dim(term_name: str, act_dim: int) -> int:
    if term_name in {"dof_pos", "dof_vel", "actions"}:
        return int(act_dim)
    if term_name in _TERM_FIXED_DIMS:
        return int(_TERM_FIXED_DIMS[term_name])
    raise ValueError(f"Unsupported compact term '{term_name}' for raw obs scaling.")


def _apply_raw_obs_preprocess(raw_obs: np.ndarray, preprocess: RawObsPreprocess) -> np.ndarray:
    if raw_obs.ndim != 2:
        raise ValueError(f"raw_obs must be rank-2 [T, obs_dim], got shape {raw_obs.shape}")

    start = 0
    out = np.empty_like(raw_obs, dtype=np.float32)
    for term in preprocess.term_order:
        dim = _term_dim(term, preprocess.act_dim)
        end = start + dim
        if end > raw_obs.shape[1]:
            raise ValueError(f"Raw obs term slicing overflow for term '{term}': end={end}, obs_dim={raw_obs.shape[1]}")
        scale = float(preprocess.term_scale.get(term, 1.0))
        out[:, start:end] = raw_obs[:, start:end].astype(np.float32, copy=False) * scale
        start = end

    if start != raw_obs.shape[1]:
        raise ValueError(
            "Raw obs preprocessing term dims mismatch: "
            f"sum={start}, obs_dim={raw_obs.shape[1]}, term_order={preprocess.term_order}"
        )
    if raw_obs.shape[1] != preprocess.obs_dim:
        raise ValueError(f"Raw obs dim mismatch: got {raw_obs.shape[1]}, expected {preprocess.obs_dim}")
    return out


def _ensure_step_env_feature(array: np.ndarray, *, name: str) -> np.ndarray:
    if array.ndim == 2:
        return array[:, np.newaxis, :]
    if array.ndim == 3:
        return array
    if array.ndim > 3:
        return array.reshape(array.shape[0], array.shape[1], -1)
    raise ValueError(f"Expected {name} with ndim >= 2, got shape {array.shape}")


def _ensure_step_env_done(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array[:, np.newaxis].astype(np.bool_, copy=False)
    if array.ndim == 2:
        return array.astype(np.bool_, copy=False)
    if array.ndim == 3:
        return np.any(array.astype(np.bool_, copy=False), axis=-1)
    raise ValueError(f"Expected dones with ndim in [1,2,3], got shape {array.shape}")


def _resolve_obs_key_for_episode(ep_group: Any, obs_key: str) -> str:
    if obs_key != "auto":
        if obs_key not in ep_group:
            raise KeyError(
                f"Observation key '{obs_key}' not found in episode group. Available: {list(ep_group.keys())}"
            )
        return obs_key

    if "raw_dynamics_obs" in ep_group:
        return "raw_dynamics_obs"

    if "dynamics_obs" in ep_group:
        return "dynamics_obs"

    raise KeyError(
        "obs_key='auto' requires 'raw_dynamics_obs' or 'dynamics_obs', but neither is found in episode group. "
        f"Available keys: {list(ep_group.keys())}. "
        "Pass --obs-key <key> explicitly."
    )


def load_trajectories_from_h5(
    dataset_paths: list[Path],
    *,
    obs_key: str,
    command_key: str,
    obs_dim: int,
    act_dim: int,
    cmd_dim: int,
    raw_obs_preprocess: RawObsPreprocess | None,
    pred_horizon_k: int,
    trim_head_steps: int = 0,
    trim_tail_steps: int = 0,
    require_command_key: bool = False,
) -> tuple[list[Trajectory], dict[str, Any]]:
    h5py = _load_h5py()
    trim_head_steps = int(trim_head_steps)
    trim_tail_steps = int(trim_tail_steps)
    if trim_head_steps < 0 or trim_tail_steps < 0:
        raise ValueError(
            f"trim_head_steps/trim_tail_steps must be non-negative, got {trim_head_steps}/{trim_tail_steps}"
        )

    trajectories: list[Trajectory] = []
    stats = {
        "num_files": 0,
        "num_episodes": 0,
        "num_env_trajectories": 0,
        "num_skipped_short": 0,
        "num_skipped_trimmed_empty": 0,
        "obs_key_usage": {},
        # Termination provenance. A dataset with no `dones` and one whose robot never
        # fell produce byte-identical training windows here; these counters are what
        # distinguishes them.
        "num_episodes_with_dones_field": 0,
        "num_episodes_without_dones_field": 0,
        "num_env_trajectories_truncated_at_done": 0,
        "fall_reports": [],
        # Collection provenance, per file: an interrupted collection still closes cleanly
        # and produces a structurally complete H5. The collector stamps what it planned,
        # what it got, and whether it reached close(); this restates that.
        "collection_provenance": [],
    }

    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as f:
            if "episodes" not in f:
                raise ValueError(f"Expected 'episodes' group in {dataset_path}")
            episodes = f["episodes"]
            ep_names = sorted(episodes.keys())
            if not ep_names:
                raise ValueError(f"No episodes found in {dataset_path}")

            stats["num_files"] += 1
            stats["num_episodes"] += len(ep_names)
            size_mb = Path(dataset_path).stat().st_size / 1024 / 1024
            print(
                f"  [data] {Path(dataset_path).name} ({size_mb:.0f} MB, {len(ep_names)} episodes)",
                flush=True,
            )
            report = _read_collection_fall_report(Path(dataset_path))
            if report is not None:
                stats["fall_reports"].append(report)
            stats["collection_provenance"].append(_read_collection_provenance(f, Path(dataset_path)))

            # Buffer data (from training NPZ) carries skip_trim=True — no
            # simulator startup transient to remove.
            file_skip_trim = bool(f.attrs.get("skip_trim", False))
            file_trim_head = 0 if file_skip_trim else trim_head_steps
            file_trim_tail = 0 if file_skip_trim else trim_tail_steps

            for ep_name in ep_names:
                ep_group = episodes[ep_name]
                resolved_obs_key = _resolve_obs_key_for_episode(ep_group, obs_key)
                stats["obs_key_usage"][resolved_obs_key] = stats["obs_key_usage"].get(resolved_obs_key, 0) + 1

                if "actions" not in ep_group:
                    raise KeyError(f"Required actions dataset missing in {dataset_path}::{ep_name}")

                obs_arr = _ensure_step_env_feature(np.asarray(ep_group[resolved_obs_key]), name=resolved_obs_key)
                actions_arr = _ensure_step_env_feature(np.asarray(ep_group["actions"]), name="actions")
                if command_key in ep_group:
                    current_command_arr = _ensure_step_env_feature(np.asarray(ep_group[command_key]), name=command_key)
                elif require_command_key:
                    # The caller conditions the IDM on the command, so a missing key is
                    # fatal rather than zero-filled.
                    raise KeyError(
                        f"command_key={command_key!r} not found in {dataset_path}::{ep_name}, and "
                        "command conditioning is enabled (idm_use_current_command_for_history=True), "
                        "so substituting zeros would train the IDM on an all-zero command. "
                        f"Available keys: {sorted(ep_group.keys())}. Pass --command-key <key>."
                    )
                else:
                    # Command key absent — use zeros. Reachable only when the model does not
                    # consume the command at all: _resolve_idm_current_command returns None
                    # when idm_use_current_command_for_history=False, so nothing reads these.
                    current_command_arr = np.zeros((obs_arr.shape[0], obs_arr.shape[1], cmd_dim), dtype=np.float32)

                if obs_arr.shape[0] != actions_arr.shape[0] or obs_arr.shape[0] != current_command_arr.shape[0]:
                    raise ValueError(
                        "Step dimension mismatch across obs/actions/current_command in "
                        f"{dataset_path}::{ep_name}: obs={obs_arr.shape}, actions={actions_arr.shape}, "
                        f"current_command={current_command_arr.shape}"
                    )
                if obs_arr.shape[1] != actions_arr.shape[1] or obs_arr.shape[1] != current_command_arr.shape[1]:
                    raise ValueError(
                        "Env dimension mismatch across obs/actions/current_command in "
                        f"{dataset_path}::{ep_name}: obs={obs_arr.shape}, actions={actions_arr.shape}, "
                        f"current_command={current_command_arr.shape}"
                    )

                if "dones" in ep_group:
                    dones_arr = _ensure_step_env_done(np.asarray(ep_group["dones"]))
                    stats["num_episodes_with_dones_field"] += 1
                else:
                    # No `dones` dataset: synthesize all-False and count the substitution,
                    # so it is distinguishable from "recorded, and never terminated".
                    dones_arr = np.zeros((obs_arr.shape[0], obs_arr.shape[1]), dtype=np.bool_)
                    stats["num_episodes_without_dones_field"] += 1

                num_envs = int(obs_arr.shape[1])
                for env_idx in range(num_envs):
                    done_env = dones_arr[:, env_idx]
                    done_indices = np.flatnonzero(done_env)
                    cutoff = int(done_indices[0]) if done_indices.size > 0 else int(obs_arr.shape[0])
                    if done_indices.size > 0:
                        stats["num_env_trajectories_truncated_at_done"] += 1
                    start_idx = int(file_trim_head)
                    end_idx = int(cutoff - file_trim_tail)
                    if end_idx <= start_idx:
                        stats["num_skipped_trimmed_empty"] += 1
                        continue

                    effective_len = int(end_idx - start_idx)
                    if effective_len <= pred_horizon_k:
                        stats["num_skipped_short"] += 1
                        continue

                    obs_env = obs_arr[start_idx:end_idx, env_idx, :].astype(np.float32, copy=False)
                    actions_env = actions_arr[start_idx:end_idx, env_idx, :].astype(np.float32, copy=False)
                    current_command_env = current_command_arr[start_idx:end_idx, env_idx, :].astype(
                        np.float32,
                        copy=False,
                    )

                    obs_env = obs_env.reshape(obs_env.shape[0], -1)
                    actions_env = actions_env.reshape(actions_env.shape[0], -1)
                    current_command_env = current_command_env.reshape(current_command_env.shape[0], -1)

                    if obs_env.shape[1] != obs_dim:
                        raise ValueError(
                            f"obs_dim mismatch for {dataset_path}::{ep_name}/env{env_idx}: "
                            f"got {obs_env.shape[1]}, expected {obs_dim}"
                        )
                    if actions_env.shape[1] != act_dim:
                        raise ValueError(
                            f"act_dim mismatch for {dataset_path}::{ep_name}/env{env_idx}: "
                            f"got {actions_env.shape[1]}, expected {act_dim}"
                        )
                    if current_command_env.shape[1] != cmd_dim:
                        raise ValueError(
                            f"cmd_dim mismatch for {dataset_path}::{ep_name}/env{env_idx}: "
                            f"got {current_command_env.shape[1]}, expected {cmd_dim}"
                        )

                    if resolved_obs_key == "raw_dynamics_obs":
                        if raw_obs_preprocess is None:
                            raise RuntimeError(
                                "raw_dynamics_obs found but no raw obs preprocessing metadata is available from checkpoint."
                            )
                        obs_env = _apply_raw_obs_preprocess(obs_env, raw_obs_preprocess)

                    trajectories.append(
                        Trajectory(
                            obs=obs_env,
                            actions=actions_env,
                            current_command=current_command_env,
                            source=f"{dataset_path}:{ep_name}:env{env_idx}",
                        )
                    )
                    stats["num_env_trajectories"] += 1

    if not trajectories:
        raise RuntimeError("No valid trajectories loaded from dataset paths.")
    for line in describe_collection_completeness(stats):
        print(line, flush=True)
    for line in describe_termination_provenance(stats):
        print(line, flush=True)
    return trajectories, stats


def _as_int(value: Any, default: Any = -1) -> Any:
    """Coerce an H5 attribute / JSON field to ``int``, preserving a genuine zero.

    Only ``None`` and unconvertible values fall back to ``default``; a recorded
    ``episodes=0`` / ``collected_steps=0`` comes back as ``0``.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_collection_provenance(h5_file: Any, dataset_path: Path) -> dict[str, Any]:
    """Read the collector's completion stamps off an open H5, tolerating their absence.

    Total: every dataset produces a row, including one written before these attributes
    existed, whose fields then read as "not recorded".
    """

    def _attr(name: str, default: Any) -> Any:
        value = h5_file.attrs.get(name, default)
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return value

    status = _attr("collection_status", None)
    planned = _as_int(_attr("planned_steps", None))
    collected = _as_int(_attr("collected_steps", None))
    # The accumulating per-session history, when the writer recorded one. The attributes
    # above are rewritten on every reopen, so in a file more than one session appended to
    # they describe the LAST session while the file holds every episode of all of them.
    sessions: list[dict[str, Any]] = []
    raw_sessions = _attr("collection_sessions", None)
    if raw_sessions is not None:
        try:
            parsed = json.loads(raw_sessions)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            sessions = [entry for entry in parsed if isinstance(entry, dict)]

    # The episodes themselves, bucketed by the session tag each one carries. A session
    # record states what its writer believed; these counts come from the file.
    file_episodes = -1
    session_tags: dict[int, int] = {}
    untagged_episodes = 0
    unparseable_tags = 0
    try:
        if "episodes" in h5_file:
            groups = h5_file["episodes"]
            file_episodes = len(groups.keys())
            for name in groups.keys():
                raw = groups[name].attrs.get("collection_session", None)
                if raw is None:
                    untagged_episodes += 1
                    continue
                # `_as_session_index`, not `_as_int`: a non-integral tag such as `0.1` is
                # rejected rather than truncated to `0`.
                index = _as_session_index(raw)
                if index is None:
                    unparseable_tags += 1
                    continue
                session_tags[index] = session_tags.get(index, 0) + 1
    except Exception:  # a reporting path must not be what fails a valid dataset
        file_episodes = -1
        session_tags = {}
        untagged_episodes = 0
        unparseable_tags = 0

    return {
        "dataset": str(dataset_path),
        "collection_status": str(status) if status is not None else None,
        "planned_steps": planned,
        "planned_steps_source": str(_attr("planned_steps_source", "unknown")),
        "collected_steps": collected,
        "sessions": sessions,
        "file_episodes": file_episodes,
        "episodes_by_session": session_tags,
        "untagged_episodes": untagged_episodes,
        "unparseable_session_tags": unparseable_tags,
    }


def _describe_one_session(prefix: str, status: Any, planned: int, collected: int, source: str) -> str:
    """One session's verdict, or the contradiction between its own attributes.

    The consistency check runs first: `collection_status=complete` together with
    `collected_steps < planned_steps` is reported as a contradiction, not as a verdict.
    """
    if status == "complete" and planned > 0 and 0 <= collected < planned - PLANNED_STEPS_TOLERANCE:
        return (
            f"{prefix}WARNING: records collection_status=complete but collected_steps={collected} "
            f"of planned_steps={planned} ({source}). Those two attributes contradict each other, so "
            "completeness is UNKNOWN for this session -- the status is not evidence and neither is "
            "the count. The data is used unchanged."
        )
    if status == "complete":
        return f"{prefix}collection complete ({collected}/{planned} planned steps, {source})."
    if status == "truncated":
        return (
            f"{prefix}WARNING: collected SHORT: {collected} of a planned {planned} step(s) "
            f"({source}), then closed cleanly. Training on it is allowed and unchanged; be aware it is "
            "a partial rollout, not the run that was requested."
        )
    if status == "in_progress":
        return (
            f"{prefix}WARNING: still marked collection_status=in_progress: the collector "
            f"never reached close() (SIGKILL/OOM/crash). {collected} step(s) were stamped before it "
            f"died, of a planned {planned} ({source}). Whatever survived in the file is being used."
        )
    return (
        f"{prefix}collection_status={status} -- closed cleanly, but no planned step count "
        "was recorded, so completeness is unknown."
    )


def _as_session_index(value: Any) -> int | None:
    """A `collection_session` index, or ``None`` when the value is not one.

    Rejects rather than coerces: a non-integral number (``0.9``), a `bool` (which is an
    `int` subclass), and a negative value all return ``None``. ``None`` routes an episode
    tag into `unparseable_session_tags` and a session record into the warning in
    `describe_session_reconciliation`.
    """
    if value is None:
        return None
    # Checked before the int branch: bool IS an int, and `int(True) == 1`.
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        index = int(value)
    elif isinstance(value, (float, np.floating)):
        as_float = float(value)
        # `1.0` is an integer written as a float (h5py hands back float64 routinely);
        # `0.9` is not an index at all.
        if not as_float.is_integer():
            return None
        index = int(as_float)
    elif isinstance(value, (str, bytes, np.bytes_)):
        text = value.decode("utf-8", "replace") if isinstance(value, (bytes, np.bytes_)) else str(value)
        try:
            index = int(text.strip())  # rejects "0.9", unlike int(float("0.9"))
        except (TypeError, ValueError):
            return None
    else:
        return None
    return index if index >= 0 else None


def invalid_session_index_records(sessions: list[dict[str, Any]]) -> list[tuple[int, Any]]:
    """`(position, raw value)` for every record whose `session_index` cannot be read.

    A record with no `session_index` at all is not in here: its index is its position,
    which is the documented fallback. This is only the records that stated one and stated
    something that is not an index.
    """
    bad: list[tuple[int, Any]] = []
    for position, session in enumerate(sessions):
        raw = session.get("session_index")
        if raw is None:
            continue
        if _as_session_index(raw) is None:
            bad.append((position, raw))
    return bad


def _session_slot_index(session: dict[str, Any], position: int) -> int:
    """The 0-based index an episode's ``collection_session`` tag would carry for this record."""
    recorded = _as_session_index(session.get("session_index"))
    return position if recorded is None else recorded


def describe_session_reconciliation(row: dict[str, Any]) -> list[str]:
    """Cross-check the session records against the episodes the file actually holds.

    A `collection_sessions` record states what its writer believed, in an attribute it
    rewrites on every reopen. Each episode carries a `collection_session` tag naming the
    record it belongs to; this compares the two and reports every disagreement:
    untagged episodes, unreadable tags, per-record counts that do not match the tagged
    episodes, and records whose indices collide.

    Reporting only. Nothing here changes `dones`, the episodes, or which windows the
    finetune trains on.
    """
    file_episodes = _as_int(row.get("file_episodes"))
    if file_episodes < 0:
        return []  # the episode groups could not be read; nothing to reconcile against

    tags: dict[int, int] = dict(row.get("episodes_by_session") or {})
    untagged = _as_int(row.get("untagged_episodes"), default=0)
    unparseable = _as_int(row.get("unparseable_session_tags"), default=0)
    sessions = row.get("sessions") or []
    name = Path(str(row.get("dataset", "?"))).name

    if not sessions and not tags and untagged == file_episodes:
        # A file written entirely before session provenance existed: nothing to
        # reconcile, and the file-level verdict already reports it.
        return []

    lines: list[str] = []
    if untagged:
        lines.append(
            f"  [data] WARNING: {name}: {untagged} of {file_episodes} episode(s) carry no "
            "collection_session tag, so they cannot be attributed to any session record. They predate "
            "session tagging (or were written by a collector that did not tag them); no session's "
            "verdict below covers them."
        )
    if unparseable:
        lines.append(
            f"  [data] WARNING: {name}: {unparseable} episode(s) carry a collection_session tag that "
            "is not a session index (non-integral, boolean or negative). Treated as unattributed."
        )

    # The record side of the same rejection: a record that stated an unreadable index
    # would otherwise fall back to its position in the list, indistinguishable from one
    # that stated no index at all.
    for position, raw in invalid_session_index_records(sessions):
        lines.append(
            f"  [data] WARNING: {name}: collection_sessions[{position}] records "
            f"session_index={raw!r}, which is not a session index (a non-negative integer). "
            f"It is being read as its position in the list ({position}) instead, so its "
            "episode attribution below may be wrong."
        )

    slots_by_position = [_session_slot_index(session, position) for position, session in enumerate(sessions)]
    claimed_slots = set(slots_by_position)
    # An episode tag names an index, not a record, so two records claiming the same slot
    # both take credit for the same episodes: each record's count can match while the file
    # holds only one record's worth. Counted over the list, not the set.
    duplicated_slots = sorted({slot for slot in claimed_slots if slots_by_position.count(slot) > 1})
    if duplicated_slots:
        detail = ", ".join(
            f"session_index={slot} claimed by {slots_by_position.count(slot)} records "
            f"({tags.get(slot, 0)} episode(s) carry that tag)"
            for slot in duplicated_slots
        )
        lines.append(
            f"  [data] WARNING: {name}: more than one record in collection_sessions claims the same "
            f"session index: {detail}. An episode tag names an index, not a record, so those "
            "episodes cannot be attributed to either record and the per-session counts below "
            "double-count them."
        )
    for position, session in enumerate(sessions):
        slot = _session_slot_index(session, position)
        recorded = _as_int(session.get("episodes"), default=None)
        actual = tags.get(slot, 0)
        label = f"  [data]   session {position + 1}/{len(sessions)}:"
        if recorded is None:
            lines.append(
                f"{label} records no episode count; {actual} episode(s) in the file are tagged with it."
            )
        elif recorded != actual:
            lines.append(
                f"{label} WARNING: records episodes={recorded} but {actual} episode(s) in the file "
                "carry its tag. The record and the file disagree, so neither number is evidence for "
                "this session."
            )
        elif recorded == 0:
            lines.append(
                f"{label} WARNING: recorded zero episodes, and the file confirms it -- this session "
                "contributed nothing to the data being trained on."
            )

    orphan_slots = sorted(slot for slot in tags if slot not in claimed_slots)
    if orphan_slots:
        counts = ", ".join(f"session_index={slot} ({tags[slot]} episode(s))" for slot in orphan_slots)
        lines.append(
            f"  [data] WARNING: {name}: episodes are tagged with session indices that no record in "
            f"collection_sessions claims: {counts}. Those episodes have no recorded verdict at all."
        )

    attributed = sum(tags.values())
    if attributed + untagged + unparseable != file_episodes:
        lines.append(
            f"  [data] WARNING: {name}: {file_episodes} episode(s) in the file, but "
            f"{attributed} tagged + {untagged} untagged + {unparseable} unparseable does not add up. "
            "The reconciliation above is incomplete."
        )
    return lines


def _sessions_account_for_every_episode(row: dict[str, Any]) -> bool:
    """True when every episode in the file is attributed to a session record.

    Gates the "holds the episodes of all of them" headline.
    """
    file_episodes = _as_int(row.get("file_episodes"))
    if file_episodes < 0:
        return False
    if _as_int(row.get("untagged_episodes"), default=0) or _as_int(row.get("unparseable_session_tags"), default=0):
        return False
    tags: dict[int, int] = dict(row.get("episodes_by_session") or {})
    sessions = row.get("sessions") or []
    slots_by_position = [_session_slot_index(session, position) for position, session in enumerate(sessions)]
    claimed = set(slots_by_position)
    # Distinct records must claim distinct slots: otherwise the same tagged episodes are
    # attributed to every record sharing the index, each record's `episodes` count can
    # equal that shared total, and `sum(tags.values()) == file_episodes` still holds.
    # Checked before the per-record loop, which cannot see it.
    if len(claimed) != len(slots_by_position):
        return False
    if any(slot not in claimed for slot in tags):
        return False
    for position, session in enumerate(sessions):
        if _as_int(session.get("episodes"), default=None) != tags.get(_session_slot_index(session, position), 0):
            return False
    return sum(tags.values()) == file_episodes


def _describe_multi_session_file(name: str, sessions: list[dict[str, Any]], row: dict[str, Any]) -> list[str]:
    """A file that more than one collection session wrote to.

    The file-level attributes describe only the newest session, because reopening the H5
    rewrites them, while the file keeps every episode from every session. The headline is
    conditional on the records and the episodes agreeing; see
    :func:`describe_session_reconciliation`, whose lines follow this block.
    """
    file_episodes = _as_int(row.get("file_episodes"))
    newest_eps = _as_int(sessions[-1].get("episodes"))
    reconciled = _sessions_account_for_every_episode(row)
    tags: dict[int, int] = dict(row.get("episodes_by_session") or {})
    empty = sum(1 for position, session in enumerate(sessions) if not tags.get(_session_slot_index(session, position)))
    if not reconciled:
        holds = "holds episodes that its session records do NOT fully account for (see the reconciliation below)"
    elif empty:
        # Sessions that contributed no episodes are named separately, so the count in
        # the headline is of sessions whose episodes the file holds.
        holds = (
            f"holds the episodes of {len(sessions) - empty} of them "
            f"({empty} contributed no episodes at all)"
        )
    else:
        holds = "holds the episodes of all of them"
    lines = [
        f"  [data] WARNING: {name} was written by {len(sessions)} collection sessions and {holds}"
        + (f" ({file_episodes} episode(s) in the file" if file_episodes >= 0 else "")
        + (f", {newest_eps} from the newest session)" if file_episodes >= 0 and newest_eps >= 0 else ")")
        + ". Its file-level collection_status describes the NEWEST session only; each session's own "
        "verdict follows."
    ]
    for index, session in enumerate(sessions, start=1):
        status = session.get("collection_status")
        planned = _as_int(session.get("planned_steps"))
        collected = _as_int(session.get("collected_steps"))
        source = session.get("planned_steps_source") or "unknown"
        episodes = _as_int(session.get("episodes"), default=None)
        prefix = f"  [data]   session {index}/{len(sessions)}: "
        if status is None:
            lines.append(
                f"{prefix}no verdict recorded -- this slot was written by a collector that did not "
                "complete its own provenance entry."
            )
            continue
        line = _describe_one_session(prefix, status, planned, collected, str(source))
        if episodes is not None and episodes >= 0:
            line = f"{line} [{episodes} episode(s)]"
        lines.append(line)
    return lines


def describe_collection_completeness(stats: dict[str, Any]) -> list[str]:
    """Console block stating, per file, whether the collection that wrote it finished.

    Reporting only -- what gets trained on is unchanged.
    """
    lines: list[str] = []
    for row in stats.get("collection_provenance", []):
        name = Path(str(row.get("dataset", "?"))).name
        status = row.get("collection_status")
        planned = _as_int(row.get("planned_steps"))
        collected = _as_int(row.get("collected_steps"))
        source = row.get("planned_steps_source") or "unknown"
        sessions = row.get("sessions") or []
        if len(sessions) > 1:
            lines.extend(_describe_multi_session_file(name, sessions, row))
            # Reconciliation applies to every file, not only multi-session ones.
            lines.extend(describe_session_reconciliation(row))
            continue
        lines.extend(describe_session_reconciliation(row))
        if status is None:
            lines.append(
                f"  [data] WARNING: {name} carries no collection_status attribute -- it predates "
                "collection-provenance recording. Whether that collection ran to completion or was "
                "interrupted cannot be determined from the file; both produce a valid H5."
            )
            continue
        if status == "complete" and planned > 0 and 0 <= collected < planned - PLANNED_STEPS_TOLERANCE:
            # The status and the counts cannot both be right. Repeating them side by side
            # as a verdict ("collection complete (2/5000 planned steps)") is the reader
            # trusting a file that contradicts itself.
            lines.append(
                f"  [data] WARNING: {name} records collection_status=complete but "
                f"collected_steps={collected} of planned_steps={planned} ({source}). Those two "
                "attributes contradict each other, so completeness is UNKNOWN for this file -- the "
                "status is not evidence and neither is the count. The data is used unchanged."
            )
        elif status == "complete":
            lines.append(f"  [data] {name}: collection complete ({collected}/{planned} planned steps, {source}).")
        elif status == "truncated":
            lines.append(
                f"  [data] WARNING: {name} was collected SHORT: {collected} of a planned {planned} step(s) "
                f"({source}), then closed cleanly. Training on it is allowed and unchanged; be aware it is "
                "a partial rollout, not the run that was requested."
            )
        elif status == "in_progress":
            lines.append(
                f"  [data] WARNING: {name} is still marked collection_status=in_progress: the collector "
                f"never reached close() (SIGKILL/OOM/crash). {collected} step(s) were stamped before it "
                f"died, of a planned {planned} ({source}). Whatever survived in the file is being used."
            )
        else:
            lines.append(
                f"  [data] {name}: collection_status={status} -- closed cleanly, but no planned step count "
                "was recorded, so completeness is unknown."
            )
    return lines


def _read_collection_fall_report(dataset_path: Path) -> dict[str, Any] | None:
    """Return the collector's `collection_fall_report.json` sitting next to an H5, if any.

    The collector writes this report next to the dataset when a run ends, so the training
    step can restate the run's post-fall contamination.
    """
    report_path = Path(dataset_path).parent / "collection_fall_report.json"
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text())
    except Exception as exc:  # a malformed sidecar must not fail a valid dataset
        print(f"  [data] could not read {report_path}: {exc}", flush=True)
        return None
    if not isinstance(payload, dict):
        return None
    payload = dict(payload)
    payload["dataset"] = str(dataset_path)
    return payload


def describe_termination_provenance(stats: dict[str, Any]) -> list[str]:
    """Console block stating what the loaded training windows are, and are not.

    Three situations produce the same all-False `dones`, and therefore the same windows:
    a clean rollout, a dataset recorded before `dones` existed, and a run where the robot
    fell with `--task.collect-mark-fall-terminal` at its default. This block names which
    one each file is.

    Reporting only -- what gets trained on is unchanged.
    """
    lines: list[str] = []
    without = int(stats.get("num_episodes_without_dones_field", 0))
    truncated = int(stats.get("num_env_trajectories_truncated_at_done", 0))
    total_traj = int(stats.get("num_env_trajectories", 0))
    lines.append(
        f"  [data] termination: {truncated}/{total_traj} trajectories were truncated at a `done`; "
        f"the other {max(total_traj - truncated, 0)} were loaded to the end of the recording."
    )
    if without > 0:
        lines.append(
            f"  [data] WARNING: {without} episode(s) carry no `dones` dataset at all. They were "
            "treated as 'never terminated' -- the released behavior -- so if the robot fell during "
            "any of them, the post-fall tail is in this training set as ordinary windows. Datasets "
            "collected before `dones` was recorded cannot be checked; re-collect, or trim with "
            "--trim-tail-steps."
        )
    for report in stats.get("fall_reports", []):
        dataset = Path(str(report.get("dataset", "?"))).name
        if not report.get("fall_detection_available", True):
            lines.append(
                f"  [data] WARNING: {dataset}'s collection report says fall detection was "
                "UNAVAILABLE for that run; whether it contains post-fall data is unknown."
            )
            continue
        post_fall = int(report.get("post_fall_steps", 0) or 0)
        if post_fall <= 0:
            continue
        collected = int(report.get("collected_steps", 0) or 0)
        pct = (100.0 * post_fall / collected) if collected > 0 else float("nan")
        marked = bool(report.get("marked_terminal", False))
        lines.append(
            f"  [data] WARNING: {dataset}'s collection report says the robot fell at recorded step "
            f"{report.get('fall_step')}; {post_fall}/{collected} steps ({pct:.1f}%) are post-fall."
        )
        lines.append(
            "  [data]          Those steps were written "
            + (
                "terminal, so the loader cut the episode at the fall."
                if marked
                else "NON-terminal (--task.collect-mark-fall-terminal was off, the released default), "
                "so they are being used as ordinary IDM training windows. Re-collect, or pass "
                "--task.collect-mark-fall-terminal true when collecting."
            )
        )
    return lines


class TrajectoryObsWindowDataset(torch.utils.data.Dataset):
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
        history_valid = np.zeros((self.history_len,), dtype=np.bool_)
        start = max(0, t - self.history_len + 1)
        hist_indices = np.arange(start, t + 1, dtype=np.int64)
        offset = int(self.history_len - hist_indices.shape[0])

        history_obs[offset:] = obs[hist_indices]
        history_valid[offset:] = True
        current_command = current_command_seq[t].astype(np.float32, copy=False)

        prev_indices = hist_indices - 1
        valid_prev = prev_indices >= 0
        if np.any(valid_prev):
            valid_slots = np.flatnonzero(valid_prev)
            history_act[offset + valid_slots] = actions[prev_indices[valid_prev]]

        target_next_obs = obs[t + 1 : t + 1 + self.pred_horizon_k]
        if target_next_obs.shape[0] != self.pred_horizon_k:
            raise RuntimeError(
                f"Unexpected target horizon at index {index}: got {target_next_obs.shape[0]}, expected {self.pred_horizon_k}"
            )

        sample = {
            "history_obs": torch.from_numpy(history_obs),
            "history_act": torch.from_numpy(history_act),
            "current_command": torch.from_numpy(current_command),
            "history_valid_mask": torch.from_numpy(history_valid),
            "target_next_observations": torch.from_numpy(target_next_obs.astype(np.float32, copy=False)),
        }
        if self.include_metadata:
            sample["trajectory_index"] = torch.tensor(traj_idx, dtype=torch.int64)
            sample["window_index"] = torch.tensor(t, dtype=torch.int64)
            sample["source"] = traj.source
        return sample


def _normalize_if_enabled(
    history_obs: torch.Tensor,
    history_act: torch.Tensor,
    current_command: torch.Tensor,
    target_next_obs: torch.Tensor,
    *,
    io_norm_enabled: bool,
    obs_stats: dict[str, torch.Tensor] | None,
    action_stats: dict[str, torch.Tensor] | None,
    command_stats: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not io_norm_enabled:
        return history_obs, history_act, current_command, target_next_obs

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
        (target_next_obs - obs_mean) / obs_std,
    )


def _run_epoch(
    *,
    model: HistoryPolicy,
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
) -> dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    # PyTorch Transformer eval fastpath (`torch._transformer_encoder_layer_fwd`) reads
    # encoder linear weights directly and bypasses wrapped module forward calls.
    # With LoRA wrappers, that would ignore adapter A/B branches during validation.
    lora_active = any(isinstance(m, (LoRALinear, LoRAMultiheadAttention)) for m in model.modules())
    disable_eval_fastpath = (not is_train) and lora_active and bool(torch.backends.mha.get_fastpath_enabled())
    fastpath_prev = bool(torch.backends.mha.get_fastpath_enabled())
    if disable_eval_fastpath:
        torch.backends.mha.set_fastpath_enabled(False)

    total_obs_loss = 0.0
    total_action_loss = 0.0
    total_loss = 0.0
    total_steps = 0

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
            target_obs = batch["target_next_observations"].to(device=device, dtype=torch.float32)

            history_obs, history_act, current_command, target_obs = _normalize_if_enabled(
                history_obs,
                history_act,
                current_command,
                target_obs,
                io_norm_enabled=io_norm_enabled,
                obs_stats=obs_stats,
                action_stats=action_stats,
                command_stats=command_stats,
            )

            with torch.set_grad_enabled(is_train):
                _, pred_next_obs = model(
                    history_obs,
                    history_act,
                    current_command,
                    history_valid_mask=history_valid,
                    return_obs=True,
                )
                pred_next_obs = pred_next_obs[:, :pred_horizon_k, :]
                obs_loss = F.mse_loss(pred_next_obs, target_obs)
                action_loss = torch.zeros((), device=obs_loss.device, dtype=obs_loss.dtype)
                total = obs_loss

                if is_train:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    if grad_clip > 0.0:
                        params = [p for p in model.parameters() if p.requires_grad]
                        if params:
                            torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()

            total_obs_loss += float(obs_loss.detach().cpu().item())
            total_action_loss += float(action_loss.detach().cpu().item())
            total_loss += float(total.detach().cpu().item())
            total_steps += 1
    finally:
        if disable_eval_fastpath:
            torch.backends.mha.set_fastpath_enabled(fastpath_prev)

    if total_steps <= 0:
        return {
            "loss_total": float("nan"),
            "loss_obs": float("nan"),
            "loss_action": float("nan"),
            "steps": 0.0,
        }

    return {
        "loss_total": total_loss / total_steps,
        "loss_obs": total_obs_loss / total_steps,
        "loss_action": total_action_loss / total_steps,
        "steps": float(total_steps),
    }
