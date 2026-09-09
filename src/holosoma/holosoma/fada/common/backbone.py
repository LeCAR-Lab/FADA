from __future__ import annotations

import math
from typing import Any

from holosoma.utils.safe_torch_import import nn, torch


def normalize_backbone_type(value: str | None) -> str:
    token = str(value or "transformer").strip().lower().replace("-", "_")
    if token in {"transformer", "mlp"}:
        return token
    raise ValueError(f"Unsupported backbone_type: {value!r}. Expected one of: transformer, mlp")


def normalize_mlp_hidden_dims(
    value: object | None,
    *,
    default: tuple[int, ...] = (1024, 512, 256),
    fallback_hidden_dim: int | None = None,
    fallback_num_layers: int | None = None,
) -> tuple[int, ...]:
    dims: tuple[int, ...]
    if value is None:
        if fallback_hidden_dim is None and fallback_num_layers is None:
            dims = tuple(int(v) for v in default)
        else:
            resolved_hidden_dim = int(fallback_hidden_dim if fallback_hidden_dim is not None else default[0])
            resolved_num_layers = int(fallback_num_layers if fallback_num_layers is not None else len(default))
            dims = (resolved_hidden_dim,) * resolved_num_layers
    elif isinstance(value, str):
        tokens = tuple(part.strip() for part in value.split(",") if part.strip())
        dims = tuple(int(token) for token in tokens)
    elif isinstance(value, (list, tuple)):
        dims = tuple(int(v) for v in value)
    else:
        dims = (int(value),)

    if not dims:
        raise ValueError("mlp_hidden_dims must contain at least one hidden dimension")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"mlp_hidden_dims must be positive, got {dims}")
    return dims


def infer_backbone_type_from_cfg(cfg: dict[str, Any] | None) -> str:
    if not isinstance(cfg, dict):
        return "transformer"
    return normalize_backbone_type(cfg.get("backbone_type"))


# Removed: `get_policy_display_name`, `get_policy_onnx_basename`,
# `get_policy_metadata_prefix`, `build_policy` and `build_policy_from_train_cfg`.
# They existed to select and export a `TransformerPolicy`-or-`MLPPolicy` from a
# checkpoint's `backbone_type`, and their only caller was
# `fada/common/eval_checkpoint.py`, which in turn had no caller but one unit test of
# itself. Nothing in the six documented FADA steps reached any of them: step 3 is
# `python -m holosoma.fada.planner_idm.eval_checkpoint`, which builds and exports
# `PlannerIDMPolicy` and never consults `backbone_type`.
#
# `MLPPolicy` itself stays: `lora_utils.py`'s backbone dispatch still branches on it
# and `tests/fada/test_finetune_obs_lora.py` still exercises that branch directly.


class _CheckpointCompatiblePolicy(nn.Module):
    predict_future_obs: bool

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        model_keys = set(self.state_dict().keys())
        state_keys = set(state_dict.keys())

        if self.predict_future_obs:
            missing = sorted(model_keys - state_keys)
            unexpected = sorted(state_keys - model_keys)
            if missing:
                raise RuntimeError(
                    "Checkpoint is missing required keys for predict_future_obs=True. "
                    f"Missing {len(missing)} keys, first few: {missing[:8]}"
                )
            if unexpected:
                raise RuntimeError(
                    "Checkpoint contains unexpected keys for current model. "
                    f"Unexpected {len(unexpected)} keys, first few: {unexpected[:8]}"
                )
            self.load_state_dict(state_dict, strict=True)
            return

        filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("obs_head.")}
        filtered_keys = set(filtered_state.keys())
        missing = sorted(model_keys - filtered_keys)
        unexpected = sorted(filtered_keys - model_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint is incompatible with predict_future_obs=False model. "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        self.load_state_dict(filtered_state, strict=True)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_len > self.pe.shape[1]:
            raise ValueError(f"seq_len={seq_len} exceeds max_len={self.pe.shape[1]}")
        return self.pe[:, :seq_len, :].to(device=device, dtype=dtype)


class TransformerPolicy(_CheckpointCompatiblePolicy):
    """Encoder-only transformer policy with configurable prediction horizon."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        history_len: int,
        pred_horizon: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_learned_positional_encoding: bool = False,
        predict_future_obs: bool = False,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or act_dim <= 0 or cmd_dim <= 0:
            raise ValueError("obs_dim, act_dim, cmd_dim must be positive")
        if history_len <= 0 or pred_horizon <= 0:
            raise ValueError("history_len and pred_horizon must be positive")
        if d_model <= 0 or nhead <= 0 or num_layers <= 0:
            raise ValueError("d_model, nhead, num_layers must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.backbone_type = "transformer"
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cmd_dim = int(cmd_dim)
        self.history_len = int(history_len)
        self.pred_horizon = int(pred_horizon)
        self.d_model = int(d_model)
        self.predict_future_obs = bool(predict_future_obs)

        self.obs_embed = nn.Linear(self.obs_dim, self.d_model)
        self.act_embed = nn.Linear(self.act_dim, self.d_model)
        self.cmd_embed = nn.Linear(self.cmd_dim, self.d_model)

        self.use_learned_positional_encoding = bool(use_learned_positional_encoding)
        if self.use_learned_positional_encoding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.history_len, self.d_model))
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
            self.sinusoidal_positional_encoding = None
        else:
            self.pos_embedding = None
            self.sinusoidal_positional_encoding = SinusoidalPositionalEncoding(self.d_model, max_len=self.history_len)

        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.action_head = nn.Linear(self.d_model, self.pred_horizon * self.act_dim)
        self.obs_head = nn.Linear(self.d_model, self.pred_horizon * self.obs_dim) if self.predict_future_obs else None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        history_valid_mask: torch.Tensor | None = None,
    ) -> None:
        if history_obs.ndim != 3 or history_act.ndim != 3:
            raise ValueError("history_obs and history_act must be rank-3")
        if current_command.ndim != 2:
            raise ValueError("current_command must be rank-2 [B, C]")
        if history_obs.shape[0] != history_act.shape[0] or history_obs.shape[0] != current_command.shape[0]:
            raise ValueError("Batch dimension mismatch across model inputs")
        if history_obs.shape[1] != self.history_len:
            raise ValueError(f"history_obs length mismatch: got {history_obs.shape[1]}, expected {self.history_len}")
        if history_act.shape[1] != self.history_len:
            raise ValueError(f"history_act length mismatch: got {history_act.shape[1]}, expected {self.history_len}")
        if history_obs.shape[2] != self.obs_dim:
            raise ValueError(f"history_obs feature mismatch: got {history_obs.shape[2]}, expected {self.obs_dim}")
        if history_act.shape[2] != self.act_dim:
            raise ValueError(f"history_act feature mismatch: got {history_act.shape[2]}, expected {self.act_dim}")
        if current_command.shape[1] != self.cmd_dim:
            raise ValueError(
                f"current_command feature mismatch: got {current_command.shape[1]}, expected {self.cmd_dim}"
            )
        if history_valid_mask is not None:
            if history_valid_mask.ndim != 2:
                raise ValueError("history_valid_mask must be rank-2 [B, H]")
            if history_valid_mask.shape[0] != history_obs.shape[0]:
                raise ValueError(
                    "history_valid_mask batch mismatch: "
                    f"got {history_valid_mask.shape[0]}, expected {history_obs.shape[0]}"
                )
            if history_valid_mask.shape[1] != self.history_len:
                raise ValueError(
                    "history_valid_mask length mismatch: "
                    f"got {history_valid_mask.shape[1]}, expected {self.history_len}"
                )

    def forward(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        *,
        history_valid_mask: torch.Tensor | None = None,
        return_obs: bool = False,
        skip_empty_history_check: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(history_obs, history_act, current_command, history_valid_mask)

        current_command_embed = self.cmd_embed(current_command).unsqueeze(1).expand(-1, self.history_len, -1)
        tokens = self.obs_embed(history_obs) + self.act_embed(history_act) + current_command_embed
        if self.pos_embedding is not None:
            pos = self.pos_embedding.to(device=tokens.device, dtype=tokens.dtype)
        else:
            assert self.sinusoidal_positional_encoding is not None
            pos = self.sinusoidal_positional_encoding(self.history_len, device=tokens.device, dtype=tokens.dtype)
        tokens = self.embed_dropout(tokens + pos)

        src_key_padding_mask: torch.Tensor | None = None
        if history_valid_mask is not None:
            valid_mask = history_valid_mask.to(device=tokens.device)
            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask > 0
            src_key_padding_mask = ~valid_mask
            if not skip_empty_history_check and torch.any(torch.all(src_key_padding_mask, dim=1)):
                raise ValueError(
                    "history_valid_mask contains an empty history sample (all tokens invalid). "
                    "At least one token must be valid per sample."
                )

        features = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        context = features[:, -1, :]
        pred_actions = self.action_head(context).view(-1, self.pred_horizon, self.act_dim)
        if not return_obs:
            return pred_actions
        if self.obs_head is None:
            raise RuntimeError("return_obs=True requested but model was built with predict_future_obs=False")
        pred_next_obs = self.obs_head(context).view(-1, self.pred_horizon, self.obs_dim)
        return pred_actions, pred_next_obs


class MLPPolicy(_CheckpointCompatiblePolicy):
    """Flattened-history MLP policy with the same DAgger IO contract as TransformerPolicy."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        history_len: int,
        pred_horizon: int,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1,
        predict_future_obs: bool = False,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or act_dim <= 0 or cmd_dim <= 0:
            raise ValueError("obs_dim, act_dim, cmd_dim must be positive")
        if history_len <= 0 or pred_horizon <= 0:
            raise ValueError("history_len and pred_horizon must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        self.backbone_type = "mlp"
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cmd_dim = int(cmd_dim)
        self.history_len = int(history_len)
        self.pred_horizon = int(pred_horizon)
        self.hidden_dims = normalize_mlp_hidden_dims(
            hidden_dims,
            fallback_hidden_dim=hidden_dim,
            fallback_num_layers=num_layers,
        )
        self.hidden_dim = int(self.hidden_dims[0])
        self.num_layers = int(len(self.hidden_dims))
        self.dropout = float(dropout)
        self.predict_future_obs = bool(predict_future_obs)

        input_dim = self.history_len * (self.obs_dim + self.act_dim) + self.cmd_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for layer_hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, int(layer_hidden_dim)))
            layers.append(nn.GELU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = int(layer_hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, self.pred_horizon * self.act_dim)
        self.obs_head = nn.Linear(prev_dim, self.pred_horizon * self.obs_dim) if self.predict_future_obs else None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        history_valid_mask: torch.Tensor | None = None,
    ) -> None:
        if history_obs.ndim != 3 or history_act.ndim != 3:
            raise ValueError("history_obs and history_act must be rank-3")
        if current_command.ndim != 2:
            raise ValueError("current_command must be rank-2 [B, C]")
        if history_obs.shape[0] != history_act.shape[0] or history_obs.shape[0] != current_command.shape[0]:
            raise ValueError("Batch dimension mismatch across model inputs")
        if history_obs.shape[1] != self.history_len:
            raise ValueError(f"history_obs length mismatch: got {history_obs.shape[1]}, expected {self.history_len}")
        if history_act.shape[1] != self.history_len:
            raise ValueError(f"history_act length mismatch: got {history_act.shape[1]}, expected {self.history_len}")
        if history_obs.shape[2] != self.obs_dim:
            raise ValueError(f"history_obs feature mismatch: got {history_obs.shape[2]}, expected {self.obs_dim}")
        if history_act.shape[2] != self.act_dim:
            raise ValueError(f"history_act feature mismatch: got {history_act.shape[2]}, expected {self.act_dim}")
        if current_command.shape[1] != self.cmd_dim:
            raise ValueError(
                f"current_command feature mismatch: got {current_command.shape[1]}, expected {self.cmd_dim}"
            )
        if history_valid_mask is not None:
            if history_valid_mask.ndim != 2:
                raise ValueError("history_valid_mask must be rank-2 [B, H]")
            if history_valid_mask.shape[0] != history_obs.shape[0]:
                raise ValueError(
                    "history_valid_mask batch mismatch: "
                    f"got {history_valid_mask.shape[0]}, expected {history_obs.shape[0]}"
                )
            if history_valid_mask.shape[1] != self.history_len:
                raise ValueError(
                    "history_valid_mask length mismatch: "
                    f"got {history_valid_mask.shape[1]}, expected {self.history_len}"
                )

    def forward(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        *,
        history_valid_mask: torch.Tensor | None = None,
        return_obs: bool = False,
        skip_empty_history_check: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(history_obs, history_act, current_command, history_valid_mask)

        if history_valid_mask is not None:
            valid_mask = history_valid_mask.to(device=history_obs.device)
            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask > 0
            if not skip_empty_history_check and torch.any(torch.all(~valid_mask, dim=1)):
                raise ValueError(
                    "history_valid_mask contains an empty history sample (all tokens invalid). "
                    "At least one token must be valid per sample."
                )
            mask = valid_mask.unsqueeze(-1).to(dtype=history_obs.dtype)
            history_obs = history_obs * mask
            history_act = history_act * mask.to(dtype=history_act.dtype)

        flattened_history_obs = history_obs.reshape(history_obs.shape[0], -1)
        flattened_history_act = history_act.reshape(history_act.shape[0], -1)
        mlp_input = torch.cat([flattened_history_obs, flattened_history_act, current_command], dim=1)
        features = self.backbone(mlp_input)

        pred_actions = self.action_head(features).view(-1, self.pred_horizon, self.act_dim)
        if not return_obs:
            return pred_actions
        if self.obs_head is None:
            raise RuntimeError("return_obs=True requested but model was built with predict_future_obs=False")
        pred_next_obs = self.obs_head(features).view(-1, self.pred_horizon, self.obs_dim)
        return pred_actions, pred_next_obs


HistoryPolicy = TransformerPolicy | MLPPolicy
