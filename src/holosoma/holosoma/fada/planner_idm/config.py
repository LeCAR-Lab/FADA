from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import math
import re
from pathlib import Path

from holosoma.fada.common import cli_validation
from holosoma.fada.common.current_command import DEFAULT_COMMAND_PROFILE, normalize_command_profile

# Leading "YYYYmmdd_HHMMSS_" stamp that holosoma run directories carry.
_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{8}_\d{6}_")

# Fallback used when the expert checkpoint's run directory yields no usable token.
_FALLBACK_RUN_TOKEN = "fada_planner_idm"

# Subdirectory (under the expert's run directory) that holds derived DAgger runs.
_DERIVED_OUTPUT_SUBDIR = "fada_dagger"


def _default_run_name() -> str:
    return f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_t1_fada_planner_idm"


def expert_run_dir(expert_checkpoint: str) -> Path:
    """Return the oracle PPO *run directory* implied by ``--expert-checkpoint``.

    ``--expert-checkpoint`` may be either a ``model_*.pt`` file or the directory that
    holds the run's whole checkpoint history (see the README's FADA section -- the
    suboptimal-data source needs the full history, so the directory form is the
    documented one). Both spellings, plus a ``checkpoints/`` subdirectory layout,
    resolve to the same run directory so the derived defaults below do not depend on
    which spelling the user typed.
    """
    path = Path(str(expert_checkpoint)).expanduser()
    run_dir = path if path.is_dir() else path.parent
    if run_dir.name == "checkpoints":
        run_dir = run_dir.parent
    return run_dir


def default_run_name_from_expert(expert_checkpoint: str) -> str:
    """Derive a stable, expert-scoped ``--run-name`` default.

    Scheme: the expert run directory's name with any leading ``YYYYmmdd_HHMMSS_``
    stamp stripped, plus a ``_fada`` suffix -- e.g. an oracle run
    ``logs/t1_oracle/20250101_120000_t1_23dof_waist50_oracle`` yields
    ``t1_23dof_waist50_oracle_fada``. train.py prefixes a *fresh* timestamp
    onto this for the run directory, so repeated runs never collide; the un-prefixed
    name is what scopes the offline replay cache, so re-running against the same
    expert reuses that cache instead of re-collecting it.
    """
    token = _TIMESTAMP_PREFIX_RE.sub("", expert_run_dir(expert_checkpoint).resolve().name).strip()
    if token == "":
        token = _FALLBACK_RUN_TOKEN
    return f"{token}_fada"


def default_output_dir_from_expert(expert_checkpoint: str) -> str:
    """Derive an absolute ``--output-dir`` default: ``<expert run dir>_fada_dagger``, a
    *sibling* of the oracle run directory rather than a subdirectory of it.

    Absolute, not relative: sibling FADA entry points resolve a *relative*
    ``--output-dir`` against the checkpoint's directory rather than the cwd (see
    ``fada/common/lora_utils.py::_resolve_output_root``).

    A sibling of the oracle run directory, not a subdirectory of it, so nothing is
    written into the downloaded oracle checkpoint tree.
    """
    run_dir = expert_run_dir(expert_checkpoint).resolve()
    return str(run_dir.parent / f"{run_dir.name}_{_DERIVED_OUTPUT_SUBDIR}")


@dataclasses.dataclass
class FADAConfig:
    # Core sequence setup
    history_len: int = 30
    pred_horizon: int = 6
    predict_future_obs: bool = True
    planner_use_action_history: bool = False
    use_generalized_idm: bool = True
    planner_predict_delta: bool = True
    idm_use_teacher_forcing: bool = True
    idm_teacher_forcing_ratio: float = 1.0
    idm_detach_planner_future_obs: bool = True
    idm_use_current_command_for_history: bool = False

    # Runtime dimensions. If None, infer from environment.
    obs_dim: int | None = None
    act_dim: int | None = None
    cmd_dim: int | None = None
    command_profile: str = DEFAULT_COMMAND_PROFILE
    # compact_obs_term_scale/compact_obs_term_noise defaults below are robot-independent
    # (T1 and G1 use the same values) and are normally overwritten at train.py startup
    # with values extracted from --expert-checkpoint's own experiment config; the defaults
    # here matter mainly for direct FADAConfig construction (tests, eval/finetune entry
    # points) where no expert checkpoint payload is consulted.
    compact_obs_term_scale: dict[str, float] | None = dataclasses.field(
        default_factory=lambda: {
            "base_ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "projected_gravity": 1.0,
        }
    )
    # Per-term noise magnitudes. Also the magnitude source for augment_obs_noise (a
    # separate, always-on training-batch augmentation). train.py resolves this from
    # --expert-checkpoint's own config, so it is read on the default path.
    compact_obs_term_noise: dict[str, float] | None = dataclasses.field(
        default_factory=lambda: {
            "base_ang_vel": 0.3,
            "dof_pos": 0.01,
            "dof_vel": 1.0,
            "projected_gravity": 0.2,
        }
    )

    # Planner model
    planner_mlp_hidden_dims: tuple[int, ...] = (266, 266, 266)
    planner_d_model: int = 128
    planner_nhead: int = 4
    planner_num_layers: int = 3
    planner_dim_feedforward: int = 512
    planner_dropout: float = 0.1
    planner_use_learned_positional_encoding: bool = False

    # IDM model (transformer encoder/decoder; the fine-grained MLP/hybrid
    # encoder-decoder switches have been folded away — always transformer).
    idm_encoder_mlp_hidden_dims: tuple[int, ...] = (675, 675)
    idm_decoder_mlp_hidden_dims: tuple[int, ...] = (123,)
    idm_mlp_hidden_dims: tuple[int, ...] = (1024, 512, 256)
    # MLP/hybrid IDM decoder hyperparameters. Unused now that the fine-grained
    # idm_model_type switch has been folded away (always transformer).
    idm_decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
    idm_decoder_activation: str = "elu"
    idm_d_model: int = 128
    idm_nhead: int = 4
    idm_encoder_num_layers: int = 3
    idm_decoder_num_layers: int = 2
    idm_dim_feedforward: int = 512
    idm_dropout: float = 0.1
    idm_use_learned_positional_encoding: bool = False

    # Optimization
    batch_size: int = 4096
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    planner_obs_loss_coef: float = 0.0
    idm_action_loss_coef: float = 1.0
    idm_action_loss_horizon_gamma: float = 0.0
    idm_planner_pass_action_loss_coef: float = 1.0
    obs_loss_coef: float = 0.0  # internal alias for reused base trainer logging
    # normalize_io constant-folded away (always False; FADATrainer._use_io_normalization
    # is hardcoded to False for the same reason).
    mixed_batch_offline_ratio: float = 0.2
    idm_suboptimal_batch_ratio: float = 0.375
    planner_suboptimal_batch_ratio: float = 0.375
    # Fraction of the online source batch routed through trajectory_target_* (real
    # executed causality) instead of expert_future_obs_chunk (oracle shadow).
    # Applies to IDM only; planner always uses oracle-relabeled targets.
    online_trajectory_batch_ratio: float = 0.5
    # Fraction of the suboptimal source batch routed through expert_* targets
    # (oracle shadow) instead of trajectory_target_* (real suboptimal causality).
    # Applies to IDM only; planner always uses oracle-relabeled targets
    # and treats the full suboptimal budget as one expert-labeled source.
    suboptimal_expert_batch_ratio: float = 0.5
    # Warmup-only suboptimal ratios, separate from the DAgger-phase ratios above. Used
    # only when mode="offline" in _idm_batch_counts/_planner_batch_counts.
    #
    # These defaults give the within-offline ratio opt:sub_exp:sub_real = 1:1:2 for IDM
    # (0.75 x 1/3 = 1/4, 0.75 x 2/3 = 1/2), and 1:3 opt:sub for the planner.
    warmup_idm_suboptimal_batch_ratio: float = 0.75
    warmup_planner_suboptimal_batch_ratio: float = 0.75
    augment_obs_noise: bool = True
    # per-term noise override, e.g. {"projected_gravity": 0.02}; the defaults below are
    # robot-independent.
    augment_obs_noise_overrides: dict[str, float] | None = dataclasses.field(
        default_factory=lambda: {
            "base_ang_vel": 0.15,
            "projected_gravity": 0.02,
            "dof_vel": 0.4,
            "dof_pos": 0.02,
        }
    )
    augment_action_noise_level: float = 0.01  # uniform ±level on history/future actions
    augment_history_mask_ratio: float = 0.2
    augment_future_mask_ratio: float = 0.2
    validation_batches: int = 4
    val_trajectory_ratio: float = 0.15  # fraction of trajectories held out for validation
    shared_norm_source: str = "idm_visible_union"  # idm_visible_union | optimal_like | optimal_offline

    # Buffer
    replay_capacity: int = 5_000_000
    online_buffer_eviction_policy: str = "random"  # random | fifo
    online_rollout_sample_ratio: float = 0.5  # fraction of each rollout to add to online buffer
    save_online_buffer: bool = True  # save online buffer npz at end of training

    # Resume from a previous run
    resume_checkpoint: str | None = None  # path to checkpoint (model + optimizer state)
    resume_online_buffer: str | None = None  # path to online_buffer.npz

    # Warmup stage
    warmup_episodes: int = 5
    warmup_max_steps_per_episode: int = 500
    warmup_collect_steps: int | None = None
    warmup_train_steps: int = 2_000
    suboptimal_data_ratio: float = 1.0
    suboptimal_num_checkpoints: int = 20
    suboptimal_checkpoint_sampling: str = "reward_aware"  # reward_aware | logspace

    # DAgger stage
    dagger_iters: int = 30
    rollout_steps_per_iter: int = 500
    train_steps_per_iter: int = 1000
    dagger_sigma: float = 0.05
    command_resample_interval: int = 200
    strict_chunk_labels: bool = True
    strict_label_worker_timeout_s: float = 60.0

    # Eval stage
    eval_num_episodes: int = 1
    eval_max_steps: int = 500
    eval_command_resample_interval: int = 200
    eval_exp_name: str = "dagger_eval"

    # Environment / expert. No default: must be supplied via --expert-checkpoint
    # (enforced by validate() below); there is no sensible repo-relative fallback.
    expert_checkpoint: str | None = None
    simulator_override: str = "keep"  # keep | isaacgym | isaacsim
    num_envs: int = 1024
    headless: bool = True
    device: str = "cuda:0"
    seed: int = 42

    # Smart cache. These literal fallbacks only matter for direct FADAConfig
    # construction (tests, non-CLI callers); config_from_args() below derives the
    # real values from --output-dir/--run-name (see _derive_path_fields()).
    offline_cache_path: str = "logs/fada_planner_idm/cache/t1_oracle_offline_step1.npz"
    offline_suboptimal_cache_path: str = "logs/fada_planner_idm/cache/t1_oracle_offline_step1_suboptimal.npz"
    warmup_ckpt_path: str = "logs/fada_planner_idm/cache/t1_oracle_warmup_step1.pt"
    load_warmup_ckpt: bool = True
    force_warmup_train: bool = True
    force_recollect_offline: bool = False

    # Outputs
    output_dir: str = "logs/fada_planner_idm"
    run_name: str = dataclasses.field(default_factory=_default_run_name)
    save_every_iter: int = 1
    show_progress: bool = True

    # W&B
    wandb_enable: bool = True
    wandb_project: str = "fada_planner_idm"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"  # online | offline | disabled
    wandb_tags: str = "fada,planner_idm,dagger"

    @property
    def planner_suboptimal_enabled(self) -> bool:
        return float(self.planner_suboptimal_batch_ratio) > 0.0 and float(self.suboptimal_data_ratio) > 0.0

    def validate(self) -> FADAConfig:
        effective_suboptimal_batch_ratio = float(self.idm_suboptimal_batch_ratio) if self.use_generalized_idm else 0.0
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if self.pred_horizon <= 0:
            raise ValueError("pred_horizon must be positive")
        if not bool(self.predict_future_obs):
            raise ValueError("planner+idm pipeline requires predict_future_obs=True")
        if not self.planner_mlp_hidden_dims:
            raise ValueError("planner_mlp_hidden_dims must be non-empty")
        if not (0.0 <= self.idm_teacher_forcing_ratio <= 1.0):
            raise ValueError("idm_teacher_forcing_ratio must be within [0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 for mixed offline/online sampling")
        if self.batch_size % 2 != 0 and abs(self.mixed_batch_offline_ratio - 0.5) < 1e-8:
            raise ValueError("batch_size must be even when mixed_batch_offline_ratio is 0.5")
        if self.batch_size % 2 != 0 and abs(effective_suboptimal_batch_ratio - 0.5) < 1e-8:
            raise ValueError("batch_size must be even when idm_suboptimal_batch_ratio is 0.5")
        if self.planner_obs_loss_coef < 0.0:
            raise ValueError("planner_obs_loss_coef must be non-negative")
        if self.idm_action_loss_coef < 0.0:
            raise ValueError("idm_action_loss_coef must be non-negative")
        if not (0.0 <= self.idm_action_loss_horizon_gamma <= 1.0):
            raise ValueError(
                f"idm_action_loss_horizon_gamma must be in [0, 1], got {self.idm_action_loss_horizon_gamma}"
            )
        if not self.idm_encoder_mlp_hidden_dims:
            raise ValueError("idm_encoder_mlp_hidden_dims must be non-empty")
        if not self.idm_decoder_mlp_hidden_dims:
            raise ValueError("idm_decoder_mlp_hidden_dims must be non-empty")
        if self.idm_planner_pass_action_loss_coef < 0.0:
            raise ValueError("idm_planner_pass_action_loss_coef must be non-negative")
        if not (0.0 <= self.augment_history_mask_ratio < 1.0):
            raise ValueError("augment_history_mask_ratio must be in [0, 1)")
        if not (0.0 <= self.augment_future_mask_ratio < 1.0):
            raise ValueError("augment_future_mask_ratio must be in [0, 1)")
        if not (0.0 < self.mixed_batch_offline_ratio < 1.0):
            raise ValueError("mixed_batch_offline_ratio must be in (0, 1)")
        if not (0.0 <= self.idm_suboptimal_batch_ratio < 1.0):
            raise ValueError("idm_suboptimal_batch_ratio must be in [0, 1)")
        if not (0.0 <= self.planner_suboptimal_batch_ratio < 1.0):
            raise ValueError("planner_suboptimal_batch_ratio must be in [0, 1)")
        if not (0.0 <= self.online_trajectory_batch_ratio <= 1.0):
            raise ValueError("online_trajectory_batch_ratio must be in [0, 1]")
        if not (0.0 <= self.suboptimal_expert_batch_ratio <= 1.0):
            raise ValueError("suboptimal_expert_batch_ratio must be in [0, 1]")
        if not (0.0 <= self.warmup_idm_suboptimal_batch_ratio < 1.0):
            raise ValueError("warmup_idm_suboptimal_batch_ratio must be in [0, 1)")
        if not (0.0 <= self.warmup_planner_suboptimal_batch_ratio < 1.0):
            raise ValueError("warmup_planner_suboptimal_batch_ratio must be in [0, 1)")
        if self.validation_batches <= 0:
            raise ValueError("validation_batches must be positive")
        if self.shared_norm_source not in {"idm_visible_union", "optimal_like", "optimal_offline"}:
            raise ValueError("shared_norm_source must be one of: idm_visible_union, optimal_like, optimal_offline")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.online_buffer_eviction_policy not in {"fifo", "random"}:
            raise ValueError("online_buffer_eviction_policy must be one of: fifo, random")
        if not (0.0 < self.online_rollout_sample_ratio <= 1.0):
            raise ValueError("online_rollout_sample_ratio must be in (0, 1]")
        if self.warmup_episodes <= 0:
            raise ValueError("warmup_episodes must be positive")
        if self.warmup_max_steps_per_episode <= 0:
            raise ValueError("warmup_max_steps_per_episode must be positive")
        if self.warmup_collect_steps is not None and self.warmup_collect_steps <= 0:
            raise ValueError("warmup_collect_steps must be positive when set")
        if self.warmup_train_steps < 0:
            raise ValueError("warmup_train_steps must be non-negative")
        if self.suboptimal_data_ratio < 0.0:
            raise ValueError("suboptimal_data_ratio must be non-negative")
        if self.suboptimal_num_checkpoints <= 0:
            raise ValueError("suboptimal_num_checkpoints must be positive")
        if self.suboptimal_checkpoint_sampling not in {"reward_aware", "logspace"}:
            raise ValueError("suboptimal_checkpoint_sampling must be one of: reward_aware, logspace")
        if self.dagger_iters < 0:
            raise ValueError("dagger_iters must be non-negative")
        if self.rollout_steps_per_iter <= 0:
            raise ValueError("rollout_steps_per_iter must be positive")
        if self.train_steps_per_iter < 0:
            raise ValueError("train_steps_per_iter must be non-negative")
        if self.eval_num_episodes <= 0:
            raise ValueError("eval_num_episodes must be positive")
        if self.eval_max_steps <= 0:
            raise ValueError("eval_max_steps must be positive")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.command_profile = normalize_command_profile(self.command_profile)
        if self.save_every_iter <= 0:
            raise ValueError("save_every_iter must be positive")
        if self.strict_label_worker_timeout_s <= 0.0:
            raise ValueError("strict_label_worker_timeout_s must be positive")
        if self.simulator_override not in {"keep", "isaacgym", "isaacsim"}:
            raise ValueError("simulator_override must be one of: keep, isaacgym, isaacsim")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be one of: online, offline, disabled")
        if (
            self.use_generalized_idm
            and str(self.offline_cache_path).strip() == str(self.offline_suboptimal_cache_path).strip()
        ):
            raise ValueError("offline_suboptimal_cache_path must differ from offline_cache_path")
        for prefix, d_model, nhead, num_layers in (
            ("planner", self.planner_d_model, self.planner_nhead, self.planner_num_layers),
            ("idm_encoder", self.idm_d_model, self.idm_nhead, self.idm_encoder_num_layers),
            ("idm_decoder", self.idm_d_model, self.idm_nhead, self.idm_decoder_num_layers),
        ):
            if d_model <= 0 or nhead <= 0 or num_layers <= 0:
                raise ValueError(f"{prefix} d_model/nhead/num_layers must be positive")
            if d_model % nhead != 0:
                raise ValueError(f"{prefix} d_model must be divisible by {prefix} nhead")

        if self.expert_checkpoint is None or str(self.expert_checkpoint).strip() == "":
            raise ValueError("expert_checkpoint must be provided")
        return self


_BOOL_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "on"})
_BOOL_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n", "off"})


def parse_bool_flag(value: str) -> bool:
    """argparse ``type=`` for explicit ``--flag true`` / ``--flag false`` booleans.

    Not ``store_true``/``store_false``: every override flag below uses ``default=None``
    as a "not supplied" sentinel, so an un-passed flag leaves the FADAConfig dataclass
    default untouched (see ``config_from_args``). ``store_true`` cannot express "leave
    the default alone" for a field whose default is already True, and ``store_false``
    cannot for a False default.
    """
    token = str(value).strip().lower()
    if token in _BOOL_TRUE_TOKENS:
        return True
    if token in _BOOL_FALSE_TOKENS:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (true/false), got {value!r}")


def parse_term_float_map(value: str) -> dict[str, float]:
    """argparse ``type=`` for the one dict-valued overridable field.

    Spelling: comma-separated ``term=value`` pairs, e.g.
    ``--augment-obs-noise-overrides dof_pos=0.02,projected_gravity=0.02``. The map
    replaces the default wholesale (it is not merged into it), matching how
    ``FADAConfig.augment_obs_noise_overrides`` is consumed.
    """
    result: dict[str, float] = {}
    for chunk in str(value).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "=" not in token:
            raise argparse.ArgumentTypeError(f"expected comma-separated 'term=value' pairs, got {value!r}")
        key, _, raw = token.partition("=")
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"empty term name in {value!r}")
        try:
            parsed = float(raw.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"non-numeric value for term {key!r} in {value!r}") from exc
        # Per-term augmentation noise levels: finite and >= 0 (a symmetric scale).
        if not math.isfinite(parsed) or parsed < 0.0:
            raise argparse.ArgumentTypeError(
                f"noise level for term {key!r} must be a finite float >= 0, got {parsed!r} in {value!r}"
            )
        result[key] = parsed
    if not result:
        raise argparse.ArgumentTypeError(f"expected at least one 'term=value' pair, got {value!r}")
    return result


# Every FADAConfig field a user may reasonably want to change from the command line,
# as (field name, argparse type, choices or None, help). Flag spelling is the field
# name with underscores turned into dashes, so `--lr`, `--batch-size`,
# `--dagger-iters`, ... map 1:1 onto FADAConfig attributes.
#
# Fields NOT here, and why each has no flag:
#   * obs_dim / act_dim / cmd_dim / compact_obs_term_scale / compact_obs_term_noise
#     -- resolved at startup from the expert checkpoint and its environment
#     (train.py::_resolve_dimensions / _extract_compact_scale_noise_from_expert_checkpoint
#     overwrite whatever was configured).
#   * predict_future_obs -- validate() rejects False outright; the planner+IDM
#     pipeline has no other mode.
#   * strict_chunk_labels -- every reader spells it
#     `strict_chunk_labels or predict_future_obs` (train.py::main, trainer.py's two
#     `_use_teacher_aligned_labels` assignments), and `predict_future_obs` is
#     mandatory-true by the line above, so the disjunction is a constant. The field
#     stays: it is written into the checkpoint's `cfg` payload.
#   * planner_mlp_hidden_dims / idm_encoder_mlp_hidden_dims / idm_decoder_mlp_hidden_dims
#     / idm_mlp_hidden_dims / idm_decoder_hidden_dims / idm_decoder_activation /
#     obs_loss_coef / save_online_buffer / eval_num_episodes /
#     eval_command_resample_interval / eval_exp_name -- read by nothing on the release
#     path (the MLP/hybrid model switches were constant-folded away, and
#     obs_loss_coef's only consumer, _compute_batch_losses, is reachable only from
#     the caller-less _estimate_validation_losses).
#   * wandb_run_name -- train.py::main() unconditionally overwrites it with the
#     timestamp-aligned run name; use --run-name.
#   * idm_use_teacher_forcing / idm_teacher_forcing_ratio /
#     idm_detach_planner_future_obs -- the supported training step is
#     `_train_step_separate_pass` (trainer.py), and it passes explicit per-pass
#     overrides for all three: the IDM pass calls
#     `_compute_idm_batch_loss_from_source_batches(..., force_teacher_forcing_ratio=1.0)`
#     and the planner pass calls `_compute_idm_batch_loss_from_batch(...,
#     use_teacher_forcing=False, teacher_forcing_ratio=0.0, detach_override=False)`.
#     `_resolve_idm_future_obs` reads the cfg fields only when its override argument
#     is None, which on that path never happens, so the fields have a reader but no
#     reachable one. The fields themselves stay: they are serialized into the
#     checkpoint's `cfg` payload and into the exported ONNX metadata, which
#     `eval_checkpoint.py` reads back, and they are the defaults for callers driving
#     the loss helpers directly (which the trainer unit tests do).
#   * expert_checkpoint / run_name / output_dir / device / wandb_mode -- already flags.
_OVERRIDABLE_FIELDS: tuple[tuple[str, object, tuple[str, ...] | None, str], ...] = (
    # ── Sequence setup ────────────────────────────────────────────────────────
    (
        "history_len",
        int,
        None,
        "History length fed to planner and IDM. The IDM always consumes both obs and action "
        "history; the planner consumes obs history, and action history only under "
        "--planner-use-action-history (default false).",
    ),
    (
        "pred_horizon",
        int,
        None,
        "Number of future steps the planner predicts and the IDM emits actions for. NOT the "
        "number of steps supervised: --idm-action-loss-horizon-gamma (default 0.0) weights step "
        "0 alone, so at the shipped default only the first action is supervised.",
    ),
    (
        "planner_use_action_history",
        parse_bool_flag,
        None,
        "Feed action history to the planner. Default false, and false in both released students.",
    ),
    ("planner_predict_delta", parse_bool_flag, None, "Planner predicts deltas from the current obs."),
    ("use_generalized_idm", parse_bool_flag, None, "Use the generalized (chunked) IDM batch composition."),
    (
        "idm_use_current_command_for_history",
        parse_bool_flag,
        None,
        "Broadcast current_command into IDM history tokens. Default false, and false in both "
        "released students, so the IDM history carries no command at the shipped default.",
    ),
    ("command_profile", str, None, "Command component profile (see fada/common/current_command.py)."),
    # ── Planner model ─────────────────────────────────────────────────────────
    ("planner_d_model", int, None, "Planner transformer width."),
    ("planner_nhead", int, None, "Planner attention heads (must divide --planner-d-model)."),
    ("planner_num_layers", int, None, "Planner transformer layers."),
    ("planner_dim_feedforward", int, None, "Planner feedforward width."),
    ("planner_dropout", float, None, "Planner dropout."),
    ("planner_use_learned_positional_encoding", parse_bool_flag, None, "Learned (not sinusoidal) planner pos. enc."),
    # ── IDM model ─────────────────────────────────────────────────────────────
    ("idm_d_model", int, None, "IDM transformer width."),
    ("idm_nhead", int, None, "IDM attention heads (must divide --idm-d-model)."),
    ("idm_encoder_num_layers", int, None, "IDM history-encoder layers."),
    ("idm_decoder_num_layers", int, None, "IDM decoder layers."),
    ("idm_dim_feedforward", int, None, "IDM feedforward width."),
    ("idm_dropout", float, None, "IDM dropout."),
    ("idm_use_learned_positional_encoding", parse_bool_flag, None, "Learned (not sinusoidal) IDM pos. enc."),
    # ── Optimization ──────────────────────────────────────────────────────────
    ("batch_size", int, None, "Training batch size (planner and IDM passes)."),
    ("lr", float, None, "AdamW learning rate for both the planner and IDM optimizers."),
    ("weight_decay", float, None, "AdamW weight decay."),
    ("grad_clip", float, None, "Gradient-norm clip; <=0 disables clipping."),
    (
        "planner_obs_loss_coef",
        float,
        None,
        "Weight on the planner future-observation loss. Default 0.0: the planner is trained by the "
        "IDM action loss in the planner pass, not by its own observation loss.",
    ),
    ("idm_action_loss_coef", float, None, "Weight on the IDM action loss."),
    (
        "idm_action_loss_horizon_gamma",
        float,
        None,
        "Per-horizon discount gamma^t for the IDM action loss, in [0,1]. Default 0.0, which is "
        "single-step supervision (0^0 == 1, every later step weight 0), not a mild discount.",
    ),
    ("idm_planner_pass_action_loss_coef", float, None, "Weight on the IDM action loss in the planner pass."),
    ("mixed_batch_offline_ratio", float, None, "Offline share of each mixed DAgger batch, in (0,1)."),
    ("idm_suboptimal_batch_ratio", float, None, "Suboptimal share of the IDM batch, in [0,1)."),
    ("planner_suboptimal_batch_ratio", float, None, "Suboptimal share of the planner batch, in [0,1)."),
    ("online_trajectory_batch_ratio", float, None, "Share of the online IDM batch using real executed causality."),
    ("suboptimal_expert_batch_ratio", float, None, "Share of the suboptimal IDM batch using oracle-shadow targets."),
    ("warmup_idm_suboptimal_batch_ratio", float, None, "Warmup-only suboptimal share of the IDM batch."),
    ("warmup_planner_suboptimal_batch_ratio", float, None, "Warmup-only suboptimal share of the planner batch."),
    # ── Augmentation ──────────────────────────────────────────────────────────
    ("augment_obs_noise", parse_bool_flag, None, "Apply per-term observation noise to training batches."),
    (
        "augment_obs_noise_overrides",
        parse_term_float_map,
        None,
        "Per-term augmentation noise as comma-separated term=value pairs "
        "(e.g. 'dof_pos=0.02,projected_gravity=0.02'). Replaces the default map wholesale; "
        "unknown term names are rejected at startup.",
    ),
    ("augment_action_noise_level", float, None, "Uniform +/- noise level on history/future actions."),
    ("augment_history_mask_ratio", float, None, "Random history-token mask ratio, in [0,1)."),
    ("augment_future_mask_ratio", float, None, "Random future-token mask ratio, in [0,1)."),
    # ── Validation / normalization ────────────────────────────────────────────
    ("validation_batches", int, None, "Batches per validation pass."),
    ("val_trajectory_ratio", float, None, "Fraction of trajectories held out for validation."),
    (
        "shared_norm_source",
        str,
        ("idm_visible_union", "optimal_like", "optimal_offline"),
        "Which data the shared normalization statistics are estimated from.",
    ),
    # ── Replay buffers ────────────────────────────────────────────────────────
    ("replay_capacity", int, None, "Per-buffer replay capacity in transitions."),
    ("online_buffer_eviction_policy", str, ("random", "fifo"), "Online replay-buffer eviction policy."),
    ("online_rollout_sample_ratio", float, None, "Fraction of each rollout added to the online buffer, in (0,1]."),
    # ── Resume ────────────────────────────────────────────────────────────────
    ("resume_checkpoint", str, None, "Resume from this student checkpoint (model + optimizer state)."),
    (
        "resume_online_buffer",
        str,
        None,
        "Resume the online buffer from this online_buffer.npz. Only meaningful together with "
        "--resume-checkpoint: the restore lives inside that branch, so this flag alone is "
        "ignored. Passing it without --resume-checkpoint, or pointing it at a path that does "
        "not exist, is now an error rather than a silent empty buffer.",
    ),
    # ── Warmup stage ──────────────────────────────────────────────────────────
    (
        "warmup_episodes",
        int,
        None,
        "Warmup collection PARALLEL ROLLOUTS, not episodes: each one steps all --num-envs "
        "environments for --warmup-max-steps-per-episode steps, so the collection target is "
        "warmup_episodes * num_envs * warmup_max_steps_per_episode transition slots and up to "
        "num_envs episodes finish per rollout. At the shipped defaults (num_envs 1024, "
        "500 steps) a value of 1 means ~512,000 slots and up to 1,024 completed episodes.",
    ),
    ("warmup_max_steps_per_episode", int, None, "Max steps per warmup episode."),
    (
        "warmup_collect_steps",
        int,
        None,
        "Explicit warmup collection step budget, replacing the --warmup-episodes derivation "
        "(default: derived). A floor, not a cap: collection runs whole parallel rollouts and "
        "stops once the total is reached, so the result can exceed it by up to one rollout "
        "(num_envs * warmup_max_steps_per_episode steps).",
    ),
    ("warmup_train_steps", int, None, "Gradient steps taken during warmup."),
    ("suboptimal_data_ratio", float, None, "Suboptimal-to-optimal offline data ratio; 0 disables the source."),
    ("suboptimal_num_checkpoints", int, None, "Number of weak-policy checkpoints sampled from the expert history."),
    (
        "suboptimal_checkpoint_sampling",
        str,
        ("reward_aware", "logspace"),
        "How the suboptimal checkpoints are drawn from the oracle run's history.",
    ),
    # ── DAgger stage ──────────────────────────────────────────────────────────
    ("dagger_iters", int, None, "Number of DAgger iterations."),
    ("rollout_steps_per_iter", int, None, "Environment steps collected per DAgger iteration."),
    ("train_steps_per_iter", int, None, "Gradient steps per DAgger iteration."),
    ("dagger_sigma", float, None, "Student action noise used during DAgger rollouts."),
    ("command_resample_interval", int, None, "Steps between command resamples during rollouts."),
    ("strict_label_worker_timeout_s", float, None, "Timeout for the strict-label worker process."),
    ("eval_max_steps", int, None, "max_eval_steps written into the training environment config."),
    # ── Environment ───────────────────────────────────────────────────────────
    ("simulator_override", str, ("keep", "isaacgym", "isaacsim"), "Override the expert's simulator backend."),
    ("num_envs", int, None, "Parallel environments used for rollout collection."),
    ("headless", parse_bool_flag, None, "Run the simulator headless."),
    ("seed", int, None, "Global RNG seed."),
    # ── Cache / warmup reuse ──────────────────────────────────────────────────
    (
        "load_warmup_ckpt",
        parse_bool_flag,
        None,
        "Reuse a cached warmup checkpoint when one exists. Both this and --force-warmup-train "
        "default true and the load condition is `load_warmup_ckpt and not force_warmup_train`, "
        "so at the shipped defaults the cached checkpoint is NEVER loaded. Reusing one takes "
        "--force-warmup-train false; this flag only turns the reuse further off.",
    ),
    (
        "force_warmup_train",
        parse_bool_flag,
        None,
        "Run warmup training rather than reusing a cached warmup checkpoint. True by default, "
        "which is what makes --load-warmup-ckpt inert at the shipped defaults; pass false to "
        "let --load-warmup-ckpt take effect.",
    ),
    ("force_recollect_offline", parse_bool_flag, None, "Ignore the offline cache and re-collect it."),
    ("offline_cache_path", str, None, "Override the derived optimal offline cache .npz path."),
    ("offline_suboptimal_cache_path", str, None, "Override the derived suboptimal offline cache .npz path."),
    ("warmup_ckpt_path", str, None, "Override the derived warmup checkpoint .pt path."),
    # ── Outputs ───────────────────────────────────────────────────────────────
    ("save_every_iter", int, None, "Save a student checkpoint every N DAgger iterations."),
    ("show_progress", parse_bool_flag, None, "Show tqdm progress bars."),
    # ── W&B ───────────────────────────────────────────────────────────────────
    ("wandb_enable", parse_bool_flag, None, "Enable Weights & Biases logging."),
    ("wandb_project", str, None, "Override the derived W&B project name."),
    ("wandb_entity", str, None, "W&B entity."),
    ("wandb_group", str, None, "W&B group."),
    ("wandb_tags", str, None, "Comma-separated W&B tags."),
)

# Numeric domain for every numeric flag above, enforced in config_from_args() (see
# fada/common/cli_validation.py). A flag missing from this map is unchecked;
# tests/fada/test_cli_numeric_validation.py asserts the map covers every int/float
# field of _OVERRIDABLE_FIELDS.
_NUMERIC_DOMAINS: dict[str, cli_validation.NumericRange] = {
    # ── Sequence setup ────────────────────────────────────────────────────────
    "history_len": cli_validation.POSITIVE_INT,
    "pred_horizon": cli_validation.POSITIVE_INT,
    # ── Planner / IDM model ───────────────────────────────────────────────────
    "planner_d_model": cli_validation.POSITIVE_INT,
    "planner_nhead": cli_validation.POSITIVE_INT,
    "planner_num_layers": cli_validation.POSITIVE_INT,
    "planner_dim_feedforward": cli_validation.POSITIVE_INT,
    "planner_dropout": cli_validation.UNIT_HALF_OPEN,
    "idm_d_model": cli_validation.POSITIVE_INT,
    "idm_nhead": cli_validation.POSITIVE_INT,
    "idm_encoder_num_layers": cli_validation.POSITIVE_INT,
    "idm_decoder_num_layers": cli_validation.POSITIVE_INT,
    "idm_dim_feedforward": cli_validation.POSITIVE_INT,
    "idm_dropout": cli_validation.UNIT_HALF_OPEN,
    # ── Optimization ──────────────────────────────────────────────────────────
    "batch_size": cli_validation.POSITIVE_INT,
    "lr": cli_validation.POSITIVE_FLOAT,
    "weight_decay": cli_validation.NON_NEGATIVE_FLOAT,
    # grad_clip's documented contract is "<=0 disables clipping", so the sign is free --
    # only non-finite values are rejected.
    "grad_clip": cli_validation.FINITE_FLOAT,
    "planner_obs_loss_coef": cli_validation.non_negative_float_disabled_by_zero(
        "0 drops the planner observation loss term."
    ),
    "idm_action_loss_coef": cli_validation.non_negative_float_disabled_by_zero("0 drops the IDM action loss term."),
    "idm_action_loss_horizon_gamma": cli_validation.PROBABILITY,
    "idm_planner_pass_action_loss_coef": cli_validation.non_negative_float_disabled_by_zero(
        "0 drops the planner-pass IDM action loss term."
    ),
    "mixed_batch_offline_ratio": cli_validation.UNIT_OPEN,
    "idm_suboptimal_batch_ratio": cli_validation.UNIT_HALF_OPEN,
    "planner_suboptimal_batch_ratio": cli_validation.UNIT_HALF_OPEN,
    "online_trajectory_batch_ratio": cli_validation.PROBABILITY,
    "suboptimal_expert_batch_ratio": cli_validation.PROBABILITY,
    "warmup_idm_suboptimal_batch_ratio": cli_validation.UNIT_HALF_OPEN,
    "warmup_planner_suboptimal_batch_ratio": cli_validation.UNIT_HALF_OPEN,
    # ── Augmentation ──────────────────────────────────────────────────────────
    # `_apply_action_noise` (trainer_batching.py) returns the actions untouched for any
    # `level <= 0.0`, so every non-positive value means "off".
    "augment_action_noise_level": cli_validation.finite_float(
        disabled_note="Any value <= 0 disables action-history noise."
    ),
    "augment_history_mask_ratio": cli_validation.UNIT_HALF_OPEN,
    "augment_future_mask_ratio": cli_validation.UNIT_HALF_OPEN,
    # ── Validation / normalization ────────────────────────────────────────────
    "validation_batches": cli_validation.POSITIVE_INT,
    # The split is guarded by `if val_ratio > 0.0:` in three places (trainer.py), so any
    # non-positive ratio is the original's "no validation split". 1.0 stays rejected: it is
    # degenerate (everything held out), not "off".
    "val_trajectory_ratio": cli_validation.finite_float(
        hi=1.0,
        hi_inclusive=False,
        disabled_note="Any value <= 0 disables the trajectory-level validation split.",
    ),
    # ── Replay buffers ────────────────────────────────────────────────────────
    "replay_capacity": cli_validation.POSITIVE_INT,
    "online_rollout_sample_ratio": cli_validation.UNIT_LEFT_OPEN,
    # ── Warmup stage ──────────────────────────────────────────────────────────
    "warmup_episodes": cli_validation.POSITIVE_INT,
    "warmup_max_steps_per_episode": cli_validation.POSITIVE_INT,
    "warmup_collect_steps": cli_validation.POSITIVE_INT,
    "warmup_train_steps": cli_validation.non_negative_int_disabled_by_zero("0 skips warmup training entirely."),
    "suboptimal_data_ratio": cli_validation.non_negative_float_disabled_by_zero(
        "0 disables the weak-policy (suboptimal) data source."
    ),
    "suboptimal_num_checkpoints": cli_validation.POSITIVE_INT,
    # ── DAgger stage ──────────────────────────────────────────────────────────
    "dagger_iters": cli_validation.non_negative_int_disabled_by_zero("0 stops after warmup."),
    "rollout_steps_per_iter": cli_validation.POSITIVE_INT,
    "train_steps_per_iter": cli_validation.non_negative_int_disabled_by_zero(
        "0 collects rollouts without taking gradient steps."
    ),
    # `rollout_and_collect` perturbs the action only under `if sigma > 0.0:`
    # (trainer_rollout.py), so every non-positive sigma is the deterministic rollout.
    "dagger_sigma": cli_validation.finite_float(
        disabled_note="Any value <= 0 rolls out the deterministic student action."
    ),
    # _set_command_resampling_interval_from_steps() returns early for interval_steps <= 0,
    # leaving whatever resampling time the experiment config already carried. 0 and
    # negatives are accepted and mean that, not "never resample"; the note below states
    # it on every --help and in the run's own echo of the resolved config.
    "command_resample_interval": cli_validation.unbounded_int(
        disabled_note=(
            "Any value <= 0 leaves the experiment config's own command-resampling time in "
            "place; it does not mean 'never resample'."
        )
    ),
    "strict_label_worker_timeout_s": cli_validation.POSITIVE_FLOAT,
    "eval_max_steps": cli_validation.POSITIVE_INT,
    # ── Environment ───────────────────────────────────────────────────────────
    "num_envs": cli_validation.POSITIVE_INT,
    # numpy's legacy seeding rejects negatives outright.
    "seed": cli_validation.NON_NEGATIVE_INT,
    # ── Outputs ───────────────────────────────────────────────────────────────
    "save_every_iter": cli_validation.POSITIVE_INT,
}


# Overrides applied *after* _derive_path_fields(), which recomputes these four from
# --output-dir/--run-name.
_POST_DERIVATION_FIELDS = frozenset(
    {
        "offline_cache_path",
        "offline_suboptimal_cache_path",
        "warmup_ckpt_path",
        "wandb_project",
    }
)


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "hyperparameter overrides",
        "Every flag below defaults to the FADAConfig release value; omitting one leaves that "
        "default untouched. Booleans take an explicit true/false argument.",
    )
    for name, arg_type, choices, help_text in _OVERRIDABLE_FIELDS:
        domain = _NUMERIC_DOMAINS.get(name)
        kwargs: dict[str, object] = {
            "type": arg_type,
            "default": None,
            "dest": name,
            "help": help_text + (domain.help_suffix() if domain is not None else ""),
        }
        if choices is not None:
            kwargs["choices"] = list(choices)
        group.add_argument(f"--{name.replace('_', '-')}", **kwargs)  # type: ignore[arg-type]


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI surface for FADA (planner+IDM) DAgger training.

    The run-identifying flags (--expert-checkpoint/--run-name/--output-dir) and the
    infrastructure ones (--device/--wandb-mode) are declared explicitly below; every
    other overridable FADAConfig field gets a flag generated from
    ``_OVERRIDABLE_FIELDS``. All of those use ``default=None`` as an "unset" sentinel,
    so the dataclass defaults remain byte-for-byte what they are today unless a flag is
    passed. T1 and G1 share every one of these defaults, so there is no per-robot
    override baked in either.
    """
    parser = argparse.ArgumentParser(description="FADA (planner+IDM) DAgger training entrypoint")
    parser.add_argument("--expert-checkpoint", type=str, required=True, help="Oracle PPO checkpoint path")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run identity used to derive the offline cache/warmup checkpoint paths and "
        "the W&B project (e.g. 't1_loco', 'g1_loco'). Do not include a timestamp or a "
        "'_planner_idm' suffix -- both are added automatically. Optional: defaults to "
        "the expert checkpoint's run directory name with any leading 'YYYYmmdd_HHMMSS_' "
        "stamp stripped, plus a '_fada' suffix (e.g. an oracle run "
        "'.../20250101_120000_t1_23dof_waist50_oracle' gives "
        "'t1_23dof_waist50_oracle_fada'). A fresh timestamp is prefixed onto the "
        "run directory either way, so repeated runs never overwrite each other.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Parent log directory for this run. Optional: defaults to the absolute path "
        "'<expert checkpoint's run directory>_fada_dagger' -- a sibling of the oracle run "
        "directory, not a subdirectory of it, so nothing is written into the (possibly "
        "read-only or re-downloadable) released oracle checkpoint tree. The final run "
        "directory is '<output-dir>/<timestamp>_<run-name>'.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
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


def _derive_path_fields(cfg: FADAConfig) -> None:
    """Derive the run-identity-scoped path/naming fields from output_dir/run_name.

    Must run with cfg.run_name/cfg.output_dir still equal to the raw --run-name/
    --output-dir CLI values (i.e. before train.py's timestamp-prefixing renames
    cfg.run_name for the run directory) so that offline_cache_path lands on a
    timestamp-free, run-name-scoped path:
    {output_dir}/dataset/{run_name}_planner_idm_optimal.npz. Getting this wrong causes
    a silent, expensive re-collection of the offline cache instead of reusing it.
    """
    cfg.offline_cache_path = f"{cfg.output_dir}/dataset/{cfg.run_name}_planner_idm_optimal.npz"
    cfg.offline_suboptimal_cache_path = f"{cfg.output_dir}/dataset/{cfg.run_name}_planner_idm_optimal_suboptimal.npz"
    cfg.warmup_ckpt_path = f"{cfg.output_dir}/{cfg.run_name}_planner_idm_warmup.pt"
    cfg.wandb_project = f"{cfg.run_name.split('_')[0]}_loco_dagger"
    cfg.wandb_run_name = cfg.run_name


def _validate_resume_flags(overrides: dict[str, object]) -> None:
    """`--resume-online-buffer` is only read from inside the `--resume-checkpoint` branch.

    Two silent no-ops came out of that, both of which look like a successful resume:

    * passed alone, the flag is never read at all (`train.py`'s restore lives inside
      `if cfg.resume_checkpoint:`), so training starts from scratch with an empty online
      buffer and nothing in the output says the flag was ignored;
    * passed with `--resume-checkpoint` but pointing at a path that does not exist, the
      restore printed a note and continued with an empty buffer -- a typo in the path is
      indistinguishable from a run that genuinely had no buffer to restore.

    A path the user typed is a statement that the file matters, so both are errors. The
    *derived* fallbacks inside the resume branch (`online_buffer_latest.npz` next to the
    checkpoint, then `online_buffer.npz` a level up) are untouched: those are guesses the
    code makes on the user's behalf, and their absence is genuinely not an error.
    """
    buffer_path = overrides.get("resume_online_buffer")
    if buffer_path is None or str(buffer_path).strip() == "":
        return
    if not overrides.get("resume_checkpoint"):
        raise ValueError(
            "--resume-online-buffer requires --resume-checkpoint: the online-buffer restore only "
            "runs when a resume checkpoint is being loaded, so on its own this flag is ignored and "
            "training silently starts with an empty online buffer."
        )
    if not Path(str(buffer_path)).expanduser().exists():
        raise ValueError(f"--resume-online-buffer path does not exist: {buffer_path}")


def config_from_args(args: argparse.Namespace) -> FADAConfig:
    # --run-name / --output-dir are optional; when omitted they are derived from
    # --expert-checkpoint (see default_run_name_from_expert /
    # default_output_dir_from_expert). Derivation happens before _derive_path_fields(),
    # so the cache/warmup/W&B paths are built from the resolved values.
    run_name = getattr(args, "run_name", None)
    if run_name is None or str(run_name).strip() == "":
        run_name = default_run_name_from_expert(args.expert_checkpoint)
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None or str(output_dir).strip() == "":
        output_dir = default_output_dir_from_expert(args.expert_checkpoint)
    # Hyperparameter overrides: only fields the user actually passed are forwarded to
    # FADAConfig, so every un-passed field keeps its dataclass default untouched
    # (`default=None` is the "unset" sentinel -- see _add_override_arguments).
    overrides: dict[str, object] = {}
    post_derivation_overrides: dict[str, object] = {}
    for name, _arg_type, _choices, _help in _OVERRIDABLE_FIELDS:
        value = getattr(args, name, None)
        if value is None:
            continue
        if name in _POST_DERIVATION_FIELDS:
            post_derivation_overrides[name] = value
        else:
            overrides[name] = value
    # Range-check every numeric flag the user passed, before anything is constructed or
    # written. FADAConfig.validate() covers a subset, runs on the *resolved* config (so
    # it cannot name the flag), and does not reject nan/inf.
    cli_validation.validate_numeric_values(_NUMERIC_DOMAINS, {**overrides, **post_derivation_overrides})
    _validate_resume_flags(overrides)
    cfg = FADAConfig(
        expert_checkpoint=args.expert_checkpoint,
        run_name=run_name,
        output_dir=output_dir,
        device=args.device,
        wandb_mode=args.wandb_mode,
        **overrides,  # type: ignore[arg-type]
    )
    _derive_path_fields(cfg)
    # Applied after _derive_path_fields(), which recomputes these four from
    # output_dir/run_name, so an explicit flag wins.
    for name, value in post_derivation_overrides.items():
        setattr(cfg, name, value)
    applied = {**overrides, **post_derivation_overrides}
    print(
        "[CLI] FADAConfig overrides: "
        + (
            ", ".join(f"{name}={value!r}" for name, value in sorted(applied.items()))
            or "(none -- all release defaults)"
        )
    )
    return cfg.validate()
