from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values import logger as logger_defaults
from holosoma.config_values import simulator as simulator_defaults
from holosoma.fada.common import cli_validation
from holosoma.fada.common.compact_obs import (
    canonicalize_compact_term_noise,
    canonicalize_compact_term_scale,
    get_compact_obs_preprocess,
)
from holosoma.fada.common.current_command import (
    DEFAULT_COMMAND_PROFILE,
    command_components_for_profile,
    normalize_command_profile,
)
from holosoma.fada.common.norm_stats import DEFAULT_NORM_EPS, validate_norm_stats
from holosoma.fada.common.utils import _find_motor_gains
from holosoma.fada.planner_idm.eval import evaluate_policy
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.safe_torch_load import load_checkpoint as safe_load_checkpoint
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


class _EvalDefaults:
    """Hardcoded values for the flags that are not on the CLI.

    Overriding one means editing this class or calling the underlying functions directly.
    The ablation-only knobs (force_scale_eval_level, fixed_ee_force_*,
    future_obs_mask_*) default to disabled; they are not part of the eval/export path.
    """

    simulator_override = "isaacsim"
    num_envs = 1024
    headless = True
    device = "cuda:0"
    seed = 42
    num_episodes = 1
    max_steps = 1000
    command_resample_interval = 200
    open_loop_deploy_steps = 1
    show_progress = True
    force_scale_eval_level = -1
    fixed_ee_force_left: tuple[float, float, float] | None = None
    fixed_ee_force_right: tuple[float, float, float] | None = None
    future_obs_mask_mode = "none"
    future_obs_mask_step: int | None = None
    future_obs_mask_fill = "current"
    export_onnx = True
    onnx_output_path: str | None = None
    eval_exp_name = "dagger_eval"


_BOOL_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "on"})
_BOOL_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n", "off"})


def parse_bool_flag(value: str) -> bool:
    """argparse ``type=`` for explicit ``--flag true`` / ``--flag false`` booleans.

    Same "unset sentinel" reasoning as the other two FADA entry points: every override
    flag below defaults to ``None`` so an un-passed flag leaves the ``_EvalDefaults``
    value untouched.
    """
    token = str(value).strip().lower()
    if token in _BOOL_TRUE_TOKENS:
        return True
    if token in _BOOL_FALSE_TOKENS:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (true/false), got {value!r}")


def parse_force_triplet(value: str) -> tuple[float, float, float]:
    """argparse ``type=`` for the two tuple-valued fields (``fixed_ee_force_*``).

    Spelling: three comma-separated floats, e.g. ``--fixed-ee-force-left 0,0,-30``.
    """
    parts = [token.strip() for token in str(value).split(",") if token.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected three comma-separated floats (x,y,z), got {value!r}")
    try:
        triplet = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"non-numeric component in {value!r}") from exc
    # A non-finite force is applied to the simulated end effector every step; it makes
    # the whole rollout nan without any error being raised.
    if not all(math.isfinite(component) for component in triplet):
        raise argparse.ArgumentTypeError(f"force components must be finite (nan/inf rejected), got {value!r}")
    return triplet


# `_EvalDefaults` fields a user may reasonably want to change, as
# (field name, argparse type, choices or None, help). Every field of `_EvalDefaults` is
# covered: the standard eval/export knobs plus the ablation-only ones
# (force_scale_eval_level, fixed_ee_force_*, future_obs_mask_*), which are ablation-only
# by default value, not dead -- `main()` reads all of them and forwards them into
# `evaluate_policy`.
_OVERRIDABLE_DEFAULTS: tuple[tuple[str, object, tuple[str, ...] | None, str], ...] = (
    # ── Environment ───────────────────────────────────────────────────────────
    (
        "simulator_override",
        str,
        ("keep", "isaacsim", "isaacgym", "mjwarp"),
        "Simulator backend the checkpoint's experiment config is evaluated in.",
    ),
    ("num_envs", int, None, "Parallel eval environments."),
    ("headless", parse_bool_flag, None, "Run the simulator headless."),
    ("device", str, None, "Torch/simulator device."),
    ("seed", int, None, "Eval RNG seed."),
    # ── Episode schedule ──────────────────────────────────────────────────────
    ("num_episodes", int, None, "Number of eval episodes."),
    ("max_steps", int, None, "Steps per eval episode (also written as max_eval_steps into the env config)."),
    ("command_resample_interval", int, None, "Steps between command resamples during eval."),
    ("open_loop_deploy_steps", int, None, "Actions executed per IDM inference; must not exceed pred_horizon."),
    ("show_progress", parse_bool_flag, None, "Show tqdm progress bars."),
    # ── Export ────────────────────────────────────────────────────────────────
    ("export_onnx", parse_bool_flag, None, "Export the checkpoint to ONNX after evaluating."),
    ("onnx_output_path", str, None, "Explicit ONNX output path (default: <output-dir>/planner_idm_policy.onnx)."),
    ("eval_exp_name", str, None, "Subdirectory name used when --output-dir is omitted."),
    # ── Ablation knobs ────────────────────────────────────────────────────────
    ("force_scale_eval_level", int, None, "External-force ablation level; <0 disables it."),
    ("fixed_ee_force_left", parse_force_triplet, None, "Constant left end-effector force as 'x,y,z' newtons."),
    ("fixed_ee_force_right", parse_force_triplet, None, "Constant right end-effector force as 'x,y,z' newtons."),
    (
        "future_obs_mask_mode",
        str,
        ("none", "full", "drop", "only", "prefix"),
        "Planner future-observation masking ablation mode.",
    ),
    (
        "future_obs_mask_step",
        int,
        None,
        "Horizon index the drop/only/prefix mask modes act on. ONE-BASED: valid values are "
        "1..pred_horizon (6 in both released students), so 1 is the first predicted step and "
        "there is no 0. Required by --future-obs-mask-mode drop/only/prefix and ignored by "
        "none/full.",
    ),
    ("future_obs_mask_fill", str, ("current", "nearest"), "What masked future observations are replaced with."),
)


# Numeric domain for every numeric flag of this entry point, enforced uniformly in
# resolve_eval_defaults() (see fada/common/cli_validation.py).
_NUMERIC_DOMAINS: dict[str, cli_validation.NumericRange] = {
    "num_envs": cli_validation.POSITIVE_INT,
    "seed": cli_validation.NON_NEGATIVE_INT,
    "num_episodes": cli_validation.POSITIVE_INT,
    "max_steps": cli_validation.POSITIVE_INT,
    # `_set_command_resampling_interval_from_steps` (trainer.py) returns early for any
    # interval <= 0, so 0 and negatives select "leave the experiment config's own
    # resampling time alone" and are accepted. See config.py's copy.
    "command_resample_interval": cli_validation.unbounded_int(
        disabled_note=(
            "Any value <= 0 leaves the experiment config's own command-resampling time in "
            "place; it does not mean 'never resample'."
        )
    ),
    "open_loop_deploy_steps": cli_validation.POSITIVE_INT,
    # The consumer is `if int(D.force_scale_eval_level) >= 0:` (see evaluate_policy's
    # caller below), so every negative level disables the ablation.
    "force_scale_eval_level": cli_validation.unbounded_int(
        disabled_note="Any value < 0 disables the force ablation."
    ),
    # A ONE-based horizon index into [1, pred_horizon] -- `_apply_future_obs_mask`
    # (eval.py) rejects anything below 1. The upper bound is checked there, once the
    # checkpoint's pred_horizon is known.
    "future_obs_mask_step": cli_validation.POSITIVE_INT,
}


def resolve_eval_defaults(args: argparse.Namespace) -> type[_EvalDefaults]:
    """Return the `_EvalDefaults`-shaped object `main()` reads its config from.

    Returns `_EvalDefaults` itself when no override flag was passed, and a thin subclass
    carrying only the overridden attributes otherwise.
    """
    overrides = {
        name: getattr(args, name)
        for name, _arg_type, _choices, _help in _OVERRIDABLE_DEFAULTS
        if getattr(args, name, None) is not None
    }
    # Range-check every numeric flag the user passed, before the simulator is brought
    # up.
    cli_validation.validate_numeric_values(_NUMERIC_DOMAINS, overrides)
    # The pairing between the two mask flags, checked here rather than in
    # `_apply_future_obs_mask`, which runs per rollout step (after the checkpoint is
    # loaded and the simulator is up). The upper bound (<= pred_horizon) stays there,
    # since pred_horizon comes from the checkpoint.
    mask_mode = str(overrides.get("future_obs_mask_mode", _EvalDefaults.future_obs_mask_mode)).lower()
    if mask_mode in {"drop", "only", "prefix"} and overrides.get("future_obs_mask_step") is None:
        raise ValueError(
            f"--future-obs-mask-mode {mask_mode} requires --future-obs-mask-step "
            "(a one-based horizon index in [1, pred_horizon])"
        )
    if not overrides:
        return _EvalDefaults
    return type("_EvalDefaultsWithCLIOverrides", (_EvalDefaults,), overrides)


# Everything the eval CLI does not recognise is handed to `tyro.cli(ExperimentConfig,
# ...)` verbatim, and tyro only type-checks. Those overrides are applied *after*
# `_build_eval_config` wrote the validated values, so the passthrough spelling wins.
# Three rules are enforced on the resolved config:
#
#   1. the fields the eval CLI owns are re-checked against the same domains, so the two
#      spellings cannot disagree;
#   2. every float anywhere in the config tree must be finite (`nan` passes every
#      ordinary comparison);
#   3. the simulator's own step/timing counts must be positive: frame rate, control
#      decimation, substep count and render interval are ints, so rule 2 does not reach
#      them. `max_episode_length_s` is included as a positive float.
#
# Nothing beyond those three is bounded: the passthrough exists so a reader can override
# anything in the experiment config, and physics coefficients, friction and gains are the
# simulator's domain, not this CLI's.
#
# The third tuple element is a note appended to the field name in the error message,
# naming the equivalent flag or the reason for the bound.
_PASSTHROUGH_DOMAINS: tuple[tuple[str, cli_validation.NumericRange, str], ...] = (
    ("training.num_envs", _NUMERIC_DOMAINS["num_envs"], "same quantity as --num-envs"),
    ("training.seed", _NUMERIC_DOMAINS["seed"], "same quantity as --seed"),
    ("training.max_eval_steps", _NUMERIC_DOMAINS["max_steps"], "same quantity as --max-steps"),
    ("simulator.config.sim.fps", cli_validation.POSITIVE_INT, "simulator frame rate"),
    ("simulator.config.sim.control_decimation", cli_validation.POSITIVE_INT, "physics steps per control step"),
    ("simulator.config.sim.substeps", cli_validation.POSITIVE_INT, "physics substeps per frame"),
    ("simulator.config.sim.render_interval", cli_validation.POSITIVE_INT, "frames between rendered frames"),
    ("simulator.config.sim.max_episode_length_s", cli_validation.POSITIVE_FLOAT, "episode length in seconds"),
)


def _iter_config_numbers(node: Any, path: str = "", _seen: set[int] | None = None) -> Any:
    """Yield `(dotted path, value)` for every int/float reachable in a config tree."""
    _seen = _seen if _seen is not None else set()
    if id(node) in _seen:
        return
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield path, node
        return
    if isinstance(node, (str, bytes)) or node is None:
        return
    if isinstance(node, dict):
        _seen.add(id(node))
        for key, value in node.items():
            yield from _iter_config_numbers(value, f"{path}[{key!r}]", _seen)
        return
    if isinstance(node, (list, tuple, set)):
        _seen.add(id(node))
        for index, value in enumerate(node):
            yield from _iter_config_numbers(value, f"{path}[{index}]", _seen)
        return
    if dataclasses.is_dataclass(node):
        _seen.add(id(node))
        for field in dataclasses.fields(node):
            child = getattr(node, field.name, None)
            yield from _iter_config_numbers(child, f"{path}.{field.name}" if path else field.name, _seen)


def _resolve_dotted(node: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def validate_experiment_config(eval_cfg: ExperimentConfig) -> None:
    """Range-check the resolved experiment config, after tyro's passthrough overrides.

    Raises `ValueError` naming the offending path and the equivalent flag.
    """
    for dotted, domain, note in _PASSTHROUGH_DOMAINS:
        try:
            value = _resolve_dotted(eval_cfg, dotted)
        except AttributeError:  # a config revision that no longer carries the field
            continue
        if value is None:
            # `None` is these fields' "unset" state in a stock ExperimentConfig
            # (`_build_eval_config` fills them). Unset is not out of range.
            continue
        flag = "--" + dotted.replace("_", "-")
        domain.check(value, field=f"{dotted} ({note})", flag=flag)

    non_finite = [
        (path, value)
        for path, value in _iter_config_numbers(eval_cfg)
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if non_finite:
        listed = ", ".join(f"{path}={value!r}" for path, value in non_finite[:6])
        more = "" if len(non_finite) <= 6 else f" (and {len(non_finite) - 6} more)"
        raise ValueError(
            f"Non-finite value(s) in the resolved experiment config: {listed}{more}. "
            "NaN and inf pass every ordinary comparison, so the eval would run to completion "
            "and write an eval_summary.json of NaNs instead of failing."
        )


def report_config_passthrough(
    base_cfg: ExperimentConfig,
    resolved_cfg: ExperimentConfig,
    remaining_args: list[str],
) -> list[str]:
    """Print, and return, what the un-validated tyro passthrough actually changed.

    The eval CLI prints its own overrides separately. Returned as well as printed so
    tests can assert on it.
    """
    if not remaining_args:
        return []
    base_numbers = dict(_iter_config_numbers(base_cfg))
    changed: list[str] = []
    for path, value in _iter_config_numbers(resolved_cfg):
        if path in base_numbers and base_numbers[path] != value:
            changed.append(f"{path}: {base_numbers[path]!r} -> {value!r}")
    print(
        "[CLI] experiment-config passthrough: " + " ".join(remaining_args),
        flush=True,
    )
    for line in changed:
        print(f"[CLI]   numeric change {line}", flush=True)
    return changed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate planner+IDM FADA checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where eval artifacts (eval_summary.json, planner_idm_policy.onnx) are written. "
        "Optional: defaults to '<checkpoint's run directory>/eval/"
        f"{_EvalDefaults.eval_exp_name}', where the run directory is the checkpoint's parent "
        "(or its grandparent when the checkpoint sits in a 'checkpoints/' subdirectory).",
    )
    group = parser.add_argument_group(
        "eval overrides",
        "Every flag below defaults to the _EvalDefaults release value; omitting one leaves that "
        "default untouched. Booleans take an explicit true/false argument. Unrecognized flags are "
        "passed through to tyro as ExperimentConfig overrides.",
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
    return parser


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = safe_load_checkpoint(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(payload)}")
    required = ("cfg", "obs_dim", "act_dim", "cmd_dim")
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {missing}")
    if "planner_state_dict" not in payload and "model_state_dict" not in payload:
        raise KeyError("Checkpoint missing planner_state_dict/model_state_dict")
    if "experiment_config" not in payload:
        raise KeyError(
            "Checkpoint missing 'experiment_config'. Please retrain/export with the updated planner_idm trainer."
        )
    return payload


def _build_eval_config(
    experiment_config_payload: dict[str, Any],
    *,
    simulator_override: str,
    num_envs: int,
    headless: bool,
    max_steps: int,
    seed: int,
) -> ExperimentConfig:
    base_cfg = ExperimentConfig(**experiment_config_payload)
    # Match the DAgger training/data-collection path: evaluate the student in the
    # oracle training environment, not in the oracle's visualization eval_overrides.
    # The student replay buffer was collected from this train-time distribution.
    eval_cfg = base_cfg

    if simulator_override == "isaacgym":
        eval_cfg = dataclasses.replace(eval_cfg, simulator=simulator_defaults.isaacgym)
    elif simulator_override == "isaacsim":
        eval_cfg = dataclasses.replace(eval_cfg, simulator=simulator_defaults.isaacsim)
    elif simulator_override == "mjwarp":
        eval_cfg = dataclasses.replace(eval_cfg, simulator=simulator_defaults.mjwarp)

    training_cfg = dataclasses.replace(
        eval_cfg.training,
        num_envs=int(num_envs),
        headless=bool(headless),
        max_eval_steps=int(max_steps),
        export_onnx=False,
        collect_data=False,
        seed=int(seed),
    )
    return dataclasses.replace(eval_cfg, training=training_cfg, logger=logger_defaults.disabled)


def _ensure_eval_randomization_param_defaults(eval_cfg: ExperimentConfig) -> None:
    mass_term = eval_cfg.randomization.setup_terms.get("mass_randomizer")
    if mass_term is None:
        return
    mass_term.params.setdefault("fixed_payload_body_names", [])
    mass_term.params.setdefault("fixed_payload_added_masses", [])
    mass_term.params.setdefault("replace_fixed_payload_dr", True)


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
    return {mean_key: mean, std_key: std}


def _extract_checkpoint_compact_obs_config(payload: dict[str, Any]) -> dict[str, Any]:
    compact_payload = payload.get("compact_obs")
    if not isinstance(compact_payload, dict):
        raise RuntimeError("Checkpoint missing required 'compact_obs' payload.")
    if "term_scale" not in compact_payload or "term_noise" not in compact_payload:
        raise RuntimeError("Checkpoint compact_obs payload missing required term_scale/term_noise fields.")
    term_scale = canonicalize_compact_term_scale(compact_payload.get("term_scale"))
    term_noise = canonicalize_compact_term_noise(compact_payload.get("term_noise"))
    add_noise = bool(compact_payload.get("add_noise", False))
    noise_seed_raw = compact_payload.get("noise_seed")
    noise_seed = int(noise_seed_raw) if noise_seed_raw is not None else None
    return {
        "term_scale": term_scale,
        "term_noise": term_noise,
        "add_noise": add_noise,
        "noise_seed": noise_seed,
    }


def _build_model_from_checkpoint(
    payload: dict[str, Any],
    device: torch.device,
) -> tuple[PlannerIDMPolicy, dict[str, int | bool | str]]:
    train_cfg = payload["cfg"]
    if not isinstance(train_cfg, dict):
        raise ValueError(f"Expected payload['cfg'] to be dict, got {type(train_cfg)}")

    obs_dim = int(payload["obs_dim"])
    act_dim = int(payload["act_dim"])
    cmd_dim = int(payload["cmd_dim"])
    history_len = int(train_cfg["history_len"])
    pred_horizon = int(train_cfg["pred_horizon"])

    idm_encoder_num_layers = int(train_cfg.get("idm_encoder_num_layers", train_cfg.get("idm_num_layers", 6)))
    idm_decoder_num_layers = int(train_cfg.get("idm_decoder_num_layers", train_cfg.get("idm_num_layers", 6)))

    model = PlannerIDMPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        cmd_dim=cmd_dim,
        history_len=history_len,
        pred_horizon=pred_horizon,
        planner_d_model=int(train_cfg["planner_d_model"]),
        planner_nhead=int(train_cfg["planner_nhead"]),
        planner_num_layers=int(train_cfg["planner_num_layers"]),
        planner_dim_feedforward=int(train_cfg["planner_dim_feedforward"]),
        planner_dropout=float(train_cfg["planner_dropout"]),
        planner_use_learned_positional_encoding=bool(train_cfg["planner_use_learned_positional_encoding"]),
        idm_use_current_command_for_history=bool(train_cfg.get("idm_use_current_command_for_history", False)),
        planner_use_action_history=bool(train_cfg.get("planner_use_action_history", False)),
        idm_d_model=int(train_cfg["idm_d_model"]),
        idm_nhead=int(train_cfg["idm_nhead"]),
        idm_encoder_num_layers=idm_encoder_num_layers,
        idm_decoder_num_layers=idm_decoder_num_layers,
        idm_dim_feedforward=int(train_cfg["idm_dim_feedforward"]),
        idm_dropout=float(train_cfg["idm_dropout"]),
        idm_use_learned_positional_encoding=bool(train_cfg["idm_use_learned_positional_encoding"]),
        planner_predict_delta=bool(train_cfg.get("planner_predict_delta", False)),
    ).to(device)

    planner_state_dict = payload.get("planner_state_dict")
    idm_state_dict = payload.get("idm_state_dict")
    if isinstance(planner_state_dict, dict) and isinstance(idm_state_dict, dict):
        model.load_split_state_dict(
            planner_state_dict=planner_state_dict,
            idm_state_dict=idm_state_dict,
        )
    else:
        model.load_compatible_state_dict(payload["model_state_dict"])
    model.eval()

    dims: dict[str, int | bool | str] = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "cmd_dim": cmd_dim,
        "command_profile": normalize_command_profile(str(train_cfg.get("command_profile", DEFAULT_COMMAND_PROFILE))),
        "history_len": history_len,
        "pred_horizon": pred_horizon,
        "predict_future_obs": True,
        "policy_arch": "planner_idm",
        "planner_use_action_history": bool(train_cfg.get("planner_use_action_history", False)),
        "idm_use_current_command_for_history": bool(train_cfg.get("idm_use_current_command_for_history", False)),
        "idm_use_teacher_forcing": bool(train_cfg.get("idm_use_teacher_forcing", True)),
        "idm_teacher_forcing_ratio": float(train_cfg.get("idm_teacher_forcing_ratio", 1.0)),
        "idm_encoder_num_layers": idm_encoder_num_layers,
        "idm_decoder_num_layers": idm_decoder_num_layers,
        "fdm_enabled": False,
        "planner_predict_delta": bool(train_cfg.get("planner_predict_delta", False)),
        "fdm_predict_delta": False,
    }
    return model, dims


def _default_output_dir(ckpt_path: Path, eval_exp_name: str) -> Path:
    if ckpt_path.parent.name == "checkpoints":
        run_dir = ckpt_path.parent.parent
    else:
        run_dir = ckpt_path.parent
    return run_dir / "eval" / eval_exp_name


class _PlannerIDMOnnxWrapper(torch.nn.Module):
    def __init__(
        self,
        *,
        model: PlannerIDMPolicy,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        io_normalization: bool,
        obs_norm_stats: dict[str, torch.Tensor] | None,
        action_norm_stats: dict[str, torch.Tensor] | None,
        command_norm_stats: dict[str, torch.Tensor] | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cmd_dim = int(cmd_dim)
        self.io_normalization = bool(io_normalization)

        self.register_buffer("obs_mean", torch.zeros((1, 1, self.obs_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("obs_std", torch.ones((1, 1, self.obs_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("act_mean", torch.zeros((1, 1, self.act_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("act_std", torch.ones((1, 1, self.act_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("cmd_mean", torch.zeros((1, self.cmd_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("cmd_std", torch.ones((1, self.cmd_dim), dtype=torch.float32), persistent=False)

        if self.io_normalization:
            if obs_norm_stats is None or action_norm_stats is None or command_norm_stats is None:
                raise RuntimeError("Missing normalization stats for io_normalization=True ONNX export.")
            self.obs_mean.copy_(obs_norm_stats["obs_mean"].view(1, 1, -1))
            self.obs_std.copy_(obs_norm_stats["obs_std"].view(1, 1, -1))
            self.act_mean.copy_(action_norm_stats["action_mean"].view(1, 1, -1))
            self.act_std.copy_(action_norm_stats["action_std"].view(1, 1, -1))
            self.cmd_mean.copy_(command_norm_stats["command_mean"].view(1, -1))
            self.cmd_std.copy_(command_norm_stats["command_std"].view(1, -1))

    def forward(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        history_valid_mask: torch.Tensor,
        teacher_future_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        history_obs_model = history_obs
        history_act_model = history_act
        current_command_model = current_command
        teacher_future_obs_model = teacher_future_obs
        if self.io_normalization:
            history_obs_model = (history_obs - self.obs_mean) / self.obs_std
            history_act_model = (history_act - self.act_mean) / self.act_std
            current_command_model = (current_command - self.cmd_mean) / self.cmd_std
            teacher_future_obs_model = (teacher_future_obs - self.obs_mean) / self.obs_std

        pred_actions, pred_future_obs = self.model(
            history_obs_model,
            history_act_model,
            current_command_model,
            history_valid_mask=history_valid_mask,
            return_obs=True,
            skip_empty_history_check=True,
        )
        idm_current_command_model = self.model._resolve_idm_current_command(
            current_command_model,
            idm_use_current_command_for_history=self.model.idm_use_current_command_for_history,
        )
        idm_teacher_actions = self.model.predict_idm_actions(
            history_obs_model,
            history_act_model,
            idm_current_command_model,
            teacher_future_obs_model,
            history_valid_mask=history_valid_mask,
        )

        if self.io_normalization:
            pred_actions = pred_actions * self.act_std + self.act_mean
            pred_future_obs = pred_future_obs * self.obs_std + self.obs_mean
            idm_teacher_actions = idm_teacher_actions * self.act_std + self.act_mean
        return pred_actions, pred_future_obs, idm_teacher_actions


def _export_planner_idm_onnx(
    *,
    model: PlannerIDMPolicy,
    onnx_output_path: Path,
    dims: dict[str, int | bool | str],
    io_normalization: bool,
    obs_norm_stats: dict[str, torch.Tensor] | None,
    action_norm_stats: dict[str, torch.Tensor] | None,
    command_norm_stats: dict[str, torch.Tensor] | None,
    compact_obs_term_scale: dict[str, float],
    compact_obs_term_noise: dict[str, float],
    payload: dict[str, Any],
) -> Path:
    wrapper = _PlannerIDMOnnxWrapper(
        model=model,
        obs_dim=int(dims["obs_dim"]),
        act_dim=int(dims["act_dim"]),
        cmd_dim=int(dims["cmd_dim"]),
        io_normalization=bool(io_normalization),
        obs_norm_stats=obs_norm_stats,
        action_norm_stats=action_norm_stats,
        command_norm_stats=command_norm_stats,
    ).eval()

    device = next(model.parameters()).device
    batch = 1
    history_len = int(dims["history_len"])
    obs_dim = int(dims["obs_dim"])
    act_dim = int(dims["act_dim"])
    cmd_dim = int(dims["cmd_dim"])
    dummy_history_obs = torch.zeros((batch, history_len, obs_dim), device=device, dtype=torch.float32)
    dummy_history_act = torch.zeros((batch, history_len, act_dim), device=device, dtype=torch.float32)
    dummy_current_command = torch.zeros((batch, cmd_dim), device=device, dtype=torch.float32)
    dummy_history_valid_mask = torch.ones((batch, history_len), device=device, dtype=torch.bool)
    dummy_teacher_future_obs = torch.zeros(
        (batch, int(dims["pred_horizon"]), obs_dim),
        device=device,
        dtype=torch.float32,
    )

    input_names = ["history_obs", "history_act", "current_command", "history_valid_mask", "teacher_future_obs"]
    output_names = ["actions", "pred_future_obs", "idm_teacher_actions"]
    export_args: tuple[torch.Tensor, ...] = (
        dummy_history_obs,
        dummy_history_act,
        dummy_current_command,
        dummy_history_valid_mask,
        dummy_teacher_future_obs,
    )
    dynamic_axes: dict[str, dict[int, str]] = {
        "history_obs": {0: "batch"},
        "history_act": {0: "batch"},
        "current_command": {0: "batch"},
        "history_valid_mask": {0: "batch"},
        "teacher_future_obs": {0: "batch"},
        "actions": {0: "batch"},
        "pred_future_obs": {0: "batch"},
        "idm_teacher_actions": {0: "batch"},
    }

    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        export_args,
        str(onnx_output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx

    model_proto = onnx.load(str(onnx_output_path))
    compact_preprocess = get_compact_obs_preprocess(
        term_scale=compact_obs_term_scale,
        term_noise=compact_obs_term_noise,
    )
    runtime_metadata = {
        # compact_obs_add_noise (env-extraction noise) constant-folded to False: it was
        # never enabled in any trained checkpoint.
        "compact_obs_add_noise": False,
        "compact_obs_term_order": list(compact_preprocess.get("term_order", [])),
        "compact_obs_space": "scaled_compact_obs",
        "io_norm_fused": bool(io_normalization),
        "predict_future_obs": True,
        "policy_mode": "fada",
        "command_input_name": "current_command",
        "command_profile": str(dims.get("command_profile", DEFAULT_COMMAND_PROFILE)),
        "command_components": list(
            command_components_for_profile(str(dims.get("command_profile", DEFAULT_COMMAND_PROFILE)))
        ),
        "command_input_semantics": "current_command_broadcast_to_history_tokens",
        "planner_use_action_history": bool(dims["planner_use_action_history"]),
        "idm_use_current_command_for_history": bool(dims.get("idm_use_current_command_for_history", False)),
        "idm_use_teacher_forcing": bool(dims.get("idm_use_teacher_forcing", True)),
        "idm_teacher_forcing_ratio": float(dims.get("idm_teacher_forcing_ratio", 1.0)),
        "idm_inverse_metric_definition": "predict future action chunk a_t:t+K-1 from real trajectory history and future obs o_t+1:t+K",
        "fdm_enabled": bool(dims.get("fdm_enabled", False)),
        "fdm_forward_metric_definition": "predict future observation chunk o_t+1:t+K from real trajectory history and future actions a_t:t+K-1",
        "planner_predict_delta": bool(dims.get("planner_predict_delta", False)),
        "fdm_predict_delta": bool(dims.get("fdm_predict_delta", False)),
    }
    dims_metadata = {
        "obs_dim": int(dims["obs_dim"]),
        "act_dim": int(dims["act_dim"]),
        "cmd_dim": int(dims["cmd_dim"]),
        "command_profile": str(dims.get("command_profile", DEFAULT_COMMAND_PROFILE)),
        "history_len": int(dims["history_len"]),
        "pred_horizon": int(dims["pred_horizon"]),
    }
    metadata_payload: dict[str, Any] = {
        "transformer_dims": dims_metadata,
        "transformer_runtime": runtime_metadata,
        "transformer_preprocess": compact_preprocess,
    }
    kp_kd = _find_motor_gains(payload.get("experiment_config"))
    if kp_kd is not None:
        metadata_payload["kp"] = kp_kd[0]
        metadata_payload["kd"] = kp_kd[1]
    # The metadata written is exactly `metadata_payload`; there is no caller-supplied
    # passthrough.

    for key, value in metadata_payload.items():
        meta = model_proto.metadata_props.add()
        meta.key = str(key)
        meta.value = json.dumps(value)

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(onnx_output_path))

    # Postcheck. `dynamic_axes` above declares axis 0 of every input and output dynamic,
    # but `torch.onnx.export` traces at batch 1 and with `nhead=1` the attention reshapes
    # constant-fold to that traced batch. The postcheck therefore EXECUTES the graph at
    # batch 1 (what the trace saw) and batch 2 (which only works if the axis is really
    # dynamic), not just constructs a session.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_output_path))
    except Exception as exc:
        raise RuntimeError(f"ONNX export succeeded but ONNX Runtime load failed: {exc}") from exc

    def _feed(n: int) -> dict[str, np.ndarray]:
        return {
            "history_obs": np.zeros((n, history_len, obs_dim), dtype=np.float32),
            "history_act": np.zeros((n, history_len, act_dim), dtype=np.float32),
            "current_command": np.zeros((n, cmd_dim), dtype=np.float32),
            "history_valid_mask": np.ones((n, history_len), dtype=bool),
            "teacher_future_obs": np.zeros((n, int(dims["pred_horizon"]), obs_dim), dtype=np.float32),
        }

    for probe_batch in (1, 2):
        try:
            outputs = session.run(output_names, _feed(probe_batch))
        except Exception as exc:
            raise RuntimeError(
                f"ONNX export succeeded and loaded, but running it at batch {probe_batch} failed: {exc}. "
                "The graph declares a dynamic batch axis it does not actually have -- with "
                "planner_nhead/idm_nhead of 1 the attention reshapes fold to the traced batch size. "
                "Re-export with nhead >= 2, or drop the dynamic batch declaration."
            ) from exc
        for out_name, array in zip(output_names, outputs):
            if int(np.asarray(array).shape[0]) != probe_batch:
                raise RuntimeError(
                    f"ONNX output {out_name!r} has batch {np.asarray(array).shape[0]} for an input "
                    f"batch of {probe_batch}: the declared dynamic batch axis is not honoured."
                )

    return onnx_output_path


def main() -> None:
    args, remaining_args = _build_arg_parser().parse_known_args()
    # `_EvalDefaults` itself when no override flag was passed; a thin subclass carrying
    # only the passed values otherwise (see resolve_eval_defaults).
    D = resolve_eval_defaults(args)
    print(
        "[CLI] eval overrides: "
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
    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = _load_checkpoint(ckpt_path)
    eval_cfg_base = _build_eval_config(
        payload["experiment_config"],
        simulator_override=D.simulator_override,
        num_envs=D.num_envs,
        headless=D.headless,
        max_steps=D.max_steps,
        seed=D.seed,
    )
    _ensure_eval_randomization_param_defaults(eval_cfg_base)
    eval_cfg = tyro.cli(
        ExperimentConfig,
        default=eval_cfg_base,
        args=remaining_args,
        description="Evaluate planner+idm transformer checkpoint with optional config overrides.",
        config=TYRO_CONIFG,
    )
    report_config_passthrough(eval_cfg_base, eval_cfg, remaining_args)
    validate_experiment_config(eval_cfg)

    simulation_app = None
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None else _default_output_dir(ckpt_path, D.eval_exp_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        env, resolved_device, simulation_app = setup_simulation_environment(eval_cfg, device=D.device)
        device = torch.device(resolved_device)
        # Match the DAgger training path more closely. During training, constructing
        # the expert PPO wrapper performs one env.reset_all() before student rollout
        # collection starts; the standalone evaluator otherwise begins from the
        # environment's first reset after simulator creation.
        _ = env.reset_all()
        # Model construction initializes modules before loading checkpoint weights.
        # Preserve simulator/eval RNG so different architectures see the same reset.
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        model, dims = _build_model_from_checkpoint(payload, device=device)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        normalization_payload = payload.get("normalization")
        io_normalization = (
            bool(normalization_payload.get("io_enabled", False)) if isinstance(normalization_payload, dict) else False
        )
        obs_norm_stats = _extract_norm_stats(
            payload,
            key="obs_norm_stats",
            mean_key="obs_mean",
            std_key="obs_std",
            dim=int(dims["obs_dim"]),
            required=bool(io_normalization),
            device=device,
        )
        action_norm_stats = _extract_norm_stats(
            payload,
            key="action_norm_stats",
            mean_key="action_mean",
            std_key="action_std",
            dim=int(dims["act_dim"]),
            required=bool(io_normalization),
            device=device,
        )
        command_norm_stats = _extract_norm_stats(
            payload,
            key="command_norm_stats",
            mean_key="command_mean",
            std_key="command_std",
            dim=int(dims["cmd_dim"]),
            required=bool(io_normalization),
            device=device,
        )

        ckpt_compact_cfg = _extract_checkpoint_compact_obs_config(payload)
        compact_obs_term_scale = dict(ckpt_compact_cfg["term_scale"])
        compact_obs_term_noise = dict(ckpt_compact_cfg["term_noise"])

        force_scale = None
        if int(D.force_scale_eval_level) >= 0:
            force_scale = float(D.force_scale_eval_level) / 2.0
            if not hasattr(env, "apply_force_scale"):
                raise AttributeError("--force-scale-eval-level requested but env has no apply_force_scale attribute")
            env.apply_force_scale.fill_(force_scale)

        state_log_path = evaluate_policy(
            env,
            model,
            history_len=int(dims["history_len"]),
            pred_horizon=int(dims["pred_horizon"]),
            act_dim=int(dims["act_dim"]),
            cmd_dim=int(dims["cmd_dim"]),
            command_profile=str(dims.get("command_profile", DEFAULT_COMMAND_PROFILE)),
            predict_future_obs=bool(dims.get("predict_future_obs", True)),
            policy_arch=str(dims.get("policy_arch", "planner_idm")),
            idm_use_current_command_for_history=bool(dims.get("idm_use_current_command_for_history", False)),
            obs_norm_stats=obs_norm_stats,
            action_norm_stats=action_norm_stats,
            command_norm_stats=command_norm_stats,
            io_normalization=bool(io_normalization),
            num_episodes=int(D.num_episodes),
            max_steps=int(D.max_steps),
            command_resample_interval=int(D.command_resample_interval),
            open_loop_deploy_steps=int(D.open_loop_deploy_steps),
            output_dir=output_dir,
            device=device,
            compact_obs_term_scale=compact_obs_term_scale,
            force_scale=force_scale,
            fixed_ee_force_left=D.fixed_ee_force_left,
            fixed_ee_force_right=D.fixed_ee_force_right,
            show_progress=bool(D.show_progress),
            future_obs_mask_mode=str(D.future_obs_mask_mode),
            future_obs_mask_step=D.future_obs_mask_step,
            future_obs_mask_fill=str(D.future_obs_mask_fill),
        )

        onnx_path: Path | None = None
        if bool(D.export_onnx):
            onnx_path = (
                Path(D.onnx_output_path).expanduser()
                if D.onnx_output_path is not None
                else output_dir / "planner_idm_policy.onnx"
            )
            onnx_path = _export_planner_idm_onnx(
                model=model,
                onnx_output_path=onnx_path,
                dims=dims,
                io_normalization=bool(io_normalization),
                obs_norm_stats=obs_norm_stats,
                action_norm_stats=action_norm_stats,
                command_norm_stats=command_norm_stats,
                compact_obs_term_scale=compact_obs_term_scale,
                compact_obs_term_noise=compact_obs_term_noise,
                payload=payload,
            )

        summary = {
            "checkpoint": str(ckpt_path),
            "output_dir": str(output_dir),
            "state_log_path": str(state_log_path),
            "num_envs": int(D.num_envs),
            "num_episodes": int(D.num_episodes),
            "max_steps": int(D.max_steps),
            "seed": int(D.seed),
            "command_resample_interval": int(D.command_resample_interval),
            "open_loop_deploy_steps": int(D.open_loop_deploy_steps),
            "simulator_override": D.simulator_override,
            "io_normalization": bool(io_normalization),
            "compact_obs_term_scale": compact_obs_term_scale,
            "compact_obs_term_noise": compact_obs_term_noise,
            "compact_obs_add_noise": False,
            "force_scale_eval_level": int(D.force_scale_eval_level),
            "force_scale": force_scale,
            "fixed_ee_force_left": D.fixed_ee_force_left,
            "fixed_ee_force_right": D.fixed_ee_force_right,
            "future_obs_mask_mode": str(D.future_obs_mask_mode),
            "future_obs_mask_step": D.future_obs_mask_step,
            "future_obs_mask_fill": str(D.future_obs_mask_fill),
            "checkpoint_compact_obs": ckpt_compact_cfg,
            "eval_primary_metric": "idm_inverse_action_mse",
            "idm_inverse_metric_definition": "predict future action chunk a_t:t+K-1 from real trajectory history and future obs o_t+1:t+K",
            "idm_use_teacher_forcing": bool(payload["cfg"].get("idm_use_teacher_forcing", True)),
            "idm_teacher_forcing_ratio": float(payload["cfg"].get("idm_teacher_forcing_ratio", 1.0)),
            "fdm_enabled": bool(payload["cfg"].get("fdm_enabled", False)),
            "fdm_forward_metric_definition": "predict future observation chunk o_t+1:t+K from real trajectory history and future actions a_t:t+K-1",
            "planner_predict_delta": bool(payload["cfg"].get("planner_predict_delta", False)),
            "fdm_predict_delta": bool(payload["cfg"].get("fdm_predict_delta", False)),
            "export_onnx": bool(D.export_onnx),
            "onnx_output_path": str(onnx_path) if onnx_path is not None else None,
            "config_override_args": remaining_args,
            "resolved_device": str(device),
            **dims,
        }
        (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        err_msg = f"Eval failed: {exc}\n\n{traceback.format_exc()}"
        try:
            (output_dir / "eval_error.txt").write_text(err_msg, encoding="utf-8")
            print("Traceback also written to:", output_dir / "eval_error.txt", file=sys.stderr)
        except Exception as write_err:
            print(f"Could not write eval_error.txt: {write_err}", file=sys.stderr)
        raise
    finally:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
