from __future__ import annotations

import copy
import dataclasses
import json
import multiprocessing as mp
import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values import logger as logger_defaults
from holosoma.config_values import simulator as simulator_defaults
from holosoma.fada.common.backbone import HistoryPolicy
from holosoma.fada.common.compact_obs import (
    COMPACT_TERM_ORDER,
    canonicalize_compact_term_noise,
    canonicalize_compact_term_scale,
    extract_compact_obs,
    get_compact_obs_preprocess,
)
from holosoma.fada.common.current_command import (
    extract_current_command_torch,
    extract_tracking_command_torch,
)
from holosoma.fada.common.dataset import (
    ReplayBuffer,
    ReplayCacheLegacyMetadataMissingError,
    ReplayCacheMetadataMismatchError,
)
from holosoma.fada.common.lora_utils import (
    _missing_wandb_credentials_message,
    _wandb_credentials_preflight,
    _WandbCommGuard,
)
from holosoma.fada.planner_idm.config import FADAConfig
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.managers.observation.terms import locomotion as obs_terms
from holosoma.fada.common.utils import resolve_algo_class
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.safe_torch_import import F, optim, torch
from holosoma.utils.safe_torch_load import load_checkpoint as safe_load_checkpoint
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clip_actions(env: Any, actions: torch.Tensor) -> torch.Tensor:
    control_cfg = env.robot_config.control
    if bool(getattr(control_cfg, "clip_actions", False)):
        clip_val = float(control_cfg.action_clip_value)
        return torch.clamp(actions, -clip_val, clip_val)
    return actions


def _extract_compact_obs(
    env: Any,
    *,
    compact_obs_term_scale: dict[str, float] | None,
) -> torch.Tensor:
    return extract_compact_obs(
        env,
        term_scale=compact_obs_term_scale,
    )


def _extract_current_command(env: Any) -> torch.Tensor:
    return extract_current_command_torch(env, profile=getattr(env, "transformer_command_profile", None))


def _extract_command_for_logging(env: Any) -> torch.Tensor:
    return extract_tracking_command_torch(env)


def _set_command_resampling_interval_from_steps(env: Any, *, interval_steps: int) -> None:
    """Align command resampling with manager callback logic.

    PPO training uses command manager callbacks with a time-based interval
    (`locomotion_command_resampling_time`). FADA exposes this as
    a step interval, so we convert here to keep command distribution consistent.
    """
    if interval_steps <= 0:
        return
    if not hasattr(env, "dt"):
        return
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return
    command_cfg = getattr(command_manager, "command_cfg", None)
    if command_cfg is None:
        return
    resample_time = float(interval_steps) * float(env.dt)
    setattr(command_cfg, "locomotion_command_resampling_time", float(resample_time))


def init_wandb_run(cfg: FADAConfig, *, run_dir: Path) -> Any | None:
    if not cfg.wandb_enable or cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        # A missing wandb package with W&B requested (wandb_enable=True, mode !=
        # "disabled") raises rather than degrading, matching finetune_idm_lora's
        # _init_wandb_run. --wandb-mode disabled (or offline) opts out.
        raise RuntimeError(
            "W&B logging is on (the default) but the `wandb` package is not installed. "
            "Either install it (`pip install wandb`, or re-run scripts/setup_isaacsim.sh, "
            "which pins it), or pass `--wandb-mode disabled` to train without W&B."
        ) from exc

    run_name = cfg.wandb_run_name if cfg.wandb_run_name else cfg.run_name
    entity = cfg.wandb_entity if cfg.wandb_entity else None
    group = cfg.wandb_group if cfg.wandb_group else None
    # `--wandb-tags` spelling, matching fada/common/lora_utils.py: comma-separated,
    # whitespace-trimmed, empty entries dropped, and the kwarg omitted entirely when
    # nothing survives (W&B treats `tags=[]` and no `tags` differently on resume).
    tags = [part.strip() for part in str(cfg.wandb_tags or "").split(",") if part.strip()]

    # Check credentials before calling wandb.init(), so the try/except below can narrow
    # to CommError only. `wandb.init()` raises the same `UsageError` class for missing
    # credentials and for API misuse (e.g. an invalid project name). See
    # `_wandb_credentials_preflight` (lora_utils.py).
    preflight = _wandb_credentials_preflight(cfg.wandb_mode)
    if preflight == "missing":
        print(_missing_wandb_credentials_message(mode=cfg.wandb_mode, project=cfg.wandb_project))
        return None

    def _degrade_on_init_failure(exc: BaseException) -> None:
        print(
            "[W&B] wandb.init() failed "
            f"(mode='{cfg.wandb_mode}', project='{cfg.wandb_project}'): {exc}. "
            "Continuing training with W&B logging disabled. To avoid this, either fix your "
            "W&B credentials/network access, or pass --wandb-mode disabled to skip W&B "
            "entirely (or --wandb-mode offline to keep W&B running locally, writing run "
            "files to disk without any network calls)."
        )

    wandb_kwargs: dict[str, Any] = {
        "project": cfg.wandb_project,
        "entity": entity,
        "group": group,
        "name": run_name,
        "mode": cfg.wandb_mode,
        "config": dataclasses.asdict(cfg),
        "dir": str(run_dir),
        "reinit": True,
    }
    if tags:
        wandb_kwargs["tags"] = tags

    try:
        run = wandb.init(**wandb_kwargs)
    except wandb.errors.CommError as exc:
        _degrade_on_init_failure(exc)
        return None
    except wandb.errors.UsageError as exc:
        if preflight != "unknown":
            # The preflight confirmed credentials are present, so this UsageError is
            # API misuse (bad/mutually-exclusive kwargs).
            raise
        # The preflight could not determine credential state, so UsageError is also
        # treated as a possible missing-credentials signal.
        _degrade_on_init_failure(exc)
        return None
    # _WandbCommGuard protects .log()/.finish() the way this try/except protects
    # .init(): a mid-run W&B communication failure stops logging instead of raising out
    # of the training loop. Its docstring lists which exceptions are swallowed.
    return _WandbCommGuard(run, wandb)


def _safe_wandb_log(wandb_run: Any | None, payload: dict[str, float | int], *, step: int | None = None) -> None:
    if wandb_run is None:
        return
    if step is None:
        wandb_run.log(payload)
        return
    wandb_run.log(payload, step=step)


def _to_sum_and_count(value: Any) -> tuple[float, int]:
    if value is None:
        return 0.0, 0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0, 0
        tensor = value.detach().float()
        return float(tensor.sum().item()), int(tensor.numel())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0, 0
        array = np.asarray(value, dtype=np.float32)
        return float(array.sum()), int(array.size)
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value), 1
    return 0.0, 0


def _accumulate_prefixed_metrics(
    source: Any,
    *,
    prefix: str,
    sums: dict[str, float],
    counts: dict[str, int],
) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        value_sum, value_count = _to_sum_and_count(value)
        if value_count <= 0:
            continue
        full_key = f"{prefix}{key}"
        sums[full_key] = sums.get(full_key, 0.0) + float(value_sum)
        counts[full_key] = counts.get(full_key, 0) + int(value_count)


def _finalize_prefixed_metrics(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, total in sums.items():
        denom = counts.get(key, 0)
        if denom <= 0:
            continue
        out[key] = float(total / float(denom))
    return out


def _progress_range(
    total: int,
    *,
    desc: str,
    enabled: bool,
    leave: bool = False,
):
    if total <= 0:
        return range(0)
    if not enabled:
        return range(total)
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return range(total)
    return tqdm(range(total), total=total, desc=desc, leave=leave, dynamic_ncols=True)


def _nested_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {k: _nested_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nested_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_nested_to_cpu(v) for v in value)
    return copy.deepcopy(value)


def _nested_to_device(value: Any, *, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: _nested_to_device(v, device=device) for k, v in value.items()}
    if isinstance(value, list):
        return [_nested_to_device(v, device=device) for v in value]
    if isinstance(value, tuple):
        return tuple(_nested_to_device(v, device=device) for v in value)
    return copy.deepcopy(value)


def _apply_experiment_overrides(
    eval_cfg: ExperimentConfig,
    *,
    override_args: list[str] | None,
) -> ExperimentConfig:
    if not override_args:
        return eval_cfg
    return tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=override_args,
        description="FADA Planner-IDM DAgger training with optional experiment config overrides.",
        config=TYRO_CONIFG,
    )


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path is not a file or directory: {path}")

    numbered_models: list[tuple[int, float, Path]] = []
    fallback_models: list[tuple[float, Path]] = []
    for candidate in sorted(path.glob("model_*.pt")):
        stem = candidate.stem
        suffix = stem.split("model_", 1)[1] if "model_" in stem else ""
        if suffix.isdigit():
            numbered_models.append((int(suffix), candidate.stat().st_mtime, candidate))
        else:
            fallback_models.append((candidate.stat().st_mtime, candidate))
    if numbered_models:
        numbered_models.sort(key=lambda x: (x[0], x[1]))
        return numbered_models[-1][2]
    if fallback_models:
        fallback_models.sort(key=lambda x: x[0])
        return fallback_models[-1][1]
    raise FileNotFoundError(f"No checkpoint file matching 'model_*.pt' found under directory: {path}")


def _has_non_empty_path(path: str | None) -> bool:
    return path is not None and str(path).strip() != ""


def _resolve_expert_checkpoint_map(cfg: FADAConfig) -> dict[str, Path]:
    if not _has_non_empty_path(cfg.expert_checkpoint):
        raise ValueError("expert_checkpoint is required")
    return {"default": _resolve_checkpoint_path(str(cfg.expert_checkpoint))}


def _resolve_reference_checkpoint_for_env_config(cfg: FADAConfig) -> Path:
    return _resolve_checkpoint_path(str(cfg.expert_checkpoint))


def _resolve_eval_config(
    cfg: FADAConfig,
    *,
    experiment_override_args: list[str] | None = None,
) -> tuple[ExperimentConfig, Any, Any]:
    reference_checkpoint = _resolve_reference_checkpoint_for_env_config(cfg)
    checkpoint_cfg = CheckpointConfig(checkpoint=str(reference_checkpoint), eval_exp_name=None)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    # DAgger data collection should inherit the oracle's training environment by
    # default so that collected trajectories match the source policy's train-time
    # spawn / episode distribution. We still apply explicit runtime overrides
    # below (num_envs, headless, simulator override, CLI experiment overrides).
    eval_cfg = saved_cfg

    if cfg.simulator_override == "isaacgym":
        eval_cfg = dataclasses.replace(eval_cfg, simulator=simulator_defaults.isaacgym)
    elif cfg.simulator_override == "isaacsim":
        eval_cfg = dataclasses.replace(eval_cfg, simulator=simulator_defaults.isaacsim)

    eval_cfg = _apply_experiment_overrides(eval_cfg, override_args=experiment_override_args)

    # --- DAgger-specific curriculum override -----------------------------------------
    # If the saved oracle config uses TerrainLevelCurriculum, force it into "frozen
    # random init" mode for DAgger: each env starts on a uniformly random terrain row
    # and stays there. DAgger rollouts are shorter than the curriculum's
    # level_up_threshold, so without this every env stays at L0. Applied
    # unconditionally.
    curriculum_cfg = getattr(eval_cfg, "curriculum", None)
    if curriculum_cfg is not None:
        setup_terms = getattr(curriculum_cfg, "setup_terms", None) or {}
        terrain_term = setup_terms.get("terrain_level_curriculum") if isinstance(setup_terms, dict) else None
        if terrain_term is not None:
            params = dict(terrain_term.params or {})
            if not params.get("randomize_init_levels", False):
                params["randomize_init_levels"] = True
                # CurriculumTermCfg is frozen: rebuild via dataclasses.replace and rebuild
                # the parent CurriculumManagerCfg with a new setup_terms dict.
                new_terrain_term = dataclasses.replace(terrain_term, params=params)
                new_setup_terms = dict(setup_terms)
                new_setup_terms["terrain_level_curriculum"] = new_terrain_term
                new_curriculum_cfg = dataclasses.replace(curriculum_cfg, setup_terms=new_setup_terms)
                eval_cfg = dataclasses.replace(eval_cfg, curriculum=new_curriculum_cfg)
                print(
                    "[DAgger] Forcing TerrainLevelCurriculum.randomize_init_levels=True "
                    "(env spawns uniformly across all terrain rows; level up/down disabled)."
                )
    # --------------------------------------------------------------------------------

    training_cfg = dataclasses.replace(
        eval_cfg.training,
        num_envs=int(cfg.num_envs),
        headless=bool(cfg.headless),
        max_eval_steps=int(cfg.eval_max_steps),
        export_onnx=False,
        collect_data=False,
        seed=int(cfg.seed),
    )
    eval_cfg = dataclasses.replace(eval_cfg, training=training_cfg, logger=logger_defaults.disabled)
    return eval_cfg, saved_cfg, saved_wandb_path


class ExpertPolicyInterface:
    """Placeholder base interface for expert/oracle policy wrappers."""

    def act(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class ExpertPolicyWrapper(ExpertPolicyInterface):
    def __init__(self, algo: Any):
        self.algo = algo
        self.policy = algo.get_inference_policy()
        self.actor_obs_keys = self.algo.actor_obs_keys

    def act(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(self.actor_obs_keys, dict):
            policy_input = {
                actor_key: torch.cat([obs_dict[k] for k in obs_keys if isinstance(k, str)], dim=1)
                for actor_key, obs_keys in self.actor_obs_keys.items()
            }
        else:
            actor_obs_keys = [k for k in self.actor_obs_keys if isinstance(k, str)]
            actor_obs = torch.cat([obs_dict[k] for k in actor_obs_keys], dim=1)
            policy_input: dict[str, torch.Tensor] = {"actor_obs": actor_obs}

        with torch.inference_mode():
            actions = self.policy(policy_input)
        return actions

    @classmethod
    def from_checkpoint_with_env(
        cls,
        *,
        checkpoint_path: str | Path,
        env: Any,
        device: str,
        log_dir: str,
    ) -> ExpertPolicyWrapper:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_cfg = CheckpointConfig(checkpoint=str(checkpoint_path), eval_exp_name=None)
        saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
        eval_cfg = saved_cfg.get_eval_config()

        algo_cls = resolve_algo_class(eval_cfg.algo._target_)
        algo = algo_cls(
            device=device,
            env=env,
            config=eval_cfg.algo.config,
            log_dir=log_dir,
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
        algo.load(str(checkpoint_path))
        if hasattr(algo, "_eval_mode"):
            algo._eval_mode()
        return cls(algo)


def build_env_and_expert(
    cfg: FADAConfig,
    *,
    runtime_log_dir: Path,
    experiment_override_args: list[str] | None = None,
) -> tuple[Any, Any, ExpertPolicyInterface, str, dict[str, Any]]:
    eval_cfg, saved_cfg, saved_wandb_path = _resolve_eval_config(
        cfg,
        experiment_override_args=experiment_override_args,
    )

    env, resolved_device, simulation_app = setup_simulation_environment(eval_cfg, device=cfg.device)
    if hasattr(cfg, "command_profile"):
        env.transformer_command_profile = getattr(cfg, "command_profile")
    expert_checkpoint_map = _resolve_expert_checkpoint_map(cfg)
    single_checkpoint = expert_checkpoint_map["default"]
    algo_cls = resolve_algo_class(eval_cfg.algo._target_)
    expert_algo = algo_cls(
        device=resolved_device,
        env=env,
        config=eval_cfg.algo.config,
        log_dir=str(runtime_log_dir),
        multi_gpu_cfg=None,
    )
    expert_algo.setup()
    expert_algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    expert_algo.load(str(single_checkpoint))
    if hasattr(expert_algo, "_eval_mode"):
        expert_algo._eval_mode()
    expert_policy: ExpertPolicyInterface = ExpertPolicyWrapper(expert_algo)
    print(f"[Expert] single expert loaded from {single_checkpoint}")

    if hasattr(eval_cfg, "to_serializable_dict"):
        experiment_config_payload = eval_cfg.to_serializable_dict()
    else:
        experiment_config_payload = dataclasses.asdict(eval_cfg)

    return env, simulation_app, expert_policy, resolved_device, experiment_config_payload


def build_env_only(
    cfg: FADAConfig,
    *,
    experiment_override_args: list[str] | None = None,
) -> tuple[Any, Any, str]:
    eval_cfg, _, _ = _resolve_eval_config(
        cfg,
        experiment_override_args=experiment_override_args,
    )
    env, resolved_device, simulation_app = setup_simulation_environment(eval_cfg, device=cfg.device)
    if hasattr(cfg, "command_profile"):
        env.transformer_command_profile = getattr(cfg, "command_profile")
    return env, simulation_app, resolved_device


def _strict_label_worker_entry(
    conn: Any,
    *,
    cfg: FADAConfig,
    runtime_log_dir: str,
    experiment_override_args: list[str] | None,
) -> None:
    env = None
    simulation_app = None
    try:
        set_seed(int(cfg.seed))
        env, simulation_app, expert_policy, resolved_device, _ = build_env_and_expert(
            cfg,
            runtime_log_dir=Path(runtime_log_dir),
            experiment_override_args=experiment_override_args,
        )

        runtime = object.__new__(FADATrainer)
        runtime.cfg = cfg
        runtime.expert_policy = expert_policy
        runtime.device = torch.device(resolved_device)
        runtime.obs_dim = int(
            _extract_compact_obs(
                env,
                compact_obs_term_scale=cfg.compact_obs_term_scale,
            ).shape[1]
        )
        runtime.act_dim = int(env.robot_config.actions_dim)
        runtime.compact_obs_term_scale = canonicalize_compact_term_scale(cfg.compact_obs_term_scale)
        # compact_obs_add_noise (env-extraction noise) is constant-folded to False.
        # term_noise stays cfg-derived rather than hardcoded to the default dict:
        # augment_obs_noise (training-batch noise) reuses this value, and it varies per
        # expert checkpoint.
        runtime.compact_obs_term_noise = canonicalize_compact_term_noise(cfg.compact_obs_term_noise)
        runtime._use_teacher_aligned_labels = bool(cfg.strict_chunk_labels or cfg.predict_future_obs)
        runtime.strict_label_env = env
        runtime.strict_label_worker = None

        while True:
            request = conn.recv()
            op = str(request.get("op", ""))
            if op == "close":
                break
            if op != "compute":
                raise ValueError(f"Unknown strict-label worker op: {op}")

            snapshot = request["snapshot"]
            anchor_obs = _nested_to_device(request["anchor_obs"], device=runtime.device)
            runtime._restore_env_snapshot(env, snapshot)
            expert_chunk, expert_future_obs_chunk, strict_label_valid = (
                runtime._compute_strict_expert_chunk_labels(env, anchor_obs)
            )
            if expert_chunk is None or expert_future_obs_chunk is None or strict_label_valid is None:
                conn.send(
                    {
                        "ok": True,
                        "expert_chunk": None,
                        "expert_future_obs_chunk": None,
                        "strict_label_valid": None,
                    }
                )
            else:
                conn.send(
                    {
                        "ok": True,
                        "expert_chunk": expert_chunk.detach().to(device="cpu"),
                        "expert_future_obs_chunk": expert_future_obs_chunk.detach().to(device="cpu"),
                        "strict_label_valid": strict_label_valid.detach().to(device="cpu"),
                    }
                )
    except EOFError:
        pass
    except Exception:
        try:
            conn.send({"ok": False, "error": traceback.format_exc()})
        except Exception:
            pass
    finally:
        close_simulation_app(simulation_app)
        try:
            conn.close()
        except Exception:
            pass


class StrictLabelWorkerClient:
    def __init__(
        self,
        *,
        cfg: FADAConfig,
        runtime_log_dir: Path,
        experiment_override_args: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._timeout_s = float(timeout_s)
        if self._timeout_s <= 0.0:
            raise ValueError("StrictLabelWorkerClient timeout_s must be positive")
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        process = ctx.Process(
            target=_strict_label_worker_entry,
            kwargs={
                "conn": child_conn,
                "cfg": cfg,
                "runtime_log_dir": str(runtime_log_dir),
                "experiment_override_args": list(experiment_override_args) if experiment_override_args else None,
            },
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._conn = parent_conn
        self._process = process

    def compute(
        self,
        *,
        snapshot: dict[str, Any],
        anchor_obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self._process.is_alive():
            raise RuntimeError("Strict-label worker process is not alive")
        self._conn.send(
            {
                "op": "compute",
                "snapshot": _nested_to_cpu(snapshot),
                "anchor_obs": _nested_to_cpu(anchor_obs),
            }
        )
        if not self._conn.poll(self._timeout_s):
            if not self._process.is_alive():
                raise RuntimeError("Strict-label worker process died while waiting for response")
            raise RuntimeError(f"Strict-label worker timed out after {self._timeout_s:.1f}s")
        response = self._conn.recv()
        if not isinstance(response, dict) or not bool(response.get("ok", False)):
            error_text = ""
            if isinstance(response, dict):
                error_text = str(response.get("error", "")).strip()
            raise RuntimeError(f"Strict-label worker failed.\n{error_text}")
        return (
            response.get("expert_chunk"),
            response.get("expert_future_obs_chunk"),
            response.get("strict_label_valid"),
        )

    def close(self) -> None:
        try:
            if self._process.is_alive():
                self._conn.send({"op": "close"})
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)


from holosoma.fada.planner_idm.trainer_batching import _BatchingMixin
from holosoma.fada.planner_idm.trainer_logging import _LoggingMixin
from holosoma.fada.planner_idm.trainer_rollout import _RolloutMixin


class _StateMixin:
    """Base-class initialization/checkpoint-loading state that FADATrainer.__init__ and
    FADATrainer.load_student_checkpoint extend via super(). A separate MRO link, since
    both overrides in FADATrainer call up into these bodies verbatim.
    """

    def __init__(
        self,
        *,
        cfg: FADAConfig,
        student_model: HistoryPolicy,
        expert_policy: ExpertPolicyInterface,
        offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
        optimizer: optim.Optimizer,
        lr_scheduler: Any | None,
        device: torch.device,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        compact_obs_source_checkpoint: str | None = None,
        experiment_config_payload: dict[str, Any] | None = None,
        wandb_run: Any | None = None,
        strict_label_env: Any | None = None,
        strict_label_worker: StrictLabelWorkerClient | None = None,
    ) -> None:
        self.cfg = cfg
        self.student_model = student_model
        self.expert_policy = expert_policy
        self.offline_buffer = offline_buffer
        self.online_buffer = online_buffer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cmd_dim = int(cmd_dim)
        self.compact_obs_source_checkpoint = str(compact_obs_source_checkpoint) if compact_obs_source_checkpoint else ""
        self.experiment_config_payload = experiment_config_payload
        self.wandb_run = wandb_run
        self.strict_label_env = strict_label_env
        self.strict_label_worker = strict_label_worker
        if self.cfg.compact_obs_term_scale is None or self.cfg.compact_obs_term_noise is None:
            raise ValueError(
                "compact_obs term_scale/term_noise must be loaded from the expert checkpoint before trainer init."
            )
        self.compact_obs_term_scale = canonicalize_compact_term_scale(self.cfg.compact_obs_term_scale)
        # compact_obs_add_noise (env-extraction noise) is constant-folded to False.
        # term_noise stays cfg-derived rather than hardcoded to the default dict:
        # augment_obs_noise (training-batch noise) reuses this value, and it varies per
        # expert checkpoint.
        self.compact_obs_term_noise = canonicalize_compact_term_noise(self.cfg.compact_obs_term_noise)

        self.train_step_count = 0
        self.rollout_step_count = 0
        # Persist per-env episode counters across repeated collection calls so
        # (env_id, episode_id) stays globally unique inside replay buffers.
        self._online_rollout_episode_ids: torch.Tensor | None = None
        self._expert_collect_episode_ids: torch.Tensor | None = None
        self._suboptimal_collect_episode_ids: torch.Tensor | None = None
        self.obs_norm_eps = 1e-6
        self.obs_norm_mean: torch.Tensor | None = None
        self.obs_norm_std: torch.Tensor | None = None
        self.obs_norm_source: str | None = None
        self.action_norm_eps = 1e-6
        self.action_norm_mean: torch.Tensor | None = None
        self.action_norm_std: torch.Tensor | None = None
        self.action_norm_source: str | None = None
        self.command_norm_eps = 1e-6
        self.command_norm_mean: torch.Tensor | None = None
        self.command_norm_std: torch.Tensor | None = None
        self.command_norm_source: str | None = None
        self._use_teacher_aligned_labels = bool(self.cfg.strict_chunk_labels or self.cfg.predict_future_obs)
        source_checkpoint_abs = (
            str(Path(self.compact_obs_source_checkpoint).expanduser().resolve())
            if self.compact_obs_source_checkpoint
            else ""
        )
        self._offline_cache_compact_obs_metadata: dict[str, object] = {
            "term_scale": dict(self.compact_obs_term_scale),
            "term_noise": dict(self.compact_obs_term_noise),
            "add_noise": False,
            "noise_seed": None,
            "source_checkpoint_abs": source_checkpoint_abs,
        }
        model_predict_future_obs = bool(getattr(self.student_model, "predict_future_obs", False))
        if model_predict_future_obs != bool(self.cfg.predict_future_obs):
            raise ValueError(
                "Model/config mismatch for predict_future_obs: "
                f"model={model_predict_future_obs}, cfg={self.cfg.predict_future_obs}"
            )
        if self.cfg.predict_future_obs and (
            not self.offline_buffer.require_future_obs_targets
            or not self.online_buffer.require_future_obs_targets
        ):
            raise ValueError("predict_future_obs=True requires replay buffers with require_future_obs_targets=True")
        if self._use_teacher_aligned_labels and self.strict_label_env is None and self.strict_label_worker is None:
            raise ValueError(
                "Teacher-aligned labels require strict_label_env or strict_label_worker. "
                "This is needed when strict_chunk_labels or predict_future_obs is enabled."
            )

    def load_student_checkpoint(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        path = Path(path)
        payload = safe_load_checkpoint(path, map_location=self.device)
        normalization_payload = payload.get("normalization")
        ckpt_io_enabled = bool(normalization_payload.get("io_enabled", False)) if isinstance(normalization_payload, dict) else False
        cfg_io_enabled = bool(self._use_io_normalization())
        if ckpt_io_enabled != cfg_io_enabled:
            raise RuntimeError(
                "Checkpoint IO-normalization mode mismatch: "
                f"checkpoint io_enabled={ckpt_io_enabled}, current_cfg io_enabled={cfg_io_enabled}. "
                "Use a checkpoint trained with the same normalization mode."
            )
        ckpt_term_scale, ckpt_term_noise, ckpt_compact_add_noise, _ckpt_compact_noise_seed = (
            self._resolve_compact_obs_checkpoint_payload(payload)
        )
        if ckpt_term_scale != self.compact_obs_term_scale:
            raise RuntimeError(
                "Checkpoint compact-obs term_scale mismatch: "
                f"checkpoint={ckpt_term_scale}, current_cfg={self.compact_obs_term_scale}"
            )
        if ckpt_term_noise != self.compact_obs_term_noise:
            raise RuntimeError(
                "Checkpoint compact-obs term_noise mismatch: "
                f"checkpoint={ckpt_term_noise}, current_cfg={self.compact_obs_term_noise}"
            )
        if ckpt_compact_add_noise:
            raise RuntimeError(
                "Checkpoint compact-obs noise toggle mismatch: checkpoint was trained with "
                "add_noise=True, but compact_obs_add_noise (env-extraction noise) has been "
                "constant-folded to False -- it was never enabled in any current checkpoint."
            )
        self.student_model.load_compatible_state_dict(payload["model_state_dict"])
        norm_stats = payload.get("obs_norm_stats")
        if isinstance(norm_stats, dict) and "obs_mean" in norm_stats and "obs_std" in norm_stats:
            loaded_mean = torch.as_tensor(norm_stats["obs_mean"], device=self.device, dtype=torch.float32).flatten()
            loaded_std = torch.as_tensor(norm_stats["obs_std"], device=self.device, dtype=torch.float32).flatten()
            if loaded_mean.numel() != self.obs_dim or loaded_std.numel() != self.obs_dim:
                raise RuntimeError(
                    "Checkpoint obs_norm_stats shape mismatch: "
                    f"mean={tuple(loaded_mean.shape)}, std={tuple(loaded_std.shape)}, expected=({self.obs_dim},)"
                )
            loaded_std = torch.clamp(loaded_std, min=float(norm_stats.get("eps", self.obs_norm_eps)))
            self.obs_norm_mean = loaded_mean
            self.obs_norm_std = loaded_std
            self.obs_norm_source = str(norm_stats.get("source", "checkpoint"))
        action_norm_stats = payload.get("action_norm_stats")
        if isinstance(action_norm_stats, dict) and "action_mean" in action_norm_stats and "action_std" in action_norm_stats:
            loaded_mean = torch.as_tensor(
                action_norm_stats["action_mean"], device=self.device, dtype=torch.float32
            ).flatten()
            loaded_std = torch.as_tensor(
                action_norm_stats["action_std"], device=self.device, dtype=torch.float32
            ).flatten()
            if loaded_mean.numel() != self.act_dim or loaded_std.numel() != self.act_dim:
                raise RuntimeError(
                    "Checkpoint action_norm_stats shape mismatch: "
                    f"mean={tuple(loaded_mean.shape)}, std={tuple(loaded_std.shape)}, expected=({self.act_dim},)"
                )
            loaded_std = torch.clamp(loaded_std, min=float(action_norm_stats.get("eps", self.action_norm_eps)))
            self.action_norm_mean = loaded_mean
            self.action_norm_std = loaded_std
            self.action_norm_source = str(action_norm_stats.get("source", "checkpoint"))
        command_norm_stats = payload.get("command_norm_stats")
        if isinstance(command_norm_stats, dict) and "command_mean" in command_norm_stats and "command_std" in command_norm_stats:
            loaded_mean = torch.as_tensor(
                command_norm_stats["command_mean"], device=self.device, dtype=torch.float32
            ).flatten()
            loaded_std = torch.as_tensor(
                command_norm_stats["command_std"], device=self.device, dtype=torch.float32
            ).flatten()
            if loaded_mean.numel() != self.cmd_dim or loaded_std.numel() != self.cmd_dim:
                raise RuntimeError(
                    "Checkpoint command_norm_stats shape mismatch: "
                    f"mean={tuple(loaded_mean.shape)}, std={tuple(loaded_std.shape)}, expected=({self.cmd_dim},)"
                )
            loaded_std = torch.clamp(loaded_std, min=float(command_norm_stats.get("eps", self.command_norm_eps)))
            self.command_norm_mean = loaded_mean
            self.command_norm_std = loaded_std
            self.command_norm_source = str(command_norm_stats.get("source", "checkpoint"))
        if load_optimizer and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.train_step_count = int(payload.get("train_step_count", 0))
        return payload


class FADATrainer(_LoggingMixin, _RolloutMixin, _BatchingMixin, _StateMixin):
    def __init__(
        self,
        *,
        cfg: FADAConfig,
        student_model: PlannerIDMPolicy,
        expert_policy: ExpertPolicyInterface,
        offline_buffer: ReplayBuffer,
        suboptimal_offline_buffer: ReplayBuffer,
        online_buffer: ReplayBuffer,
        optimizer: optim.Optimizer,
        lr_scheduler: Any | None,
        idm_optimizer: optim.Optimizer | None = None,
        idm_lr_scheduler: Any | None = None,
        device: torch.device,
        obs_dim: int,
        act_dim: int,
        cmd_dim: int,
        resolved_expert_checkpoint: str | None = None,
        compact_obs_source_checkpoint: str | None = None,
        experiment_config_payload: dict[str, Any] | None = None,
        wandb_run: Any | None = None,
        strict_label_env: Any | None = None,
        strict_label_worker: StrictLabelWorkerClient | None = None,
    ) -> None:
        if not isinstance(student_model, PlannerIDMPolicy):
            raise TypeError(f"student_model must be PlannerIDMPolicy, got {type(student_model)}")
        super().__init__(
            cfg=cfg,
            student_model=student_model,
            expert_policy=expert_policy,
            offline_buffer=offline_buffer,
            online_buffer=online_buffer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            obs_dim=obs_dim,
            act_dim=act_dim,
            cmd_dim=cmd_dim,
            compact_obs_source_checkpoint=compact_obs_source_checkpoint,
            experiment_config_payload=experiment_config_payload,
            wandb_run=wandb_run,
            strict_label_env=strict_label_env,
            strict_label_worker=strict_label_worker,
        )
        self.idm_optimizer = idm_optimizer
        self.idm_lr_scheduler = idm_lr_scheduler
        self.suboptimal_offline_buffer = suboptimal_offline_buffer
        if self.suboptimal_offline_buffer.require_future_obs_targets != self.offline_buffer.require_future_obs_targets:
            raise ValueError("offline_buffer and suboptimal_offline_buffer must agree on require_future_obs_targets")
        if self.suboptimal_offline_buffer.history_len != self.offline_buffer.history_len:
            raise ValueError("offline_buffer and suboptimal_offline_buffer must share history_len")
        if self.suboptimal_offline_buffer.pred_horizon != self.offline_buffer.pred_horizon:
            raise ValueError("offline_buffer and suboptimal_offline_buffer must share pred_horizon")

        # Trajectory-level validation buffers (populated after freeze in warmup).
        self.offline_val_buffer: ReplayBuffer | None = None
        self.suboptimal_offline_val_buffer: ReplayBuffer | None = None
        self.online_val_buffer: ReplayBuffer | None = None

        configured_checkpoint = (
            Path(resolved_expert_checkpoint).expanduser().resolve()
            if resolved_expert_checkpoint is not None
            else _resolve_reference_checkpoint_for_env_config(cfg).expanduser().resolve()
        )
        self.configured_expert_checkpoint = configured_checkpoint
        self._currently_loaded_expert_checkpoint: Path | None = configured_checkpoint
        idm_suboptimal = bool(self.cfg.use_generalized_idm) and float(self.cfg.suboptimal_data_ratio) > 0.0
        planner_suboptimal = bool(self.cfg.planner_suboptimal_enabled)
        self.suboptimal_enabled = idm_suboptimal or planner_suboptimal
        self.planner_suboptimal_enabled = planner_suboptimal
        # When planner_suboptimal is enabled, oracle-relabeled data goes into
        # suboptimal_offline_buffer directly (expert_chunk / expert_future_obs_chunk
        # store oracle data; IDM reads real dynamics from trajectory_target_* keys).
        if not isinstance(self.expert_policy, ExpertPolicyWrapper):
            raise RuntimeError(
                "planner+idm generalized warmup requires a single ExpertPolicyWrapper so checkpoints can be swapped."
            )
        if self.planner_suboptimal_enabled and strict_label_worker is None and strict_label_env is None:
            raise RuntimeError(
                "planner_suboptimal_batch_ratio > 0 requires a strict_label_worker or strict_label_env "
                "for oracle relabeling of suboptimal planner data."
            )

    @property
    def student_policy(self) -> PlannerIDMPolicy:
        return self.student_model

    def _extract_current_command(self, env: Any) -> torch.Tensor:
        return extract_current_command_torch(env, profile=self.cfg.command_profile)

    def close(self) -> None:
        if self.strict_label_worker is not None:
            self.strict_label_worker.close()
        return

    def _current_lr(self) -> float:
        if not self.optimizer.param_groups:
            return float("nan")
        return float(self.optimizer.param_groups[0].get("lr", float("nan")))

    def train_step(self, *, mode: str = "mixed") -> tuple[float, dict[str, int], dict[str, float]]:
        # The separate IDM/planner pass is the only training step; the
        # `idm_planner_separate_pass` switch is constant-folded to True and is not a
        # config field.
        return self._train_step_separate_pass(mode=mode)

    def _train_step_separate_pass(self, *, mode: str) -> tuple[float, dict[str, int], dict[str, float]]:
        assert self.idm_optimizer is not None, "idm_optimizer required for separate-pass mode"
        self.student_model.train()

        planner_batch, planner_counts = self._sample_planner_batch(mode=mode)
        idm_batches_by_source, idm_counts = self._sample_idm_batches_by_source(mode=mode)

        idm_params = list(self.student_model.idm_parameters())

        # ── IDM pass (first): all sources, forced TF=1.0, pure inverse dynamics ──
        idm_loss_total, idm_loss, idm_loss_unweighted = self._compute_idm_batch_loss_from_source_batches(
            idm_batches_by_source, force_teacher_forcing_ratio=1.0
        )
        self.idm_optimizer.zero_grad(set_to_none=True)
        idm_loss_total.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(idm_params, self.cfg.grad_clip)
        self.idm_optimizer.step()

        # ── Planner pass (second): planner obs loss + IDM action loss (oracle-shadow, TF=0) ──
        # Freeze IDM params so autograd skips their gradient computation entirely.
        # Save prior requires_grad to safely restore even if backward() raises or IDM is partially frozen.
        prev_requires_grad = [p.requires_grad for p in idm_params]
        for p in idm_params:
            p.requires_grad_(False)

        planner_loss_total, planner_loss, planner_per_term = self._compute_planner_batch_loss_from_batch(planner_batch)

        idm_planner_loss = torch.zeros((), device=self.device, dtype=planner_loss_total.dtype)
        idm_planner_loss_uw = torch.zeros((), device=self.device, dtype=planner_loss_total.dtype)
        idm_planner_loss_total_weighted = torch.zeros((), device=self.device, dtype=planner_loss_total.dtype)
        _, idm_planner_loss, idm_planner_loss_uw = self._compute_idm_batch_loss_from_batch(
            planner_batch,
            use_teacher_forcing=False,
            teacher_forcing_ratio=0.0,
            detach_override=False,
        )
        coef = float(self.cfg.idm_planner_pass_action_loss_coef)
        idm_planner_loss_total_weighted = coef * float(self.cfg.idm_action_loss_coef) * idm_planner_loss
        total_planner_loss = planner_loss_total + idm_planner_loss_total_weighted

        self.optimizer.zero_grad(set_to_none=True)
        try:
            total_planner_loss.backward()
        finally:
            for p, prev in zip(idm_params, prev_requires_grad):
                p.requires_grad_(prev)
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(self.student_model.planner_parameters()), self.cfg.grad_clip)
        self.optimizer.step()

        self.train_step_count += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.idm_lr_scheduler is not None:
            self.idm_lr_scheduler.step()

        loss_total_combined = total_planner_loss + idm_loss_total
        sample_stats = self._build_sample_stats(planner_counts=planner_counts, idm_counts=idm_counts)
        loss_metrics = {
            "loss_total": float(loss_total_combined.detach().cpu().item()),
            "planner_loss_total": float(planner_loss_total.detach().cpu().item()),
            "planner_future_obs_loss": float(planner_loss.detach().cpu().item()),
            "planner_future_obs_loss_weighted": float(float(self.cfg.planner_obs_loss_coef) * planner_loss.detach().cpu().item()),
            "idm_loss_total": float(idm_loss_total.detach().cpu().item()),
            "idm_action_loss": float(idm_loss.detach().cpu().item()),
            "idm_action_loss_unweighted": float(idm_loss_unweighted.detach().cpu().item()),
            "idm_action_loss_weighted": float(float(self.cfg.idm_action_loss_coef) * idm_loss.detach().cpu().item()),
            "idm_planner_pass_action_loss": float(idm_planner_loss.detach().cpu().item()),
            "idm_planner_pass_action_loss_unweighted": float(idm_planner_loss_uw.detach().cpu().item()),
            "idm_planner_pass_action_loss_weighted": float(idm_planner_loss_total_weighted.detach().cpu().item()),
        }
        if planner_per_term is not None:
            for term_name, term_loss in planner_per_term.items():
                loss_metrics[f"planner_term/{term_name}"] = float(term_loss)
        return float(loss_total_combined.detach().cpu().item()), sample_stats, loss_metrics

    def load_student_checkpoint(self, path, *, load_optimizer: bool = True):
        payload = super().load_student_checkpoint(path, load_optimizer=False)
        if not load_optimizer:
            return payload
        if "planner_optimizer_state_dict" not in payload:
            raise ValueError(
                f"Checkpoint {path} was not saved with idm_planner_separate_pass=True "
                "(missing 'planner_optimizer_state_dict'). Cannot load into separate-pass mode."
            )
        self.optimizer.load_state_dict(payload["planner_optimizer_state_dict"])
        assert self.idm_optimizer is not None
        self.idm_optimizer.load_state_dict(payload["idm_optimizer_state_dict"])
        return payload

    def save_student_checkpoint(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self.idm_optimizer is not None
        optimizer_entries: dict = {
            "planner_optimizer_state_dict": self.optimizer.state_dict(),
            "idm_optimizer_state_dict": self.idm_optimizer.state_dict(),
        }
        payload = {
            "model_state_dict": self.student_model.state_dict(),
            **optimizer_entries,
            "train_step_count": self.train_step_count,
            "cfg": dataclasses.asdict(self.cfg),
            "compact_obs": self._build_compact_obs_checkpoint_payload(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "cmd_dim": self.cmd_dim,
            "offline_buffer_size": len(self.offline_buffer),
            "optimal_offline_buffer_size": len(self.offline_buffer),
            "suboptimal_offline_buffer_size": len(self.suboptimal_offline_buffer),
            "online_buffer_size": len(self.online_buffer),
            "online_val_buffer_size": len(self.online_val_buffer) if self.online_val_buffer is not None else 0,
            "offline_valid_chunks": self.offline_buffer.num_valid_chunks(),
            "optimal_offline_valid_chunks": self.offline_buffer.num_valid_chunks(),
            "suboptimal_offline_valid_chunks": self.suboptimal_offline_buffer.num_valid_chunks(),
            "online_valid_chunks": self.online_buffer.num_valid_chunks(),
            "online_val_valid_chunks": (
                self.online_val_buffer.num_valid_chunks() if self.online_val_buffer is not None else 0
            ),
            "normalization": {
                "io_enabled": bool(self._use_io_normalization()),
            },
        }
        payload["planner_state_dict"] = self.student_model.planner_state_dict()
        payload["idm_state_dict"] = self.student_model.idm_state_dict()
        if self.obs_norm_mean is not None and self.obs_norm_std is not None:
            payload["obs_norm_stats"] = {
                "obs_mean": self.obs_norm_mean.detach().to(device="cpu", dtype=torch.float32),
                "obs_std": self.obs_norm_std.detach().to(device="cpu", dtype=torch.float32),
                "enabled": bool(self._use_io_normalization()),
                "source": self.obs_norm_source if self.obs_norm_source is not None else "shared",
                "eps": float(self.obs_norm_eps),
            }
        if self.action_norm_mean is not None and self.action_norm_std is not None:
            payload["action_norm_stats"] = {
                "action_mean": self.action_norm_mean.detach().to(device="cpu", dtype=torch.float32),
                "action_std": self.action_norm_std.detach().to(device="cpu", dtype=torch.float32),
                "enabled": bool(self._use_io_normalization()),
                "source": self.action_norm_source if self.action_norm_source is not None else "shared",
                "eps": float(self.action_norm_eps),
            }
        if self.command_norm_mean is not None and self.command_norm_std is not None:
            payload["command_norm_stats"] = {
                "command_mean": self.command_norm_mean.detach().to(device="cpu", dtype=torch.float32),
                "command_std": self.command_norm_std.detach().to(device="cpu", dtype=torch.float32),
                "enabled": bool(self._use_io_normalization()),
                "source": self.command_norm_source if self.command_norm_source is not None else "shared",
                "eps": float(self.command_norm_eps),
            }
        if self.experiment_config_payload is not None:
            payload["experiment_config"] = self.experiment_config_payload
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)
        return path

    def _build_compact_obs_checkpoint_payload(self) -> dict[str, Any]:
        preprocess = get_compact_obs_preprocess(
            term_scale=self.compact_obs_term_scale,
            term_noise=self.compact_obs_term_noise,
        )
        payload: dict[str, Any] = {
            "add_noise": False,
            "term_order": list(preprocess.get("term_order", [])),
            "term_scale": dict(preprocess.get("term_scale", {})),
            "term_noise": dict(preprocess.get("term_noise", {})),
            "term_noise_effective": dict(preprocess.get("term_noise_effective", {})),
            "noise_enabled": bool(preprocess.get("noise_enabled", False)),
        }
        return payload

    @staticmethod
    def _resolve_compact_obs_checkpoint_payload(payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], bool, int | None]:
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
        return term_scale, term_noise, add_noise, noise_seed

    def warmup(self, env: Any) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "offline_cache_hit": False,
            "warmup_ckpt_loaded": False,
            "warmup_train_steps": 0,
            "warmup_mean_loss": None,
        }
        cache_stats = self._load_offline_caches()
        stats.update(cache_stats)

        optimal_target_steps = self._target_optimal_offline_steps(
            num_envs=int(env.num_envs),
            horizon=int(self.cfg.warmup_max_steps_per_episode),
        )

        if not bool(stats["optimal_offline_cache_hit"]):
            self.offline_buffer.clear()
            optimal_stats = self._collect_optimal_offline_data(env)
            self.offline_buffer.save_npz(
                self.cfg.offline_cache_path,
                compact_obs_metadata=self._offline_cache_compact_obs_metadata,
            )
            stats["optimal_offline_collect"] = optimal_stats
            stats["optimal_offline_cache_saved"] = True
            stats["offline_collect_mode"] = str(optimal_stats["collect_mode"])
            stats["offline_collected_steps"] = int(optimal_stats["kept_steps"])
            stats["offline_completed_episodes"] = int(optimal_stats["kept_episodes"])
            stats["offline_env_done_events"] = int(optimal_stats["dropped_episodes"])
            stats["offline_rollout_steps"] = int(optimal_stats["rollouts"]) * int(optimal_stats["horizon"]) * int(env.num_envs)
        else:
            stats["optimal_offline_cache_saved"] = False

        if self.suboptimal_enabled and not bool(stats["suboptimal_offline_cache_hit"]):
            self.suboptimal_offline_buffer.clear()
            suboptimal_stats = self._collect_suboptimal_offline_data(
                env,
                optimal_target_steps=int(optimal_target_steps),
            )
            self.suboptimal_offline_buffer.save_npz(
                self.cfg.offline_suboptimal_cache_path,
                compact_obs_metadata=self._offline_cache_compact_obs_metadata,
            )
            stats["suboptimal_offline_collect"] = suboptimal_stats
            stats["suboptimal_offline_cache_saved"] = True
        elif self.suboptimal_enabled:
            stats["suboptimal_offline_cache_saved"] = False

        if not self.suboptimal_enabled:
            self.suboptimal_offline_buffer.clear()
            stats["suboptimal_offline_cache_saved"] = False

        if not bool(stats["offline_cache_hit"]):
            self._maybe_load_expert_checkpoint(self.configured_expert_checkpoint)
            stats["offline_cache_hit"] = False

        # Compute normalization stats from the FULL buffer (before trajectory split)
        # so that norm stats cover the complete data distribution.
        if self.cfg.predict_future_obs or self._use_io_normalization():
            self._refresh_obs_norm_stats(include_online=False)
            stats["offline_obs_norm_stats_ready"] = True
        if self._use_io_normalization():
            self._refresh_action_norm_stats(include_online=False)
            self._refresh_command_norm_stats(include_online=False)
            stats["offline_action_norm_stats_ready"] = True
            stats["offline_command_norm_stats_ready"] = True

        # Trajectory-level train/val split (before freeze).
        val_ratio = float(self.cfg.val_trajectory_ratio)
        if val_ratio > 0.0 and len(self.offline_buffer) > 0:
            full_size = len(self.offline_buffer)
            train_buf, val_buf = self.offline_buffer.split_by_trajectory(
                val_ratio=val_ratio, seed=int(self.cfg.seed),
            )
            self.offline_buffer = train_buf
            self.offline_val_buffer = val_buf
            stats["val_trajectory_split"] = True
            stats["optimal_train_size"] = len(train_buf)
            stats["optimal_val_size"] = len(val_buf)
            print(
                f"[VAL SPLIT] Optimal offline: {full_size} → "
                f"train={len(train_buf)} val={len(val_buf)} "
                f"(ratio={val_ratio:.2f})"
            )
        else:
            self.offline_buffer.freeze()
            stats["val_trajectory_split"] = False

        if val_ratio > 0.0 and len(self.suboptimal_offline_buffer) > 0:
            sub_full_size = len(self.suboptimal_offline_buffer)
            sub_train_buf, sub_val_buf = self.suboptimal_offline_buffer.split_by_trajectory(
                val_ratio=val_ratio, seed=int(self.cfg.seed) + 1,
            )
            self.suboptimal_offline_buffer = sub_train_buf
            self.suboptimal_offline_val_buffer = sub_val_buf
            stats["suboptimal_train_size"] = len(sub_train_buf)
            stats["suboptimal_val_size"] = len(sub_val_buf)
            print(
                f"[VAL SPLIT] Suboptimal offline: {sub_full_size} → "
                f"train={len(sub_train_buf)} val={len(sub_val_buf)} "
                f"(ratio={val_ratio:.2f})"
            )
        else:
            self.suboptimal_offline_buffer.freeze()

        stats["offline_read_only"] = True
        stats["optimal_offline_read_only"] = True
        stats["suboptimal_offline_read_only"] = True

        warmup_ckpt = Path(self.cfg.warmup_ckpt_path)
        should_load_warmup = (
            warmup_ckpt.exists()
            and self.cfg.load_warmup_ckpt
            and not self.cfg.force_warmup_train
        )
        # `load_warmup_ckpt` and `force_warmup_train` both default True, so at the
        # shipped defaults this condition is False whatever the cache holds and the cached
        # checkpoint is not loaded. Report that case.
        if warmup_ckpt.exists() and self.cfg.load_warmup_ckpt and self.cfg.force_warmup_train:
            print(
                f"[Warmup] Cached warmup checkpoint exists ({warmup_ckpt}) and load_warmup_ckpt is "
                "true, but force_warmup_train is also true (both default true), so it is NOT being "
                "loaded and warmup training will run. Pass --force-warmup-train false to reuse it."
            )
        if should_load_warmup:
            self.load_student_checkpoint(warmup_ckpt, load_optimizer=True)
            if self.cfg.predict_future_obs or self._use_io_normalization():
                if self.obs_norm_mean is None or self.obs_norm_std is None:
                    self._refresh_obs_norm_stats(include_online=False)
                    stats["offline_obs_norm_stats_recomputed"] = True
            if self._use_io_normalization():
                if self.action_norm_mean is None or self.action_norm_std is None:
                    self._refresh_action_norm_stats(include_online=False)
                    stats["offline_action_norm_stats_recomputed"] = True
                if self.command_norm_mean is None or self.command_norm_std is None:
                    self._refresh_command_norm_stats(include_online=False)
                    stats["offline_command_norm_stats_recomputed"] = True
            stats["warmup_ckpt_loaded"] = True
            stats["offline_valid_chunks"] = int(self.offline_buffer.num_valid_chunks())
            stats["optimal_offline_valid_chunks"] = int(self.offline_buffer.num_valid_chunks())
            stats["suboptimal_offline_valid_chunks"] = int(self.suboptimal_offline_buffer.num_valid_chunks())
            return stats

        if self.cfg.warmup_train_steps > 0:
            losses_total: list[float] = []
            planner_losses: list[float] = []
            idm_losses: list[float] = []
            for _ in _progress_range(
                self.cfg.warmup_train_steps,
                desc="Warmup Train",
                enabled=bool(self.cfg.show_progress),
                leave=False,
            ):
                loss, sample_stats, loss_metrics = self.train_step(mode="offline")
                losses_total.append(float(loss_metrics["loss_total"]))
                planner_losses.append(float(loss_metrics["planner_future_obs_loss"]))
                idm_losses.append(float(loss_metrics["idm_action_loss"]))
                warmup_payload = {
                    "warmup/loss": float(loss),
                    "warmup/loss_total": float(loss_metrics["loss_total"]),
                    "warmup/planner_loss_total": float(loss_metrics["planner_loss_total"]),
                    "warmup/planner_future_obs_loss": float(loss_metrics["planner_future_obs_loss"]),
                    "warmup/planner_future_obs_loss_weighted": float(
                        loss_metrics["planner_future_obs_loss_weighted"]
                    ),
                    "warmup/idm_loss_total": float(loss_metrics["idm_loss_total"]),
                    "warmup/idm_action_loss": float(loss_metrics["idm_action_loss"]),
                    "warmup/idm_action_loss_unweighted": float(loss_metrics["idm_action_loss_unweighted"]),
                    "warmup/idm_action_loss_weighted": float(loss_metrics["idm_action_loss_weighted"]),
                    "warmup/optimal_batch_size": int(sample_stats["optimal_batch_size"]),
                    "warmup/suboptimal_batch_size": int(sample_stats["suboptimal_batch_size"]),
                    "warmup/online_batch_size": int(sample_stats["online_batch_size"]),
                    "warmup/lr": self._current_lr(),
                }
                warmup_payload.update(
                    {
                        "warmup/idm_planner_pass_action_loss": float(
                            loss_metrics.get("idm_planner_pass_action_loss", 0.0)
                        ),
                        "warmup/idm_planner_pass_action_loss_unweighted": float(
                            loss_metrics.get("idm_planner_pass_action_loss_unweighted", 0.0)
                        ),
                        "warmup/idm_planner_pass_action_loss_weighted": float(
                            loss_metrics.get("idm_planner_pass_action_loss_weighted", 0.0)
                        ),
                    }
                )
                for key, value in loss_metrics.items():
                    if key.startswith("planner_term/"):
                        warmup_payload[f"warmup/{key}"] = float(value)
                _safe_wandb_log(
                    self.wandb_run,
                    warmup_payload,
                    step=int(self.train_step_count),
                )
            mean_loss = float(np.mean(losses_total)) if losses_total else None
            stats["warmup_train_steps"] = int(self.cfg.warmup_train_steps)
            stats["warmup_mean_loss"] = mean_loss
            stats["warmup_mean_planner_future_obs_loss"] = float(np.mean(planner_losses)) if planner_losses else None
            stats["warmup_mean_idm_action_loss"] = float(np.mean(idm_losses)) if idm_losses else None
            self.save_student_checkpoint(
                warmup_ckpt,
                extra={
                    "phase": "warmup",
                    "mean_loss": mean_loss,
                },
            )

        planner_val_losses = self._estimate_planner_validation_losses(
            mode="offline",
            num_batches=int(self.cfg.validation_batches),
        )
        idm_val_losses = self._estimate_idm_validation_losses(
            mode="offline",
            num_batches=int(self.cfg.validation_batches),
        )
        stats["planner_validation_loss"] = (
            None if planner_val_losses is None else float(planner_val_losses["total"])
        )
        stats["planner_validation_future_obs_loss"] = (
            None if planner_val_losses is None else float(planner_val_losses["planner_future_obs_loss"])
        )
        stats["idm_validation_loss"] = None if idm_val_losses is None else float(idm_val_losses["total"])
        stats["idm_validation_action_loss"] = (
            None if idm_val_losses is None else float(idm_val_losses["idm_action_loss"])
        )
        stats["idm_validation_action_loss_unweighted"] = (
            None if idm_val_losses is None else float(idm_val_losses["idm_action_loss_unweighted"])
        )
        stats["offline_valid_chunks"] = int(self.offline_buffer.num_valid_chunks())
        stats["optimal_offline_valid_chunks"] = int(self.offline_buffer.num_valid_chunks())
        stats["suboptimal_offline_valid_chunks"] = int(self.suboptimal_offline_buffer.num_valid_chunks())
        return stats

    def rollout_and_collect(self, env: Any, sigma: float) -> dict[str, Any]:
        val_ratio = float(self.cfg.val_trajectory_ratio)
        sample_ratio = float(self.cfg.online_rollout_sample_ratio)

        if val_ratio <= 0.0 and sample_ratio >= 1.0:
            # No val split, no subsampling — collect directly into online_buffer.
            return self._collect_with_policy(
                env,
                num_steps=self.cfg.rollout_steps_per_iter,
                sigma=sigma,
                execute_expert=False,
                target_buffer=self.online_buffer,
            )

        # Collect into a temporary buffer first.
        temp_buffer = ReplayBuffer(
            capacity=int(self.cfg.rollout_steps_per_iter * self.cfg.num_envs) + 4096,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            cmd_dim=self.cmd_dim,
            history_len=self.cfg.history_len,
            pred_horizon=self.cfg.pred_horizon,
            require_future_obs_targets=True,
        )
        stats = self._collect_with_policy(
            env,
            num_steps=self.cfg.rollout_steps_per_iter,
            sigma=sigma,
            execute_expert=False,
            target_buffer=temp_buffer,
        )

        # Subsample trajectories if sample_ratio < 1.0.
        if sample_ratio < 1.0:
            keep_buf, _discard = temp_buffer.split_by_trajectory(
                val_ratio=1.0 - sample_ratio,
                seed=int(self.cfg.seed) + 2000 + int(self.rollout_step_count),
            )
            temp_buffer = keep_buf

        if val_ratio <= 0.0:
            # No val split — add subsampled data directly.
            self.online_buffer.add_from_buffer(temp_buffer)
            print(
                f"[ONLINE] rollout sampled {len(temp_buffer)} steps "
                f"(ratio={sample_ratio}) | "
                f"cumulative online={len(self.online_buffer)}"
            )
            return stats

        # Split into train and val by trajectory.
        train_buf, val_buf = temp_buffer.split_by_trajectory(
            val_ratio=val_ratio, seed=int(self.cfg.seed) + 1000 + int(self.rollout_step_count),
        )

        # Merge into persistent online buffers.
        self.online_buffer.add_from_buffer(train_buf)
        if self.online_val_buffer is None:
            self.online_val_buffer = ReplayBuffer(
                capacity=max(int(self.cfg.replay_capacity * val_ratio) + 1024, 4096),
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                cmd_dim=self.cmd_dim,
                history_len=self.cfg.history_len,
                pred_horizon=self.cfg.pred_horizon,
                require_future_obs_targets=True,
                eviction_policy=self.online_buffer.eviction_policy,
            )
        self.online_val_buffer.add_from_buffer(val_buf)

        print(
            f"[ONLINE VAL SPLIT] rollout {len(temp_buffer)} steps "
            f"(sample_ratio={sample_ratio}) → "
            f"train={len(train_buf)} val={len(val_buf)} | "
            f"cumulative online_train={len(self.online_buffer)} "
            f"online_val={len(self.online_val_buffer)}"
        )

        return stats

    def run_loop(
        self,
        env: Any,
        *,
        n_iters: int,
        checkpoint_dir: Path,
        start_iter: int = 0,
    ) -> list[dict[str, Any]]:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logs: list[dict[str, Any]] = []

        for it_idx in _progress_range(
            n_iters,
            desc="DAgger Iterations",
            enabled=bool(self.cfg.show_progress),
            leave=True,
        ):
            it = int(it_idx) + 1
            if it <= start_iter:
                continue
            rollout_stats = self.rollout_and_collect(env, sigma=self.cfg.dagger_sigma)
            self.rollout_step_count += int(rollout_stats["collected_steps"])

            rollout_wandb: dict[str, float | int] = {
                "dagger/expert_correction_energy": float(rollout_stats["expert_correction_energy"]),
                "Train/mean_reward": float(rollout_stats["episode_return_mean"]),
                "Train/mean_reward/time": float(rollout_stats["episode_return_mean"]),
                "Train/mean_episode_length": float(rollout_stats["episode_length_mean"]),
                "Train/mean_episode_length/time": float(rollout_stats["episode_length_mean"]),
                "Train/num_samples": int(self.rollout_step_count),
            }
            strict_ratio = float(rollout_stats.get("strict_label_valid_ratio", float("nan")))
            if np.isfinite(strict_ratio):
                rollout_wandb["dagger/strict_label_valid_ratio"] = strict_ratio
                rollout_wandb["dagger/strict_label_valid_count"] = int(
                    rollout_stats.get("strict_label_valid_count", 0)
                )
                rollout_wandb["dagger/strict_label_total_count"] = int(
                    rollout_stats.get("strict_label_total_count", 0)
                )
            for extra_key, extra_value in rollout_stats.get("episode_metric_means", {}).items():
                rollout_wandb[str(extra_key)] = float(extra_value)
            for extra_key, extra_value in rollout_stats.get("raw_episode_metric_means", {}).items():
                rollout_wandb[str(extra_key)] = float(extra_value)
            for extra_key, extra_value in rollout_stats.get("env_metric_means", {}).items():
                rollout_wandb[str(extra_key)] = float(extra_value)
            if rollout_stats.get("env_average_episode_length") is not None:
                rollout_wandb["Env/average_episode_length"] = float(rollout_stats["env_average_episode_length"])
            _safe_wandb_log(self.wandb_run, rollout_wandb, step=int(self.train_step_count))

            if self.cfg.predict_future_obs or self._use_io_normalization():
                self._refresh_obs_norm_stats(include_online=True)
            if self._use_io_normalization():
                self._refresh_action_norm_stats(include_online=True)
                self._refresh_command_norm_stats(include_online=True)

            losses_total: list[float] = []
            planner_losses: list[float] = []
            idm_losses: list[float] = []
            sample_stats = self._build_sample_stats(
                planner_counts=self._planner_batch_counts(mode="mixed"),
                idm_counts=self._idm_batch_counts(mode="mixed"),
            )
            for _ in _progress_range(
                self.cfg.train_steps_per_iter,
                desc=f"Iter {it}/{n_iters} Train",
                enabled=bool(self.cfg.show_progress),
                leave=False,
            ):
                _, sample_stats, loss_metrics = self.train_step(mode="mixed")
                losses_total.append(float(loss_metrics["loss_total"]))
                planner_losses.append(float(loss_metrics["planner_future_obs_loss"]))
                idm_losses.append(float(loss_metrics["idm_action_loss"]))
                train_payload = {
                    "dagger/loss": float(loss_metrics["loss_total"]),
                    "dagger/loss_total": float(loss_metrics["loss_total"]),
                    "dagger/planner_loss_total": float(loss_metrics["planner_loss_total"]),
                    "dagger/planner_future_obs_loss": float(loss_metrics["planner_future_obs_loss"]),
                    "dagger/planner_future_obs_loss_weighted": float(
                        loss_metrics["planner_future_obs_loss_weighted"]
                    ),
                    "dagger/idm_loss_total": float(loss_metrics["idm_loss_total"]),
                    "dagger/idm_action_loss": float(loss_metrics["idm_action_loss"]),
                    "dagger/idm_action_loss_unweighted": float(loss_metrics["idm_action_loss_unweighted"]),
                    "dagger/idm_action_loss_weighted": float(loss_metrics["idm_action_loss_weighted"]),
                    "dagger/lr": self._current_lr(),
                }
                train_payload.update(
                    {
                        "dagger/idm_planner_pass_action_loss": float(
                            loss_metrics.get("idm_planner_pass_action_loss", 0.0)
                        ),
                        "dagger/idm_planner_pass_action_loss_unweighted": float(
                            loss_metrics.get("idm_planner_pass_action_loss_unweighted", 0.0)
                        ),
                        "dagger/idm_planner_pass_action_loss_weighted": float(
                            loss_metrics.get("idm_planner_pass_action_loss_weighted", 0.0)
                        ),
                    }
                )
                for key, value in loss_metrics.items():
                    if key.startswith("planner_term/"):
                        train_payload[f"dagger/{key}"] = float(value)
                _safe_wandb_log(
                    self.wandb_run,
                    train_payload,
                    step=int(self.train_step_count),
                )

            mean_loss = float(np.mean(losses_total)) if losses_total else float("nan")
            std_loss = float(np.std(losses_total)) if losses_total else float("nan")
            mean_planner_loss = float(np.mean(planner_losses)) if planner_losses else float("nan")
            mean_idm_loss = float(np.mean(idm_losses)) if idm_losses else float("nan")
            planner_val_losses = self._estimate_planner_validation_losses(
                mode="mixed",
                num_batches=int(self.cfg.validation_batches),
            )
            idm_val_losses = self._estimate_idm_validation_losses(
                mode="mixed",
                num_batches=int(self.cfg.validation_batches),
            )

            iter_log: dict[str, Any] = {
                "iter": int(it),
                "train_step_count": int(self.train_step_count),
                "mean_train_loss": mean_loss,
                "std_train_loss": std_loss,
                "mean_train_planner_future_obs_loss": mean_planner_loss,
                "mean_train_idm_action_loss": mean_idm_loss,
                "planner_validation_loss": (
                    None if planner_val_losses is None else float(planner_val_losses["total"])
                ),
                "planner_validation_future_obs_loss": (
                    None
                    if planner_val_losses is None
                    else float(planner_val_losses["planner_future_obs_loss"])
                ),
                "idm_validation_loss": None if idm_val_losses is None else float(idm_val_losses["total"]),
                "idm_validation_action_loss": (
                    None if idm_val_losses is None else float(idm_val_losses["idm_action_loss"])
                ),
                "idm_validation_action_loss_unweighted": (
                    None if idm_val_losses is None else float(idm_val_losses["idm_action_loss_unweighted"])
                ),
                "offline_buffer_size": int(len(self.offline_buffer)),
                "optimal_offline_buffer_size": int(len(self.offline_buffer)),
                "suboptimal_offline_buffer_size": int(len(self.suboptimal_offline_buffer)),
                "online_buffer_size": int(len(self.online_buffer)),
                "online_val_buffer_size": int(len(self.online_val_buffer)) if self.online_val_buffer is not None else 0,
                "offline_valid_chunks": int(self.offline_buffer.num_valid_chunks()),
                "optimal_offline_valid_chunks": int(self.offline_buffer.num_valid_chunks()),
                "suboptimal_offline_valid_chunks": int(self.suboptimal_offline_buffer.num_valid_chunks()),
                "online_valid_chunks": int(self.online_buffer.num_valid_chunks()),
                "online_val_valid_chunks": (
                    int(self.online_val_buffer.num_valid_chunks()) if self.online_val_buffer is not None else 0
                ),
                **sample_stats,
                **rollout_stats,
            }
            iter_log["rollout_step_count"] = int(self.rollout_step_count)
            logs.append(iter_log)

            print(
                f"[DAgger] iter={it:04d} "
                f"train_loss={iter_log['mean_train_loss']:.6f} "
                f"planner_val={float(iter_log['planner_validation_loss']) if iter_log['planner_validation_loss'] is not None else float('nan'):.6f} "
                f"idm_val={float(iter_log['idm_validation_loss']) if iter_log['idm_validation_loss'] is not None else float('nan'):.6f} "
                f"optimal_buf={iter_log['optimal_offline_buffer_size']} "
                f"suboptimal_buf={iter_log['suboptimal_offline_buffer_size']} "
                f"online_buf={iter_log['online_buffer_size']} "
                f"online_val_buf={iter_log['online_val_buffer_size']} "
                f"corr_energy={float(iter_log['expert_correction_energy']):.6f}"
            )

            wandb_payload: dict[str, float | int] = {
                "dagger/optimal_batch_size": int(sample_stats["optimal_batch_size"]),
                "dagger/suboptimal_batch_size": int(sample_stats["suboptimal_batch_size"]),
                "dagger/online_batch_size": int(sample_stats["online_batch_size"]),
                "dagger/online_buffer_size": int(iter_log["online_buffer_size"]),
                "dagger/online_val_buffer_size": int(iter_log["online_val_buffer_size"]),
            }
            if iter_log["planner_validation_loss"] is not None:
                wandb_payload["dagger/planner_validation_loss"] = float(iter_log["planner_validation_loss"])
                wandb_payload["dagger/planner_validation_future_obs_loss"] = float(
                    iter_log["planner_validation_future_obs_loss"]
                )
            if planner_val_losses is not None:
                for key, value in planner_val_losses.items():
                    if key.startswith("term/") or key == "naive_baseline":
                        wandb_payload[f"dagger/planner_val_{key}"] = float(value)
            if iter_log["idm_validation_loss"] is not None:
                wandb_payload["dagger/idm_validation_loss"] = float(iter_log["idm_validation_loss"])
                wandb_payload["dagger/idm_validation_action_loss"] = float(iter_log["idm_validation_action_loss"])
            _safe_wandb_log(self.wandb_run, wandb_payload, step=int(iter_log["train_step_count"]))

            if it % self.cfg.save_every_iter == 0:
                ckpt_path = checkpoint_dir / f"student_iter_{it:04d}.pt"
                self.save_student_checkpoint(ckpt_path, extra={"iter_log": iter_log})
                if len(self.online_buffer) > 0:
                    self.online_buffer.save_npz(checkpoint_dir / "online_buffer_latest.npz")
                if self.online_val_buffer is not None and len(self.online_val_buffer) > 0:
                    self.online_val_buffer.save_npz(checkpoint_dir / "online_val_buffer_latest.npz")

        return logs

    def _maybe_load_expert_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if self._currently_loaded_expert_checkpoint == checkpoint_path:
            return
        wrapper = self.expert_policy
        assert isinstance(wrapper, ExpertPolicyWrapper)
        wrapper.algo.load(str(checkpoint_path))
        if hasattr(wrapper.algo, "_eval_mode"):
            wrapper.algo._eval_mode()
        wrapper.policy = wrapper.algo.get_inference_policy()
        self._currently_loaded_expert_checkpoint = checkpoint_path

    def _use_io_normalization(self) -> bool:
        # normalize_io is constant-folded to False, so every branch gated on this method
        # alone (training-time IO normalization stats application) is unreachable. The
        # branches stay: some are also reached via the independent predict_future_obs
        # flag (see the `self.cfg.predict_future_obs or self._use_io_normalization()`
        # call sites).
        return False

    def _ensure_obs_norm_stats_for_training(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.obs_norm_mean is None or self.obs_norm_std is None:
            if len(self.offline_buffer) <= 0:
                raise RuntimeError(
                    "Observation normalization stats are required but offline buffer is empty; "
                    "cannot compute obs_mean/obs_std."
                )
            self._compute_obs_norm_stats_from_offline()

        assert self.obs_norm_mean is not None
        assert self.obs_norm_std is not None
        return self.obs_norm_mean, self.obs_norm_std

    def _ensure_action_norm_stats_for_training(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.action_norm_mean is None or self.action_norm_std is None:
            if len(self.offline_buffer) <= 0:
                raise RuntimeError(
                    "Action normalization stats are required but offline buffer is empty; "
                    "cannot compute action_mean/action_std."
                )
            self._compute_action_norm_stats_from_offline()

        assert self.action_norm_mean is not None
        assert self.action_norm_std is not None
        return self.action_norm_mean, self.action_norm_std

    def _ensure_command_norm_stats_for_training(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.command_norm_mean is None or self.command_norm_std is None:
            if len(self.offline_buffer) <= 0:
                raise RuntimeError(
                    "Command normalization stats are required but offline buffer is empty; "
                    "cannot compute command_mean/command_std."
                )
            self._compute_command_norm_stats_from_offline()

        assert self.command_norm_mean is not None
        assert self.command_norm_std is not None
        return self.command_norm_mean, self.command_norm_std
