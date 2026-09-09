from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
from pathlib import Path

import yaml

from holosoma.fada.common.compact_obs import DEFAULT_COMPACT_TERM_NOISE, extract_compact_obs
from holosoma.fada.common.current_command import (
    command_components_for_profile,
    extract_current_command_torch,
)
from holosoma.fada.common.dataset import ReplayBuffer
from holosoma.fada.planner_idm.config import build_arg_parser, config_from_args
from holosoma.fada.planner_idm.model import (
    PlannerIDMPolicy,
    build_student_policy,
)
from holosoma.fada.planner_idm.trainer import (
    FADATrainer,
    StrictLabelWorkerClient,
    _extract_compact_obs,
    _resolve_reference_checkpoint_for_env_config,
    build_env_and_expert,
    init_wandb_run,
    set_seed,
)
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.safe_torch_load import load_checkpoint as safe_load_checkpoint
from holosoma.utils.sim_utils import close_simulation_app


def _extract_compact_scale_noise_from_expert_payload(
    experiment_config_payload: dict[str, object],
) -> tuple[dict[str, float], dict[str, float]]:
    compact_terms = ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")
    observation = experiment_config_payload.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("Missing observation config in expert checkpoint payload.")
    groups = observation.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("Missing observation.groups in expert checkpoint payload.")
    source_group: dict[str, object] | None = None
    source_group_name: str | None = None
    for group_name in ("actor_obs", "actor_state_obs", "oracle_state_obs", "critic_state_obs", "critic_obs"):
        candidate = groups.get(group_name)
        if not isinstance(candidate, dict):
            continue
        candidate_terms = candidate.get("terms")
        if isinstance(candidate_terms, dict) and all(term_name in candidate_terms for term_name in compact_terms):
            source_group = candidate
            source_group_name = group_name
            break
    if source_group is None:
        raise ValueError(
            "Could not find compact obs terms in expert checkpoint observation groups. "
            f"Checked groups: {sorted(groups.keys())}"
        )
    source_group_enable_noise = bool(source_group.get("enable_noise", True))
    terms = source_group.get("terms")
    if not isinstance(terms, dict):
        raise ValueError(f"Missing observation.groups.{source_group_name}.terms in expert checkpoint payload.")

    term_scale: dict[str, float] = {}
    term_noise: dict[str, float] = {}
    for term_name in compact_terms:
        term_cfg = terms.get(term_name)
        if not isinstance(term_cfg, dict):
            raise ValueError(f"Missing {source_group_name} term config for compact term '{term_name}'.")
        if "scale" not in term_cfg or "noise" not in term_cfg:
            raise ValueError(f"{source_group_name} term '{term_name}' is missing scale/noise.")
        term_scale[term_name] = float(term_cfg["scale"])
        # If the expert disabled noise (enable_noise=False) or declared noise=0,
        # fall back to the PPO default so that augment_obs_noise /
        # compact_obs_add_noise can apply meaningful noise levels.
        raw_noise = float(term_cfg["noise"])
        if raw_noise > 0.0 and source_group_enable_noise:
            term_noise[term_name] = raw_noise
        else:
            term_noise[term_name] = float(DEFAULT_COMPACT_TERM_NOISE[term_name])
    return term_scale, term_noise


def _resolve_dimensions(env, cfg) -> tuple[int, int, int]:
    obs_dim_infer = int(
        extract_compact_obs(
            env,
            term_scale=cfg.compact_obs_term_scale,
        ).shape[1]
    )
    act_dim_infer = int(env.robot_config.actions_dim)
    cmd_dim_infer = int(extract_current_command_torch(env, profile=cfg.command_profile).shape[1])

    obs_dim = int(cfg.obs_dim) if cfg.obs_dim is not None else obs_dim_infer
    act_dim = int(cfg.act_dim) if cfg.act_dim is not None else act_dim_infer
    cmd_dim = int(cfg.cmd_dim) if cfg.cmd_dim is not None else cmd_dim_infer

    if cfg.obs_dim is not None and obs_dim != obs_dim_infer:
        raise ValueError(f"Configured obs_dim={cfg.obs_dim} but inferred obs_dim={obs_dim_infer}")
    if cfg.act_dim is not None and act_dim != act_dim_infer:
        raise ValueError(f"Configured act_dim={cfg.act_dim} but inferred act_dim={act_dim_infer}")
    if cfg.cmd_dim is not None and cmd_dim != cmd_dim_infer:
        raise ValueError(f"Configured cmd_dim={cfg.cmd_dim} but inferred current_command dim={cmd_dim_infer}")
    return obs_dim, act_dim, cmd_dim


def _extract_compact_scale_noise_from_expert_checkpoint(cfg) -> tuple[dict[str, float], dict[str, float], Path]:
    reference_checkpoint = _resolve_reference_checkpoint_for_env_config(cfg)
    checkpoint_cfg = CheckpointConfig(checkpoint=str(reference_checkpoint), eval_exp_name=None)
    saved_cfg, _ = load_saved_experiment_config(checkpoint_cfg)
    if hasattr(saved_cfg, "to_serializable_dict"):
        saved_payload = saved_cfg.to_serializable_dict()
    else:
        saved_payload = dataclasses.asdict(saved_cfg)
    term_scale, term_noise = _extract_compact_scale_noise_from_expert_payload(saved_payload)
    return term_scale, term_noise, reference_checkpoint


def _is_timestamp_prefixed(name: str) -> bool:
    return bool(re.match(r"^\d{8}_\d{6}(?:_|$)", name))


def _build_timestamp_prefixed_name(base_name: str) -> str:
    if _is_timestamp_prefixed(base_name):
        return base_name
    return f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{base_name}"


def _resolve_aligned_run_name(cfg) -> str:
    base_name = cfg.wandb_run_name if cfg.wandb_run_name else cfg.run_name
    return _build_timestamp_prefixed_name(str(base_name))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> None:
    parser = build_arg_parser()
    args, experiment_override_args = parser.parse_known_args()
    cfg = config_from_args(args)
    aligned_run_name = _resolve_aligned_run_name(cfg)
    cfg.run_name = aligned_run_name
    cfg.wandb_run_name = aligned_run_name
    set_seed(cfg.seed)
    compact_term_scale, compact_term_noise, compact_source_checkpoint = (
        _extract_compact_scale_noise_from_expert_checkpoint(cfg)
    )
    cfg.compact_obs_term_scale = compact_term_scale
    cfg.compact_obs_term_noise = compact_term_noise
    if cfg.augment_obs_noise_overrides:
        # Validate term names early, but do NOT modify cfg.compact_obs_term_noise here.
        # Overrides are applied in _build_augment_noise_scales() so that cache metadata
        # (which includes term_noise) matches the cached dataset.
        for term in cfg.augment_obs_noise_overrides:
            if term not in cfg.compact_obs_term_noise:
                raise ValueError(f"augment_obs_noise_overrides: unknown term '{term}'")
        print(f"[Noise] Overrides registered (applied at augmentation time): {cfg.augment_obs_noise_overrides}")

    run_dir = Path(cfg.output_dir) / cfg.run_name
    runtime_log_dir = run_dir / "runtime_logs"
    checkpoint_dir = run_dir / "checkpoints"
    summary_path = run_dir / "summary.json"
    config_snapshot_path = run_dir / "config_snapshot.json"
    holosoma_config_path = run_dir / CONFIG_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_time = dt.datetime.now().isoformat(timespec="seconds")

    _write_json(
        config_snapshot_path,
        {
            **dataclasses.asdict(cfg),
            "resolved_device": None,
            "experiment_override_args": list(experiment_override_args),
            "status": "started",
            "start_time": start_time,
        },
    )
    _write_json(
        summary_path,
        {
            "run_dir": str(run_dir),
            "status": "started",
            "start_time": start_time,
            "history_len": int(cfg.history_len),
            "pred_horizon": int(cfg.pred_horizon),
            "command_input_name": "current_command",
            "command_profile": str(cfg.command_profile),
            "command_components": list(command_components_for_profile(cfg.command_profile)),
            "command_input_semantics": "current_command_broadcast_to_history_tokens",
            "planner_use_action_history": bool(cfg.planner_use_action_history),
            "fdm_enabled": False,
            "use_generalized_idm": bool(cfg.use_generalized_idm),
            "idm_use_teacher_forcing": bool(cfg.idm_use_teacher_forcing),
            "idm_teacher_forcing_ratio": float(cfg.idm_teacher_forcing_ratio),
            "idm_use_current_command_for_history": bool(cfg.idm_use_current_command_for_history),
            "idm_suboptimal_batch_ratio": float(cfg.idm_suboptimal_batch_ratio),
            "online_trajectory_batch_ratio": float(cfg.online_trajectory_batch_ratio),
            "suboptimal_expert_batch_ratio": float(cfg.suboptimal_expert_batch_ratio),
            "warmup_idm_suboptimal_batch_ratio": float(cfg.warmup_idm_suboptimal_batch_ratio),
            "warmup_planner_suboptimal_batch_ratio": float(cfg.warmup_planner_suboptimal_batch_ratio),
            "compact_obs_term_scale": cfg.compact_obs_term_scale,
            "compact_obs_term_noise": cfg.compact_obs_term_noise,
            "compact_obs_add_noise": False,
            "compact_obs_source_checkpoint": str(compact_source_checkpoint),
            "shared_norm_source": str(cfg.shared_norm_source),
            "suboptimal_data_ratio": float(cfg.suboptimal_data_ratio),
            "offline_cache_path": str(cfg.offline_cache_path),
            "offline_suboptimal_cache_path": str(cfg.offline_suboptimal_cache_path),
            "holosoma_config_path": str(holosoma_config_path),
            "experiment_override_args": list(experiment_override_args),
        },
    )

    simulation_app = None
    wandb_run = None
    summary: dict[str, object] = {}
    trainer: FADATrainer | None = None
    strict_label_worker: StrictLabelWorkerClient | None = None
    completed_normally = False

    try:
        wandb_run = init_wandb_run(cfg, run_dir=run_dir)

        env, simulation_app, expert_policy, resolved_device, experiment_config_payload = build_env_and_expert(
            cfg,
            runtime_log_dir=runtime_log_dir,
            experiment_override_args=experiment_override_args,
        )
        print(
            f"[CompactObs] term_scale from expert checkpoint {compact_source_checkpoint}: "
            f"{cfg.compact_obs_term_scale}"
        )
        print(
            f"[CompactObs] term_noise from expert checkpoint {compact_source_checkpoint}: "
            f"{cfg.compact_obs_term_noise}"
        )
        _write_yaml(holosoma_config_path, experiment_config_payload)
        device = torch.device(resolved_device)
        print(f"[Setup] device={device} num_envs={env.num_envs}")

        obs_dim, act_dim, cmd_dim = _resolve_dimensions(env, cfg)
        print(f"[Setup] obs_dim={obs_dim} act_dim={act_dim} cmd_dim={cmd_dim}")

        use_teacher_aligned_labels = bool(cfg.strict_chunk_labels or cfg.predict_future_obs)
        if use_teacher_aligned_labels:
            strict_worker_cfg = dataclasses.replace(cfg, device=str(device))
            strict_label_worker = StrictLabelWorkerClient(
                cfg=strict_worker_cfg,
                runtime_log_dir=runtime_log_dir / "strict_shadow_worker",
                experiment_override_args=experiment_override_args,
                timeout_s=float(cfg.strict_label_worker_timeout_s),
            )
            print(
                "[Setup] strict_label_worker enabled (separate process) "
                f"device={strict_worker_cfg.device}"
            )

        model = build_student_policy(
            cfg,
            obs_dim=obs_dim,
            act_dim=act_dim,
            cmd_dim=cmd_dim,
        ).to(device)
        total_params = sum(int(p.numel()) for p in model.parameters())
        trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        print(f"[Setup] student_params={total_params:,} trainable_params={trainable_params:,}")
        optimizer = torch.optim.AdamW(list(model.planner_parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
        idm_optimizer = torch.optim.AdamW(list(model.idm_parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # lr_scheduler/idm_lr_scheduler are constant-folded to None: cfg.lr_scheduler was
        # always "none" (LinearWarmupCosineScheduler was never instantiated in any trained
        # checkpoint).
        lr_scheduler = None
        idm_lr_scheduler = None

        offline_buffer = ReplayBuffer(
            capacity=cfg.replay_capacity,
            obs_dim=obs_dim,
            act_dim=act_dim,
            cmd_dim=cmd_dim,
            history_len=cfg.history_len,
            pred_horizon=cfg.pred_horizon,
            require_future_obs_targets=True,
            growable=True,
        )
        suboptimal_offline_buffer = ReplayBuffer(
            capacity=cfg.replay_capacity,
            obs_dim=obs_dim,
            act_dim=act_dim,
            cmd_dim=cmd_dim,
            history_len=cfg.history_len,
            pred_horizon=cfg.pred_horizon,
            require_future_obs_targets=True,
            growable=True,
        )
        online_buffer = ReplayBuffer(
            capacity=cfg.replay_capacity,
            obs_dim=obs_dim,
            act_dim=act_dim,
            cmd_dim=cmd_dim,
            history_len=cfg.history_len,
            pred_horizon=cfg.pred_horizon,
            require_future_obs_targets=True,
            eviction_policy=cfg.online_buffer_eviction_policy,
        )

        # ── Resume: restore model, optimizer, and online buffer ──
        resume_start_iter = 0
        if cfg.resume_checkpoint:
            resume_ckpt_path = Path(cfg.resume_checkpoint)
            if not resume_ckpt_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")
            print(f"[Resume] Loading checkpoint: {resume_ckpt_path}")
            resume_ckpt = safe_load_checkpoint(resume_ckpt_path, map_location=device)
            model.load_state_dict(resume_ckpt["model_state_dict"])
            if "planner_optimizer_state_dict" not in resume_ckpt:
                raise ValueError(
                    f"Resume checkpoint {cfg.resume_checkpoint} was not saved with "
                    "idm_planner_separate_pass=True (missing 'planner_optimizer_state_dict'). "
                    "Cannot resume separate-pass training from a joint-optimizer checkpoint."
                )
            optimizer.load_state_dict(resume_ckpt["planner_optimizer_state_dict"])
            assert idm_optimizer is not None
            idm_optimizer.load_state_dict(resume_ckpt["idm_optimizer_state_dict"])
            resume_extra = resume_ckpt.get("extra", {})
            resume_start_iter = int(resume_extra.get("num_iters", 0))
            if resume_start_iter == 0:
                # Infer from checkpoint filename: student_iter_NNNN.pt
                stem = resume_ckpt_path.stem
                import re as _re
                m = _re.search(r"iter_(\d+)", stem)
                if m:
                    resume_start_iter = int(m.group(1))
            print(
                f"[Resume] Restored model + optimizer (train_step_count="
                f"{resume_ckpt.get('train_step_count', '?')}), will resume from iter {resume_start_iter + 1}"
            )
            # Restore online buffer
            resume_buf_path = cfg.resume_online_buffer
            if resume_buf_path is None:
                # Prefer latest per-iter buffer (saved alongside checkpoints) over end-of-run buffer
                latest_buf = resume_ckpt_path.parent / "online_buffer_latest.npz"
                if latest_buf.exists():
                    resume_buf_path = str(latest_buf)
                else:
                    candidate = resume_ckpt_path.parent.parent / "online_buffer.npz"
                    if candidate.exists():
                        resume_buf_path = str(candidate)
            if resume_buf_path and Path(resume_buf_path).exists():
                print(f"[Resume] Loading online buffer: {resume_buf_path}")
                loaded_count = online_buffer.load_npz(resume_buf_path)
                print(f"[Resume] Online buffer restored: {loaded_count} transitions")
            elif cfg.resume_online_buffer:
                # Unreachable via the CLI: `_validate_resume_flags` rejects a
                # non-existent explicit path before the simulator starts. Reached only by
                # a config built in code, where an explicitly named buffer is missing.
                raise FileNotFoundError(f"resume_online_buffer path does not exist: {resume_buf_path}")
            else:
                print("[Resume] No online buffer found, starting with empty online buffer")
            # Restore validation buffer if available (stored as a separate npz per iter)
            resume_val_buf_path = resume_ckpt_path.parent / "online_val_buffer_latest.npz"
            _resume_val_buf_npz = str(resume_val_buf_path) if resume_val_buf_path.exists() else None

        trainer = FADATrainer(
            cfg=cfg,
            student_model=model,
            expert_policy=expert_policy,
            offline_buffer=offline_buffer,
            suboptimal_offline_buffer=suboptimal_offline_buffer,
            online_buffer=online_buffer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            idm_optimizer=idm_optimizer,
            idm_lr_scheduler=idm_lr_scheduler,
            device=device,
            obs_dim=obs_dim,
            act_dim=act_dim,
            cmd_dim=cmd_dim,
            resolved_expert_checkpoint=str(compact_source_checkpoint),
            compact_obs_source_checkpoint=str(compact_source_checkpoint),
            experiment_config_payload=experiment_config_payload,
            wandb_run=wandb_run,
            strict_label_worker=strict_label_worker,
        )

        if resume_start_iter > 0:
            # Restore internal counters
            trainer.train_step_count = int(resume_ckpt.get("train_step_count", 0))
            # rollout_step_count is not saved in checkpoint; estimate from iters
            trainer.rollout_step_count = resume_start_iter * cfg.num_envs * cfg.rollout_steps_per_iter
            # Force warmup to skip model loading and training (we already loaded the resume checkpoint)
            cfg_for_warmup = dataclasses.replace(
                cfg,
                load_warmup_ckpt=False,
                force_warmup_train=False,
                warmup_train_steps=0,
            )
            trainer.cfg = cfg_for_warmup
            warmup_stats = trainer.warmup(env)
            trainer.cfg = cfg  # restore original config
            if _resume_val_buf_npz:
                trainer.online_val_buffer = ReplayBuffer(
                    capacity=max(int(cfg.replay_capacity * cfg.val_trajectory_ratio) + 1024, 4096),
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    cmd_dim=cmd_dim,
                    history_len=cfg.history_len,
                    pred_horizon=cfg.pred_horizon,
                    require_future_obs_targets=True,
                    eviction_policy=online_buffer.eviction_policy,
                )
                loaded_val = trainer.online_val_buffer.load_npz(_resume_val_buf_npz)
                print(f"[Resume] Online val buffer restored: {loaded_val} transitions from {_resume_val_buf_npz}")
            print(f"[Resume] Warmup (offline cache only): {json.dumps({k: v for k, v in warmup_stats.items() if 'cache' in k or 'split' in k}, indent=2)}")
            print(f"[Resume] Resuming DAgger from iter {resume_start_iter + 1}/{cfg.dagger_iters}")
        else:
            warmup_stats = trainer.warmup(env)
            print(f"[Warmup] {json.dumps(warmup_stats, indent=2)}")

        loop_logs = trainer.run_loop(
            env, n_iters=cfg.dagger_iters, checkpoint_dir=checkpoint_dir, start_iter=resume_start_iter,
        )
        train_log_path = trainer.dump_training_log(run_dir / "training_log.json", logs=loop_logs)
        final_ckpt = trainer.save_student_checkpoint(
            run_dir / "model_final.pt",
            extra={"stage": "final", "num_iters": cfg.dagger_iters},
        )

        summary = {
            "run_dir": str(run_dir),
            "status": "completed",
            "start_time": start_time,
            "end_time": dt.datetime.now().isoformat(timespec="seconds"),
            "resolved_device": str(device),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "cmd_dim": cmd_dim,
            "history_len": cfg.history_len,
            "pred_horizon": cfg.pred_horizon,
            "predict_future_obs": bool(cfg.predict_future_obs),
            "policy_arch": "planner_idm",
            "command_input_name": "current_command",
            "command_profile": str(cfg.command_profile),
            "command_components": list(command_components_for_profile(cfg.command_profile)),
            "command_input_semantics": "current_command_broadcast_to_history_tokens",
            "planner_use_action_history": bool(cfg.planner_use_action_history),
            "fdm_enabled": False,
            "use_generalized_idm": bool(cfg.use_generalized_idm),
            "idm_use_teacher_forcing": bool(cfg.idm_use_teacher_forcing),
            "idm_teacher_forcing_ratio": float(cfg.idm_teacher_forcing_ratio),
            "idm_use_current_command_for_history": bool(cfg.idm_use_current_command_for_history),
            "idm_suboptimal_batch_ratio": float(cfg.idm_suboptimal_batch_ratio),
            "online_trajectory_batch_ratio": float(cfg.online_trajectory_batch_ratio),
            "suboptimal_expert_batch_ratio": float(cfg.suboptimal_expert_batch_ratio),
            "warmup_idm_suboptimal_batch_ratio": float(cfg.warmup_idm_suboptimal_batch_ratio),
            "warmup_planner_suboptimal_batch_ratio": float(cfg.warmup_planner_suboptimal_batch_ratio),
            "idm_detach_planner_future_obs": bool(cfg.idm_detach_planner_future_obs),
            "compact_obs_term_scale": cfg.compact_obs_term_scale,
            "compact_obs_term_noise": cfg.compact_obs_term_noise,
            "compact_obs_add_noise": False,
            "compact_obs_source_checkpoint": str(compact_source_checkpoint),
            "planner_obs_loss_coef": float(cfg.planner_obs_loss_coef),
            "idm_action_loss_coef": float(cfg.idm_action_loss_coef),
            "normalize_io": False,
            "shared_norm_source": str(cfg.shared_norm_source),
            "strict_chunk_labels": bool(cfg.strict_chunk_labels),
            "lr_scheduler": "none",
            "warmup_stats": warmup_stats,
            "num_dagger_iters": cfg.dagger_iters,
            "final_checkpoint": str(final_ckpt),
            "offline_buffer_size": int(len(offline_buffer)),
            "optimal_offline_buffer_size": int(len(offline_buffer)),
            "suboptimal_offline_buffer_size": int(len(suboptimal_offline_buffer)),
            "online_buffer_size": int(len(online_buffer)),
            "offline_valid_chunks": int(offline_buffer.num_valid_chunks()),
            "optimal_offline_valid_chunks": int(offline_buffer.num_valid_chunks()),
            "suboptimal_offline_valid_chunks": int(suboptimal_offline_buffer.num_valid_chunks()),
            "online_valid_chunks": int(online_buffer.num_valid_chunks()),
            "offline_read_only": bool(offline_buffer.is_read_only),
            "suboptimal_offline_read_only": bool(suboptimal_offline_buffer.is_read_only),
            "training_log": str(train_log_path),
            "holosoma_config": str(holosoma_config_path),
            "eval_required_separate_step": True,
            "experiment_override_args": list(experiment_override_args),
        }
        _write_json(summary_path, summary)
        _write_json(
            config_snapshot_path,
            {
                **dataclasses.asdict(cfg),
                "resolved_device": str(device),
                "experiment_override_args": list(experiment_override_args),
                "status": "completed",
                "start_time": start_time,
                "end_time": dt.datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(json.dumps(summary, indent=2))
        completed_normally = True
    except Exception as exc:
        _write_json(
            summary_path,
            {
                "run_dir": str(run_dir),
                "status": "failed",
                "start_time": start_time,
                "end_time": dt.datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
                "holosoma_config_path": str(holosoma_config_path),
                "experiment_override_args": list(experiment_override_args),
            },
        )
        raise
    finally:
        if trainer is not None:
            trainer.close()
        elif strict_label_worker is not None:
            strict_label_worker.close()
        if wandb_run is not None and completed_normally:
            wandb_run.finish()
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
