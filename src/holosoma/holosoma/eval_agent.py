from __future__ import annotations

import dataclasses
import inspect
import os
from pathlib import Path

import tyro
from loguru import logger

from holosoma.utils.safe_torch_import import torch

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
):
    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    assert checkpoint_cfg.checkpoint is not None
    checkpoint_str = str(checkpoint_cfg.checkpoint)

    if checkpoint_str.startswith("wandb://"):
        # Download W&B checkpoint into a temporary eval folder
        download_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
        download_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = load_checkpoint(checkpoint_str, str(download_dir))
        checkpoint_path = str(checkpoint)
        checkpoint_dir = os.path.dirname(checkpoint_path)
    else:
        checkpoint_path = str(Path(checkpoint_str).expanduser())
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path, checkpoint_dir)
        checkpoint_path = str(checkpoint)

    # Derive eval log dir next to checkpoint
    eval_subdir = checkpoint_cfg.eval_exp_name or get_timestamp()
    eval_log_dir = Path(checkpoint_dir) / "eval" / eval_subdir
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    # Prepare data collection config if enabled
    collect_data_config = None
    if tyro_config.training.collect_data:
        collect_data_config = {
            "collect_data": True,
            "output_dir": tyro_config.training.data_collection_output_dir,
            "dataset_name": tyro_config.training.data_collection_dataset_name,
            "compress": tyro_config.training.data_collection_compress,
            "batch_size": 10,  # Default batch size
            "skip_obs_keys": tyro_config.training.data_collection_skip_obs_keys,
        }
        logger.info(f"Data collection enabled: {collect_data_config['output_dir']}/{collect_data_config['dataset_name']}.h5")
    
    # Force-adaptive eval freeze: pin ``apply_force_scale`` to a chosen level and
    # disable the curriculum term so subsequent resets cannot bump it back up. No-op for
    # non-force-adaptive envs (which do not own ``apply_force_scale``).
    _ALLOWED_FORCE_LEVELS = (-1, 0, 1, 2)
    if checkpoint_cfg.force_scale_eval_level not in _ALLOWED_FORCE_LEVELS:
        raise ValueError(
            f"--force-scale-eval-level must be one of {_ALLOWED_FORCE_LEVELS} "
            f"(-1=no override, 0/1/2 = freeze at 0.0/0.5/1.0), "
            f"got {checkpoint_cfg.force_scale_eval_level}. "
            f"Higher values would push apply_force_scale beyond the curriculum max (1.0)."
        )
    if checkpoint_cfg.force_scale_eval_level >= 0:
        if hasattr(env, "apply_force_scale"):
            scale = checkpoint_cfg.force_scale_eval_level / 2.0  # 0 -> 0.0, 1 -> 0.5, 2 -> 1.0
            term = None
            if env.curriculum_manager is not None:
                term = env.curriculum_manager.get_term("force_scale_curriculum")
            if term is not None:
                term.enabled = False
            env.apply_force_scale.fill_(scale)
            logger.info(
                f"[eval] force_scale frozen at {scale} (level={checkpoint_cfg.force_scale_eval_level})"
            )
        else:
            logger.warning(
                f"--force-scale-eval-level={checkpoint_cfg.force_scale_eval_level} requested but env "
                f"{type(env).__name__} has no apply_force_scale attribute; ignoring."
            )

    # Optional eval mode: pin ``ref_upper_dof_pos`` to all zeros for the whole run.
    if checkpoint_cfg.upper_pose_zero_only:
        _flag_label = "--upper-pose-zero-only"
        cm = getattr(env, "command_manager", None)
        if cm is None:
            logger.warning(f"[eval] {_flag_label} requested but env has no command_manager; skipping.")
        else:
            ref_term = None
            for name in ("upper_body_ref_pose",):
                t = cm.get_state(name) if hasattr(cm, "get_state") else None
                if t is not None:
                    ref_term = t
                    break
            if ref_term is None:
                logger.warning(
                    f"[eval] {_flag_label} requested but no 'upper_body_ref_pose' "
                    "term found on command manager; skipping."
                )
            else:
                upper_idx = ref_term.upper_dof_indices
                static_upper = torch.zeros_like(env.default_dof_pos[:, upper_idx])
                _pose_label = "all zeros"
                ref_term.ref_upper_dof_pos.copy_(static_upper)
                env.ref_upper_dof_pos.copy_(static_upper)

                def _noop_step(self=ref_term):
                    self.ref_upper_dof_pos.copy_(static_upper)
                    self.env.ref_upper_dof_pos.copy_(static_upper)

                def _noop_reset(env_ids=None, self=ref_term):
                    self.ref_upper_dof_pos.copy_(static_upper)
                    self.env.ref_upper_dof_pos.copy_(static_upper)

                ref_term.step = _noop_step  # type: ignore[assignment]
                ref_term.reset = _noop_reset  # type: ignore[assignment]
                logger.info(
                    f"[eval] {_flag_label}=True → ref_upper_dof_pos pinned to "
                    f"{_pose_label}; AMASS motion clip playback disabled."
                )

    # DEBUG: optional override to bypass is_evaluating-gated branches in env reset
    # so motion sampling, command randomization, and curriculum scaling all run as
    # in training. Use to isolate whether eval-specific code paths are breaking a
    # force-adaptive policy's behavior.
    if os.environ.get("DEBUG_TRAIN_MODE_EVAL", "0") == "1":
        env.is_evaluating = False
        env.set_is_evaluating = lambda *args, **kwargs: None
        logger.info("[eval] DEBUG: DEBUG_TRAIN_MODE_EVAL=1 → forcing is_evaluating=False")

    # Inject eval callbacks if the algo has no eval_callbacks set (e.g., ppo_ma).
    # Loco/slope PPO evals normally use AnalysisPlotLocomotion to produce
    # state_log.npz for plot_eval_logs; keep the same artifact shape for other
    # PPO variants instead of only writing eval_rewards.json.
    if not getattr(algo, "eval_callbacks", None) and (
        getattr(algo, "config", None) is None
        or getattr(algo.config, "eval_callbacks", None) is None
    ):
        try:
            from holosoma.agents.callbacks.analysis_plot_locomotion import AnalysisPlotLocomotion
            from holosoma.agents.callbacks.eval_reward_logger import EvalRewardLogger

            class _AnalysisPlotCfg:
                sim_dt = getattr(env, "dt", 0.02)
                log_dir = None
                log_single_robot = False
                command_seed = tyro_config.training.seed
                command_resample_interval = None
                plot_update_interval = 100

            class _RewLogCfg:
                print_every = 100
                log_dir = None

            algo.eval_callbacks.append(AnalysisPlotLocomotion(_AnalysisPlotCfg(), algo))
            algo.eval_callbacks.append(EvalRewardLogger(_RewLogCfg(), algo))
            logger.info("[eval] auto-injected AnalysisPlotLocomotion + EvalRewardLogger callbacks")
        except Exception as e:
            logger.warning(f"[eval] failed to inject eval callbacks: {e}")

    # Call evaluate_policy with signature-based dispatch so algorithms without
    # data collection support still work.
    eval_policy_sig = inspect.signature(algo.evaluate_policy)
    if "collect_data_config" in eval_policy_sig.parameters:
        algo.evaluate_policy(
            max_eval_steps=tyro_config.training.max_eval_steps,
            collect_data_config=collect_data_config,
        )
    else:
        if collect_data_config is not None:
            logger.warning(
                f"{algo_class.__name__} does not support data collection during eval; running without collection."
            )
        algo.evaluate_policy(
            max_eval_steps=tyro_config.training.max_eval_steps,
        )

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    # Overlay DR and terrain from another checkpoint (e.g. phase2) when using a phase1 checkpoint
    if checkpoint_cfg.dr_terrain_from_checkpoint:
        dr_cfg, _ = load_saved_experiment_config(
            CheckpointConfig(checkpoint=checkpoint_cfg.dr_terrain_from_checkpoint)
        )
        overwritten_tyro_config = dataclasses.replace(
            overwritten_tyro_config,
            randomization=dr_cfg.randomization,
            terrain=dr_cfg.terrain,
        )
        logger.info(
            f"Using randomization and terrain from {checkpoint_cfg.dr_terrain_from_checkpoint}"
        )
    if checkpoint_cfg.randomize_commands and overwritten_tyro_config.command is not None:
        flipped: list[str] = []
        for term_name, term_cfg in overwritten_tyro_config.command.setup_terms.items():
            params = term_cfg.params if isinstance(term_cfg.params, dict) else {}
            if "command_ranges" not in params:
                continue  # Not a LocomotionCommand-style term — skip.
            params["allow_eval_randomization"] = True
            flipped.append(term_name)
        if flipped:
            logger.info(
                f"--randomize-commands=True → enabled allow_eval_randomization on: {', '.join(flipped)}"
            )

    # When pinning the upper-body reference to zero, the AMASS motion clip is never read
    # — but ``UpperBodyRefPoseCommand.setup`` would still try to construct a MotionLibRobot
    # and load files from the saved ``robot.motion`` path. On a machine that doesn't have
    # those motion files, env construction fails before our reset/step override kicks in.
    # Drop ``robot.motion`` here so MotionLib is skipped entirely.
    if (
        checkpoint_cfg.upper_pose_zero_only
        and overwritten_tyro_config.robot is not None
        and getattr(overwritten_tyro_config.robot, "motion", None) is not None
    ):
        overwritten_tyro_config = dataclasses.replace(
            overwritten_tyro_config,
            robot=dataclasses.replace(overwritten_tyro_config.robot, motion=None),
        )
        logger.info(
            "--upper-pose-zero-only → cleared robot.motion in saved config; motion files no longer required."
        )
    if checkpoint_cfg.force_randomize_init_levels and overwritten_tyro_config.curriculum is not None:
        new_setup = {}
        for name, term in overwritten_tyro_config.curriculum.setup_terms.items():
            if "terrain_level_curriculum" in name:
                new_setup[name] = dataclasses.replace(term, params={**term.params, "randomize_init_levels": True})
                logger.info(f"[eval] --force-randomize-init-levels: injected randomize_init_levels=True into '{name}'")
            else:
                new_setup[name] = term
        overwritten_tyro_config = dataclasses.replace(
            overwritten_tyro_config,
            curriculum=dataclasses.replace(overwritten_tyro_config.curriculum, setup_terms=new_setup),
        )

    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
