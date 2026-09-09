#!/usr/bin/env python3
"""
Policy Runner Script with Tyro Configuration

This script uses Tyro configuration system to run different policy types.

Usage:
    python run_policy.py inference:g1-29dof-loco --task.model-path path/to/model.onnx
    python run_policy.py inference:g1-29dof-loco-transformer --task.model-path path/to/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path wandb://project/run/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path https://wandb-url/files/model.onnx
"""

from __future__ import annotations

import atexit
import random
import signal
import sys
import traceback
import warnings
from pathlib import Path

# Pydantic v2 emits UnsupportedFieldAttributeWarning for frozen/repr on dataclass
# fields. Filter before any holosoma_inference imports.
warnings.filterwarnings("ignore", category=UserWarning, message=".*UnsupportedFieldAttribute.*")

import numpy as np
import torch
import tyro
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import get_annotated_inference_config
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.dual_mode import DualModePolicy, _select_policy_class
from holosoma_inference.policies.locomotion import (
    LocomotionPolicy,
    LocomotionPolicy_Deploy,
)
from holosoma_inference.policies.locomotion_fada import (
    LocomotionPolicy_FADA,
)
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.utils.misc import restore_terminal_settings

_MODEL_CONTRACT_MISMATCH_MARKERS = (
    "ONNX model missing required",
    "MLP inputs",
    "MLP output",
    "MLP metadata missing required fields",
    "Transformer inputs",
    "Transformer output",
    "Transformer metadata missing required fields",
)


def _is_model_contract_mismatch(error: Exception) -> bool:
    msg = str(error)
    return any(marker in msg for marker in _MODEL_CONTRACT_MISMATCH_MARKERS)


def _auto_policy_candidates(config: InferenceConfig) -> list[type]:
    actor_obs = config.observation.obs_dict.get("actor_obs", [])

    if "motion_command" in actor_obs:
        return [WholeBodyTrackingPolicy]
    return [LocomotionPolicy_Deploy]


def _resolve_policy_candidates(config: InferenceConfig) -> tuple[str, list[type], bool]:
    mode = getattr(config.task, "policy_mode", "auto")

    if mode == "auto":
        return mode, _auto_policy_candidates(config), True

    explicit_map: dict[str, list[type]] = {
        # "transformer" (plain compact-obs transformer, no planner/IDM teacher heads) and "fada"
        # both dispatch to LocomotionPolicy_FADA -- it gracefully degrades to plain-transformer
        # behavior when the loaded ONNX has no IDM/FDM teacher I/O. See dual_mode.py.
        "transformer": [LocomotionPolicy_FADA],
        "fada": [LocomotionPolicy_FADA],
        "deploy": [LocomotionPolicy_Deploy],
        "wbt": [WholeBodyTrackingPolicy],
    }
    if mode not in explicit_map:
        supported = ", ".join(["auto", *explicit_map.keys()])
        raise ValueError(f"Unsupported task.policy_mode='{mode}'. Supported values: {supported}")
    return mode, explicit_map[mode], False


def _print_control_guide(policy_class, use_joystick: bool, dual_mode: bool = False, session_mode: bool = False):
    """Print control guide for users."""
    is_wbt = policy_class.__name__ == "WholeBodyTrackingPolicy"

    logger.info("=" * 80)
    logger.info("🎮 POLICY CONTROLS")
    logger.info("=" * 80)
    logger.info("")

    if use_joystick:
        logger.info("📝 Using JOYSTICK control mode")
        logger.info("")
        logger.info("General Controls:")
        logger.info("  A button       - Start the policy")
        logger.info("  B button       - Stop the policy")
        logger.info("  Y button       - Set robot to default pose")
        logger.info("  L1+R1 (LB+RB)  - Kill controller program")

        if is_wbt:
            logger.info("")
            logger.info("Whole-Body Tracking Controls:")
            logger.info("  Start button   - Start motion clip")
        else:
            logger.info("")
            logger.info("Locomotion Controls:")
            logger.info("  Start button   - Switch walking/standing mode")
            logger.info("  Left stick     - Adjust linear velocity (forward/backward/left/right)")
            logger.info("  Right stick    - Adjust angular velocity (turn left/right)")
    else:
        logger.info("⌨️  Using KEYBOARD control mode")
        logger.info("")
        logger.info("⚠️  IMPORTANT: Make sure THIS TERMINAL is active to receive keyboard input!")
        logger.info("⚠️  All commands below must be entered in THIS terminal window.")
        logger.info("")
        logger.info("General Controls:")
        logger.info("  ]  - Start the policy")
        logger.info("  o  - Stop the policy")
        logger.info("  i  - Set robot to default pose")

        if is_wbt:
            logger.info("")
            logger.info("Whole-Body Tracking Controls:")
            logger.info("  s  - Start motion clip")
        else:
            logger.info("")
            logger.info("Locomotion Controls:")
            logger.info("  =          - Switch walking/standing mode")
            logger.info("  w/s        - Increase/decrease forward velocity")
            logger.info("  a/d        - Increase/decrease lateral velocity")
            logger.info("  q/e        - Increase/decrease angular velocity (turn left/right)")
            logger.info("  z          - Set all velocities to zero")

    logger.info("")
    logger.info("🎬 MuJoCo Simulator Controls (⚠️  ONLY in MuJoCo window, NOT this terminal!):")
    logger.info("  7/8        - Decrease/increase elastic band length")
    logger.info("  9          - Toggle elastic band enable/disable")
    logger.info("  BACKSPACE  - Reset simulation")

    if dual_mode:
        logger.info("")
        logger.info("🔀 Dual-Mode Controls:")
        if session_mode:
            if use_joystick:
                logger.info("  X button       - Toggle robust secondary <-> test primary session")
            else:
                logger.info("  x              - Toggle robust secondary <-> test primary session")
            logger.info("  Startup        - Robot begins in robust secondary standing mode")
            logger.info("  Test session   - Primary starts from command sequence step 0 every time")
            logger.info("  Exit           - Test completion returns to robust; Ctrl+C ends the process")
        else:
            if use_joystick:
                logger.info("  X button       - Switch between primary and secondary policy")
            else:
                logger.info("  x              - Switch between primary and secondary policy")

    logger.info("")
    logger.info("=" * 80)
    logger.info("👆 Press the appropriate button/key to begin!")
    logger.info("=" * 80)
    logger.info("")


def _set_global_seed(seed: int):
    """Set seeds for reproducible command generation and randomness."""
    logger.info(f"Setting global seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def run_policy(config: InferenceConfig):
    """Run policy with Tyro configuration."""
    logger.info("🚀 Starting Policy with Tyro configuration...")
    logger.info(f"🤖 Robot: {config.robot.robot_type}")
    logger.info(f"📋 Observation groups: {list(config.observation.obs_dict.keys())}")
    logger.info(f"⚙️ RL Rate: {config.task.rl_rate} Hz")
    logger.info(f"📁 Model path: {config.task.model_path}")

    if config.task.seed is not None:
        _set_global_seed(config.task.seed)

    do_plot_on_exit = getattr(config.task, "plot_mocap_raw_on_exit", True)
    checkpoint_path = getattr(config.task, "tracking_reward_checkpoint_path", None)
    proc = None
    save_path = None
    exit_plot_invoked = False
    previous_sigint_handler = None
    previous_sigterm_handler = None

    def _invoke_exit_plot_once() -> None:
        nonlocal exit_plot_invoked
        if exit_plot_invoked or proc is None or not do_plot_on_exit:
            return
        exit_plot_invoked = True
        proc.plot_mocap_raw_on_exit(
            save_path=save_path,
            tracking_reward_checkpoint_path=checkpoint_path,
        )

    def _handle_termination(signum, _frame) -> None:
        _invoke_exit_plot_once()
        restore_terminal_settings()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    try:
        dual_mode = config.secondary is not None
        session_mode = dual_mode and bool(getattr(config.task, "dual_mode_session_enabled", False))

        if dual_mode:
            policy_class = _select_policy_class(config)
            logger.info(f"Using {policy_class.__name__} (dual-mode enabled)")
            policy = DualModePolicy(primary_config=config, secondary_config=config.secondary)
        else:
            # Explicit task.policy_mode and/or auto fallback (deploy / fada / transformer / …).
            policy_class = None
            policy_mode, candidate_classes, allow_fallback = _resolve_policy_candidates(config)
            logger.info(
                f"Policy selection mode: {policy_mode} | candidates: {[cls.__name__ for cls in candidate_classes]}"
            )

            last_error = None
            policy = None
            for cls in candidate_classes:
                try:
                    logger.info(f"Trying policy class {cls.__name__}")
                    policy = cls(config=config)
                    policy_class = cls
                    break
                except ValueError as exc:
                    last_error = exc
                    if allow_fallback and _is_model_contract_mismatch(exc):
                        logger.warning(
                            f"{cls.__name__} rejected due to model contract mismatch, trying fallback: {exc}"
                        )
                        continue
                    raise

            if policy_class is None:
                if last_error is not None:
                    raise RuntimeError(
                        f"Failed to initialize policy in mode '{policy_mode}'. Last error: {last_error}"
                    ) from last_error
                raise RuntimeError(f"Failed to initialize any policy class for mode '{policy_mode}'.")

            logger.info(f"Using {policy_class.__name__}")

        plot_owner = policy.primary if dual_mode and hasattr(policy, "primary") else policy
        proc = getattr(getattr(plot_owner, "interface", None), "vel_state_processor", None)
        if proc is not None and do_plot_on_exit:
            log_dir = (
                plot_owner._resolve_log_output_dir()
                if hasattr(plot_owner, "_resolve_log_output_dir")
                else "logs/inference"
            )
            save_path = str(Path(log_dir) / "mocap_raw.png")
            atexit.register(_invoke_exit_plot_once)
            previous_sigint_handler = signal.signal(signal.SIGINT, _handle_termination)
            previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_termination)
            logger.info(
                "Unified log and exit plots enabled: mocap_unified.npz + PNGs will be saved on exit; figures will pop up."
            )
        elif do_plot_on_exit and proc is None:
            logger.warning(
                "plot_mocap_raw_on_exit is True but no vel_state_processor (vel_state_source may be 'none'). "
                "Set --task.vel-state-source zmq and connect mocap to save log and plots."
            )

        logger.info("✅ Policy initialized successfully!")
        _print_control_guide(policy_class, config.task.use_joystick, dual_mode=dual_mode, session_mode=session_mode)
        policy.run()
        logger.info("✅ Policy execution completed!")

    except Exception as e:
        logger.error(f"❌ Error running policy: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            _invoke_exit_plot_once()
        finally:
            if previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, previous_sigint_handler)
            if previous_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)
            restore_terminal_settings()


def _split_secondary_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split --secondary.* args out of argv, renaming them for standalone parsing.

    Returns (primary_argv, secondary_argv) where secondary args have the
    ``--secondary.`` prefix stripped, e.g. ``--secondary.task.model-path X``
    becomes ``--task.model-path X``.
    """
    primary = []
    secondary = []
    expect_secondary_value = False
    for arg in argv:
        if arg.startswith("--secondary."):
            renamed = "--" + arg[len("--secondary.") :]
            secondary.append(renamed)
            # If not --key=value form, the next token might be the value
            expect_secondary_value = "=" not in renamed
        elif expect_secondary_value and not arg.startswith("--"):
            secondary.append(arg)
            expect_secondary_value = False
        else:
            primary.append(arg)
            expect_secondary_value = False
    return primary, secondary


def main(annotated_config=None):
    """Main entry point. Extensions can pass their own AnnotatedInferenceConfig."""
    import argparse

    from holosoma_inference.config.config_values.inference import DEFAULTS

    # Pre-parse --secondary-preset and --secondary none before tyro.
    # Tyro can't build a CLI parser for InferenceConfig | None when it
    # contains dict[str, Any] fields, so we handle secondary selection ourselves.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument(
        "--secondary-preset",
        default=None,
        metavar="NAME",
        help=f"Select a preset for the secondary policy. Choices: {list(DEFAULTS.keys())}",
    )
    pre.add_argument("--secondary", default=None, help="Set to 'none' to disable dual-mode.")
    known, remaining = pre.parse_known_args()

    disable_secondary = known.secondary is not None and known.secondary.lower() == "none"
    secondary_preset = known.secondary_preset

    # Strip --secondary.* args from remaining so tyro doesn't see them
    primary_argv, secondary_argv = _split_secondary_args(remaining)
    sys.argv = [sys.argv[0]] + primary_argv

    if annotated_config is None:
        # Use factory function to lazily load extension configs
        annotated_config = get_annotated_inference_config()
    config = tyro.cli(annotated_config, config=TYRO_CONFIG)

    from dataclasses import replace as _replace

    if disable_secondary:
        config = _replace(config, secondary=None)
    elif secondary_preset:
        preset = DEFAULTS.get(secondary_preset)
        if preset is None:
            logger.error(f"Unknown secondary preset: {secondary_preset}")
            logger.info(f"Available presets: {list(DEFAULTS.keys())}")
            sys.exit(1)
        preset = _replace(preset, secondary=None)

        # Parse secondary overrides against the preset defaults
        if secondary_argv:
            sys.argv = [sys.argv[0]] + secondary_argv
            secondary = tyro.cli(InferenceConfig, default=preset, config=TYRO_CONFIG)
        else:
            secondary = preset
        config = _replace(config, secondary=secondary)
    elif secondary_argv:
        # --secondary.* overrides on the config's default secondary
        if config.secondary is not None:
            sys.argv = [sys.argv[0]] + secondary_argv
            secondary = tyro.cli(InferenceConfig, default=config.secondary, config=TYRO_CONFIG)
            config = _replace(config, secondary=secondary)
        else:
            logger.warning("--secondary.* args ignored: no default secondary in this config")

    run_policy(config)


if __name__ == "__main__":
    main()
