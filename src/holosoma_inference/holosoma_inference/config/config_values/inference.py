"""Default inference configurations for holosoma_inference."""

from dataclasses import replace
from importlib.metadata import entry_points

import tyro
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import observation, robot, task

g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.locomotion,
)

t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# fmt: off
g1_29dof_wbt = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0, 0.0, 0.0,                          # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
# fmt: on
    observation=observation.wbt,
    task=task.wbt,
)

DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
}

# Auto-discover inference configs from installed extensions
for ep in entry_points(group="holosoma.config.inference"):
    DEFAULTS[ep.name] = ep.load()

AnnotatedInferenceConfig = Annotated[
    InferenceConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults({f"inference:{k}": v for k, v in DEFAULTS.items()})
        ),
]

# ---------------------------------------------------------------------------
# FADA: additional robot/task variants, a shared safety secondary policy, and
# lazily-loaded extension discovery (avoids circular imports when extensions
# import from holosoma_inference.config at module load time). Everything below
# only adds to DEFAULTS.
# ---------------------------------------------------------------------------

# Shared safety secondary for all G1 configs — FastSAC locomotion.
# Each config references the same object; users can override any field
# with --secondary.task.model-path etc., or disable with --secondary none.
_g1_safety_secondary = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.safety_locomotion_g1,
)

g1_29dof_loco = replace(g1_29dof_loco, secondary=_g1_safety_secondary)
g1_29dof_wbt = replace(g1_29dof_wbt, secondary=_g1_safety_secondary)

# G1 Locomotion — Deploy / Transformer (29 DoF; same interface as T1-23dof)
g1_29dof_loco_deploy = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof_deploy,
    task=replace(task.locomotion, policy_mode="deploy"),
)

g1_29dof_loco_transformer = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof_transformer,
    task=replace(task.locomotion, policy_mode="transformer"),
)

# T1 Locomotion (23 DoF)
t1_23dof_loco = InferenceConfig(
    robot=robot.t1_23dof,
    observation=observation.loco_t1_23dof,
    task=replace(task.locomotion, policy_mode="auto"),
)

t1_23dof_loco_deploy = InferenceConfig(
    robot=robot.t1_23dof,
    observation=observation.loco_t1_23dof_deploy,
    task=replace(task.locomotion, policy_mode="deploy"),
)

t1_23dof_loco_transformer = InferenceConfig(
    robot=robot.t1_23dof,
    observation=observation.loco_t1_23dof_transformer,
    task=replace(task.locomotion, policy_mode="transformer"),
)

DEFAULTS.update(
    {
        "g1-29dof-loco-deploy": g1_29dof_loco_deploy,
        "g1-29dof-loco-transformer": g1_29dof_loco_transformer,
        "t1-23dof-loco": t1_23dof_loco,
        "t1-23dof-loco-deploy": t1_23dof_loco_deploy,
        "t1-23dof-loco-transformer": t1_23dof_loco_transformer,
    }
)

# FADA deployment presets (README FADA steps 4 and 6).
#
# On top of `*-loco-deploy` these set two task fields:
#   * policy_mode="fada"      -> dispatches to LocomotionPolicy_FADA
#   * randomize_commands=True -> scripted randomized velocity commands
# The rest comes from the `task.locomotion` preset (command_resampling_time=4.0,
# auto_start_policy=True).
#
# Per-run choices are not set here and must still be passed:
# --task.model-path, --task.seed, --task.max-steps, --task.collect-data.
#
# robot=/observation= mirror the corresponding `*_loco_deploy` preset, so a FADA rollout
# observes exactly what the deploy preset observes.
t1_23dof_loco_fada = InferenceConfig(
    robot=robot.t1_23dof,
    observation=observation.loco_t1_23dof_deploy,
    task=replace(task.locomotion, policy_mode="fada", randomize_commands=True),
)

g1_29dof_loco_fada = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof_deploy,
    task=replace(task.locomotion, policy_mode="fada", randomize_commands=True),
)

DEFAULTS.update(
    {
        "t1-23dof-loco-fada": t1_23dof_loco_fada,
        "g1-29dof-loco-fada": g1_29dof_loco_fada,
    }
)

# Track whether (a second round of) extensions have been loaded lazily.
_extensions_loaded = False


def _load_extensions() -> None:
    """Lazily (re)load extension configs from entry points.

    This is deferred to avoid circular imports when extensions import
    from holosoma_inference.config at module load time.
    """
    global _extensions_loaded  # noqa: PLW0603
    if _extensions_loaded:
        return
    _extensions_loaded = True
    for ep in entry_points(group="holosoma.config.inference"):
        DEFAULTS[ep.name] = ep.load()


def get_annotated_inference_config() -> type:
    """Build the annotated InferenceConfig type with all discovered configs.

    This function loads extension configs lazily and returns a tyro-compatible
    annotated type for CLI subcommand generation.

    Returns:
        Annotated type suitable for use with tyro.cli()
    """
    _load_extensions()
    return Annotated[
        InferenceConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"inference:{k}": v for k, v in DEFAULTS.items()}
            )
        ),
    ]


def get_defaults() -> dict:
    """Get all inference config defaults, including extensions.

    Returns:
        Dictionary mapping config names to InferenceConfig instances.
    """
    _load_extensions()
    return DEFAULTS
