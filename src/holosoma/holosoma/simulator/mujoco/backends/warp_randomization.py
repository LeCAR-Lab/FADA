"""MuJoCo Warp domain randomization utilities.

Adapted from mjlab (Apache 2.0): https://github.com/mujocolab/mjlab/blob/main/src/mjlab/sim/randomization.py
See THIRD_PARTY_LICENSES for full license text.

Provides functions for randomizing model fields across batched environments
in GPU-accelerated MuJoCo Warp simulations.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, cast

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp


@wp.kernel(module="unique")
def repeat_array_kernel(
    src: wp.array(dtype=Any),  # type: ignore[valid-type]
    nelems_per_world: int,
    dst: wp.array(dtype=Any),  # type: ignore[valid-type]
):
    """Warp kernel to repeat array elements across worlds."""
    tid = wp.tid()
    src_idx = tid % nelems_per_world
    dst[tid] = src[src_idx]  # type: ignore[index]


def expand_model_fields(
    model: mjwarp.Model,
    nworld: int,
    fields_to_expand: list[str],
) -> None:
    """Expand model fields to support per-environment randomization.

    Tiles single-world model fields across all environments, enabling
    per-environment physics parameter randomization. This must be called
    BEFORE the randomization manager is initialized.

    Parameters
    ----------
    model : mjwarp.Model
        MuJoCo Warp model to expand
    nworld : int
        Number of parallel environments
    fields_to_expand : list[str]
        List of field names to expand (e.g., ['body_mass', 'geom_friction'])
    """
    if nworld == 1:
        return

    # Initialize registry to track which fields have been expanded
    if not hasattr(model, "_expanded_fields"):
        model._expanded_fields = set()  # type: ignore[attr-defined]

    def tile(x: wp.array) -> wp.array:
        """Tile a Warp array across environments."""
        # Create new array with same shape but first dim multiplied by nworld.
        new_shape = list(x.shape)
        new_shape[0] = nworld
        wp_array = cast(
            "Callable[..., Any]", {1: wp.array, 2: wp.array2d, 3: wp.array3d, 4: wp.array4d}[len(new_shape)]
        )
        dst = wp_array(shape=new_shape, dtype=x.dtype, device=x.device)

        src_flat = x.flatten()
        dst_flat = dst.flatten()

        # Launch kernel to repeat data, one thread per destination element.
        n_elems_per_world = dst_flat.shape[0] // nworld
        wp.launch(
            repeat_array_kernel,
            dim=dst_flat.shape[0],
            inputs=[src_flat, n_elems_per_world],
            outputs=[dst_flat],
            device=x.device,
        )
        return dst

    for field in model.__dataclass_fields__:
        if field in fields_to_expand:
            array = getattr(model, field)
            setattr(model, field, tile(array))
            # Register this field as expanded for validation
            model._expanded_fields.add(field)  # type: ignore[attr-defined]


def resolve_entity_ids(mj_model: mujoco.MjModel, names: list[str], entity_type: str) -> list[int]:
    """Resolve entity names to MuJoCo indices.

    Parameters
    ----------
    mj_model : mujoco.MjModel
        The CPU MuJoCo model
    names : List[str]
        List of entity names to resolve
    entity_type : str
        The type of entity ("body", "geom", "joint", "site", "actuator", etc.)

    Returns
    -------
    List[int]
        List of MuJoCo indices corresponding to the entity names

    Raises
    ------
    ValueError
        If entity type is unknown or entity name is not found
    """
    # Map string type to MuJoCo enum
    type_map = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "camera": mujoco.mjtObj.mjOBJ_CAMERA,
        "sensor": mujoco.mjtObj.mjOBJ_SENSOR,
        "light": mujoco.mjtObj.mjOBJ_LIGHT,
        "mesh": mujoco.mjtObj.mjOBJ_MESH,
        "texture": mujoco.mjtObj.mjOBJ_TEXTURE,
        "material": mujoco.mjtObj.mjOBJ_MATERIAL,
    }

    if entity_type.lower() not in type_map:
        raise ValueError(f"Unknown entity type: '{entity_type}'. Supported: {list(type_map.keys())}")

    obj_type = type_map[entity_type.lower()]
    indices = []

    for name in names:
        idx = mujoco.mj_name2id(mj_model, obj_type, name)
        if idx == -1:
            # Try prefixed name
            idx_prefixed = mujoco.mj_name2id(mj_model, obj_type, "robot_" + name)
            if idx_prefixed == -1:
                raise ValueError(f"Entity '{name}' of type '{entity_type}' not found in model.")
            idx = idx_prefixed
        indices.append(idx)

    return indices


def randomize_field(
    simulator: Any,
    field: str,
    ranges: tuple[float, float] | dict[int, tuple[float, float]],
    env_ids: torch.Tensor | None = None,
    entity_ids: torch.Tensor | None = None,
    entity_names: list[str] | None = None,
    entity_type: str | None = None,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    operation: Literal["add", "scale", "abs"] = "abs",
):
    """Unified model randomization function for MuJoCo Warp.

    Randomizes physics parameters in the MuJoCo model for specified environments
    and entities. Supports vectorized operations for efficient GPU execution.

    Parameters
    ----------
    simulator : Any
        The simulator instance with backend.warp_model_bridge
    field : str
        Model field name to randomize (e.g., 'body_mass', 'geom_friction', 'body_ipos')
    ranges : Union[Tuple[float, float], Dict[int, Tuple[float, float]]]
        Range(s) for randomization. Can be:
        - Single tuple (min, max) for scalar fields or all axes
        - Dict mapping axis indices to ranges for vector fields
    env_ids : Optional[torch.Tensor]
        Environment IDs to randomize (default: all environments)
    entity_ids : Optional[torch.Tensor]
        Entity IDs to randomize (default: all entities)
    entity_names : Optional[List[str]]
        Entity names to resolve to IDs (mutually exclusive with entity_ids)
    entity_type : Optional[str]
        Type of entity for name resolution (e.g., 'body', 'geom')
        Required if entity_names is provided, otherwise inferred from field
    distribution : Literal["uniform", "log_uniform", "gaussian"]
        Distribution to sample from (default: "uniform")
    operation : Literal["add", "scale", "abs"]
        Operation to apply: add to current, scale current, or set absolute value

    Raises
    ------
    ValueError
        If both entity_ids and entity_names are specified
        If entity_type cannot be inferred from field name
    """
    device = simulator.sim_device

    # -----------------------------------------------------------
    # 0. Pre-resolution: Name -> ID Logic
    # -----------------------------------------------------------
    if entity_names is not None:
        if entity_ids is not None:
            raise ValueError("Cannot specify both 'entity_ids' and 'entity_names'. Choose one.")
        # 1. Access the CPU model to look up names
        mj_model = simulator.backend.model

        # 2. Infer entity type if not provided
        if entity_type is None:
            # Simple heuristic based on common naming conventions
            if field.startswith("body_"):
                entity_type = "body"
            elif field.startswith("geom_"):
                entity_type = "geom"
            elif field.startswith(("jnt_", "joint_")):
                entity_type = "joint"
            elif field.startswith("site_"):
                entity_type = "site"
            elif field.startswith(("actuator_", "gear")):
                entity_type = "actuator"
            else:
                raise ValueError(
                    f"Could not infer entity type for field '{field}'. "
                    "Please provide explicit 'entity_type' (e.g., 'body', 'geom')."
                )

        # 3. Resolve names to integer list
        ids_list = resolve_entity_ids(mj_model, entity_names, entity_type)
        entity_ids = torch.tensor(ids_list, device=device, dtype=torch.long)

    # -----------------------------------------------------------
    # 1. Retrieve the Field and Determine Shapes
    # -----------------------------------------------------------
    model_field = getattr(simulator.backend.warp_model_bridge, field)
    full_shape = model_field.shape

    ndim = len(full_shape)
    n_world = full_shape[0]
    n_total_entities = full_shape[1]

    # -----------------------------------------------------------
    # 1.5. Validate Field Expansion
    # -----------------------------------------------------------
    # Check if field has been explicitly expanded via expand_model_fields()
    # when there are multiple environments
    num_envs = simulator.num_envs
    if num_envs > 1:
        mjw_model = simulator.backend.mjw_model
        expanded_fields: set[str] = getattr(mjw_model, "_expanded_fields", set())

        if field not in expanded_fields:
            raise ValueError(
                f"Field '{field}' has not been expanded for per-environment randomization. "
                f"Did you forget to add @mujoco_required_field('{field}') to your randomization function? "
                f"Currently expanded fields: {sorted(expanded_fields) if expanded_fields else 'none'}"
            )

    # -----------------------------------------------------------
    # 2. Resolve Indices (Broadcasting Prep)
    # -----------------------------------------------------------
    if env_ids is None:
        env_ids = torch.arange(n_world, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device, dtype=torch.long)

    if entity_ids is None:
        entity_ids = torch.arange(n_total_entities, device=device, dtype=torch.long)
    else:
        entity_ids = entity_ids.to(device, dtype=torch.long)

    # -- Target Axes & Ranges --
    target_axes: torch.Tensor | None = None

    if ndim == 3:
        # Vector field
        if isinstance(ranges, dict):
            axes_list = sorted(ranges.keys())
            target_axes = torch.tensor(axes_list, device=device, dtype=torch.long)
            range_vals = [ranges[ax] for ax in axes_list]
        else:
            target_axes = torch.arange(full_shape[2], device=device, dtype=torch.long)
            range_vals = [ranges] * full_shape[2]

        axis_ranges = torch.tensor(range_vals, device=device, dtype=torch.float32)

    else:
        # Scalar field
        if isinstance(ranges, dict):
            raise ValueError("Cannot specify axis dict for a scalar (2D) field.")
        target_axes = None
        axis_ranges = torch.tensor([ranges], device=device, dtype=torch.float32)

    # -----------------------------------------------------------
    # 3. Create Broadcasting Views
    # -----------------------------------------------------------
    idx_env = env_ids.view(-1, 1, 1)  # (N, 1, 1)
    idx_ent = entity_ids.view(1, -1, 1)  # (1, M, 1)

    indexer: tuple[torch.Tensor, ...]
    if target_axes is not None:
        idx_ax = target_axes.view(1, 1, -1)  # (1, 1, K)
        indexer = (idx_env, idx_ent, idx_ax)
    else:
        indexer = (idx_env.squeeze(-1), idx_ent.squeeze(-1))

    # -----------------------------------------------------------
    # 4. Generate Random Values
    # -----------------------------------------------------------
    n_e = len(env_ids)
    n_n = len(entity_ids)
    n_a = len(target_axes) if target_axes is not None else 1

    lo = axis_ranges[:, 0].view(1, 1, -1)
    hi = axis_ranges[:, 1].view(1, 1, -1)
    shape = (n_e, n_n, n_a)

    if distribution == "uniform":
        rv = torch.rand(shape, device=device)
        random_values = lo + (hi - lo) * rv

    elif distribution == "log_uniform":
        log_lo = torch.log(lo)
        log_hi = torch.log(hi)
        rv = torch.rand(shape, device=device)
        random_values = torch.exp(log_lo + (log_hi - log_lo) * rv)

    elif distribution == "gaussian":
        mean = 0.5 * (lo + hi)
        std = (hi - lo) / 6.0
        random_values = torch.randn(shape, device=device) * std + mean
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    if target_axes is None:
        random_values = random_values.squeeze(-1)

    # -----------------------------------------------------------
    # 5. Apply Operation
    # -----------------------------------------------------------
    current_data = model_field[indexer]

    if operation == "add":
        model_field[indexer] = current_data + random_values
    elif operation == "scale":
        model_field[indexer] = current_data * random_values
    elif operation == "abs":
        model_field[indexer] = random_values
    else:
        raise ValueError(f"Unknown operation: {operation}")
