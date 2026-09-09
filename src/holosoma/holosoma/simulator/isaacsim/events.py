from __future__ import annotations

from typing import Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def resolve_dist_fn(
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    dist_fn = math_utils.sample_uniform

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")

    return dist_fn


def randomize_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_envs: int = 1,  # number of environments
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples
    random values from the given distribution parameters and adds, scales or sets the values into the
    physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        # Check if body_names is provided, if so resolve them to indices
        if asset_cfg.body_names is not None:
            body_ids, _ = asset.find_bodies(asset_cfg.body_names)
            body_ids = torch.tensor(body_ids, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms()
    # apply randomization on default values
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (distribution_params[0].to(coms.device), distribution_params[1].to(coms.device))

    env.base_com_bias[env_ids, :] = dist_fn(
        *distribution_params, (env_ids.shape[0], env.base_com_bias.shape[1]), device=coms.device
    )

    # sample from the given range
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    inertia_distribution_params is a tuple of (min, max) or (min_tensor, max_tensor) for the inertia values.
    The inertia is a 3x3 matrix, which is symmetric, we only need to randomize the diagonal elements.
    The inertia matrix is stored in the order of Ixx, Iyx, Izx, Ixy, Iyy, Izy, Ixz, Iyz, Izz.
    https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectData.default_inertia


    The inertia distribution params is stored in the order of Ixx, Iyy, Izz, Ixy, Iyz, Ixz.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        # Check if body_names is provided, if so resolve them to indices
        if asset_cfg.body_names is not None:
            body_ids, _ = asset.find_bodies(asset_cfg.body_names)
            body_ids = torch.tensor(body_ids, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    dist_fn = resolve_dist_fn(distribution)
    inertias_original = asset.root_physx_view.get_inertias()  # (num_envs, 9) or (num_envs, num_bodies, 9)

    if inertias_original.ndim == 2:
        inertias = inertias_original.unsqueeze(1).clone()  # Add body_ids dimension (num_envs, 1, 9)
    else:
        inertias = inertias_original.clone()  # (num_envs, num_bodies, 9)

    # The inertia distribution params is stored in the order of Ixx, Iyy, Izz, Ixy, Iyz, Ixz.
    inertia_random = dist_fn(
        *inertia_distribution_params, (env_ids.shape[0], body_ids.shape[0], 6), device=inertias.device
    )

    # Storage order: Ixx, Iyx, Izx, Ixy, Iyy, Izy, Ixz, Iyz, Izz (indices 0-8)
    inertias_bias = torch.ones_like(inertias)
    # Diagonal elements: (param_idx, matrix_idx)
    diagonal_elements = [(0, 0), (1, 4), (2, 8)]  # Ixx, Iyy, Izz
    # Off-diagonal elements: (param_idx, matrix_idx_primary, matrix_idx_symmetric)
    off_diagonal_elements = [(3, 1, 3), (4, 7, 5), (5, 6, 2)]  # Ixy, Iyz, Ixz

    # Sample diagonal elements
    for param_idx, matrix_idx in diagonal_elements:
        inertias_bias[:, :, matrix_idx] = inertia_random[:, :, param_idx]

    # Sample off-diagonal elements and set symmetric pairs
    for param_idx, primary_idx, symmetric_idx in off_diagonal_elements:
        inertias_bias[:, :, primary_idx] = inertia_random[:, :, param_idx]
        inertias_bias[:, :, symmetric_idx] = inertias_bias[:, :, primary_idx]

    if operation == "add":
        inertias[env_ids[:, None], body_ids] += inertias_bias
    elif operation == "scale":
        inertias[env_ids[:, None], body_ids] *= inertias_bias
    elif operation == "abs":
        inertias[env_ids[:, None], body_ids] = inertias_bias
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )

    if inertias_original.ndim == 2:
        inertias = inertias.squeeze(1)

    asset.root_physx_view.set_inertias(inertias, env_ids)
