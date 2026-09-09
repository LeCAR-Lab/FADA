# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab v2.0.0 (https://github.com/isaac-sim/IsaacLab)
# Contributors: https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md
#
# This file contains utilities adapted from Isaac Lab's spawner code.

"""Custom USD file spawners with source prim path support.

This module provides custom USD file spawning functionality that supports loading
specific prims from USD files.

Key Differences from IsaacLab's Built-in from_files.py:

1. **Source Path Support**: Added 'source_path' parameter to load specific prims from USD files
   instead of loading the entire file. This enables selective loading from complex USD scenes.

2. **Enhanced Schema Handling**: Implements USD schema (API) handling for properties that
   may not exist in source USD files. Uses ensure_api_and_modify() utility to automatically
   apply required APIs before setting properties.

3. **Schema "Ensure" Implementation Status**:
   - **rigid_props**: Uses ensure_api_and_modify() with UsdPhysics.RigidBodyAPI
   - **collision_props**: Uses ensure_api_and_modify() with UsdPhysics.CollisionAPI
   - **mass_props**: Uses ensure_api_and_modify() with UsdPhysics.MassAPI
   - **articulation_props**: Uses standard schemas.modify_* (may fail if API missing)
   - **fixed_tendons_props**: Uses standard schemas.modify_* (may fail if API missing)
   - **joint_drive_props**: Uses standard schemas.modify_* (may fail if API missing)
   - **deformable_props**: Uses standard schemas.modify_* (may fail if API missing)

4. **Optional Instanceable Handling**: By default, disables instanceable flag before applying
   physics properties to avoid USD instancing conflicts. Can be disabled by setting
   `disable_instanceable=False` in the configuration.

Background:
IsaacLab's original from_files.py uses schemas.modify_*() functions which assume the required
USD APIs already exist in the source files. This fails for prims that only have collision APIs
but no rigid body APIs. Our enhanced version ensures the required APIs exist before modification.

The "ensure" functionality is currently implemented for the most commonly used properties
(rigid body, collision, mass). Other properties still use the original IsaacLab approach
and may fail silently if the required APIs don't exist in the source USD files.

NOTE: This is copied and adapted from IsaacLab with patches for 'source_path' and enhanced
schema handling. This is a temporary work-around until IsaacLab has these capabilities.
"""

from __future__ import annotations

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.kit.commands
import omni.log
from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_visual_material, clone, select_usd_variants
from pxr import Usd, UsdPhysics

from holosoma.simulator.isaacsim.spawners import from_files_cfg
from holosoma.simulator.isaacsim.spawners.schema_utils import ensure_api_and_modify
from holosoma.simulator.isaacsim.prim_utils import set_instanceable

import carb


def create_prim(target_path, usd_path, source_path=None, position=None, translation=None, orientation=None, scale=None):
    """Create a prim at the target path by referencing a USD file.

    Parameters
    ----------
    target_path : str
        The prim path where the asset should be created.
    usd_path : str
        Path to the USD file to reference.
    source_path : str, optional
        Optional prim path within the USD file to load specifically.
    position : array-like, optional
        Position to apply to the prim.
    translation : array-like, optional
        Translation to apply to the prim.
    orientation : array-like, optional
        Orientation to apply to the prim.
    scale : array-like, optional
        Scale to apply to the prim.

    Returns
    -------
    XFormPrim or None
        The created prim or None if creation failed.

    Raises
    ------
    FileNotFoundError
        If the USD file is not found at the specified path.
    """
    # Check if prim exists
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(target_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(target_path, "Xform")
    if not prim:
        return None

    # Add reference with logging
    if source_path:
        carb.log_info(f"Loading {source_path} from {usd_path}")
        success = prim.GetReferences().AddReference(assetPath=usd_path, primPath=source_path)
    else:
        carb.log_info(f"Loading from {usd_path}")
        success = prim.GetReferences().AddReference(usd_path)

    if not success:
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    from isaacsim.core.api.simulation_context.simulation_context import SimulationContext
    from isaacsim.core.prims import XFormPrim

    if SimulationContext.instance() is None:
        import isaacsim.core.utils.numpy as backend_utils

        device = "cpu"
    else:
        backend_utils = SimulationContext.instance().backend_utils
        device = SimulationContext.instance().device

    if position is not None:
        position = backend_utils.expand_dims(backend_utils.convert(position, device), 0)
    if translation is not None:
        translation = backend_utils.expand_dims(backend_utils.convert(translation, device), 0)
    if orientation is not None:
        orientation = backend_utils.expand_dims(backend_utils.convert(orientation, device), 0)
    if scale is not None:
        scale = backend_utils.expand_dims(backend_utils.convert(scale, device), 0)
    return XFormPrim(target_path, positions=position, translations=translation, orientations=orientation, scales=scale)
    # return prim


@clone
def spawn_from_usd(
    prim_path: str,
    cfg: from_files_cfg.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override settings with the given config.

    In the case of a USD file, the asset is spawned at the default prim specified in the USD file.
    If a default prim is not specified, then the asset is spawned at the root prim.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Parameters
    ----------
    prim_path : str
        The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
        then the asset is spawned at all the matching prim paths.
    cfg : from_files_cfg.UsdFileCfg
        The configuration instance.
    translation : tuple[float, float, float], optional
        The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
        case the translation specified in the USD file is used.
    orientation : tuple[float, float, float, float], optional
        The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
        in which case the orientation specified in the USD file is used.

    Returns
    -------
    Usd.Prim
        The prim of the spawned asset.

    Raises
    ------
    FileNotFoundError
        If the USD file does not exist at the given path.
    """
    return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, cfg.source_path, translation, orientation)


def _spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: from_files_cfg.FileCfg,
    source_path: str = "/",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override settings with the given config.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    Parameters
    ----------
    prim_path : str
        The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
        then the asset is spawned at all the matching prim paths.
    usd_path : str
        The path to the USD file to spawn the asset from.
    cfg : from_files_cfg.FileCfg
        The configuration instance.
    source_path : str, optional
        The prim path within the USD file to load. Defaults to "/" (root).
    translation : tuple[float, float, float], optional
        The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
        case the translation specified in the generated USD file is used.
    orientation : tuple[float, float, float, float], optional
        The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
        in which case the orientation specified in the generated USD file is used.

    Returns
    -------
    Usd.Prim
        The prim of the spawned asset.

    Raises
    ------
    FileNotFoundError
        If the USD file does not exist at the given path.
    """
    # check file path exists
    stage: Usd.Stage = stage_utils.get_current_stage()
    if not stage.ResolveIdentifierToEditTarget(usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")
    # spawn asset if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        # add prim as reference to stage, replaces built-in create_prim to support source_path
        prim = create_prim(
            prim_path,
            usd_path=usd_path,
            source_path=source_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
    else:
        omni.log.warn(f"A prim already exists at prim path: '{prim_path}'.")
        prim = prim_utils.get_prim_at_path(prim_path)

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # Optionally disable instanceable before modifying physics properties to avoid instancing issues
    if cfg.disable_instanceable:
        set_instanceable(stage, prim_path, False)

    # modify rigid body properties
    if cfg.rigid_props is not None:
        ensure_api_and_modify(
            prim_path, cfg.rigid_props, UsdPhysics.RigidBodyAPI, schemas.modify_rigid_body_properties, stage
        )

    # modify collision properties
    if cfg.collision_props is not None:
        ensure_api_and_modify(
            prim_path, cfg.collision_props, UsdPhysics.CollisionAPI, schemas.modify_collision_properties, stage
        )

    # modify mass properties
    if cfg.mass_props is not None:
        ensure_api_and_modify(prim_path, cfg.mass_props, UsdPhysics.MassAPI, schemas.modify_mass_properties, stage)

    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    # modify tendon properties
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    # define drive API on the joints
    # note: these are only for setting low-level simulation properties. all others should be set or are
    #  and overridden by the articulation/actuator properties.
    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)

    # modify deformable body properties
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)
