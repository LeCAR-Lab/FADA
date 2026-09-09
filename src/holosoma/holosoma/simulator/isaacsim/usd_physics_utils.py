"""
USD Physics Utilities for IsaacSim

This module provides utilities for directly modifying USD files to apply physics properties
before spawning objects. This approach is more reliable than trying to pass physics materials
through spawner configurations.
"""

from __future__ import annotations
from typing import Optional
from pxr import Usd, UsdPhysics, PhysxSchema
from loguru import logger


def set_physics_material(
    stage: Usd.Stage,
    parent_path: str,
    dynamic_friction: Optional[float] = None,
    static_friction: Optional[float] = None,
    restitution: Optional[float] = None,
    friction_combine_mode: Optional[str] = None,
    restitution_combine_mode: Optional[str] = None,
    create_if_missing: bool = True,
) -> bool:
    """Set parameters on a physics material prim.

    Searches the entire prim tree for existing physics materials and applies
    the specified parameters. If no material is found and create_if_missing is True,
    creates a new physics material at the parent path.

    Parameters
    ----------
    stage : Usd.Stage
        USD Stage containing the prim hierarchy.
    parent_path : str
        Parent prim path where physics material should exist or be created.
    dynamic_friction : float, optional
        Dynamic friction coefficient, by default None.
    static_friction : float, optional
        Static friction coefficient, by default None.
    restitution : float, optional
        Restitution coefficient (bounciness), by default None.
    friction_combine_mode : str, optional
        How to combine friction values ('multiply', 'min', 'max', 'average'), by default None.
    restitution_combine_mode : str, optional
        How to combine restitution values ('multiply', 'min', 'max', 'average'), by default None.
    create_if_missing : bool, optional
        Create physics material if it doesn't exist, by default True.

    Returns
    -------
    bool
        True if successful, False if failed.

    Raises
    ------
    RuntimeError
        If there's an error setting physics material parameters.
    """
    try:
        # First, search for existing physics materials in the prim tree
        material_prim = _find_existing_physics_material(stage, parent_path)

        if not material_prim:
            if not create_if_missing:
                logger.warning(f"Physics material not found under {parent_path}")
                return False

            # Create new physics material at the parent path
            material_path = f"{parent_path}/Physics/PhysicsMaterial"

            # Create the physics scope if it doesn't exist
            physics_path = f"{parent_path}/Physics"
            _ = stage.DefinePrim(physics_path, "Scope")

            # Create the physics material
            material_prim = stage.DefinePrim(material_path, "PhysicsMaterial")

        # Create the physics material API
        physics_material = UsdPhysics.MaterialAPI(material_prim)
        if not physics_material:
            physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)

        # Set the basic parameters if provided
        if dynamic_friction is not None:
            dyn_attr = physics_material.CreateDynamicFrictionAttr()
            dyn_attr.Set(dynamic_friction)

        if static_friction is not None:
            static_attr = physics_material.CreateStaticFrictionAttr()
            static_attr.Set(static_friction)

        if restitution is not None:
            rest_attr = physics_material.CreateRestitutionAttr()
            rest_attr.Set(restitution)

        # Set PhysX-specific parameters if provided
        if friction_combine_mode is not None or restitution_combine_mode is not None:
            # Get or create PhysX material API
            physx_material = PhysxSchema.PhysxMaterialAPI(material_prim)
            if not physx_material:
                physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)

            if friction_combine_mode is not None:
                friction_attr = physx_material.CreateFrictionCombineModeAttr()
                friction_attr.Set(friction_combine_mode)

            if restitution_combine_mode is not None:
                restitution_attr = physx_material.CreateRestitutionCombineModeAttr()
                restitution_attr.Set(restitution_combine_mode)

        return True

    except Exception as e:
        raise RuntimeError(f"Error setting physics material parameters for {parent_path}: {e}")


def _find_existing_physics_material(stage: Usd.Stage, parent_path: str):
    """Search for existing physics materials in the prim tree.

    Searches the entire prim subtree under parent_path for existing physics
    materials, checking both MaterialAPI and PhysicsMaterial prim types.

    Parameters
    ----------
    stage : Usd.Stage
        USD stage containing the prim hierarchy.
    parent_path : str
        Path to start searching from.

    Returns
    -------
    Usd.Prim or None
        Prim with physics material, or None if not found.
    """
    from pxr import UsdPhysics

    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim:
        return None

    # First check the parent prim itself
    # Prioritize API check over prim type since API is what matters for physics
    if parent_prim.HasAPI(UsdPhysics.MaterialAPI):
        return parent_prim
    elif parent_prim.GetTypeName() == "PhysicsMaterial":
        return parent_prim

    # Search the entire subtree using USD's Traverse method
    for prim in stage.Traverse():
        # Only check prims under our parent path
        prim_path_str = str(prim.GetPath())
        if not prim_path_str.startswith(parent_path):
            continue

        # Skip the parent prim itself (already checked above)
        if prim_path_str == parent_path:
            continue

        # Prioritize API check over prim type since API is what matters for physics
        # This handles both patterns:
        # 1. def PhysicsMaterial "PhysicsMaterial" (type="PhysicsMaterial")
        # 2. def Material "PhysicsMaterial" (type="Material" with PhysicsMaterialAPI)
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            return prim
        elif prim.GetTypeName() == "PhysicsMaterial":
            return prim

    return None


def set_mass(stage: Usd.Stage, prim_path: str, mass: float) -> bool:
    """Set the mass of a rigid body prim.

    Args:
        stage: USD stage
        prim_path: Path to the prim
        mass: Mass value in kg

    Returns:
        bool: True if successful, False if failed
    """
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            logger.warning(f"Prim not found at {prim_path}")
            return False

        # Get or create mass API schema
        mass_api = UsdPhysics.MassAPI(prim)
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(prim)

        # Set mass value
        mass_api.CreateMassAttr().Set(mass)
        return True

    except Exception as e:
        raise RuntimeError(f"Error setting mass for {prim_path}: {e}")


def set_density(stage: Usd.Stage, prim_path: str, density: float) -> bool:
    """Set the density of a rigid body prim.

    Args:
        stage: USD stage
        prim_path: Path to the prim
        density: Density value in kg/m³

    Returns:
        bool: True if successful, False if failed
    """
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            logger.warning(f"Could not set density, prim not found at {prim_path}")
            return False

        # Get or create mass API schema
        mass_api = UsdPhysics.MassAPI(prim)
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(prim)

        # Set density value
        mass_api.CreateDensityAttr().Set(density)
        return True

    except Exception as e:
        raise RuntimeError(f"Error setting density for {prim_path}: {e}")


def set_rigid_body_properties(
    stage: Usd.Stage,
    prim_path: str,
    kinematic_enabled: Optional[bool] = None,
    linear_damping: Optional[float] = None,
    angular_damping: Optional[float] = None,
    max_linear_velocity: Optional[float] = None,
    max_angular_velocity: Optional[float] = None,
) -> bool:
    """Set rigid body properties on a prim.

    Args:
        stage: USD stage
        prim_path: Path to the prim
        kinematic_enabled: Whether the body is kinematic
        linear_damping: Linear damping coefficient
        angular_damping: Angular damping coefficient
        max_linear_velocity: Maximum linear velocity
        max_angular_velocity: Maximum angular velocity

    Returns:
        bool: True if successful, False if failed
    """
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            logger.warning(f"Prim not found at {prim_path}")
            return False

        # Get or create rigid body API
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        if not rigid_body_api:
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)

        # Set kinematic flag if provided
        if kinematic_enabled is not None:
            kinematic_attr = rigid_body_api.CreateKinematicEnabledAttr()
            kinematic_attr.Set(kinematic_enabled)
            logger.debug(f"Set kinematic enabled: {kinematic_enabled}")

        # Get or create PhysX rigid body API for additional properties
        physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_rigid_body:
            physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

        # Set damping properties
        if linear_damping is not None:
            linear_attr = physx_rigid_body.CreateLinearDampingAttr()
            linear_attr.Set(linear_damping)

        if angular_damping is not None:
            angular_attr = physx_rigid_body.CreateAngularDampingAttr()
            angular_attr.Set(angular_damping)

        # Set velocity limits
        if max_linear_velocity is not None:
            max_linear_attr = physx_rigid_body.CreateMaxLinearVelocityAttr()
            max_linear_attr.Set(max_linear_velocity)

        if max_angular_velocity is not None:
            max_angular_attr = physx_rigid_body.CreateMaxAngularVelocityAttr()
            max_angular_attr.Set(max_angular_velocity)
        return True

    except Exception as e:
        raise RuntimeError(f"Error setting rigid body properties for {prim_path}: {e}")


def set_instanceable(stage: Usd.Stage, prim_path: str, instanceable: bool = True) -> bool:
    """Set instanceable flag on a prim.

    Args:
        stage: USD stage
        prim_path: Path to the prim
        instanceable: True to make instanceable, False to make non-instanceable

    Returns:
        bool: True if successful, False if failed
    """
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            logger.warning(f"Prim not found at {prim_path}")
            return False

        # Set instanceable metadata
        prim.SetInstanceable(instanceable)
        logger.debug(f"Set instanceable={instanceable} for {prim_path}")
        return True

    except Exception as e:
        raise RuntimeError(f"Error setting instanceable flag for {prim_path}: {e}")


def apply_physics_config_to_usd(
    stage: Usd.Stage,
    prim_path: str,
    physics_config,
    disable_instanceable: bool = True,
    skip_hierarchy_check: bool = False,
) -> bool:
    """Apply a complete physics configuration to a USD prim.

    Args:
        stage: USD stage
        prim_path: Path to the prim
        physics_config: PhysicsConfig instance with all physics properties
        disable_instanceable: Whether to disable instanceable flag before applying physics

    Returns:
        bool: True if successful, False if failed
    """
    from holosoma.config_types.simulator import PhysicsConfig

    if physics_config is None:
        raise RuntimeError(f"Expected PhysicsConfig, got None")

    if not isinstance(physics_config, PhysicsConfig):
        raise RuntimeError(f"Expected PhysicsConfig, got {type(physics_config)}")

    try:
        # Check for existing physics hierarchy conflicts (unless skipped)
        if not skip_hierarchy_check and _has_physics_hierarchy_conflict(stage, prim_path):
            raise RuntimeError(
                f"Physics hierarchy conflict detected for {prim_path}. "
                f"Multiple RigidBodyAPI prims found in hierarchy. "
                f"This will cause unpredictable simulation results."
            )

        # Disable instanceable if requested (required for physics modifications)
        if disable_instanceable:
            set_instanceable(stage, prim_path, False)

        # Apply mass properties
        if physics_config.mass is not None:
            set_mass(stage, prim_path, physics_config.mass)

        if physics_config.density is not None:
            set_density(stage, prim_path, physics_config.density)

        # Apply rigid body properties
        set_rigid_body_properties(
            stage,
            prim_path,
            kinematic_enabled=physics_config.kinematic_enabled,
            linear_damping=physics_config.linear_damping,
            angular_damping=physics_config.angular_damping,
            max_linear_velocity=physics_config.max_linear_velocity,
            max_angular_velocity=physics_config.max_angular_velocity,
        )

        # Apply material properties if IsaacSim config is available
        if physics_config.isaacsim is not None:
            isaacsim_config = physics_config.isaacsim
            set_physics_material(
                stage,
                prim_path,
                dynamic_friction=isaacsim_config.dynamic_friction,
                static_friction=isaacsim_config.static_friction,
                restitution=isaacsim_config.restitution,
                friction_combine_mode=isaacsim_config.friction_combine_mode,
                restitution_combine_mode=isaacsim_config.restitution_combine_mode,
            )
        return True
    except Exception as e:
        raise RuntimeError(f"Error applying physics config to {prim_path}: {e}")


def _has_physics_hierarchy_conflict(stage: Usd.Stage, prim_path: str) -> bool:
    """Check if applying physics to this prim would create a hierarchy conflict.

    Args:
        stage: USD stage
        prim_path: Path to the prim to check

    Returns:
        bool: True if there would be a conflict, False otherwise
    """
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return False

    # Check if any parent has RigidBodyAPI
    parent = prim.GetParent()
    while parent and parent.GetPath() != "/":
        if parent.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        parent = parent.GetParent()

    # Check if any child has RigidBodyAPI
    for child in prim.GetAllChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
    return False
