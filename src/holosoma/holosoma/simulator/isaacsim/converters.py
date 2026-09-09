"""Physics configuration converters for IsaacSim.

This module provides conversion functions between the unified physics configuration
system and IsaacLab's specific configuration types.
"""

import isaaclab.sim as sim_utils
from holosoma.config_types.simulator import PhysicsConfig
from loguru import logger


def physics_to_rigid_body_material(physics: PhysicsConfig | None) -> sim_utils.RigidBodyMaterialCfg:
    """Convert PhysicsConfig to IsaacLab RigidBodyMaterialCfg.

    Parameters
    ----------
    physics : PhysicsConfig or None
        PhysicsConfig instance with unified physics properties.

    Returns
    -------
    sim_utils.RigidBodyMaterialCfg
        RigidBodyMaterialCfg instance for IsaacLab material properties.

    Raises
    ------
    TypeError
        If physics is not a PhysicsConfig instance or None.
    """
    # Handle None case
    if physics is None:
        return sim_utils.RigidBodyMaterialCfg()

    # Strict type checking
    if not isinstance(physics, PhysicsConfig):
        raise TypeError(f"Expected PhysicsConfig instance, got {type(physics)}")

    # Use IsaacSim-specific config if available
    if physics.isaacsim is not None:
        import dataclasses

        return sim_utils.RigidBodyMaterialCfg(**dataclasses.asdict(physics.isaacsim))

    # Fallback to defaults
    return sim_utils.RigidBodyMaterialCfg()


def physics_to_mass_props(physics: PhysicsConfig | None) -> sim_utils.MassPropertiesCfg | None:
    """Convert PhysicsConfig to IsaacLab MassPropertiesCfg.

    Parameters
    ----------
    physics : PhysicsConfig or None
        PhysicsConfig instance with unified physics properties.

    Returns
    -------
    sim_utils.MassPropertiesCfg or None
        MassPropertiesCfg instance for IsaacLab mass properties, or None if no mass/density specified.

    Raises
    ------
    TypeError
        If physics is not a PhysicsConfig instance or None.
    """
    # Handle None case
    if physics is None:
        return None

    # Strict type checking
    if not isinstance(physics, PhysicsConfig):
        raise TypeError(f"Expected PhysicsConfig instance, got {type(physics)}")

    # Check if we have mass or density to set
    if physics.mass is None and physics.density is None:
        return None

    # Create mass properties config
    mass_props = sim_utils.MassPropertiesCfg(mass=physics.mass, density=physics.density)

    return mass_props


def physics_to_rigid_body_props(physics: PhysicsConfig | None) -> sim_utils.RigidBodyPropertiesCfg:
    """Convert PhysicsConfig to IsaacLab RigidBodyPropertiesCfg.

    Parameters
    ----------
    physics : PhysicsConfig or None
        PhysicsConfig instance with unified physics properties.

    Returns
    -------
    sim_utils.RigidBodyPropertiesCfg
        RigidBodyPropertiesCfg instance for IsaacLab.

    Raises
    ------
    TypeError
        If physics is not a PhysicsConfig instance or None.
    """
    # Handle None case
    if physics is None:
        physics = PhysicsConfig()

    # Strict type checking - should be PhysicsConfig by now
    if not isinstance(physics, PhysicsConfig):
        raise TypeError(
            f"Expected PhysicsConfig instance, got {type(physics)}. "
            f"Physics config should be converted to PhysicsConfig in __post_init__ methods."
        )

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=physics.kinematic_enabled,
        linear_damping=physics.linear_damping,
        angular_damping=physics.angular_damping,
        max_linear_velocity=physics.max_linear_velocity,
        max_angular_velocity=physics.max_angular_velocity,
        # Note: density is handled by MaterialPropertiesCfg, not RigidBodyPropertiesCfg
    )

    return rigid_props
