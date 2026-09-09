"""Physics utilities for IsaacGym simulator.

This module provides utilities for applying physics properties to actors
in the IsaacGym simulation environment, including rigid shape properties
and mass configurations.
"""

from __future__ import annotations

from typing import Any, Dict

from isaacgym import gymapi
from loguru import logger


def apply_rigid_shape_properties(
    gym: gymapi.Gym, env_ptr: int, actor_handle: int, physics_config: Dict[str, Any], object_name: str
) -> None:
    """Apply rigid shape properties (friction, restitution, compliance).

    Applies IsaacGym-specific physics properties to the actor's rigid shapes,
    including friction coefficients, restitution, and compliance values.

    Parameters
    ----------
    gym : gymapi.Gym
        The IsaacGym API instance.
    env_ptr : int
        Environment handle for the IsaacGym environment.
    actor_handle : int
        Handle to the actor to modify.
    physics_config : Dict[str, Any]
        Physics configuration containing IsaacGym-specific properties.
    object_name : str
        Name of the object for logging purposes.

    Raises
    ------
    RuntimeError
        If no rigid shape properties are found for the specified object.
    """
    # Get current rigid shape properties
    shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)

    if not shape_props:
        raise RuntimeError(f"No rigid shape properties found for '{object_name}'")

    # Apply IsaacGym-specific friction properties if configured
    if hasattr(physics_config, "isaacgym") and physics_config.isaacgym is not None:
        gym_config = physics_config.isaacgym
        logger.debug(f"Applying IsaacGym friction properties to '{object_name}': {gym_config}")

        for i in range(len(shape_props)):
            # Apply 1:1 mapping to IsaacGym RigidShapeProperties
            shape_props[i].friction = gym_config.friction
            shape_props[i].rolling_friction = gym_config.rolling_friction
            shape_props[i].torsion_friction = gym_config.torsion_friction
            shape_props[i].restitution = gym_config.restitution
            shape_props[i].compliance = gym_config.compliance

        # Set the modified properties back to the actor
        gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, shape_props)
        logger.debug(
            f"Applied IsaacGym friction properties to '{object_name}': "
            f"friction={gym_config.friction}, rolling={gym_config.rolling_friction}, "
            f"torsion={gym_config.torsion_friction}"
        )

    else:
        logger.debug(f"No IsaacGym-specific friction config for '{object_name}', using defaults")


def apply_mass_from_config(
    gym: gymapi.Gym, env_ptr: int, actor_handle: int, physics_config: Dict[str, Any], object_name: str
) -> None:
    """Apply mass or density from physics config.

    Applies mass properties to the actor based on the physics configuration,
    with priority given to explicit mass values over density calculations.

    Parameters
    ----------
    gym : gymapi.Gym
        The IsaacGym API instance.
    env_ptr : int
        Environment handle for the IsaacGym environment.
    actor_handle : int
        Handle to the actor to modify.
    physics_config : Dict[str, Any]
        Physics configuration containing mass or density settings.
    object_name : str
        Name of the object for logging purposes.

    Raises
    ------
    RuntimeError
        If no rigid body properties are found for the specified object.
    """
    # Get current rigid body properties
    body_props = gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
    if not body_props:
        raise RuntimeError(f"No rigid body properties found for '{object_name}', cannot apply mass config")

    # Priority 1: Direct mass override
    if physics_config.get("mass") is not None:
        target_mass = physics_config["mass"]
        logger.debug(f"Setting explicit mass for '{object_name}': {target_mass}")

        for prop in body_props:
            prop.mass = target_mass

        gym.set_actor_rigid_body_properties(env_ptr, actor_handle, body_props, recomputeInertia=True)
        logger.debug(f"Applied explicit mass to '{object_name}': {target_mass}")
        return

    # Priority 2: Density-based calculation
    if physics_config.get("density") is not None:
        current_mass = sum(prop.mass for prop in body_props)

        if current_mass < 1e-6:
            # URDF has very low mass, log warning but keep original values
            target_density = physics_config["density"]
            logger.warning(
                f"URDF has very low mass ({current_mass}) for '{object_name}' "
                f"with density config {target_density}. Keeping original URDF mass values."
            )
        else:
            # Density was applied during asset loading, use existing mass
            logger.debug(f"Using density-calculated mass for '{object_name}': {current_mass}")
        return

    # Priority 3: No mass/density config - use URDF values
    current_mass = sum(prop.mass for prop in body_props)
    logger.debug(f"Using URDF mass for '{object_name}': {current_mass}")
