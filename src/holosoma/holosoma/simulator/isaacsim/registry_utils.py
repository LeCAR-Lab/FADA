"""
IsaacSim Object Registration Utilities

This module provides utilities for registering objects with the unified ObjectRegistry system
in IsaacSim. It handles the registration of three types of objects defined by ObjectType enum:

- ObjectType.ROBOT: Articulated robots (handled separately by IsaacSim)
- ObjectType.SCENE: Objects from USD scene collections (RigidObjectCollection)
- ObjectType.INDIVIDUAL: Individual rigid objects (RigidObject instances)

The registration process creates a unified indexing system that allows tasks to access
objects using consistent APIs across IsaacGym and IsaacSim simulators.
"""

from loguru import logger

from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.safe_torch_import import torch


def register_objects(sim):
    """
    Register all objects for unified indexing and cross-simulator compatibility.

    This is the main entry point for object registration in IsaacSim. It:
    1. Counts objects by type to calculate index ranges
    2. Registers objects with ObjectType enum for type safety
    3. Creates AllRootStatesProxy for IsaacGym-style tensor access

    Args:
        sim: IsaacSim simulator instance with scene and object_registry

    Usage:
        # Called automatically during IsaacSim initialization
        register_objects(sim)

        # After registration, objects can be accessed by name:
        indices = sim.get_actor_indices(["obj0", "obj1"], env_ids)
        sim.all_root_states[indices] = new_states
    """
    # Step 1: Count objects by type to calculate index ranges
    # This determines how many objects of each ObjectType exist
    robot_count = 1 if hasattr(sim, "_robot") else 0
    scene_count = 0
    individual_count = 0

    # Count scene objects from USD collections
    if hasattr(sim.scene, "rigid_objects") and "usd_scene_objects" in sim.scene.rigid_objects:
        scene_collection = sim.scene.rigid_objects["usd_scene_objects"]
        scene_count = len(scene_collection.cfg.rigid_objects)

    # Count individual rigid objects (excluding scene collections)
    for obj_key in sim.scene.rigid_objects.keys():
        if obj_key != "usd_scene_objects":
            individual_count += 1

    # Step 2: Set up index ranges in ObjectRegistry
    #
    # This creates flat index ranges for each ObjectType:
    # - [robot_range: 0 to robot_count*num_envs]
    # - [scene_range: robot_end to robot_end + scene_count*num_envs]
    # - [individual_range: scene_end to scene_end + individual_count*num_envs]
    #
    # Note: this flattening order does NOT match IsaacGym, we may want to do this, but conversely
    # I don't think envs/tasks should depend on the order and instead using indices and masks.
    sim.object_registry.setup_ranges(sim.num_envs, robot_count, scene_count, individual_count)

    # Step 3: Register objects by type using ObjectType enum
    register_robots(sim)  # ObjectType.ROBOT
    register_scene_objects(sim)  # ObjectType.SCENE
    register_individual_objects(sim)  # ObjectType.INDIVIDUAL

    # Step 4: Finalize registration to calculate actual indices
    sim.object_registry.finalize_registration()

    logger.info(f"IsaacSim: Object registry populated with {len(sim.object_registry.objects)} objects")


def register_robots(sim):
    """Register robot with pre-calculated world coordinates

    This method registers the robot with the ObjectRegistry using pre-calculated world coordinates
    that include environment origins.

    Notes
    -----
    - Must be called after sim.play() when robot data is available
    - Applies env_origins to get world coordinates for all environments
    """
    from holosoma.simulator.shared.object_registry import ObjectType

    # Get robot's base pose from config (without env_origins)
    pos = torch.tensor(sim.robot_config.init_state.pos, device=sim.sim_device, dtype=torch.float32)
    rot = torch.tensor(sim.robot_config.init_state.rot, device=sim.sim_device, dtype=torch.float32)
    base_pose = torch.cat([pos, rot])  # [7] - base pose without env_origins

    # Apply env_origins to get world coordinates for all environments
    robot_world_poses = torch.zeros(sim.num_envs, 7, device=sim.sim_device, dtype=torch.float32)
    for env_id in range(sim.num_envs):
        robot_world_poses[env_id] = base_pose.clone()
        robot_world_poses[env_id, :3] += sim.scene.env_origins[env_id]  # Add env_origin to position

    # Register robot with ObjectRegistry using enhanced interface
    sim.object_registry.register_object(
        name="robot", object_type=ObjectType.ROBOT, position_in_type=0, initial_poses=robot_world_poses
    )

    logger.debug(f"Registered robot with world coordinates for {sim.num_envs} environments")


def register_scene_objects(sim):
    """
    Register scene objects with ObjectType.SCENE using tensor-based interface.

    Scene objects come from USD scene collections (RigidObjectCollection) loaded
    from scene files. These are typically multiple objects loaded as a group
    from a single USD file containing a complete scene.

    Args:
        sim: IsaacSim simulator instance with scene.rigid_objects['usd_scene_objects']

    Process:
        1. Extract object names from scene collection configuration
        2. Get base poses from configuration (no ObjectRegistry dependency)
        3. Apply env_origins to get world coordinates for all environments
        4. Register each object with tensor-based ObjectRegistry interface

    Example:
        # After registration, scene objects can be accessed by name:
        indices = sim.get_actor_indices(["table", "chair"], env_ids)
        sim.all_root_states[indices] = new_poses
    """
    if not (hasattr(sim.scene, "rigid_objects") and "usd_scene_objects" in sim.scene.rigid_objects):
        logger.info("No USD scene objects found")
        return

    scene_collection = sim.scene.rigid_objects["usd_scene_objects"]

    # Extract object names from USD paths in scene collection config
    object_names = [usd_path.split("/")[-1] for usd_path in scene_collection.cfg.rigid_objects.keys()]

    # Get base poses from configuration (no env_origins applied yet)
    initial_poses_tensor = sim.get_actor_initial_poses(object_names)  # [num_objects * num_envs, 7]

    # Extract base poses (first environment only, since all envs have same initial pose)
    base_poses = initial_poses_tensor[:: sim.num_envs]  # [num_objects, 7] - every num_envs-th element

    # Register each scene object with ObjectType.SCENE using tensor interface
    for position, (object_name, base_pose) in enumerate(zip(object_names, base_poses)):
        # Apply env_origins to get world coordinates for all environments
        object_world_poses = torch.zeros(sim.num_envs, 7, device=sim.sim_device, dtype=torch.float32)
        for env_id in range(sim.num_envs):
            object_world_poses[env_id] = base_pose.clone()
            # object_world_poses[env_id, :3] += sim.scene.env_origins[env_id]  # Add env_origin to position

        # Register with tensor-based interface (same as IsaacGym)
        sim.object_registry.register_object(
            name=object_name,
            object_type=ObjectType.SCENE,
            position_in_type=position,  # Position within scene objects
            initial_poses=object_world_poses,  # Tensor interface with pre-calculated world coordinates
        )

        logger.debug(f"Registered scene object '{object_name}' at position {position}")


def register_individual_objects(sim):
    """
    Register individual objects with ObjectType.INDIVIDUAL using tensor-based interface.

    Individual objects are RigidObject instances loaded separately, not as part
    of a scene collection. Each object is typically loaded from its own file
    and configured independently.

    Args:
        sim: IsaacSim simulator instance with scene.rigid_objects containing individual objects

    Process:
        1. Collect individual object names (excluding scene collections)
        2. Get base poses from configuration (no ObjectRegistry dependency)
        3. Apply env_origins to get world coordinates for all environments
        4. Register each object with tensor-based ObjectRegistry interface

    Example:
        # After registration, individual objects can be accessed by name:
        indices = sim.get_actor_indices(["box1", "sphere2"], env_ids)
        sim.set_actor_states(["box1", "sphere2"], env_ids, new_states)
    """
    individual_objects = []

    # Collect individual object names (exclude scene collections like 'usd_scene_objects')
    for obj_key in sim.scene.rigid_objects.keys():
        if obj_key != "usd_scene_objects":  # Skip scene collections
            individual_objects.append(obj_key)

    if not individual_objects:
        logger.debug("No individual objects found to register")
        return

    # Get base poses from configuration (no env_origins applied yet)
    initial_poses_tensor = sim.get_actor_initial_poses(individual_objects)  # [num_objects * num_envs, 7]

    # Extract base poses (first environment only, assumes environment cloning)
    base_poses = initial_poses_tensor[:: sim.num_envs]  # [num_objects, 7]

    # Register each individual object with ObjectType.INDIVIDUAL using tensor interface
    for position, (obj_name, base_pose) in enumerate(zip(individual_objects, base_poses)):
        # Apply env_origins to get world coordinates for all environments
        object_world_poses = torch.zeros(sim.num_envs, 7, device=sim.sim_device, dtype=torch.float32)
        for env_id in range(sim.num_envs):
            object_world_poses[env_id] = base_pose.clone()
            # object_world_poses[env_id, :3] += sim.scene.env_origins[env_id]  # Add env_origin to position

        # Register with tensor-based interface (same as IsaacGym)
        sim.object_registry.register_object(
            name=obj_name,
            object_type=ObjectType.INDIVIDUAL,
            position_in_type=position,  # Position within individual objects
            initial_poses=object_world_poses,  # Tensor interface with pre-calculated world coordinates
        )

        logger.debug(f"Registered individual object '{obj_name}' at position {position}")
