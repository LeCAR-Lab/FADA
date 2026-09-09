"""
Utility functions for working with USD prims in IsaacSim.
"""

import fnmatch
from typing import Dict, List, Optional, Tuple, Union

import isaaclab.sim as sim_utils
import numpy as np
import omni
import omni.log
import omni.usd
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.assets.rigid_object_collection import RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_mul
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def get_current_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


def print_prim_tree(prim_path: str, max_depth: int = None, indent: int = 0, stage=None):
    """Print a tree visualization of a prim and its descendants.

    Args:
        prim_path: Path to the root prim to start printing from
        max_depth: Maximum depth to traverse (None for unlimited)
        indent: Current indentation level (used recursively)
        stage: Optional USD stage to use (defaults to current stage)

    Example:
        ```python
        print_prim_tree("/World/envs/env_0/robot")

        # Output:
        /World/envs/env_0/robot
        ├── base
        │   ├── link_base
        │   │   └── collision_base
        ├── right_arm
        │   ├── link_arm_0
        │   │   └── collision_arm_0
        │   ├── link_arm_1
        │   │   └── collision_arm_1
        └── camera
            ├── ZED_X
            │   ├── base_link
            │   └── CameraLeft
        ```
    """
    # Get stage if not provided
    if stage is None:
        stage = omni.usd.get_context().get_stage()

    # Get prim at path
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Invalid prim path: {prim_path}")
        return

    # Check max depth
    if max_depth is not None and indent > max_depth:
        return

    # Print current prim with indentation
    prefix = "│   " * (indent - 1) + "├── " if indent > 0 else ""
    print(f"{prefix}{prim.GetPath()}")

    # Recursively print children
    children = prim.GetChildren()
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        # Change prefix for last child
        if is_last and indent > 0:
            print("│   " * (indent - 1) + "└── " + str(child.GetPath()))
        else:
            print_prim_tree(str(child.GetPath()), max_depth, indent + 1, stage)


def find_matching_prims(root_path, pattern="*", include_root=False, stage=None):
    """Find all prims under a root path that match a pattern.

    Args:
        stage: USD stage to search
        root_path: Base path to start searching from
        pattern: Pattern to match against relative paths (using fnmatch)
        include_root: Whether to include the root prim in results if it matches

    Returns:
        List of matching Usd.Prim objects
    """
    import fnmatch

    stage = stage or get_current_stage()

    matching_prims = []
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return matching_prims

    # Process the root prim if requested
    if include_root and fnmatch.fnmatch("", pattern):
        matching_prims.append(root_prim)

    # Recursively traverse and match pattern, but only return top-level matches
    def _traverse_prims(prim, base_path, matched_ancestors=None):
        if matched_ancestors is None:
            matched_ancestors = set()

        for child in prim.GetChildren():
            path = str(child.GetPath())
            rel_path = path[len(base_path) :]

            # Check if this prim matches the pattern
            if fnmatch.fnmatch(rel_path, pattern):
                # Only add if none of its ancestors have already matched
                is_top_level_match = True
                for ancestor_path in matched_ancestors:
                    if path.startswith(ancestor_path + "/"):
                        is_top_level_match = False
                        break

                if is_top_level_match:
                    matching_prims.append(child)
                    # Add this path to matched ancestors for its descendants
                    new_matched_ancestors = matched_ancestors | {path}
                    _traverse_prims(child, base_path, new_matched_ancestors)
                else:
                    # This is a descendant of an already matched prim, skip it
                    # but continue traversing in case there are other matches
                    _traverse_prims(child, base_path, matched_ancestors)
            else:
                # This prim doesn't match, continue traversing with same matched ancestors
                _traverse_prims(child, base_path, matched_ancestors)

    _traverse_prims(root_prim, root_path)
    return matching_prims


def log_robot_properties(robot_path: str, pattern: str = "*", stage=None):
    """Log mass properties and velocity limits of robot links matching a pattern.

    Args:
        stage: USD stage
        robot_path: Base path to the robot prim
        pattern: Pattern to match link names (e.g. "*" for all, "*/hand/*" for hand links)
    """

    from prettytable import PrettyTable
    from pxr import PhysxSchema, UsdPhysics

    # Default to current stage
    stage = stage or get_current_stage()

    # Find all matching prims under the robot
    matching_prims = find_matching_prims(robot_path, pattern, stage=stage)

    # Create table for mass properties
    mass_table = PrettyTable()
    mass_table.title = f"Robot Mass Properties (Pattern: {pattern})"
    mass_table.field_names = ["Name", "Mass (kg)", "Center of Mass", "Diagonal Inertia", "Principal Axes"]
    mass_table.align["Name"] = "l"

    # Create table for velocity limits
    velocity_table = PrettyTable()
    velocity_table.title = f"Robot Velocity Limits (Pattern: {pattern})"
    velocity_table.field_names = ["Name", "Max Linear Velocity", "Max Angular Velocity", "Max Joint Velocity"]
    velocity_table.align["Name"] = "l"

    # Process each matching prim
    for prim in matching_prims:
        name = prim.GetName()

        # Check if prim has mass properties
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)

            # Get mass value - convert to float
            mass = mass_api.GetMassAttr().Get() if mass_api.GetMassAttr() else "-"
            if mass != "-":
                mass = float(mass)

            # Get center of mass - convert Gf.Vec3f to list of floats
            com = mass_api.GetCenterOfMassAttr().Get() if mass_api.GetCenterOfMassAttr() else "-"
            if com != "-":
                com = [float(x) for x in com]

            # Get diagonal inertia - convert Gf.Vec3f to list of floats
            inertia = mass_api.GetDiagonalInertiaAttr().Get() if mass_api.GetDiagonalInertiaAttr() else "-"
            if inertia != "-":
                inertia = [float(x) for x in inertia]

            # Get principal axes - convert Gf.Quatf to list of floats
            axes = mass_api.GetPrincipalAxesAttr().Get() if mass_api.GetPrincipalAxesAttr() else "-"
            if axes != "-":
                # Convert quaternion to list of floats
                axes = [float(axes.GetReal())] + [float(x) for x in axes.GetImaginary()]

            mass_table.add_row(
                [
                    name,
                    f"{mass}" if isinstance(mass, float) else mass,
                    f"[{com[0]}, {com[1]}, {com[2]}]" if isinstance(com, list) else com,
                    f"[{inertia[0]}, {inertia[1]}, {inertia[2]}]" if isinstance(inertia, list) else inertia,
                    f"[{axes[0]}, {axes[1]}, {axes[2]}, {axes[3]}]" if isinstance(axes, list) else axes,
                ]
            )

        # Get velocity limits from both APIs
        max_linear_vel = "-"
        max_angular_vel = "-"
        max_joint_vel = "-"

        # Check PhysxRigidBodyAPI for linear/angular velocity limits
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)

            # Get max linear velocity
            if physx_api.GetMaxLinearVelocityAttr():
                max_linear_vel = physx_api.GetMaxLinearVelocityAttr().Get()
                if isinstance(max_linear_vel, float):
                    max_linear_vel = f"{max_linear_vel}"

            # Get max angular velocity
            if physx_api.GetMaxAngularVelocityAttr():
                max_angular_vel = physx_api.GetMaxAngularVelocityAttr().Get()
                if isinstance(max_angular_vel, float):
                    max_angular_vel = f"{max_angular_vel}"

        # Check PhysxJointAPI for joint velocity limit
        if prim.HasAPI(PhysxSchema.PhysxJointAPI):
            joint_api = PhysxSchema.PhysxJointAPI(prim)
            if joint_api.GetMaxJointVelocityAttr():
                max_joint_vel = joint_api.GetMaxJointVelocityAttr().Get()
                if isinstance(max_joint_vel, float):
                    max_joint_vel = f"{max_joint_vel}"

        # Add velocity limits to table if any limits are defined
        if max_linear_vel != "-" or max_angular_vel != "-" or max_joint_vel != "-":
            velocity_table.add_row([name, max_linear_vel, max_angular_vel, max_joint_vel])

    # Print tables
    if mass_table.rows:
        omni.log.info("\n" + mass_table.get_string())
    else:
        omni.log.info(f"No prims with mass properties found matching pattern: {pattern}")

    if velocity_table.rows:
        omni.log.info("\n" + velocity_table.get_string())
    else:
        omni.log.info(f"No prims with velocity limits found matching pattern: {pattern}")


def list_prims(usd_path, path="/", recurse=True):
    """List prims in a USD file at the specified path.

    Args:
        usd_path: Path to the USD file
        path: Root path to start listing from
        recurse: Whether to recursively list children

    Returns:
        List of prim paths as strings
    """
    stage = Usd.Stage.Open(usd_path)
    return list_prims_in_stage(stage, path, recurse)


def list_prims_in_stage(stage, path="/", recurse=True):
    """List prims in a USD stage at the specified path.

    Args:
        stage: USD stage
        path: Root path to start listing from
        recurse: Whether to recursively list children

    Returns:
        List of prim paths as strings
    """
    root_prim = stage.GetPrimAtPath(path)

    if not root_prim.IsValid():
        omni.log.warn(f"Path {path} not found")
        return []

    # List direct children
    children = []
    for child in root_prim.GetChildren():
        children.append(str(child.GetPath()))
        if recurse:
            children.extend(list_prims_in_stage(stage, str(child.GetPath()), recurse))

    return children


def compute_world_transform(stage, prim_path):
    """Compute the world transform matrix for a prim.

    Args:
        stage: USD stage
        prim_path: Path to the prim

    Returns:
        Gf.Matrix4d: World transform matrix
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        omni.log.warn(f"Invalid prim path for transform computation: {prim_path}")
        return Gf.Matrix4d(1.0)  # Return identity matrix

    # UsdGeom.Xformable works for all transformable prims and handles hierarchy automatically
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    return Gf.Matrix4d(1.0)  # Fallback to identity


def apply_offset_transform(world_transform, offset_pos=(0, 0, 0), offset_rot=(1, 0, 0, 0)):
    """Apply offset position and rotation to a world transform.

    Args:
        world_transform: Base world transform matrix
        offset_pos: Position offset (x, y, z)
        offset_rot: Rotation offset as quaternion (w, x, y, z)

    Returns:
        Gf.Transform: Combined transform
    """
    world_tf = Gf.Transform()
    world_tf.SetMatrix(world_transform)

    offset_tf = Gf.Transform()
    offset_tf.SetTranslation(Gf.Vec3d(*offset_pos))

    if isinstance(offset_rot, tuple):
        offset_tf.SetRotation(Gf.Rotation(Gf.Quatd(*offset_rot)))
    else:
        offset_tf.SetRotation(offset_rot)

    return world_tf * offset_tf


def get_pose(transform: Gf.Transform):
    """Extract position and rotation from a transform.

    Args:
        transform: Gf.Transform object

    Returns:
        Tuple of (position_array, rotation_tuple) where:
        - position_array: numpy array of (x, y, z)
        - rotation_tuple: (w, x, y, z) quaternion
    """
    translation: Gf.Vec3d = transform.GetTranslation()
    rotation: Gf.Quatd = transform.GetRotation().GetQuat()
    rot_tuple = (rotation.real, rotation.imaginary[0], rotation.imaginary[1], rotation.imaginary[2])
    return np.array(translation), rot_tuple


def set_instanceable(stage, prim_path: str, instanceable: bool = True) -> bool:
    """Set instanceable flag on a prim.

    Args:
        stage: The USD stage containing the prim
        prim_path: Path to the prim to modify
        instanceable: True to make instanceable, False to make non-instanceable

    Returns:
        bool: True if flag was set, False if prim not found
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return False

    prim.SetInstanceable(instanceable)
    return True


class UsdSceneLoaderCfg:
    """Configuration for USD scene loader.

    Args:
        usd_path: Path to the USD file to load
        prim_configs: Dictionary mapping prim patterns to RigidObjectCfg
        strip_prefixes: Prefixes to strip from USD paths when creating spawned paths
        scene_pos_offset: Position offset applied to all loaded objects (x, y, z)
        scene_rot_offset: Rotation offset applied to all loaded objects as quaternion (w, x, y, z)
        device: Device to use for tensors
    """

    def __init__(
        self,
        usd_path: str,
        prim_configs: Dict[str, RigidObjectCfg],
        strip_prefixes: List[str] = None,
        scene_pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scene_rot_offset: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        device: str = "cuda",
    ):
        self.usd_path = usd_path
        self.prim_configs = prim_configs
        self.strip_prefixes = strip_prefixes if strip_prefixes is not None else ["/world/"]
        self.scene_pos_offset = scene_pos_offset
        self.scene_rot_offset = scene_rot_offset
        self.device = device


def apply_rigid_body_properties_to_prim(prim_path: str, rigid_props=None):
    """Apply rigid body properties to a prim, ensuring RigidBodyAPI is applied.

    This function ensures that rigid body properties are properly applied to spawned prims.
    It uses the new schema utilities for consistent API handling.

    Args:
        prim_path: Path to the prim to modify
        rigid_props: RigidBodyPropertiesCfg to apply, defaults to basic config
    """
    from isaaclab.sim import schemas
    from holosoma.simulator.isaacsim.spawners.schema_utils import ensure_api_and_modify
    from pxr import UsdPhysics

    # Use default rigid props if none provided
    if rigid_props is None:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
        )

    # Use the new utility function for consistent handling
    ensure_api_and_modify(prim_path, rigid_props, UsdPhysics.RigidBodyAPI, schemas.modify_rigid_body_properties)


def _compute_object_pose(temp_stage, prim_path, cfg):
    """Simplified pose computation for scene loading.

    Args:
        temp_stage: USD stage containing the prim
        prim_path: Path to the prim
        cfg: UsdSceneLoaderCfg with scene offset settings

    Returns:
        Tuple of (position, rotation) where position is (x,y,z) and rotation is (w,x,y,z)
    """
    world_transform = compute_world_transform(temp_stage, prim_path)
    offset_transform = apply_offset_transform(world_transform, cfg.scene_pos_offset, cfg.scene_rot_offset)

    pos, rot = get_pose(offset_transform)
    return (float(pos[0]), float(pos[1]), float(pos[2])), rot


def create_usd_scene_loader(cfg: UsdSceneLoaderCfg) -> RigidObjectCollection:
    """Create a USD scene loader that loads specific prims as RigidObjects.

    This function loads a USD scene and creates RigidObjects for prims that match
    the specified patterns. It uses USD prim paths directly as collection keys
    for deterministic semantic mapping.

    Args:
        cfg: Configuration for the USD scene loader

    Returns:
        RigidObjectCollection: Collection containing the loaded rigid objects
    """
    from holosoma.simulator.isaacsim.spawners.from_files_cfg import CustomUsdFileCfg
    from holosoma.simulator.isaacsim.path_utils import transform_usd_path_to_spawned_path

    omni.log.info(f"Loading USD scene from: {cfg.usd_path}")

    # Open USD stage to analyze prims
    temp_layer = Sdf.Layer.FindOrOpen(cfg.usd_path)
    temp_stage = Usd.Stage.Open(temp_layer)

    # Create collection config
    collection_cfg = RigidObjectCollectionCfg(rigid_objects={})

    # Process each prim pattern and its configuration
    for pattern, base_rigid_cfg in cfg.prim_configs.items():
        # Find all prims matching the pattern
        matching_prims = find_matching_prims("/", pattern, stage=temp_stage)
        for i, prim in enumerate(matching_prims):
            prim_path = str(prim.GetPath())

            # Compute object pose with simplified transform pipeline
            pos, rot = _compute_object_pose(temp_stage, prim_path, cfg)

            # Transform USD path to spawned path using configurable prefix stripping
            spawned_path_pattern = transform_usd_path_to_spawned_path(prim_path, cfg.strip_prefixes)

            # Create RigidObjectCfg for this specific prim using custom spawner
            rigid_cfg = RigidObjectCfg(
                prim_path=spawned_path_pattern,
                spawn=CustomUsdFileCfg(
                    usd_path=cfg.usd_path,
                    source_path=prim_path,  # Load only this specific prim
                    rigid_props=base_rigid_cfg.spawn.rigid_props,  # Use rigid_props from pattern config
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot=rot,
                ),
            )

            # Use USD prim path directly as collection key (deterministic)
            collection_cfg.rigid_objects[prim_path] = rigid_cfg

    # Clean up temporary stage
    del temp_stage
    del temp_layer

    # Create and return the collection
    if collection_cfg.rigid_objects:
        return RigidObjectCollection(collection_cfg), collection_cfg
    else:
        omni.log.warn("No objects were loaded from the USD scene")
        return None, None
