"""USD file loader for IsaacSim.

This module provides USD-specific scene loading for IsaacLab/IsaacSim,
handling both scene USD files and individual USD objects through a clean interface.
"""

import fnmatch
import os
from typing import Dict, Any, Tuple, List
from loguru import logger
from pxr import Usd

from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils

from holosoma.config_types.simulator import SceneConfig, SceneFileConfig, RigidObjectConfig
from holosoma.simulator.isaacsim.prim_utils import (
    UsdSceneLoaderCfg,
    create_usd_scene_loader,
    apply_rigid_body_properties_to_prim,
    find_matching_prims,
)


class USDFileLoader:
    """USD file loader for IsaacLab scene objects.

    Provides functionality to load USD scene files and individual USD objects
    through a clean interface. Handles both scene collections and individual
    rigid objects with physics property application and path resolution.

    Parameters
    ----------
    sim_context : isaaclab.sim.SimulationContext
        IsaacLab simulation context instance.
    scene : isaaclab.scene.InteractiveScene
        IsaacLab interactive scene instance.
    device : str
        Device string for tensor operations (e.g., "cuda:0", "cpu").
    """

    def __init__(self, sim_context, scene, device):
        """Initialize USD file loader.

        Parameters
        ----------
        sim_context : isaaclab.sim.SimulationContext
            IsaacLab simulation context instance.
        scene : isaaclab.scene.InteractiveScene
            IsaacLab interactive scene instance.
        device : str
            Device string for tensor operations (e.g., "cuda:0", "cpu").
        """
        self.sim = sim_context
        self.scene = scene
        self.device = device

    def load_scene_files(self, scene_files: list[SceneFileConfig], asset_root: str) -> Any:
        """Load scene files as collections of rigid objects.

        Loads USD/URDF scene files and converts them into RigidObjectCollection instances
        that can be used within the simulation environment. Handles path resolution,
        prim filtering, and physics property application.

        Parameters
        ----------
        scene_files : list[holosoma.config_types.simulator.SceneFileConfig]
            List of scene file configurations containing paths, transforms, and filtering rules.
        asset_root : str
            Root directory path for resolving relative asset paths.

        Returns
        -------
        isaaclab.assets.RigidObjectCollection
            RigidObjectCollection containing all loaded scene objects with applied physics properties.

        Raises
        ------
        ValueError
            If no scene files are provided, multiple scene files are specified (not yet supported),
            asset root is missing, or no USD assets are available in the configuration.
        """
        if not scene_files:
            raise ValueError("No scene files provided")

        if len(scene_files) > 1:
            raise ValueError(f"Multiple scene files not yet supported. Got {len(scene_files)} scene files.")

        if not asset_root:
            raise ValueError("Asset root is required")

        # Get the single scene file
        scene_file = scene_files[0]

        # Check if USD scene file is available
        if not scene_file.has_format("usd", asset_root):
            raise ValueError("No USD scene file available in scene file configuration")

        # Get USD path and transforms from the scene file
        usd_path = scene_file.get_asset_path("usd", asset_root)

        # Discover and filter prims using patterns
        filtered_prims = self._discover_and_filter_prims(usd_path, scene_file)

        # Generate prim_configs using scene_file.object_configs (moved inside scene file)
        prim_configs = self._generate_prim_configs_with_auto_paths(filtered_prims, usd_path, scene_file)

        # Create loader config
        usd_loader_config = {
            "scene_pos_offset": scene_file.position,
            "scene_rot_offset": scene_file.orientation,
            "device": self.device,
            "prim_configs": prim_configs,
            "strip_prefixes": ["/world/", "/World/", "/"],
            "scene_file": scene_file,  # Pass scene_file for pattern reconstruction
        }
        return self._load_usd_scene_objects(usd_path, usd_loader_config)

    def load_rigid_objects(self, rigid_objects: list[RigidObjectConfig], asset_root: str) -> Dict[str, RigidObject]:
        """Load individual rigid objects from configuration.

        Processes a list of rigid object configurations and creates RigidObject instances
        for each USD-based object. Handles pose conversion, physics property application,
        and path resolution.

        Parameters
        ----------
        rigid_objects : list[holosoma.config_types.simulator.RigidObjectConfig]
            List of rigid object configurations containing USD paths, poses, and physics properties.
        asset_root : str
            Root directory path for resolving relative asset paths.

        Returns
        -------
        Dict[str, isaaclab.assets.RigidObject]
            Dictionary mapping object names to their corresponding RigidObject instances.
        """
        individual_objects = {}

        for rigid_obj in rigid_objects:
            try:
                if rigid_obj.usd_path:  # Only handle USD objects in this loader
                    # Keep quaternion in wxyz format for internal consistency
                    # rigid_obj.position = [x, y, z]
                    # rigid_obj.orientation = [w, x, y, z] (IsaacLab format)
                    # Keep as [x, y, z, qw, qx, qy, qz] (wxyz format for internal consistency)
                    pos = rigid_obj.position[:3]  # [x, y, z]
                    quat_wxyz = rigid_obj.orientation[:4]  # [w, x, y, z]
                    pose = pos + quat_wxyz  # [x, y, z, qw, qx, qy, qz] - wxyz format for consistency

                    object_config = {"usd_path": rigid_obj.usd_path, "pose": pose, "physics": rigid_obj.physics}

                    rigid_object = self._load_individual_object(rigid_obj.name, object_config, asset_root)
                    individual_objects[rigid_obj.name] = rigid_object
                    logger.debug(f"Loaded individual USD object '{rigid_obj.name}'")
            except Exception as e:
                logger.error(f"USDFileLoader: Failed to load individual object '{rigid_obj.name}': {e}")
                raise

        logger.info(f"Loaded {len(individual_objects)} individual USD objects from rigid_objects")
        return individual_objects

    def _discover_and_filter_prims(self, usd_path: str, source: SceneFileConfig) -> List[str]:
        """Discover prims in USD file and filter using include/exclude patterns.

        Opens a USD file temporarily to scan for prims and applies filtering based on
        include and exclude patterns specified in the scene file configuration.

        Parameters
        ----------
        usd_path : str
            Path to the USD file to scan for prims.
        source : holosoma.config_types.simulator.SceneFileConfig
            Scene file configuration containing include_patterns and exclude_patterns.

        Returns
        -------
        List[str]
            List of prim paths that match the filtering criteria.
        """
        # Open USD file temporarily to scan prims
        temp_stage = Usd.Stage.Open(usd_path)

        # Apply include patterns (default to all if empty)
        matching_prims = []
        include_patterns = source.include_patterns or ["*"]

        for pattern in include_patterns:
            # Use existing prim_utils function!
            prims = find_matching_prims("/", pattern, stage=temp_stage)
            prim_paths = [str(prim.GetPath()) for prim in prims]
            matching_prims.extend(prim_paths)

        # Apply exclude patterns
        if source.exclude_patterns:
            filtered_prims = []
            for prim_path in matching_prims:
                excluded = any(
                    fnmatch.fnmatch(prim_path, exclude_pattern) for exclude_pattern in source.exclude_patterns
                )
                if not excluded:
                    filtered_prims.append(prim_path)
            return filtered_prims
        return matching_prims

    def _generate_prim_configs_with_auto_paths(
        self, filtered_prims: List[str], usd_path: str, scene_file: SceneFileConfig
    ) -> Dict[str, RigidObjectCfg]:
        """Generate prim configurations with automatic prim path generation.

        Creates RigidObjectCfg instances for each filtered prim by matching them against
        object configuration patterns and applying physics properties directly to the USD file.

        Parameters
        ----------
        filtered_prims : List[str]
            List of discovered prim paths from USD file that passed filtering.
        usd_path : str
            Path to the USD file containing the prims.
        scene_file : holosoma.config_types.simulator.SceneFileConfig
            Scene file configuration with object_configs patterns.

        Returns
        -------
        Dict[str, isaaclab.assets.RigidObjectCfg]
            Dictionary mapping stripped prim names to RigidObjectCfg objects.
        """
        prim_configs = {}

        if not scene_file.object_configs:
            logger.info("No object_configs found in scene_file, no objects will be loaded")
            return prim_configs

        for prim_path in filtered_prims:
            # Find matching object_config pattern
            matching_config = None
            matching_pattern = None
            for pattern, config in scene_file.object_configs.items():
                if fnmatch.fnmatch(prim_path, pattern):
                    matching_config = config
                    matching_pattern = pattern
                    break

            if not matching_config:
                logger.warning(f"No matching object_config pattern found for prim '{prim_path}', skipping")
                continue

            stripped_name = self._strip_path_prefixes(prim_path)
            target_prim_path = self._generate_target_prim_path(stripped_name)
            rigid_props, mass_props, collision_props, physics_config = self._resolve_physics_props_for_pattern(
                matching_config
            )

            # Apply some physics properties directly to USD file before spawning (TODO: why)
            temp_stage = Usd.Stage.Open(usd_path)
            if temp_stage and physics_config:
                from holosoma.simulator.isaacsim.usd_physics_utils import apply_physics_config_to_usd

                success = apply_physics_config_to_usd(temp_stage, prim_path, physics_config)
                if success:
                    temp_stage.Save()
                else:
                    raise RuntimeError(f"Failed to apply physics config to USD prim: {prim_path}")

            # Create RigidObjectCfg with basic properties (physics already applied to USD)
            from holosoma.simulator.isaacsim.spawners.from_files_cfg import CustomUsdFileCfg

            prim_configs[stripped_name] = RigidObjectCfg(
                prim_path=target_prim_path,
                spawn=CustomUsdFileCfg(
                    usd_path=usd_path,
                    source_path=prim_path,
                    rigid_props=rigid_props,
                    mass_props=mass_props,
                    collision_props=collision_props,
                    # Note: physics properties already applied directly to USD file
                ),
                # Note: init_state (position/rotation) will be computed from USD transforms
            )
        return prim_configs

    def _strip_path_prefixes(self, prim_path: str, strip_prefixes: List[str] = None) -> str:
        """Strip common prefixes from discovered prim paths.

        Parameters
        ----------
        prim_path : str
            The prim path to strip prefixes from.
        strip_prefixes : List[str], optional
            List of prefixes to strip. Defaults to ["/World", "/world", "/"].

        Returns
        -------
        str
            The prim path with prefixes stripped and leading slashes removed.
        """
        if strip_prefixes is None:
            strip_prefixes = ["/World", "/world", "/"]

        clean_path = prim_path
        for prefix in strip_prefixes:
            if clean_path.startswith(prefix):
                clean_path = clean_path[len(prefix) :]
                break

        return clean_path.lstrip("/")

    def _generate_target_prim_path(self, stripped_name: str, env_pattern: bool = True) -> str:
        """Generate target prim path for RigidObjectCfg.

        Parameters
        ----------
        stripped_name : str
            The stripped prim name to use in the target path.
        env_pattern : bool, optional
            Whether to use environment pattern matching. Defaults to True.

        Returns
        -------
        str
            The generated target prim path.
        """
        if env_pattern:
            return f"/World/envs/env_.*/{stripped_name}"
        else:
            return f"/World/envs/env_0/{stripped_name}"  # For single env

    def _create_rigid_object_cfg_with_physics(
        self,
        object_name: str,
        usd_path: str,
        source_path: str,  # None for individual, specific path for scene
        target_prim_path: str,
        physics_config,
        init_state=None,
    ) -> RigidObjectCfg:
        """Create RigidObjectCfg with physics for both scene and individual objects.

        Unified function to create RigidObjectCfg instances with physics properties
        applied for both scene objects and individual objects.

        Parameters
        ----------
        object_name : str
            Name of the object.
        usd_path : str
            Path to the USD file.
        source_path : str or None
            Specific prim path within USD file for scene objects, None for individual objects.
        target_prim_path : str
            Target prim path in the simulation scene.
        physics_config : holosoma.config_types.simulator.PhysicsConfig
            Physics configuration to apply.
        init_state : isaaclab.assets.RigidObjectCfg.InitialStateCfg, optional
            Initial state configuration for the object.

        Returns
        -------
        isaaclab.assets.RigidObjectCfg
            Configured RigidObjectCfg instance with physics properties.
        """
        from holosoma.simulator.isaacsim.spawners.from_files_cfg import CustomUsdFileCfg
        from holosoma.config_types.simulator import PhysicsConfig

        # Ensure we have a physics config
        physics_cfg = physics_config or PhysicsConfig()

        # Convert physics config to all property types
        rigid_props, mass_props, collision_props = self._convert_physics_to_props(physics_cfg)

        # Apply physics materials to USD file (unified approach)
        if physics_cfg and physics_cfg.isaacsim:
            self._apply_physics_materials_to_usd(usd_path, source_path or "/", physics_cfg)

        # Create unified RigidObjectCfg
        rigid_object_cfg = RigidObjectCfg(
            prim_path=target_prim_path,
            spawn=CustomUsdFileCfg(
                usd_path=usd_path,
                source_path=source_path,
                rigid_props=rigid_props,
                mass_props=mass_props,
                collision_props=collision_props,
            ),
            init_state=init_state,
        )

        return rigid_object_cfg

    def _convert_physics_to_props(self, physics_config):
        """Convert physics config to IsaacLab property configs.

        Parameters
        ----------
        physics_config : holosoma.config_types.simulator.PhysicsConfig
            Physics configuration to convert.

        Returns
        -------
        tuple[isaaclab.sim.RigidBodyPropertiesCfg, isaaclab.sim.MassPropertiesCfg, isaaclab.sim.CollisionPropertiesCfg]
            Tuple containing (rigid_props, mass_props, collision_props).
        """
        from holosoma.simulator.isaacsim.converters import physics_to_rigid_body_props, physics_to_mass_props
        import isaaclab.sim as sim_utils

        rigid_props = physics_to_rigid_body_props(physics_config)
        mass_props = physics_to_mass_props(physics_config)
        collision_props = sim_utils.CollisionPropertiesCfg()

        return rigid_props, mass_props, collision_props

    def _apply_physics_materials_to_usd(self, usd_path: str, prim_path: str, physics_config):
        """Apply physics materials to USD file.

        Parameters
        ----------
        usd_path : str
            Path to the USD file to modify.
        prim_path : str
            Prim path within the USD file to apply materials to.
        physics_config : holosoma.config_types.simulator.PhysicsConfig
            Physics configuration containing material properties.

        Raises
        ------
        RuntimeError
            If physics materials cannot be applied to the USD prim or USD stage cannot be opened.
        """
        temp_stage = Usd.Stage.Open(usd_path)
        if temp_stage:
            from holosoma.simulator.isaacsim.usd_physics_utils import apply_physics_config_to_usd

            # For individual objects, skip hierarchy conflict check since they're standalone USD files
            success = apply_physics_config_to_usd(
                temp_stage,
                prim_path,
                physics_config,
                disable_instanceable=True,
                skip_hierarchy_check=True,  # Skip for individual objects
            )
            if success:
                logger.debug(f"Applied physics materials to USD prim: {prim_path}")
                temp_stage.Save()
            else:
                raise RuntimeError(f"Failed to apply physics materials to USD prim: {prim_path}")
        else:
            raise RuntimeError(f"Could not open USD stage: {usd_path}")

    def _resolve_physics_props_for_pattern(self, pattern_config) -> tuple:
        """Resolve all physics properties for scene file patterns.

        Parameters
        ----------
        pattern_config : ObjectPatternConfig
            Pattern configuration containing physics properties.

        Returns
        -------
        tuple
            Tuple containing (rigid_props, mass_props, collision_props, physics_config).
        """
        from holosoma.simulator.isaacsim.converters import (
            physics_to_rigid_body_props,
            physics_to_mass_props,
            physics_to_rigid_body_material,
        )
        from holosoma.config_types.simulator import PhysicsConfig

        physics_config = pattern_config.physics or PhysicsConfig()

        # Convert to all three property types
        rigid_props = physics_to_rigid_body_props(physics_config)
        mass_props = physics_to_mass_props(physics_config)
        material_cfg = physics_to_rigid_body_material(physics_config)
        # Create collision props (without material - material goes directly in spawner)
        collision_props = sim_utils.CollisionPropertiesCfg()
        return rigid_props, mass_props, collision_props, physics_config

    def _resolve_rigid_body_props_for_object(self, object_config) -> sim_utils.RigidBodyPropertiesCfg:
        """Resolve physics for individual rigid objects.

        Parameters
        ----------
        object_config : Any
            Object configuration containing physics properties.

        Returns
        -------
        sim_utils.RigidBodyPropertiesCfg
            Resolved rigid body properties configuration.
        """
        from holosoma.simulator.isaacsim.converters import physics_to_rigid_body_props
        from holosoma.config_types.simulator import PhysicsConfig

        physics_config = object_config.physics or PhysicsConfig()
        return physics_to_rigid_body_props(physics_config)

    def load_object(self, object_name: str, object_config: Any) -> RigidObject:
        """Load a single USD object as a RigidObject.

        Parameters
        ----------
        object_name : str
            Name of the object to load.
        object_config : Any
            Object configuration with usd_path, pose, and physics properties.

        Returns
        -------
        isaaclab.assets.RigidObject
            Loaded RigidObject instance.

        Raises
        ------
        ValueError
            If USD path is not available or loading fails.
        """
        logger.debug(f"USDFileLoader: Loading individual object '{object_name}'")

        # Extract USD path from object config - handle both dict and object attribute access
        usd_path = None
        if hasattr(object_config, "usd_path"):
            usd_path = object_config.usd_path
        elif isinstance(object_config, dict) and "usd_path" in object_config:
            usd_path = object_config["usd_path"]

        if not usd_path:
            raise ValueError(f"No usd_path specified for object '{object_name}'")

        # Get pose and physics config - handle both dict and object attribute access
        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # default
        if hasattr(object_config, "pose"):
            pose = object_config.pose
        elif isinstance(object_config, dict) and "pose" in object_config:
            pose = object_config["pose"]

        physics_config = None
        if hasattr(object_config, "physics"):
            physics_config = object_config.physics
        elif isinstance(object_config, dict) and "physics" in object_config:
            physics_config = object_config["physics"]

        # Create RigidObject configuration
        rigid_object_cfg = self._create_rigid_object_config(object_name, usd_path, pose, physics_config)
        return RigidObject(rigid_object_cfg)

    def _load_usd_scene_objects(self, usd_path: str, loader_config: Dict[str, Any]) -> Any:
        """Load rigid objects from USD scene using the unified scene loader configuration."""
        logger.info(f"Loading USD scene from: {usd_path}")

        # Get the already-generated prim_configs (now RigidObjectCfg objects)
        prim_configs = loader_config.get("prim_configs", {})

        # Group prim_configs by their original patterns from scene_file.object_configs
        pattern_based_configs = {}

        # We need to reconstruct which pattern each stripped name came from
        temp_stage = Usd.Stage.Open(os.path.abspath(usd_path))

        for stripped_name, rigid_cfg in prim_configs.items():
            # Find the original USD path for this stripped name
            original_path = None
            for prim in temp_stage.Traverse():
                prim_path = str(prim.GetPath())
                if self._strip_path_prefixes(prim_path) == stripped_name:
                    original_path = prim_path
                    break

            if original_path:
                scene_file = loader_config.get("scene_file")
                if hasattr(scene_file, "include_patterns") and scene_file.include_patterns:
                    for include_pattern in scene_file.include_patterns:
                        if fnmatch.fnmatch(original_path, include_pattern):
                            # Use the include pattern as the key, not the object_configs pattern
                            if include_pattern not in pattern_based_configs:
                                pattern_based_configs[include_pattern] = rigid_cfg
                            break
                    else:
                        logger.warning(
                            f"Original path '{original_path}' doesn't match any include patterns: {scene_file.include_patterns}"
                        )
                else:
                    logger.warning(f"No include_patterns found in scene_file")

        usd_path_prim_configs = pattern_based_configs

        # Create USD scene loader configuration with original USD paths as keys
        scene_loader_cfg = UsdSceneLoaderCfg(
            usd_path=os.path.abspath(usd_path),
            prim_configs=usd_path_prim_configs,  # Use original USD paths as keys
            strip_prefixes=loader_config.get("strip_prefixes", ["/world/"]),
            scene_pos_offset=tuple(loader_config.get("scene_pos_offset", [0.0, 0.0, 0.0])),
            scene_rot_offset=tuple(loader_config.get("scene_rot_offset", [1.0, 0.0, 0.0, 0.0])),
            device=loader_config.get("device", self.device),
        )

        # Create the rigid object collection
        scene_collection, scene_collection_cfg = create_usd_scene_loader(scene_loader_cfg)
        if scene_collection is not None:
            # Debug: Print prim tree before adding to scene to help diagnose any issues
            from holosoma.simulator.isaacsim.prim_utils import print_prim_tree

            # print_prim_tree("/World/envs/env_0/", max_depth=4)
            logger.info(f"Successfully loaded {len(scene_collection.cfg.rigid_objects)} objects from USD scene")
            return scene_collection
        else:
            logger.warning(f"No objects were loaded from the USD scene: {usd_path}")
            return None

    def _load_individual_objects(self, scene_config: SceneConfig) -> Dict[str, RigidObject]:
        """
        Load individual objects from configuration.

        Args:
            scene_config: New unified SceneConfig

        Returns:
            Dictionary mapping object names to RigidObject instances
        """
        individual_objects = {}

        # Get assets root path for resolving relative paths
        assets_root = scene_config.asset_root

        for rigid_obj in scene_config.rigid_objects:
            try:
                if rigid_obj.usd_path:  # Only handle USD objects in this loader
                    rigid_object = self._load_individual_object(rigid_obj.name, rigid_obj, assets_root)
                    individual_objects[rigid_obj.name] = rigid_object
            except Exception as e:
                logger.error(f"Failed to load individual object '{rigid_obj.name}': {e}")
                raise

        logger.info(f"✅ USDFileLoader: Loaded {len(individual_objects)} individual objects")
        return individual_objects

    def _load_individual_objects_from_rigid_objects(self, scene_config: SceneConfig) -> Dict[str, RigidObject]:
        """
        Load individual objects from scene_config.rigid_objects (NEW CONFIG STRUCTURE).

        Args:
            scene_config: New unified SceneConfig with rigid_objects field

        Returns:
            Dictionary mapping object names to RigidObject instances
        """
        individual_objects = {}

        # Get assets root path for resolving relative paths
        assets_root = scene_config.asset_root

        for i, rigid_obj in enumerate(scene_config.rigid_objects):
            try:
                # Check if this object has a USD path (only handle USD objects in this loader)
                if hasattr(rigid_obj, "usd_path") and rigid_obj.usd_path:
                    # Convert RigidObjectConfig to the format expected by _load_individual_object
                    object_config = {
                        "usd_path": rigid_obj.usd_path,
                        "pose": rigid_obj.position
                        + rigid_obj.orientation,  # [x,y,z] + [w,x,y,z] = [x,y,z,w,x,y,z] - WRONG FORMAT!
                        "physics": rigid_obj.physics,
                    }

                    # Keep quaternion in wxyz format for internal consistency
                    # rigid_obj.position = [x, y, z]
                    # rigid_obj.orientation = [w, x, y, z] (IsaacLab format)
                    # Keep as [x, y, z, qw, qx, qy, qz] (wxyz format for internal consistency)
                    pos = rigid_obj.position[:3]  # [x, y, z]
                    quat_wxyz = rigid_obj.orientation[:4]  # [w, x, y, z]
                    pose = pos + quat_wxyz  # [x, y, z, qw, qx, qy, qz] - wxyz format for consistency

                    object_config["pose"] = pose

                    rigid_object = self._load_individual_object(rigid_obj.name, object_config, assets_root)
                    individual_objects[rigid_obj.name] = rigid_object
            except Exception as e:
                logger.error(f"USDFileLoader: Failed to load individual object '{rigid_obj.name}': {e}")
                raise

        logger.info(f"Loaded {len(individual_objects)} individual USD objects from rigid_objects")
        return individual_objects

    def _load_individual_object(self, object_name: str, object_config: Any, assets_root: str = None) -> RigidObject:
        """
        Load a single USD object as a RigidObject with assets root path resolution.

        Args:
            object_name: Name of the object
            object_config: Object configuration with usd_path, pose, physics
            assets_root: Assets root path for resolving relative paths

        Returns:
            RigidObject instance

        Raises:
            ValueError: If USD path is not available or loading fails
        """
        logger.debug(f"Loading individual object '{object_name}'")

        # Extract USD path from object config - handle both dict and object attribute access
        usd_path = None
        if hasattr(object_config, "usd_path"):
            usd_path = object_config.usd_path
        elif isinstance(object_config, dict) and "usd_path" in object_config:
            usd_path = object_config["usd_path"]

        if not usd_path:
            raise ValueError(f"No usd_path specified for object '{object_name}'")

        # Get pose and physics config - handle both dict and object attribute access
        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # default
        if hasattr(object_config, "pose"):
            pose = object_config.pose
        elif isinstance(object_config, dict) and "pose" in object_config:
            pose = object_config["pose"]

        physics_config = None
        if hasattr(object_config, "physics"):
            physics_config = object_config.physics
        elif isinstance(object_config, dict) and "physics" in object_config:
            physics_config = object_config["physics"]

        # Create RigidObject configuration with assets root path
        rigid_object_cfg = self._create_rigid_object_config(object_name, usd_path, pose, physics_config, assets_root)

        return RigidObject(rigid_object_cfg)

    def _create_rigid_object_config(
        self, object_name: str, usd_path: str, pose: list, physics_config: Any, assets_root: str = None
    ) -> RigidObjectCfg:
        """Create RigidObjectCfg from object parameters.

        Parameters
        ----------
        object_name : str
            Name of the object.
        usd_path : str
            Path to USD file (relative or absolute).
        pose : list[float]
            Object pose [x, y, z, qx, qy, qz, qw].
        physics_config : holosoma.config_types.simulator.PhysicsConfig or None
            Physics configuration object.
        assets_root : str, optional
            Assets root path for resolving relative paths.

        Returns
        -------
        isaaclab.assets.RigidObjectCfg
            Configured RigidObjectCfg instance.

        Raises
        ------
        ValueError
            If USD file is not found at the resolved path.
        """
        # Resolve USD path - handle both relative and absolute paths
        resolved_usd_path = usd_path
        if not os.path.isabs(usd_path) and assets_root:
            resolved_usd_path = os.path.join(assets_root, usd_path)

        # Check if file exists
        if not os.path.exists(resolved_usd_path):
            raise ValueError(f"USD file not found at path: '{resolved_usd_path}'.")

        logger.debug(f"🔗 USDFileLoader: Resolved USD path for '{object_name}': '{resolved_usd_path}'")

        # Create all physics properties using new converters
        from holosoma.simulator.isaacsim.converters import (
            physics_to_rigid_body_props,
            physics_to_mass_props,
            physics_to_rigid_body_material,
        )
        from holosoma.config_types.simulator import PhysicsConfig

        physics_cfg = physics_config or PhysicsConfig()

        # Convert to all three property types
        rigid_props = physics_to_rigid_body_props(physics_cfg)
        mass_props = physics_to_mass_props(physics_cfg)
        material_cfg = physics_to_rigid_body_material(physics_cfg)

        # Create collision props (without material - material goes directly in spawner)
        collision_props = sim_utils.CollisionPropertiesCfg()

        # Create initial state from pose
        init_state = RigidObjectCfg.InitialStateCfg(
            pos=tuple(pose[:3]),
            rot=tuple(pose[3:7]),  # IsaacLab expects xyzw quaternion format
        )

        # Use unified helper function (same as scene files)
        rigid_object_cfg = self._create_rigid_object_cfg_with_physics(
            object_name=object_name,
            usd_path=os.path.abspath(resolved_usd_path),
            source_path=None,  # None for entire USD file (individual objects)
            target_prim_path=f"/World/envs/env_.*/{object_name}",
            physics_config=physics_cfg,
            init_state=init_state,
        )

        return rigid_object_cfg
