"""
URDF Scene Loader for IsaacGym

This module handles URDF scene loading logic using the unified configuration system.

## Overview

The URDFSceneLoader processes two types of URDF objects:
1. **Scene Files**: Complex URDF files containing multiple objects (e.g., shelves with items)
2. **Rigid Objects**: Individual URDF objects with specific poses and physics properties

## Key Concepts

- **URDF Specs**: Standardized internal representation of objects to be loaded
- **Asset Configuration**: IsaacGym-specific loading parameters (physics, rendering options)
- **Physics Configuration**: Post-loading physics properties (damping, mass, etc.)
- **Pattern Matching**: Selective loading of objects based on include/exclude patterns

## Processing Flow

1. **Validation**: Check scene config structure and log key information
2. **Scene Processing**: Parse scene URDF files, apply transformations, extract individual objects
3. **Rigid Processing**: Process individual rigid objects with their poses and physics
4. **Asset Loading**: Load IsaacGym assets from URDF specs with proper configurations
5. **Storage**: Store loaded assets and initial states for environment creation

## Architecture

The loader uses a pipeline approach:
- Scene Config → URDF Specs → Loaded Assets → Environment Creation
- Each stage has focused responsibilities and clear data contracts
- Physics configs are stored separately for post-creation application
"""

from __future__ import annotations

import fnmatch
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import trimesh.transformations as tra
import yourdfpy  # type:ignore[import-untyped]
from isaacgym import gymapi
from loguru import logger
from yourdfpy.urdf import Robot  # type:ignore[import-untyped]

from holosoma.config_types.simulator import (
    RigidObjectConfig,
    SceneConfig,
    SceneFileConfig,
)

# Type aliases compatible with Python 3.8
Pose = List[float]  # [x, y, z, qx, qy, qz, qw]
AssetDict = Dict[str, gymapi.Asset]
PoseDict = Dict[str, Pose]


# Cannot freeze because we convert dynamically on load
@dataclass(frozen=False)
class AssetConfig:
    """Default configuration for IsaacGym asset loading.

    This dataclass maps directly to gymapi.AssetOptions parameters.
    Each field corresponds to an option that can be set on AssetOptions
    when loading assets through IsaacGym's load_asset() method.
    """

    default_dof_drive_mode: Any = None
    collapse_fixed_joints: bool = True
    replace_cylinder_with_capsule: bool = False
    flip_visual_attachments: bool = False
    fix_base_link: bool = False
    density: float = 1000.0
    angular_damping: float = 0.1
    linear_damping: float = 0.1
    max_angular_velocity: float = 1000.0
    max_linear_velocity: float = 1000.0
    armature: float = 0.0
    thickness: float = 0.001
    disable_gravity: bool = False


@dataclass(frozen=True)
class URDFSpec:
    """Unified URDF specification for scene loading.

    This dataclass represents a standardized specification for URDF objects
    that need to be loaded into the simulation environment. It supports both
    pre-loaded assets (for scene objects) and URDF file paths (for rigid objects).

    Parameters
    ----------
    name : str
        Unique identifier for the URDF object.
    pose : Pose
        Object pose as [x, y, z, qx, qy, qz, qw].
    asset : gymapi.Asset | None
        Pre-loaded IsaacGym asset for scene objects, by default None.
    urdf_path : str | None
        Path to URDF file for rigid objects, by default None.
    asset_config : AssetConfig | None
        IsaacGym asset configuration parameters, by default None.
    physics_config : dict[str, Any] | None
        Physics properties for post-creation application, by default None.
    """

    name: str
    pose: Pose
    asset: gymapi.Asset | None = None  # Pre-loaded asset (for scene objects)
    urdf_path: str | None = None  # Path to URDF file (for rigid objects)
    asset_config: AssetConfig | None = None
    physics_config: dict[str, Any] | None = None

    @property
    def is_preloaded(self) -> bool:
        """Check if asset is pre-loaded."""
        return self.asset is not None

    @property
    def needs_loading(self) -> bool:
        """Check if asset needs to be loaded from file."""
        return self.asset is None and self.urdf_path is not None


class URDFSceneLoader:
    """URDF scene loader working directly with typed dataclasses.

    This class handles loading URDF scenes using the unified configuration system.
    It processes both scene files (complex URDF files with multiple objects) and
    rigid objects (individual URDF objects with specific poses and physics).

    The loader uses a pipeline approach where scene configurations are converted
    to standardized URDF specs, then loaded as IsaacGym assets with proper
    physics configurations stored separately for post-creation application.

    Parameters
    ----------
    gym_instance : gymapi.Gym
        IsaacGym instance for asset loading operations.
    sim : gymapi.Sim
        IsaacGym simulation handle.
    device : str
        Device identifier for tensor operations.

    Attributes
    ----------
    gym : gymapi.Gym
        IsaacGym instance reference.
    sim : gymapi.Sim
        IsaacGym simulation handle.
    device : str
        Device identifier.
    object_physics_configs : dict
        Physics configurations for post-creation application.
    loaded_assets : dict
        Loaded IsaacGym assets indexed by object name.
    loaded_initial_states : dict
        Initial poses for loaded objects.
    """

    def __init__(self, gym_instance: gymapi.Gym, sim: gymapi.Sim, device: str) -> None:
        self.gym: gymapi.Gym = gym_instance
        self.sim: gymapi.Sim = sim
        self.device: str = device
        self._original_urdf_dir: str | None = None
        # Store physics configurations separately for post-creation application
        self.object_physics_configs: dict[str, dict[str, Any]] = {}
        # Store loaded data for access during environment creation
        self.loaded_assets: AssetDict = {}
        self.loaded_initial_states: PoseDict = {}

    # === PUBLIC INTERFACE ===

    def load_scene_files(self, scene_config: SceneConfig) -> tuple[AssetDict, PoseDict]:
        """Load URDF scenes using unified configuration structure.

        Main entry point for loading URDF scenes. Processes both scene files
        (complex URDF files with multiple objects) and rigid objects (individual
        URDF objects with specific poses and physics properties).

        Parameters
        ----------
        scene_config : SceneConfig
            Scene configuration dataclass containing scene_files and rigid_objects.

        Returns
        -------
        tuple[AssetDict, PoseDict]
            Tuple containing (object_assets, object_initial_state) where:
            - object_assets: dict mapping object names to IsaacGym assets
            - object_initial_state: dict mapping object names to poses [x,y,z,qx,qy,qz,qw]

        Raises
        ------
        ValueError
            If URDF scene loading fails due to configuration or file issues.
        """
        try:
            logger.debug(f"scene_config type: {type(scene_config)}")
            logger.debug(f"scene_config.scene_files: {getattr(scene_config, 'scene_files', None)}")
            logger.debug(f"scene_config.rigid_objects: {getattr(scene_config, 'rigid_objects', None)}")

            # Process different source types
            scene_specs = self._process_scene_files(scene_config)
            rigid_specs = self._process_rigid_objects(scene_config)

            # Combine and load assets
            all_specs = scene_specs + rigid_specs
            return self._load_assets_from_specs(all_specs)

        except Exception as e:
            logger.error(f"URDF scene loading failed: {e}")
            raise ValueError("URDF scene loading failed") from e

    # === SCENE PROCESSING ===

    def _process_scene_files(self, scene_config: SceneConfig) -> list[URDFSpec]:
        """Process scene URDF files and return URDFSpec objects"""
        if not (hasattr(scene_config, "scene_files") and scene_config.scene_files):
            return []

        # Check for multiple URDF scene files and raise exception
        urdf_scene_files = [
            sf for sf in scene_config.scene_files if self._scene_file_has_urdf_format(sf, scene_config.asset_root)
        ]

        if len(urdf_scene_files) > 1:
            urdf_paths = [sf.urdf_path for sf in urdf_scene_files]
            raise ValueError(f"Multiple URDF scene files found: {urdf_paths}. Only one URDF scene file is supported.")

        if urdf_scene_files:
            scene_file = urdf_scene_files[0]
            assert scene_config.asset_root is not None
            assert scene_file.urdf_path is not None
            urdf_path = str(Path(scene_config.asset_root) / scene_file.urdf_path)
            logger.info(f"Loading scene URDF: {urdf_path}")
            return self._generate_specs_from_scene_urdf(scene_file, scene_config)

        return []

    def _process_rigid_objects(self, scene_config) -> list[URDFSpec]:
        """Process rigid objects and return URDFSpec objects"""
        if not (hasattr(scene_config, "rigid_objects") and scene_config.rigid_objects):
            return []

        # Log the rigid object URDF files being loaded
        for obj in scene_config.rigid_objects:
            if obj.urdf_path:
                urdf_path = str(Path(scene_config.asset_root) / obj.urdf_path)
                logger.info(f"Loading rigid object URDF: {obj.name} -> {urdf_path}")

        return self._generate_specs_from_rigid_objects(scene_config.rigid_objects, scene_config.asset_root)

    def _load_assets_from_specs(self, specs: list[URDFSpec]) -> tuple[AssetDict, PoseDict]:
        """Load assets from standardized URDFSpec objects"""
        if not specs:
            logger.warning("No URDF objects to load")
            return {}, {}

        object_assets = {}
        object_initial_state = {}

        for spec in specs:
            if spec.is_preloaded:
                # Use pre-loaded asset
                object_assets[spec.name] = spec.asset
                object_initial_state[spec.name] = spec.pose
                logger.debug(f"Added pre-loaded '{spec.name}' asset")
            elif spec.needs_loading:
                # Load asset from file
                asset = self._load_and_store_asset(spec.name, spec.urdf_path, spec.asset_config)
                object_assets[spec.name] = asset
                object_initial_state[spec.name] = spec.pose
                logger.debug(f"Loaded '{spec.name}' from {spec.urdf_path}")
            else:
                logger.error(f"Invalid URDFSpec: {spec}")
                raise ValueError(f"URDFSpec missing both asset and urdf_path: {spec.name}")

        # Store the loaded data for access during environment creation
        self.loaded_assets = object_assets
        self.loaded_initial_states = object_initial_state

        logger.info(f"Successfully loaded {len(object_assets)} URDF objects")
        return object_assets, object_initial_state

    def _load_and_store_asset(self, name: str, urdf_path: str | None, asset_config: AssetConfig | None):
        """Load asset and store physics config"""
        try:
            assert urdf_path is not None
            assert asset_config is not None
            asset = self._load_asset_from_urdf(urdf_path, asset_config)
            if asset is None:
                error_msg = f"Failed to load '{name}': IsaacGym returned None asset from {urdf_path}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Store physics config for post-creation application
            physics_config = self._extract_physics_config_from_asset_config(asset_config)
            if physics_config:
                self.object_physics_configs[name] = physics_config
                logger.debug(f"Stored physics config for '{name}'")

            return asset
        except Exception as e:
            error_msg = f"Failed to load '{name}' from {urdf_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _scene_file_has_urdf_format(self, scene_file, asset_root):
        """Check if scene file has URDF format available"""
        if not hasattr(scene_file, "urdf_path") or not scene_file.urdf_path:
            return False

        urdf_path = str(Path(asset_root) / scene_file.urdf_path)
        exists = Path(urdf_path).exists()
        logger.debug(f"Checking scene file URDF: {urdf_path} -> {exists}")
        return exists

    def _generate_specs_from_scene_urdf(self, scene_file, scene_config):
        """
        Parse a URDF as a "scene" file and return standardized URDF specs.

        This is a helper because Isaacgym doesn't directly support loading URDF files
        into multiple actors. Instead, we parse top-level links and create actors
        for each.

        Scene URDF Processing Flow:
        1. Parse URDF with yourdfpy to build scene graph
        2. Apply scene-level transformations to the scene graph
        3. Extract individual links as separate objects
        4. Create temporary URDF files for each link
        5. Load IsaacGym assets immediately (while temp files exist)
        6. Return specs with pre-loaded assets
        """
        urdf_path = str(Path(scene_config.asset_root) / scene_file.urdf_path)
        logger.debug(f"Loading scene URDF file: {urdf_path}")

        # Store original URDF directory for mesh path resolution
        self._original_urdf_dir = str(Path(urdf_path).parent)

        # Step 1: Parse URDF with scene graph for transform extraction
        urdf = self._parse_scene_urdf_with_yourdfpy(urdf_path, scene_file)

        # Step 2: Apply scene-level transformations to the entire scene graph
        self._apply_scene_transform_to_urdf_scene_graph(urdf, scene_file)

        # Step 3: Process each link as an individual object
        urdf_specs = []
        for link in urdf.robot.links:
            if link.name == "world":
                continue

            # Apply pattern-based filtering
            if not self._should_load_object(link.name, scene_file):
                logger.debug(f"Skipping '{link.name}' due to scene file patterns")
                continue

            # Get IsaacGym asset configuration
            asset_config = self._get_asset_config_for_scene_object(link.name, scene_file)

            # Extract final world pose (after scene transformations)
            if urdf.scene is None:
                logger.warning(f"No scene graph for '{link.name}', using default pose")
                pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                pose = self._extract_world_transform_from_scene(urdf, link.name)

            # Step 4 & 5: Create temp URDF and load asset immediately
            with self._create_individual_link_urdf_tempfile(link, link.name, urdf_path) as temp_urdf_file:
                asset = self._load_asset_from_urdf(temp_urdf_file.name, asset_config)
                if asset is None:
                    error_msg = f"Failed to load '{link.name}': IsaacGym returned None asset from {temp_urdf_file.name}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Store physics config for post-creation application
                physics_config = self._extract_physics_config_from_asset_config(asset_config)
                if physics_config:
                    self.object_physics_configs[link.name] = physics_config
                    logger.debug(f"Stored physics config for scene object '{link.name}'")

                # Step 6: Create URDFSpec with pre-loaded asset
                urdf_spec = URDFSpec(
                    name=link.name,
                    pose=pose,
                    asset=asset,  # Pre-loaded
                    urdf_path=None,  # Not needed for scene objects
                    asset_config=asset_config,
                    physics_config=None,  # Physics config already stored separately
                )
                urdf_specs.append(urdf_spec)
                logger.debug(f"Generated URDFSpec for '{link.name}' at pose {pose[:3]}")

        logger.debug(f"Generated {len(urdf_specs)} URDFSpec objects from scene URDF file")
        return urdf_specs

    def _generate_specs_from_rigid_objects(self, rigid_objects_list, asset_root):
        """Generate URDF specs from rigid_objects list"""
        logger.debug("Loading rigid objects")

        if not rigid_objects_list:
            logger.warning("No rigid_objects configuration found")
            return []

        urdf_specs = []

        for obj in rigid_objects_list:
            try:
                # Get object name
                object_name = obj.name

                # Get URDF file path
                if not obj.urdf_path:
                    logger.warning(f"No urdf_path specified for object '{object_name}', skipping")
                    continue

                # Construct full path using asset root
                urdf_path = str(Path(asset_root) / obj.urdf_path)

                # Check if file exists
                if not Path(urdf_path).exists():
                    logger.warning(f"URDF file not found for object '{object_name}': {urdf_path}")
                    continue

                # Convert orientation from [w, x, y, z] to [x, y, z, w] for IsaacGym
                pose = obj.position + [obj.orientation[1], obj.orientation[2], obj.orientation[3], obj.orientation[0]]

                # Get asset configuration from physics config
                asset_config = self._get_asset_config_for_rigid_object(object_name, obj)

                # Create URDFSpec for rigid object
                urdf_spec = URDFSpec(
                    name=object_name,
                    pose=pose,
                    asset=None,  # Needs loading
                    urdf_path=urdf_path,
                    asset_config=asset_config,
                    physics_config=None,  # Physics config will be extracted during loading
                )
                urdf_specs.append(urdf_spec)
                logger.debug(f"Generated URDFSpec for '{object_name}' at {urdf_path}")

            except Exception as e:
                logger.error(f"Failed to process rigid object '{object_name}': {e}")
                continue

        logger.debug(f"Generated {len(urdf_specs)} URDFSpec objects for rigid objects")
        return urdf_specs

    # === UTILITIES ===

    def _should_load_object(self, object_name: str, scene_file) -> bool:
        """Determine if object should be loaded based on scene file patterns"""
        return self._should_load_object_by_patterns(
            object_name, scene_file.include_patterns, scene_file.exclude_patterns
        )

    def _should_load_object_by_patterns(
        self, object_name: str, include_patterns: list[str], exclude_patterns: list[str]
    ) -> bool:
        """Simplified pattern matching logic"""
        logger.debug(f"Checking patterns for '{object_name}': include={include_patterns}, exclude={exclude_patterns}")

        # If no patterns specified, load everything
        if not include_patterns and not exclude_patterns:
            return True

        # Check exclude patterns first
        if self._matches_any_pattern(object_name, exclude_patterns):
            logger.debug(f"Object '{object_name}' excluded by pattern")
            return False

        # Check include patterns (if any specified)
        if include_patterns:
            if self._matches_any_pattern(object_name, include_patterns):
                logger.debug(f"Object '{object_name}' included by pattern")
                return True
            logger.debug(f"Object '{object_name}' matches no include patterns")
            return False

        # No include patterns but passed exclude check
        logger.debug(f"Object '{object_name}' passed exclude check")
        return True

    def _matches_any_pattern(self, object_name: str, patterns: list[str]) -> bool:
        """Check if object name matches any of the given patterns"""
        for pattern in patterns:
            clean_pattern = pattern.replace("*/", "")  # Remove URDF path prefixes
            if fnmatch.fnmatch(object_name, clean_pattern):
                return True
        return False

    # === ASSET CONFIGURATION ===

    def _get_asset_config_for_scene_object(self, object_name: str, scene_file: SceneFileConfig) -> AssetConfig:
        """Get asset configuration for a scene object"""
        defaults = self._get_default_asset_config(is_scene_object=True)
        physics_config = self._find_physics_config_by_pattern(object_name, scene_file.object_configs)
        return self._apply_physics_config(defaults, physics_config, object_name)

    def _get_asset_config_for_rigid_object(self, object_name: str, obj: RigidObjectConfig) -> AssetConfig:
        """Get asset configuration for a rigid object"""
        defaults = self._get_default_asset_config(is_scene_object=False)
        physics_config = obj.physics
        return self._apply_physics_config(defaults, physics_config, object_name)

    def _get_default_asset_config(self, is_scene_object: bool) -> AssetConfig:
        """Create default asset config with type-appropriate defaults"""
        return AssetConfig(
            fix_base_link=is_scene_object,  # Scene objects fixed, rigid objects movable
            angular_damping=0.0 if is_scene_object else 0.1,
            linear_damping=0.0 if is_scene_object else 0.1,
        )

    def _apply_physics_config(self, asset_config, physics_config, object_name: str):
        """Apply physics config to asset config with consistent logging"""
        if not physics_config:
            return asset_config

        logger.debug(f"Applying physics config for '{object_name}'")

        # Map kinematic_enabled to fix_base_link (IsaacGym's equivalent)
        if hasattr(physics_config, "kinematic_enabled"):
            asset_config.fix_base_link = physics_config.kinematic_enabled
            logger.debug(f"Set fix_base_link={physics_config.kinematic_enabled} for '{object_name}'")

        # Apply other physics properties
        physics_properties = [
            "density",
            "angular_damping",
            "linear_damping",
            "max_angular_velocity",
            "max_linear_velocity",
        ]

        for prop in physics_properties:
            if hasattr(physics_config, prop):
                value = getattr(physics_config, prop)
                setattr(asset_config, prop, value)
                logger.debug(f"Set {prop}={value} for '{object_name}'")

        return asset_config

    def _find_physics_config_by_pattern(self, object_name: str, object_configs):
        """Find matching physics config using pattern matching"""
        if not object_configs:
            return None

        for pattern, obj_config in object_configs.items():
            clean_pattern = pattern.replace("*/", "")
            if fnmatch.fnmatch(object_name, clean_pattern):
                logger.debug(f"Object '{object_name}' matches pattern '{pattern}'")
                return obj_config.physics if obj_config.physics else None

        return None

    # === ASSET LOADING ===

    def _load_asset_from_urdf(self, urdf_path: str | None, asset_config: AssetConfig):
        """Load asset from URDF file"""
        if not urdf_path:
            raise ValueError("Missing path: {urdf_path}")

        # Store original URDF directory for scene objects (mesh path resolution)
        if self._original_urdf_dir is None:
            self._original_urdf_dir = str(Path(urdf_path).parent)

        # For individual objects, use the actual URDF path, not the scene URDF directory
        asset_root = str(Path(urdf_path).parent)
        asset_file = Path(urdf_path).name

        logger.debug(f"Loading URDF asset: asset_root='{asset_root}', asset_file='{asset_file}'")

        return self._load_gym_asset(asset_root, asset_file, asset_config)

    def _load_gym_asset(self, asset_root, asset_file, asset_cfg):
        """Load object asset using IsaacGym"""
        asset_path = str(Path(asset_root) / asset_file)
        gym_asset_root = str(Path(asset_path).parent)
        gym_asset_file = Path(asset_path).name

        asset_options = gymapi.AssetOptions()
        asset_config_options = [
            "default_dof_drive_mode",
            "collapse_fixed_joints",
            "replace_cylinder_with_capsule",
            "flip_visual_attachments",
            "fix_base_link",
            "density",
            "angular_damping",
            "linear_damping",
            "max_angular_velocity",
            "max_linear_velocity",
            "armature",
            "thickness",
            "disable_gravity",
        ]

        # Apply asset configuration options
        for option in asset_config_options:
            if hasattr(asset_cfg, option):
                value = getattr(asset_cfg, option)
                if value is not None:
                    setattr(asset_options, option, value)

        object_asset = self.gym.load_asset(self.sim, gym_asset_root, gym_asset_file, asset_options)

        if object_asset is None:
            logger.error(f"IsaacGym returned None asset for '{gym_asset_file}' - check URDF file and paths")
        else:
            logger.debug(f"Successfully loaded asset '{gym_asset_file}' with density={asset_options.density}")

        return object_asset

    # === URDF PARSING & TRANSFORMATION ===

    def _parse_scene_urdf_with_yourdfpy(self, scene_urdf_path, source):
        """Parse scene URDF using yourdfpy"""
        logger.debug(f"Parsing scene URDF with yourdfpy: {scene_urdf_path}")

        # Use proper filename handler to resolve mesh paths relative to URDF file
        filename_handler = partial(yourdfpy.filename_handler_relative_to_urdf_file, urdf_fname=scene_urdf_path)

        # Load with scene graph for transform extraction
        urdf = yourdfpy.URDF.load(
            scene_urdf_path,
            load_meshes=False,  # Skip mesh loading for faster parsing
            build_scene_graph=True,  # Build scene graph for transforms
            filename_handler=filename_handler,  # Proper path resolution
        )

        logger.debug(
            f"Parsed URDF: {urdf.robot.name} with {len(urdf.robot.links)} links, {len(urdf.robot.joints)} joints"
        )
        logger.debug(f"Scene graph built: {urdf.scene is not None}")

        return urdf

    @contextmanager
    def _create_individual_link_urdf_tempfile(self, link, link_name, scene_urdf_path):
        """Create individual URDF file for a single link"""
        # Create minimal robot with just this link
        minimal_robot = Robot(name=link_name)
        minimal_robot.links = [link]
        minimal_robot.joints = []  # No joints for single link
        minimal_robot.materials = []  # Materials embedded in link visuals

        # Use proper filename handler
        filename_handler = partial(yourdfpy.filename_handler_relative_to_urdf_file, urdf_fname=scene_urdf_path)

        minimal_urdf = yourdfpy.URDF(
            robot=minimal_robot, build_scene_graph=False, load_meshes=False, filename_handler=filename_handler
        )

        # Generate XML string
        xml_string_bytes = minimal_urdf.write_xml_string()

        # Convert bytes to string if needed
        if isinstance(xml_string_bytes, bytes):
            xml_string = xml_string_bytes.decode("utf-8")
        else:
            xml_string = xml_string_bytes

        logger.debug(f"Generated URDF for '{link_name}' ({len(xml_string)} chars)")

        # Fix mesh paths to absolute paths
        xml_string = self._fix_mesh_paths_in_urdf(xml_string, link_name)

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", prefix=f"{link_name}_", dir=self._original_urdf_dir, delete=False
        ) as temp_file:
            temp_file.write(xml_string)
            temp_file.close()

            logger.debug(f"Created temporary URDF for '{link_name}': {temp_file.name}")

            try:
                # Return a simple object with the name attribute
                yield type("TempFile", (), {"name": temp_file.name})()
            finally:
                # Clean up the temporary file
                try:
                    Path(temp_file.name).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                except OSError as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file.name}: {e}")

    def _fix_mesh_paths_in_urdf(self, xml_string, link_name):
        """Fix relative mesh paths to absolute paths in URDF XML"""
        # Find all mesh filename references in the XML
        mesh_matches = re.findall(r'filename="([^"]*)"', xml_string)

        if not mesh_matches:
            logger.debug(f"No mesh references found in URDF for '{link_name}'")
            return xml_string

        logger.debug(f"Found {len(mesh_matches)} mesh references in '{link_name}': {mesh_matches}")

        fixed_xml = xml_string
        fixes_applied = 0

        for mesh_path in mesh_matches:
            if not Path(mesh_path).is_absolute():  # Only fix relative paths
                # Convert relative path to absolute path based on original URDF directory
                absolute_mesh_path = str(Path(self._original_urdf_dir) / mesh_path)

                logger.debug(f"Fixing mesh path: '{mesh_path}' -> '{absolute_mesh_path}'")

                # Apply the fix regardless of file existence (IsaacGym will handle missing files)
                fixed_xml = fixed_xml.replace(f'filename="{mesh_path}"', f'filename="{absolute_mesh_path}"')
                fixes_applied += 1
            else:
                logger.debug(f"Already absolute path: '{mesh_path}'")

        if fixes_applied > 0:
            logger.debug(f"Applied {fixes_applied} mesh path fixes for '{link_name}'")

        return fixed_xml

    def _extract_world_transform_from_scene(self, urdf, link_name):
        """Extract world transform for a link using yourdfpy"""
        if urdf.scene is None:
            logger.error(f"No scene graph available for link '{link_name}'")
            raise RuntimeError(f"No scene graph available for link '{link_name}'")

        # Use yourdfpy's get_transform method to get world transform
        world_transform = urdf.get_transform(frame_to=link_name, frame_from="world")
        logger.debug(f"Got world transform for '{link_name}' using get_transform")

        # Extract translation and rotation
        translation = tra.translation_from_matrix(world_transform)
        quaternion_wxyz = tra.quaternion_from_matrix(world_transform)

        # Convert from [w, x, y, z] to [x, y, z, w] format for IsaacGym
        quaternion_xyzw = [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]

        pose = translation.tolist() + quaternion_xyzw
        logger.debug(f"Extracted world pose for '{link_name}': {pose[:3]} (translation)")

        return pose

    def _apply_scene_transform_to_urdf_scene_graph(self, urdf, source):
        """Apply scene transformation to the URDF scene graph using configurable root link.

        Args:
            urdf: yourdfpy URDF object with scene graph
            source: Source dataclass containing position and orientation
        """
        logger.debug("Applying scene transform to URDF scene graph")

        # Extract offsets from source
        pos_offset = source.position  # [x, y, z]
        rot_offset = source.orientation  # [w, x, y, z] format

        logger.debug(f"pos_offset={pos_offset}, rot_offset={rot_offset}")

        # Check if rotation offset is identity
        if rot_offset == [1.0, 0.0, 0.0, 0.0]:
            logger.warning("rot_offset is identity quaternion - no rotation will be applied!")

        # Create translation matrix
        translation_matrix = tra.translation_matrix(pos_offset)

        # Create rotation matrix from quaternion (wxyz format)
        rotation_matrix = tra.quaternion_matrix(rot_offset)

        # Combine translation and rotation
        scene_transform_matrix = translation_matrix @ rotation_matrix

        # Apply transform to the scene graph
        if urdf.scene is not None:
            logger.debug("Applying transform to scene graph")

            # Use configurable transform root link (validation ensures it exists)
            target_link = source.urdf_settings.transform_root_link
            logger.debug(f"Using configured transform_root_link: '{target_link}'")

            if target_link in urdf.scene.graph.nodes:
                try:
                    # Get current transform for the target link
                    current_transform = urdf.scene.graph.get(
                        frame_to=target_link, frame_from=urdf.scene.graph.base_frame
                    )[0]

                    # Apply scene transform: new_transform = scene_transform * current_transform
                    new_transform = scene_transform_matrix @ current_transform

                    # Update the scene graph with the new transform
                    urdf.scene.graph.update(
                        frame_from=urdf.scene.graph.base_frame, frame_to=target_link, matrix=new_transform
                    )

                    logger.debug(f"Applied scene transform to '{target_link}' in scene graph")

                except Exception as e:
                    logger.error(f"Failed to apply scene transform to '{target_link}': {e}")
                    raise RuntimeError(f"Scene transform application failed: {e}")
            else:
                logger.error(
                    f"Target link '{target_link}' not found in scene graph nodes: {list(urdf.scene.graph.nodes)}"
                )
                raise KeyError(f"Target link '{target_link}' not found in scene graph")

            logger.debug("Scene graph transformation completed")
        else:
            logger.warning("No scene graph available - cannot apply transform")

    def _extract_physics_config_from_asset_config(self, asset_config):
        """Extract physics configuration from asset config for post-creation application"""
        if not asset_config:
            raise RuntimeError("Asset config is required but was None")

        # Create physics config dictionary from asset config
        physics_config = {}

        # Map asset config properties to physics config - all properties are required
        if not hasattr(asset_config, "fix_base_link"):
            raise RuntimeError("Asset config missing required property 'fix_base_link'")
        physics_config["kinematic_enabled"] = asset_config.fix_base_link

        if not hasattr(asset_config, "linear_damping"):
            raise RuntimeError("Asset config missing required property 'linear_damping'")
        physics_config["linear_damping"] = asset_config.linear_damping

        if not hasattr(asset_config, "angular_damping"):
            raise RuntimeError("Asset config missing required property 'angular_damping'")
        physics_config["angular_damping"] = asset_config.angular_damping

        if not hasattr(asset_config, "density"):
            raise RuntimeError("Asset config missing required property 'density'")
        physics_config["density"] = asset_config.density

        if not hasattr(asset_config, "max_linear_velocity"):
            raise RuntimeError("Asset config missing required property 'max_linear_velocity'")
        physics_config["max_linear_velocity"] = asset_config.max_linear_velocity

        if not hasattr(asset_config, "max_angular_velocity"):
            raise RuntimeError("Asset config missing required property 'max_angular_velocity'")
        physics_config["max_angular_velocity"] = asset_config.max_angular_velocity

        return physics_config

    def get_initial_pose(self, object_name: str) -> list[float]:
        """Get initial pose for an object by name

        Args:
            object_name: Name of the object to get pose for

        Returns:
            Initial pose [x, y, z, qx, qy, qz, qw]

        Raises:
            KeyError: If object not found in loaded initial states
        """
        if object_name not in self.loaded_initial_states:
            available = list(self.loaded_initial_states.keys())
            raise KeyError(f"Object '{object_name}' not found in loaded initial states. Available: {available}")
        return self.loaded_initial_states[object_name]
