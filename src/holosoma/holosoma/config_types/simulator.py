from __future__ import annotations

from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

import tyro
from pydantic import model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated

from holosoma.config_types.viewer import ViewerConfig
from holosoma.utils.sync_rendezvous import DEFAULT_POLICY_SYNC_PORT


class MujocoBackend(str, Enum):
    """MuJoCo physics backend selection.

    Determines which MuJoCo backend to use for physics simulation.
    """

    CLASSIC = "classic"
    """CPU-based single environment backend."""

    WARP = "warp"
    """GPU-accelerated multi-environment backend."""


@dataclass(frozen=True)
class MujocoWarpConfig:
    """Configuration for MuJoCo Warp backend memory allocation.

    Controls GPU memory allocation for batched parallel simulation.
    Increase these values if you encounter overflow warnings during training.
    """

    nconmax_per_env: int = 96
    """Maximum contacts per environment (default: 96).

    Increase for:
    - Complex terrains with many contact points
    - Robots with numerous collision geometries
    - Multi-contact scenarios (manipulation, climbing)

    Memory scales as: num_envs x nconmax_per_env
    """

    njmax_per_env: int | None = None
    """Maximum constraints per environment (default: auto-calculated).

    If None (default), automatically calculated as: max(nconmax * 6, nv * 4)
    where nv is the model's velocity dimension.

    Constraints include:
    - Contact constraints (friction cones: ~6 per contact)
    - Joint limits
    - Equality constraints

    Only override if you know you need more constraint capacity.
    """


@dataclass(frozen=True)
class ResetManagerConfig:
    """Configuration for the reset event manager."""

    events: list[Any] = field(default_factory=list)
    """List of reset event configurations to be managed."""


@dataclass(frozen=True)
class PhysxConfig:
    """Low-level PhysX solver settings."""

    solver_type: int
    """Solver type identifier passed to PhysX."""

    num_position_iterations: int
    """Number of position iterations per solver step."""

    num_velocity_iterations: int
    """Number of velocity iterations per solver step."""

    num_threads: int = 4
    """Worker thread count used by PhysX."""

    enable_dof_force_sensors: bool = False
    """Whether to enable force sensors on individual DOFs."""

    bounce_threshold_velocity: float = 0.5
    """Velocity threshold below which bounce responses are suppressed."""


@dataclass(frozen=True)
class MujocoXMLFilterCfg:
    """Configuration for filtering MuJoCo MJCF/XML robot files.

    This configuration controls how robot MJCF files are processed and filtered
    when loaded into the MuJoCo simulator. It allows removal of specific elements
    that may conflict with the simulation environment or cause issues.
    """

    enable: bool = False
    """Whether to enable XML filtering."""

    remove_lights: bool = True
    """Whether to remove <light> elements from the MJCF file."""

    remove_ground: bool = True
    """Whether to remove ground/floor/plane geometries from the MJCF file.
    Assumes these are top-level worldbody geoms."""

    ground_names: list[str] = field(default_factory=lambda: ["floor", "ground", "plane"])
    """List of geometry names to identify and remove as ground elements."""


@dataclass(frozen=True)
class SimEngineConfig:
    """Top-level simulation engine settings."""

    fps: int
    """Target simulation frames per second."""

    control_decimation: int
    """Number of physics steps between agent control updates."""

    substeps: int
    """Number of substeps per physics frame."""

    physx: PhysxConfig
    """PhysX solver configuration."""

    render_mode: str = "human"
    """Rendering mode requested from the simulator."""

    render_interval: int = 1
    """Number of physics frames between rendered frames."""

    max_episode_length_s: float = 20.0
    """Maximum episode length in seconds."""


@dataclass(frozen=True)
class IsaacGymPhysicsConfig:
    """Rigid-shape material properties for the IsaacGym simulator.

    Provides 1:1 mapping to IsaacGym RigidShapeProperties for physics simulation.
    See PhysicsConfig for common physics parameter descriptions.
    """

    friction: float = 1.0
    """Static friction coefficient. Defaults to 1.0."""

    rolling_friction: float = 0.0
    """Rolling resistance coefficient. Defaults to 0.0."""

    torsion_friction: float = 0.0
    """Torsion resistance coefficient. Defaults to 0.0."""

    restitution: float = 0.0
    """Bounce coefficient. Defaults to 0.0."""

    compliance: float = 0.0
    """Shape compliance. Defaults to 0.0."""


@dataclass(frozen=True)
class IsaacSimPhysicsConfig:
    """Rigid-body material properties for the IsaacSim simulator.

    Provides 1:1 mapping to IsaacLab RigidBodyMaterialCfg for physics simulation.
    See PhysicsConfig for common physics parameter descriptions.
    """

    static_friction: float = 1.0
    """Static friction coefficient. Defaults to 1.0."""

    dynamic_friction: float = 1.0
    """Dynamic friction coefficient. Defaults to 1.0."""

    restitution: float = 0.0
    """Bounce coefficient. Defaults to 0.0."""

    friction_combine_mode: str = "multiply"
    """Friction combination mode. Options: "multiply", "max", "min", "average". Defaults to "multiply"."""

    restitution_combine_mode: str = "multiply"
    """Restitution combination mode. Options: "multiply", "max", "min", "average". Defaults to "multiply"."""


@dataclass(frozen=True)
class PhysicsConfig:
    """Unified physics configuration shared across simulators.

    Provides a unified interface for physics properties that can be used across
    different simulators, with simulator-specific sections for detailed control.
    """

    # Essential properties (supported by both simulators)
    kinematic_enabled: bool = False
    """Whether to enable kinematic behavior (static vs dynamic). Defaults to False."""

    mass: float | None = None
    """Direct mass override (highest priority). Defaults to None."""

    density: float | None = None
    """Density for mass calculation (medium priority). Defaults to None."""

    # Damping (both simulators support this!)
    linear_damping: float = 0.1
    """Linear velocity damping coefficient. Defaults to 0.1."""

    angular_damping: float = 0.1
    """Angular velocity damping coefficient. Defaults to 0.1."""

    max_linear_velocity: float = 1000.0
    """Maximum linear velocity limit. Defaults to 1000.0."""

    max_angular_velocity: float = 1000.0
    """Maximum angular velocity limit. Defaults to 1000.0."""

    isaacgym: IsaacGymPhysicsConfig | None = None
    """IsaacGym-specific physics configuration. Defaults to None."""

    isaacsim: IsaacSimPhysicsConfig | None = None
    """IsaacSim-specific physics configuration. Defaults to None."""


@dataclass(frozen=True)
class URDFSettings:
    """URDF-specific loader settings."""

    transform_root_link: str | None = None
    """Root link name for applying transforms. Defaults to None."""


@dataclass(frozen=True)
class ObjectPatternConfig:
    """Configuration for object patterns in scene files."""

    physics: PhysicsConfig | None = None
    """Physics configuration to apply to matching objects. Defaults to None."""


@dataclass(frozen=True)
class SceneFileConfig:
    """Individual scene file configuration.

    Configuration for loading scene files (USD/URDF) with transforms, filtering,
    and object-specific settings.
    """

    usd_path: str | None = None
    """Path to USD scene file. Defaults to None."""

    urdf_path: str | None = None
    """Path to URDF scene file. Defaults to None."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Position offset [x, y, z]. Defaults to [0.0, 0.0, 0.0]."""

    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    """Orientation quaternion [w, x, y, z]. Defaults to [1.0, 0.0, 0.0, 0.0]."""

    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    """Scale factors [x, y, z]. Defaults to [1.0, 1.0, 1.0]."""

    include_patterns: list[str] = field(default_factory=lambda: ["*"])
    """Patterns for including objects. Defaults to ["*"]."""

    exclude_patterns: list[str] = field(default_factory=list)
    """Patterns for excluding objects. Defaults to empty list."""

    object_configs: dict[str, ObjectPatternConfig] | None = None  # FIX: was Any
    """Object-specific configurations by pattern. Defaults to None."""

    asset_root: str | None = None
    """Root directory for resolving relative paths. Defaults to None."""

    urdf_settings: URDFSettings | None = None
    """URDF-specific settings. Defaults to None."""

    @model_validator(mode="after")
    def validate_urdf_transform_requirements(self) -> SceneFileConfig:
        """Require transform_root_link when URDF transform is applied"""
        if self.urdf_path:  # Only validate for URDF files
            has_transform = self.position != [0.0, 0.0, 0.0] or self.orientation != [1.0, 0.0, 0.0, 0.0]

            if has_transform:
                if not self.urdf_settings or not self.urdf_settings.transform_root_link:
                    raise ValueError(
                        f"URDF scene file '{self.urdf_path}' has position/orientation transform "
                        f"but missing required 'urdf_settings.transform_root_link'. "
                        f"Please specify which link to apply the transform to."
                    )
        return self

    def get_asset_path(self, format_type: str, fallback_root: str | None = None) -> str:
        """Get full path to asset file for specified format.

        Parameters
        ----------
        format_type : str
            Asset format type ('usd' or 'urdf').
        fallback_root : str, optional
            Fallback root directory if asset_root is not set.

        Returns
        -------
        str
            Full path to the asset file.

        Raises
        ------
        ValueError
            If the specified format is not configured for this source.
        """
        format_map = {
            "usd": self.usd_path,
            "urdf": self.urdf_path,
        }

        if format_type not in format_map or format_map[format_type] is None:
            raise ValueError(f"Asset format '{format_type}' not configured for this source")

        asset_path = format_map[format_type]
        assert asset_path is not None
        root_path = self.asset_root if self.asset_root is not None else fallback_root

        if not Path(asset_path).is_absolute():
            if not root_path:
                raise ValueError(f"Root path is required for relative path: {asset_path}")
            return str(Path(root_path) / asset_path)

        return asset_path

    def has_format(self, format_type: str, fallback_root: str | None = None) -> bool:
        """Check if format is configured and file exists.

        Parameters
        ----------
        format_type : str
            Asset format type ('usd' or 'urdf').
        fallback_root : str, optional
            Fallback root directory if asset_root is not set.

        Returns
        -------
        bool
            True if format is configured and file exists, False otherwise.
        """
        try:
            full_path = self.get_asset_path(format_type, fallback_root)
            return Path(full_path).exists()
        except ValueError:
            return False


@dataclass(frozen=True)
class RigidObjectConfig:
    """Configuration for individual rigid objects."""

    name: str
    """Name identifier for the rigid object."""

    urdf_path: str | None = None
    """Path to URDF file for the object. Defaults to None."""

    usd_path: str | None = None
    """Path to USD file for the object. Defaults to None."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Position [x, y, z] of the object. Defaults to [0.0, 0.0, 0.0]."""

    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    """Orientation quaternion [w, x, y, z] of the object. Defaults to [1.0, 0.0, 0.0, 0.0]."""

    physics: PhysicsConfig | None = None  # FIX: was Any
    """Physics configuration for the object. Defaults to None."""


@dataclass(frozen=True)
class SceneConfig:
    """Composition of scene assets for the simulator."""

    replicate_physics: bool = True
    """Whether to reuse physics properties across duplicated assets."""

    asset_root: str | None = None
    """Optional root directory for relative asset paths."""

    scene_files: Annotated[list[SceneFileConfig] | None, tyro.conf.Suppress] = None
    """List of scene files (USD/URDF) to load. Set programmatically, not via CLI."""

    rigid_objects: Annotated[list[RigidObjectConfig] | None, tyro.conf.Suppress] = None
    """Standalone rigid objects to instantiate. Set programmatically, not via CLI."""

    env_spacing: float = 20.0
    """Distance between parallel environments in the grid layout."""


@dataclass(frozen=True)
class VirtualGantryCfg:
    """Configuration parameters for virtual gantry system."""

    enabled: bool = False
    """Whether to enable the virtual gantry system."""

    attachment_body_names: list[str] = field(
        default_factory=lambda: ["Trunk", "torso_link", "torso", "base_link", "pelvis", "base"]
    )
    """List of body names to try for attachment (in preference order)."""

    stiffness: float = 200.0
    """Spring stiffness coefficient for elastic band force calculation."""

    damping: float = 100.0
    """Damping coefficient for velocity-based force damping."""

    height: float = 3.0
    """Default height for gantry anchor point in world coordinates."""

    point: list[float] | None = None
    """3D position of gantry anchor point [x, y, z]. If None, defaults to [0, 0, height]."""

    length: float = 0.0
    """Rest length of the elastic band (zero force distance)."""

    apply_force: float = 0.0
    """Additional force magnitude to apply (for manual force adjustment)."""

    apply_force_sign: int = -1
    """Sign multiplier for apply_force direction (-1 or 1)."""


@dataclass(frozen=True)
class BridgeConfig:
    """Configuration for robot SDK bridge integration.

    This configuration matches the parameters used in holosoma_inference's BaseSimulator
    for robot SDK communication and control.
    """

    enabled: bool = False
    """Whether to enable the bridge."""

    # Core bridge settings (from holosoma_inference BaseSimulator)
    use_joystick: bool = False
    """Whether to enable joystick/wireless controller support."""

    joystick_device: int = 0
    """Joystick device ID (Linux only)."""

    joystick_type: str = "xbox"
    """Type of joystick controller."""

    # SDK connection settings
    domain_id: int = 0
    """Domain ID for robot communication."""

    interface: str | None = None
    """Network interface for robot communication. Auto-detected if None."""

    # Rate limiting
    rate_limit_dt: float | None = None
    """Rate limiting timestep. If None, uses simulation timestep."""

    # ROS settings
    use_ros: bool = False
    """Whether to use ROS for communication."""

    # Ground-truth pose publishing for inference logging
    publish_truth_pose: bool = False
    """Publish ground-truth base pose over ZMQ for external velocity estimators."""

    truth_zmq_port: int = 6000
    """ZMQ port used for publishing ground-truth pose."""

    clock_zmq_port: int = 5555
    """ZMQ PUB port for sim clock sync. Must differ for each concurrent simulator process."""

    truth_pose_name: str = "base"
    """Name tag included in published pose messages."""


@dataclass(frozen=True)
class PredictionTwinVizConfig:
    """Configuration for simulator-side transformer prediction twin visualization."""

    enabled: bool = False
    """Whether to enable twin visualization subscriber and rendering."""

    sub_url: str = "tcp://127.0.0.1:6001"
    """ZMQ SUB endpoint used by simulator to receive chunk twin packets."""

    show_obs_prediction: bool = True
    """Whether to render observation-prediction twin trajectory when provided."""

    render_mode: str = "foot_com"
    """Rendering mode: 'skeleton' (full robot ghost) or 'foot_com' (feet + COM trajectories)."""

    max_horizon_frames: int = 10
    """Maximum number of horizon frames rendered per twin trajectory."""

    max_packet_age_sec: float = 0.5
    """Drop packets older than this threshold (seconds) to avoid stale twins."""


@dataclass(frozen=True)
class SimulatorInitConfig:
    """Top-level simulator initialisation configuration."""

    name: str
    """Name of the simulator backend (e.g. ``isaacgym``)."""

    sim: SimEngineConfig
    """Simulation engine configuration settings."""

    debug_viz: bool = True
    """Enable debug visualization (gantry lines, etc.)."""

    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """Interactive viewer camera configuration.

    Configures camera tracking for the interactive viewer with advanced features:
    - Multiple camera modes (Fixed, Spherical, Cartesian)
    - Camera smoothing for stable viewing
    - Robot tracking with configurable body attachment

    Example:
        viewer=ViewerConfig(
            enabled=True,
            camera=SphericalCameraConfig(
                distance=3.0,
                azimuth=45.0,
                elevation=30.0,
            ),
        )
    """

    scene: SceneConfig = field(default_factory=SceneConfig)
    """Scene composition and asset configuration."""

    reset_manager: ResetManagerConfig = field(default_factory=ResetManagerConfig)
    """Reset event manager configuration."""

    contact_sensor_history_length: int = 3
    """Number of frames of contact data retained for sensors."""

    robot_mjcf_filter: MujocoXMLFilterCfg = field(default_factory=MujocoXMLFilterCfg)
    """MuJoCo-specific XML filtering configuration for robot MJCF files."""

    mujoco_backend: MujocoBackend = MujocoBackend.CLASSIC
    """MuJoCo physics backend selection.

    Determines which MuJoCo backend to use for physics simulation:
    - 'classic': CPU-based single environment (backward compatible, default)
    - 'warp': GPU-accelerated multi-environment with mujoco_warp

    This setting only applies when using the MuJoCo simulator (name='mujoco').
    For other simulators (isaacgym, isaacsim), this field is ignored.

    Command line usage:
        --simulator.config.mujoco-backend=warp
        --simulator.config.mujoco-backend=classic

    Or use the syntactic sugar configs:
        simulator:mujoco   (uses classic backend)
        simulator:mjwarp   (uses warp backend)
    """

    mujoco_warp: MujocoWarpConfig = field(default_factory=MujocoWarpConfig)
    """MuJoCo Warp backend memory allocation configuration.

    Controls GPU memory allocation for the Warp backend. Only used when
    mujoco_backend='warp'. Allows tuning contact and constraint capacity
    for different scenarios.

    Command line usage:
        --simulator.config.mujoco-warp.nconmax-per-env=128
        --simulator.config.mujoco-warp.njmax-per-env=1024
    """

    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    """Robot SDK bridge configuration."""

    prediction_twin_viz: PredictionTwinVizConfig = field(default_factory=PredictionTwinVizConfig)
    """Transformer chunk/obs prediction twin visualization settings (MuJoCo run_sim)."""

    virtual_gantry: VirtualGantryCfg = field(default_factory=VirtualGantryCfg)
    """Virtual gantry system configuration."""

    control_zmq_port: int = -1
    """ZMQ REP port for programmatic sim control (gantry, reset, etc.).

    When > 0, the direct-simulation loop binds a ZMQ REP socket on this port
    and accepts JSON commands from an external orchestrator.  Set to -1 (default)
    to disable.

    Command line usage:
        --simulator.config.control-zmq-port=6100
    """

    policy_sync_zmq_port: int = DEFAULT_POLICY_SYNC_PORT
    """ZMQ PAIR port for synchronous policy-simulator stepping (on by default).

    When > 0, the direct-simulation loop binds a loopback ZMQ PAIR socket on
    this port and uses lock-step synchronization with the policy process
    instead of wall-clock rate limiting.  The policy sends its rl_rate in a
    handshake message; the sim then batches ``fps / rl_rate`` physics steps per
    policy step and exchanges DONE / STEP messages, so the number of physics
    steps per policy action is fixed by configuration and does not depend on
    the simulator's achieved throughput.

    The default is derived from the current user id -- see
    ``holosoma/utils/sync_rendezvous.py`` for why it is per-user rather than a
    single fixed number, and for the matching policy-side default.  A simulator
    started while that port is already bound fails immediately with a message
    naming the override flags; it never falls back to a shared port.

    Set to -1 to disable and use wall-clock rate limiting.

    Command line usage:
        --simulator.config.policy-sync-zmq-port=6200
        --simulator.config.policy-sync-zmq-port=-1   # opt out of lock-step
    """

    link_mass_scale: float = 1.0
    """Scale factor for robot link masses in MuJoCo (default: 1.0).
    
    Multiplies all robot body masses by this factor after model compilation.
    Example: 1.5 means all link masses become 1.5x their original values.
    
    This only applies to MuJoCo simulator. For other simulators, this field is ignored.
    
    Command line usage:
        --simulator.config.link-mass-scale=1.5
    """

    link_payloads: dict[str, float] | None = None
    """Optional inline MuJoCo link/body payload masses.

    Maps link/body name to added mass in kilograms. Each entry resolves
    directly to a MuJoCo body and adds the specified payload.

    Command line usage:
        --simulator.config.link-payloads="{'left_wrist_yaw_link': 1.5, 'right_wrist_yaw_link': 1.5}"
    """

    link_payload_config_path: str | None = None
    """Optional JSON file with fixed MuJoCo link/body payload masses.

    The file should contain either:
    - ``{"link_payloads": [{"link": "...", "added_mass": 1.5}, ...]}``
    - or a simple ``{"body_name": 1.5, ...}`` mapping.

    Each payload entry resolves directly to a MuJoCo body and adds
    ``added_mass`` kilograms to that body.

    Command line usage:
        --simulator.config.link-payload-config-path=/abs/path/link_payloads.json
    """

    joint_payloads: dict[str, float] | None = None
    """Legacy inline MuJoCo joint payload masses.

    Maps joint name to added mass in kilograms. Each joint resolves to its
    owning body in the compiled MuJoCo model and adds the specified payload.
    Prefer ``link_payloads`` for new configs.

    Command line usage:
        --simulator.config.joint-payloads="{'left_wrist_yaw_joint': 1.5, 'right_wrist_yaw_joint': 1.5}"
    """

    joint_payload_config_path: str | None = None
    """Legacy JSON file with fixed MuJoCo joint payload masses.

    The file should contain either:
    - ``{"joint_payloads": [{"joint": "...", "added_mass": 1.5}, ...]}``
    - or a simple ``{"joint_name": 1.5, ...}`` mapping.

    Each payload entry resolves the joint to its owning body in the compiled
    MuJoCo model and adds ``added_mass`` kilograms to that body.
    Prefer ``link_payload_config_path`` for new configs.

    Command line usage:
        --simulator.config.joint-payload-config-path=/abs/path/joint_payloads.json
    """

    deploy_ee_force_left: dict[str, float] | None = None
    """Constant external force at the left wrist link in MuJoCo deploy (sim2sim).

    Keys: ``'x'``, ``'y'``, ``'z'`` (Newtons, world frame). Written into MuJoCo's
    ``xfrc_applied`` slot every physics substep. The target body is resolved from
    ``left_wrist_yaw_link`` first and falls back to ``left_hand_link`` for robots
    such as T1-23DoF. Force persists until overwritten. ``None`` disables.

    Pairs with force-adaptive deploy: a trained policy that's force-blind by
    design (privileged-only critic) can be tested for passive adaptation via an
    external EE force. MuJoCo only — IsaacGym/IsaacSim training applies
    per-env external forces via the simulator's own Jacobian /
    rigid-body-force API instead (see ``jacobian`` and
    ``apply_rigid_body_force_at_pos_tensor`` on the simulator classes).

    Command line usage:
        --simulator.config.deploy-ee-force-left="{'x':0,'y':0,'z':-30}"
    """

    deploy_ee_force_right: dict[str, float] | None = None
    """Constant external force at the right wrist link in MuJoCo deploy (sim2sim).

    Same semantics as ``deploy_ee_force_left`` for ``right_wrist_yaw_link`` with
    fallback to ``right_hand_link``.

    Command line usage:
        --simulator.config.deploy-ee-force-right="{'x':0,'y':0,'z':-30}"
    """


@dataclass(frozen=True)
class SimulatorConfig:
    """Wrapper for simulator instantiation."""

    _target_: str
    """Fully-qualified simulator factory target."""

    _recursive_: bool
    """Recursive instantiation flag."""

    config: SimulatorInitConfig
    """Structured simulator configuration passed to the factory."""
