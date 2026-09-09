"""Randomization terms for locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.simulator import MujocoBackend
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.managers.randomization.base import RandomizationTermBase
from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.simulator import mujoco_required_field
from holosoma.simulator.shared.field_decorators import MUJOCO_FIELD_ATTR
from holosoma.utils.torch_utils import torch_rand_float

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg

    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


def _ensure_env_ids_tensor(env: Any, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
    """Convert environment indices to a tensor on the correct device."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _get_joint_action_term(env: Any) -> JointPositionActionTerm | None:
    """Return the joint-position action term registered with the action manager."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None

    get_term = getattr(action_manager, "get_term", None)
    if callable(get_term):
        term = get_term("joint_control")
        if isinstance(term, JointPositionActionTerm):
            return term

    iter_terms = getattr(action_manager, "iter_terms", None)
    if callable(iter_terms):
        for _, term in iter_terms():
            if isinstance(term, JointPositionActionTerm):
                return term

    return None


def _infer_num_bodies(env: Any) -> int:
    """Best-effort body-count inference for oracle randomization buffers."""
    num_bodies = getattr(env, "num_bodies", None)
    if isinstance(num_bodies, int) and num_bodies >= 0:
        return num_bodies

    simulator = getattr(env, "simulator", None)
    body_list = getattr(simulator, "_body_list", None)
    if body_list is not None:
        return len(body_list)

    body_names = getattr(simulator, "body_names", None)
    if body_names is not None:
        return len(body_names)

    return 0


def _ensure_oracle_randomization_buffers(env: Any) -> None:
    """Create oracle randomization caches with stable shapes."""
    num_envs = int(env.num_envs)
    num_bodies = _infer_num_bodies(env)
    device = env.device

    if (
        not hasattr(env, "oracle_body_mass")
        or env.oracle_body_mass.shape[0] != num_envs
        or env.oracle_body_mass.shape[1] != num_bodies
    ):
        env.oracle_body_mass = torch.zeros(num_envs, num_bodies, dtype=torch.float32, device=device)

    if (
        not hasattr(env, "oracle_nominal_body_mass")
        or env.oracle_nominal_body_mass.shape[0] != num_envs
        or env.oracle_nominal_body_mass.shape[1] != num_bodies
    ):
        env.oracle_nominal_body_mass = torch.zeros(num_envs, num_bodies, dtype=torch.float32, device=device)

    if (
        not hasattr(env, "oracle_link_mass_scale")
        or env.oracle_link_mass_scale.shape[0] != num_envs
        or env.oracle_link_mass_scale.shape[1] != num_bodies
    ):
        env.oracle_link_mass_scale = torch.ones(num_envs, num_bodies, dtype=torch.float32, device=device)

    if not hasattr(env, "oracle_base_mass_delta") or env.oracle_base_mass_delta.shape != (num_envs, 1):
        env.oracle_base_mass_delta = torch.zeros(num_envs, 1, dtype=torch.float32, device=device)

    if not hasattr(env, "oracle_friction_coeff") or env.oracle_friction_coeff.shape != (num_envs, 1):
        env.oracle_friction_coeff = torch.ones(num_envs, 1, dtype=torch.float32, device=device)

    if not hasattr(env, "oracle_base_com_bias") or env.oracle_base_com_bias.shape != (num_envs, 3):
        env.oracle_base_com_bias = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        return [part.strip().strip("'\"") for part in stripped.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _resolve_fixed_payloads(body_names: Any, added_masses: Any) -> dict[str, float]:
    names = [str(name) for name in _as_sequence(body_names)]
    masses = [float(mass) for mass in _as_sequence(added_masses)]
    if not names and not masses:
        return {}
    if len(names) != len(masses):
        raise ValueError(
            "fixed_payload_body_names and fixed_payload_added_masses must have the same length; "
            f"got {len(names)} names and {len(masses)} masses."
        )
    return dict(zip(names, masses, strict=True))


def _isaacsim_randomize_rigid_body_mass(
    simulator: IsaacSim,
    env_ids_cpu: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: str,
):
    try:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc
    func = mdp.randomize_rigid_body_mass(
        EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "env_ids": env_ids_cpu,
                "asset_cfg": asset_cfg,
                "mass_distribution_params": mass_distribution_params,
                "operation": operation,
            },
        ),
        env=simulator,
    )
    func(
        simulator,
        env_ids_cpu,
        asset_cfg=asset_cfg,
        mass_distribution_params=mass_distribution_params,
        operation=operation,
    )


def _isaacsim_randomize_rigid_body_material(
    simulator: IsaacSim,
    env_ids_cpu: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
):
    try:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc
    func = mdp.randomize_rigid_body_material(
        EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "env_ids": env_ids_cpu,
                "asset_cfg": asset_cfg,
                "static_friction_range": static_friction_range,
                "dynamic_friction_range": dynamic_friction_range,
                "restitution_range": restitution_range,
                "num_buckets": num_buckets,
            },
        ),
        simulator,
    )
    func(
        simulator,
        env_ids_cpu,
        asset_cfg=asset_cfg,
        static_friction_range=static_friction_range,
        dynamic_friction_range=dynamic_friction_range,
        restitution_range=restitution_range,
        num_buckets=num_buckets,
    )


def _get_isaacsim_root_physx_view(simulator: Any) -> Any | None:
    robot = getattr(simulator, "_robot", None)
    if robot is None:
        return None
    return getattr(robot, "root_physx_view", None)


def _read_isaacsim_body_masses(simulator: Any) -> torch.Tensor | None:
    """Read per-env per-body masses from IsaacSim root PhysX view."""
    root_physx_view = _get_isaacsim_root_physx_view(simulator)
    get_masses = getattr(root_physx_view, "get_masses", None) if root_physx_view is not None else None
    if not callable(get_masses):
        return None

    masses = get_masses()
    if masses is None:
        return None

    masses_tensor = torch.as_tensor(masses, dtype=torch.float32)
    if masses_tensor.ndim == 1:
        masses_tensor = masses_tensor.unsqueeze(0)
    elif masses_tensor.ndim == 3 and masses_tensor.shape[-1] == 1:
        masses_tensor = masses_tensor.squeeze(-1)
    elif masses_tensor.ndim > 2:
        masses_tensor = masses_tensor.reshape(masses_tensor.shape[0], -1)
    return masses_tensor


def _read_isaacsim_static_friction(simulator: Any) -> torch.Tensor | None:
    """Read average static friction per env from IsaacSim root PhysX view."""
    root_physx_view = _get_isaacsim_root_physx_view(simulator)
    get_material_properties = (
        getattr(root_physx_view, "get_material_properties", None) if root_physx_view is not None else None
    )
    if not callable(get_material_properties):
        return None

    material_properties = get_material_properties()
    if material_properties is None:
        return None

    material_tensor = torch.as_tensor(material_properties, dtype=torch.float32)
    if material_tensor.ndim == 3 and material_tensor.shape[-1] >= 1:
        static_friction = material_tensor[..., 0]
    elif material_tensor.ndim == 2 and material_tensor.shape[1] >= 1:
        static_friction = material_tensor
    else:
        return None

    if static_friction.ndim == 1:
        return static_friction.unsqueeze(1)
    return static_friction.reshape(static_friction.shape[0], -1).mean(dim=1, keepdim=True)


def _write_isaacsim_mass_oracle_cache(
    env: Any,
    idx: torch.Tensor,
    *,
    nominal_masses: torch.Tensor | None,
    randomized_masses: torch.Tensor | None,
) -> None:
    """Write mass-related oracle caches from IsaacSim mass tensors."""
    if nominal_masses is None or randomized_masses is None:
        logger.warning("IsaacSim mass oracle write-back skipped because masses could not be queried from PhysX.")
        return

    idx_nominal = idx.to(device=nominal_masses.device, dtype=torch.long)
    idx_randomized = idx.to(device=randomized_masses.device, dtype=torch.long)
    if idx_nominal.numel() == 0 or idx_randomized.numel() == 0:
        return

    if nominal_masses.shape[0] <= int(idx_nominal.max().item()) or randomized_masses.shape[0] <= int(
        idx_randomized.max().item()
    ):
        logger.warning("IsaacSim mass oracle write-back skipped due to unexpected mass tensor shape.")
        return

    nominal_selected = nominal_masses[idx_nominal].to(device=env.device, dtype=torch.float32)
    randomized_selected = randomized_masses[idx_randomized].to(device=env.device, dtype=torch.float32)

    num_bodies = min(env.oracle_body_mass.shape[1], nominal_selected.shape[1], randomized_selected.shape[1])
    if num_bodies <= 0:
        return

    env.oracle_nominal_body_mass[idx, :num_bodies] = nominal_selected[:, :num_bodies]
    env.oracle_body_mass[idx, :num_bodies] = randomized_selected[:, :num_bodies]

    body_names = list(getattr(env, "body_names", []) or [])
    body_index_map = {str(name): i for i, name in enumerate(body_names)}

    link_names = list(getattr(env.robot_config, "randomize_link_body_names", []) or [])
    for body_name in link_names:
        body_idx = body_index_map.get(body_name)
        if body_idx is None or body_idx >= num_bodies:
            continue
        nominal = nominal_selected[:, body_idx]
        randomized = randomized_selected[:, body_idx]
        scale = torch.where(torch.abs(nominal) > 1e-6, randomized / nominal, torch.ones_like(randomized))
        env.oracle_link_mass_scale[idx, body_idx] = scale

    torso_name = str(getattr(env.robot_config, "torso_name", "") or "")
    torso_idx = body_index_map.get(torso_name)
    if torso_idx is not None and torso_idx < num_bodies:
        env.oracle_base_mass_delta[idx, 0] = randomized_selected[:, torso_idx] - nominal_selected[:, torso_idx]


class PushRandomizerState(RandomizationTermBase):
    """Stateful randomizer that owns push scheduling buffers and counters."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        interval = params.get("push_interval_s", [5, 16])
        self.push_interval_range: Sequence[float] = [float(interval[0]), float(interval[1])]
        vector_max = params.get("max_push_vel")
        if vector_max is None:
            raise ValueError("PushRandomizerState requires `max_push_vel` to be specified.")
        self._max_push_vel_tensor = torch.empty(0, dtype=torch.float32, device=env.device)
        self._set_max_push_tensor(vector_max)
        self.enabled: bool = bool(params.get("enabled", True))
        logger.info(
            f"[Randomization] PushRandomizerState initialized (enabled={self.enabled}, \
                max_push_vel={self._max_push_vel_tensor.tolist()}, \
                interval_s={self.push_interval_range})",
        )

        self.push_interval_s: torch.Tensor | None = None
        self.push_robot_counter: torch.Tensor | None = None
        self.push_robot_plot_counter: torch.Tensor | None = None

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs

        self.push_interval_s = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.push_robot_counter = torch.zeros(num_envs, dtype=torch.int, device=device)
        self.push_robot_plot_counter = torch.zeros(num_envs, dtype=torch.int, device=device)

        all_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        self._resample_intervals(all_ids)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        idx = self._ensure_indices(env_ids)
        if idx.numel() == 0:
            return
        self.push_robot_counter[idx] = 0
        self.push_robot_plot_counter[idx] = 0

    def step(self) -> None:
        if not self.enabled:
            return
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        self.push_robot_counter += 1
        self.push_robot_plot_counter += 1

    # ------------------------------------------------------------------ #
    # Public helpers for other randomization hooks
    # ------------------------------------------------------------------ #

    def configure(
        self,
        *,
        enabled: bool | None = None,
        push_interval_s: Sequence[float] | None = None,
        max_push_vel: Sequence[float] | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if push_interval_s is not None:
            self.push_interval_range = [float(push_interval_s[0]), float(push_interval_s[1])]
        if max_push_vel is not None:
            self._set_max_push_tensor(max_push_vel)

    def resample(self, env_ids: torch.Tensor | None = None) -> None:
        idx = self._ensure_indices(env_ids)
        if idx.numel() == 0:
            return
        self._resample_intervals(idx)

    def due_envs(self, dt: float) -> torch.Tensor:
        if not self.enabled:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        if self.push_interval_s is None or self.push_robot_counter is None:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        interval_steps = (self.push_interval_s / dt).to(torch.int)
        return (self.push_robot_counter == interval_steps).nonzero(as_tuple=False).flatten()

    def zero_counters(self, env_ids: torch.Tensor) -> None:
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        self.push_robot_counter[env_ids] = 0
        self.push_robot_plot_counter[env_ids] = 0

    @property
    def max_push_vel(self) -> torch.Tensor:
        return self._max_push_vel_tensor

    def _ensure_indices(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)

    def _resample_intervals(self, env_ids: torch.Tensor) -> None:
        if self.push_interval_s is None:
            return
        low, high = self.push_interval_range
        low_i = max(1, int(low))
        high_i = max(low_i + 1, int(high))
        samples = torch_rand_float(low_i, high_i, (env_ids.shape[0], 1), device=self.env.device).squeeze(1)
        self.push_interval_s[env_ids] = samples

    def _set_max_push_tensor(self, values: Sequence[float]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.env.device).flatten()
        if tensor.numel() == 0:
            raise ValueError("max_push_vel must contain at least one value.")
        self._max_push_vel_tensor = tensor.clone()


class ActuatorRandomizerState(RandomizationTermBase):
    """Stateful actuator randomizer managing PD gain and RFI scales."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}

        kp_range = params.get("kp_range", [1.0, 1.0])
        kd_range = params.get("kd_range", [1.0, 1.0])
        rfi_lim_range = params.get("rfi_lim_range", [1.0, 1.0])

        self.enable_pd_gain = bool(params.get("enable_pd_gain", True))
        self.enable_rfi_lim = bool(params.get("enable_rfi_lim", False))

        self.kp_range: Sequence[float] = [float(kp_range[0]), float(kp_range[1])]
        self.kd_range: Sequence[float] = [float(kd_range[0]), float(kd_range[1])]
        self.rfi_lim_range: Sequence[float] = [float(rfi_lim_range[0]), float(rfi_lim_range[1])]

        self.rfi_lim = float(params.get("rfi_lim", 0.1))

        self.kp_scale: torch.Tensor | None = None
        self.kd_scale: torch.Tensor | None = None
        self.rfi_lim_scale: torch.Tensor | None = None

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs
        num_dof = env.num_dof

        self.kp_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)
        self.kd_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)
        self.rfi_lim_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)

        term = _get_joint_action_term(env)
        if term is not None:
            term.attach_actuator_scales(self.kp_scale, self.kd_scale, self.rfi_lim_scale)
        else:
            logger.debug(
                "JointPositionActionTerm not ready during ActuatorRandomizerState.setup(); "
                "the term will attach shared actuator scales once its setup() runs."
            )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.kp_scale is None or self.kd_scale is None or self.rfi_lim_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() must be called before reset().")

        idx = _ensure_env_ids_tensor(self.env, env_ids)
        if idx.numel() == 0:
            return

        device = self.env.device

        if self.enable_pd_gain:
            self.kp_scale[idx] = torch_rand_float(
                self.kp_range[0], self.kp_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
            self.kd_scale[idx] = torch_rand_float(
                self.kd_range[0], self.kd_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
        else:
            self.kp_scale[idx] = 1.0
            self.kd_scale[idx] = 1.0

        if self.enable_rfi_lim:
            self.rfi_lim_scale[idx] = torch_rand_float(
                self.rfi_lim_range[0], self.rfi_lim_range[1], (idx.shape[0], self.env.num_dof), device=device
            )
        else:
            self.rfi_lim_scale[idx] = 1.0

    def step(self) -> None:
        """No per-step behaviour required."""

    @property
    def kp_scale_tensor(self) -> torch.Tensor:
        if self.kp_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.kp_scale

    @property
    def kd_scale_tensor(self) -> torch.Tensor:
        if self.kd_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.kd_scale

    @property
    def rfi_lim_scale_tensor(self) -> torch.Tensor:
        if self.rfi_lim_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.rfi_lim_scale


def setup_action_delay_buffers(env, *, ctrl_delay_step_range: Sequence[int], enabled: bool = True, **_) -> None:
    """Initialize action delay index buffer during setup.

    Note: The action_queue itself is managed by the action manager.
    This only sets up the delay index that determines which queued action to use.
    """
    env._randomize_ctrl_delay = bool(enabled)
    env._ctrl_delay_step_range = list(ctrl_delay_step_range)

    if not enabled:
        return

    # Initialize action delay indices (determines which action from the queue to use)
    env.action_delay_idx = torch.randint(
        ctrl_delay_step_range[0],
        ctrl_delay_step_range[1] + 1,
        (env.num_envs,),
        device=env.device,
        requires_grad=False,
    )


def setup_torque_rfi(env, *, enabled: bool = False, rfi_lim: float = 0.1, **_) -> None:
    """Configure torque RFI at startup."""
    term = _get_joint_action_term(env)
    env._pending_torque_rfi = (bool(enabled), float(rfi_lim))
    if term is None:
        return
    term.configure_torque_rfi(enabled=env._pending_torque_rfi[0], rfi_lim=env._pending_torque_rfi[1])


def setup_dof_pos_bias(env, *, dof_pos_bias_range: Sequence[float], enabled: bool = False, **_) -> None:
    """Apply startup DOF position bias randomization."""
    env._randomize_dof_pos_bias = bool(enabled)
    env._dof_pos_bias_range = list(dof_pos_bias_range)

    if not enabled:
        return

    default_dof_pos_bias = torch_rand_float(
        dof_pos_bias_range[0],
        dof_pos_bias_range[1],
        (env.num_envs, env.num_dof),
        device=env.device,
    )
    env.default_dof_pos = env.default_dof_pos_base + default_dof_pos_bias


def randomize_push_schedule(
    env,
    env_ids,
    *,
    push_interval_s: Sequence[float] | None = None,
    enabled: bool | None = None,
    max_push_vel: Sequence[float] | None = None,
    **_,
) -> None:
    """Resample push intervals for selected environments."""
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("PushRandomizerState is not registered with the randomization manager.")

    state.configure(enabled=enabled, push_interval_s=push_interval_s, max_push_vel=max_push_vel)
    env._randomize_push_robots = state.enabled
    env._max_push_vel = state.max_push_vel.clone()

    if not state.enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    state.zero_counters(idx)
    state.resample(idx)


def randomize_pd_gains(
    env, env_ids, *, kp_range: Sequence[float], kd_range: Sequence[float], enabled: bool = True, **_
):
    """Randomize proportional and derivative gain scales."""
    state = env.randomization_manager.get_state("actuator_randomizer_state")
    term = _get_joint_action_term(env)
    if state is None:
        if term is None:
            logger.warning("JointPositionActionTerm not found; PD gain randomization skipped.")
            return

        idx = _ensure_env_ids_tensor(env, env_ids)
        if idx.numel() == 0:
            return

        if not enabled:
            kp_scale, kd_scale = term.get_pd_scale_tensors()
            term.update_pd_scales(idx, torch.ones_like(kp_scale[idx]), torch.ones_like(kd_scale[idx]))
            return

        kp_samples = torch_rand_float(kp_range[0], kp_range[1], (idx.shape[0], env.num_dof), device=env.device)
        kd_samples = torch_rand_float(kd_range[0], kd_range[1], (idx.shape[0], env.num_dof), device=env.device)
        term.update_pd_scales(idx, kp_samples, kd_samples)
        return

    state.enable_pd_gain = bool(enabled)
    state.kp_range = [float(kp_range[0]), float(kp_range[1])]
    state.kd_range = [float(kd_range[0]), float(kd_range[1])]
    state.reset(env_ids)


def randomize_rfi_limits(
    env,
    env_ids,
    *,
    rfi_lim_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize residual force injection limits."""
    state = env.randomization_manager.get_state("actuator_randomizer_state")
    term = _get_joint_action_term(env)
    if state is None:
        if term is None:
            logger.warning("JointPositionActionTerm not found; RFI randomization skipped.")
            return

        idx = _ensure_env_ids_tensor(env, env_ids)
        if idx.numel() == 0:
            return

        if not enabled:
            term.update_rfi_scales(idx, torch.ones_like(term.get_rfi_scale_tensor()[idx]))
            return

        rfi_samples = torch_rand_float(
            rfi_lim_range[0], rfi_lim_range[1], (idx.shape[0], env.num_dof), device=env.device
        )
        term.update_rfi_scales(idx, rfi_samples)
        return

    state.enable_rfi_lim = bool(enabled)
    state.rfi_lim_range = [float(rfi_lim_range[0]), float(rfi_lim_range[1])]
    state.reset(env_ids)


def randomize_action_delay(
    env,
    env_ids,
    *,
    ctrl_delay_step_range: Sequence[int] | None = None,
    enabled: bool | None = None,
    **_,
) -> None:
    """Randomize control delay indices.

    If ``ctrl_delay_step_range``/``enabled`` are omitted the values captured during
    ``setup_action_delay_buffers`` are reused.
    """
    if enabled is not None:
        env._randomize_ctrl_delay = bool(enabled)
    elif not hasattr(env, "_randomize_ctrl_delay"):
        raise AttributeError(
            "randomize_action_delay() requires setup_action_delay_buffers to run before it can infer 'enabled'."
        )

    if ctrl_delay_step_range is not None:
        env._ctrl_delay_step_range = list(ctrl_delay_step_range)
    elif not hasattr(env, "_ctrl_delay_step_range"):
        raise AttributeError(
            "randomize_action_delay() requires setup_action_delay_buffers \
                to run before it can infer ctrl_delay_step_range."
        )

    if not env._randomize_ctrl_delay:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    # Reset action queue in the action manager
    if hasattr(env.action_manager, "action_queue"):
        env.action_manager.action_queue[idx] *= 0.0

    delay_low = int(env._ctrl_delay_step_range[0])
    delay_high = int(env._ctrl_delay_step_range[1])
    if delay_high < delay_low:
        raise ValueError("ctrl_delay_step_range upper bound must be >= lower bound.")

    # Randomize delay indices
    env.action_delay_idx[idx] = torch.randint(
        delay_low,
        delay_high + 1,
        (idx.shape[0],),
        device=env.device,
        requires_grad=False,
    )


def randomize_dof_state(
    env,
    env_ids,
    *,
    joint_pos_scale_range: Sequence[float],
    joint_pos_bias_range: Sequence[float],
    joint_vel_range: Sequence[float],
    randomize_dof_pos_bias: bool = False,
    **_,
) -> None:
    """Randomize DOF positions and velocities."""
    env._joint_pos_scale_range = list(joint_pos_scale_range)
    env._joint_pos_bias_range = list(joint_pos_bias_range)
    env._joint_vel_range = list(joint_vel_range)
    env._randomize_dof_pos_bias = bool(randomize_dof_pos_bias)

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    scale_factor = torch_rand_float(
        joint_pos_scale_range[0],
        joint_pos_scale_range[1],
        (idx.shape[0], env.num_dof),
        device=env.device,
    )
    if randomize_dof_pos_bias:
        bias_offset = torch_rand_float(
            joint_pos_bias_range[0],
            joint_pos_bias_range[1],
            (idx.shape[0], env.num_dof),
            device=env.device,
        )
    else:
        bias_offset = torch.zeros((idx.shape[0], env.num_dof), device=env.device)

    env.simulator.dof_pos[idx] = env.default_dof_pos[idx] * scale_factor + bias_offset
    env.simulator.dof_vel[idx] = torch_rand_float(
        joint_vel_range[0],
        joint_vel_range[1],
        (idx.shape[0], env.num_dof),
        device=env.device,
    )


@mujoco_required_field("body_ipos")
def randomize_base_com_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    base_com_range: dict[str, Sequence[float]],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize base (torso) center of mass.

    Note: Uses ADDITION operation to offset CoM position (e.g., x: [-0.01, 0.01] m).
    """
    env._randomize_base_com = bool(enabled)
    env._base_com_range = base_com_range
    _ensure_oracle_randomization_buffers(env)
    env.oracle_base_com_bias[:] = 0.0
    if not enabled:
        return

    logger.info(
        f"[Randomization] Base CoM: "
        f"x={base_com_range.get('x', [0, 0])}, "
        f"y={base_com_range.get('y', [0, 0])}, "
        f"z={base_com_range.get('z', [0, 0])} (operation=add)"
    )

    simulator = env.simulator

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return
    env.oracle_base_com_bias[idx] = 0.0

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        torso_name = env.robot_config.torso_name
        if not hasattr(simulator, "_base_com_bias"):
            simulator._base_com_bias = torch.zeros(
                env.num_envs, 3, dtype=torch.float, device=env.device, requires_grad=False
            )

        for env_id in idx.tolist():
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            body_index = gym.find_actor_rigid_body_handle(env_ptr, actor, torso_name)
            if body_index < 0:
                raise RuntimeError(f"Body '{torso_name}' not found when randomizing base COM.")

            xrange = base_com_range["x"]
            yrange = base_com_range["y"]
            zrange = base_com_range["z"]

            bias = torch.tensor(
                [
                    torch_rand_float(xrange[0], xrange[1], (1, 1), device=env.device).item(),
                    torch_rand_float(yrange[0], yrange[1], (1, 1), device=env.device).item(),
                    torch_rand_float(zrange[0], zrange[1], (1, 1), device=env.device).item(),
                ],
                dtype=torch.float,
                device=env.device,
            )
            simulator._base_com_bias[env_id] = bias
            env.oracle_base_com_bias[env_id] = bias
            body_props[body_index].com.x += bias[0].item()
            body_props[body_index].com.y += bias[1].item()
            body_props[body_index].com.z += bias[2].item()
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=True)
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("IsaacSim base COM randomization requires isaaclab.") from exc
        from holosoma.simulator.isaacsim.events import randomize_body_com

        torso_name = env.robot_config.torso_name
        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        low = torch.tensor(
            [base_com_range["x"][0], base_com_range["y"][0], base_com_range["z"][0]],
            dtype=torch.float,
            device="cpu",
        )
        high = torch.tensor(
            [base_com_range["x"][1], base_com_range["y"][1], base_com_range["z"][1]],
            dtype=torch.float,
            device="cpu",
        )
        asset_cfg = SceneEntityCfg("robot", body_names=[torso_name])
        asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
        randomize_body_com(
            simulator,
            env_ids_cpu,
            asset_cfg,
            (low, high),
            operation="add",
            distribution="uniform",
            num_envs=simulator.training_config.num_envs,
        )
        base_com_bias = getattr(simulator, "base_com_bias", None)
        if base_com_bias is None:
            base_com_bias = getattr(simulator, "_base_com_bias", None)
        if isinstance(base_com_bias, torch.Tensor):
            idx_bias = idx.to(device=base_com_bias.device, dtype=torch.long)
            env.oracle_base_com_bias[idx] = base_com_bias[idx_bias].to(device=env.device, dtype=torch.float32)
        else:
            logger.warning("IsaacSim base COM oracle write-back skipped because base_com_bias buffer was not found.")
    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        # convert xyz to 012
        base_com_range_remapped = {}
        for key, value in base_com_range.items():
            assert len(value) == 2, f"Range for '{key}' must have exactly 2 elements, got {len(value)}"
            base_com_range_remapped["xyz".index(key)] = (value[0], value[1])
        randomize_field(
            simulator,
            field=getattr(randomize_base_com_startup, MUJOCO_FIELD_ATTR),
            ranges=base_com_range_remapped,
            env_ids=idx,
            entity_names=[env.robot_config.torso_name],
            entity_type="body",
            operation="add",
            distribution="uniform",
        )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for base COM randomization."
        )


@mujoco_required_field("body_mass")
def randomize_mass_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    enable_link_mass: bool = True,
    link_mass_range: Sequence[float] = (1.0, 1.0),
    enable_base_mass: bool = True,
    added_mass_range: Sequence[float] = (0.0, 0.0),
    fixed_payload_body_names: Sequence[str] | str | None = None,
    fixed_payload_added_masses: Sequence[float] | str | None = None,
    replace_fixed_payload_dr: bool = True,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize link and base masses at startup.

    Note: link_mass_range uses SCALING (e.g., 0.9-1.2 = 90-120% of original),
          added_mass_range uses ADDITION (e.g., -1.0 to 3.0 kg offset).
    """
    _ensure_oracle_randomization_buffers(env)
    fixed_payloads = _resolve_fixed_payloads(fixed_payload_body_names, fixed_payload_added_masses)
    fixed_payload_names = set(fixed_payloads)
    replace_fixed_payload_dr = bool(replace_fixed_payload_dr)
    torso_name_for_flags = str(getattr(env.robot_config, "torso_name", "") or "")
    link_names_for_flags = list(getattr(env.robot_config, "randomize_link_body_names", []) or [])
    if replace_fixed_payload_dr:
        link_names_for_flags = [name for name in link_names_for_flags if name not in fixed_payload_names]
    effective_link_mass = bool(enable_link_mass and link_names_for_flags)
    effective_base_mass = bool(
        enable_base_mass and not (replace_fixed_payload_dr and torso_name_for_flags in fixed_payload_names)
    )

    env.oracle_link_mass_scale[:] = 1.0
    env.oracle_base_mass_delta[:] = 0.0
    env.oracle_body_mass[:] = 0.0
    env.oracle_nominal_body_mass[:] = 0.0

    env._randomize_link_mass = bool(enabled and effective_link_mass)
    env._randomize_base_mass = bool(enabled and effective_base_mass)

    simulator = env.simulator
    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    if not enabled:
        if hasattr(simulator, "gym"):
            gym = simulator.gym
            for env_id in idx.tolist():
                env_ptr = simulator.envs[env_id]
                actor = simulator.robot_handles[env_id]
                body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
                masses = np.asarray([float(prop.mass) for prop in body_props], dtype=np.float32)
                env.oracle_body_mass[env_id, : masses.shape[0]] = torch.from_numpy(masses).to(device=env.device)
                env.oracle_nominal_body_mass[env_id, : masses.shape[0]] = torch.from_numpy(masses).to(
                    device=env.device
                )
        elif simulator.__class__.__name__ == "IsaacSim":
            masses = _read_isaacsim_body_masses(simulator)
            _write_isaacsim_mass_oracle_cache(env, idx, nominal_masses=masses, randomized_masses=masses)
        return

    logger.info(
        f"[Randomization] Mass: "
        f"link_mass={link_mass_range} (operation=scale, enabled={enable_link_mass}), "
        f"base_mass={added_mass_range} (operation=add, enabled={enable_base_mass}), "
        f"fixed_payloads={fixed_payloads} (replace_dr={replace_fixed_payload_dr})"
    )

    nominal_masses_isaacsim = None
    if simulator.__class__.__name__ == "IsaacSim":
        nominal_masses_isaacsim = _read_isaacsim_body_masses(simulator)

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        body_names = list(env.robot_config.randomize_link_body_names or [])
        torso_name = env.robot_config.torso_name
        if replace_fixed_payload_dr:
            body_names = [name for name in body_names if name not in fixed_payload_names]
        apply_base_mass = bool(enable_base_mass and not (replace_fixed_payload_dr and torso_name in fixed_payload_names))
        body_index_map = {name: i for i, name in enumerate(simulator._body_list)}
        if idx.numel() > 0:
            sample_env = idx[0].item()
            sample_env_ptr = simulator.envs[sample_env]
            sample_actor = simulator.robot_handles[sample_env]
            sample_props = gym.get_actor_rigid_body_properties(sample_env_ptr, sample_actor)
            if enable_link_mass and body_names:
                link_masses = [
                    float(sample_props[body_index_map[name]].mass)
                    for name in body_names
                    if name in body_index_map
                ]
                if link_masses:
                    logger.debug(
                        "[randomize_mass_startup][IsaacGym] default link mass range: "
                        f"min={min(link_masses):.6f}, max={max(link_masses):.6f}"
                    )
            if apply_base_mass and torso_name in body_index_map:
                base_mass = float(sample_props[body_index_map[torso_name]].mass)
                logger.debug(f"[randomize_mass_startup][IsaacGym] default torso mass: {base_mass:.6f}")
        for env_id in idx.tolist():
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            nominal_masses = np.asarray([float(prop.mass) for prop in body_props], dtype=np.float32)
            link_mass_scales = np.ones(len(body_props), dtype=np.float32)
            if enable_link_mass and body_names:
                for body_name in body_names:
                    if body_name not in body_index_map:
                        continue
                    body_index = body_index_map[body_name]
                    scale = np.random.uniform(link_mass_range[0], link_mass_range[1])
                    body_props[body_index].mass *= scale  # Scale operation: multiply by factor
                    link_mass_scales[body_index] = float(scale)

            base_mass_delta = 0.0
            if apply_base_mass and torso_name in body_index_map:
                base_index = body_index_map[torso_name]
                delta = np.random.uniform(added_mass_range[0], added_mass_range[1])
                body_props[base_index].mass += delta  # Add operation: offset by delta
                base_mass_delta = float(delta)
            for body_name, added_mass in fixed_payloads.items():
                body_index = body_index_map.get(body_name)
                if body_index is None:
                    raise RuntimeError(f"Body '{body_name}' not found when applying fixed payload mass.")
                body_props[body_index].mass += float(added_mass)
                if body_name == torso_name:
                    base_mass_delta += float(added_mass)
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=True)

            randomized_masses = np.asarray([float(prop.mass) for prop in body_props], dtype=np.float32)
            env.oracle_nominal_body_mass[env_id, : nominal_masses.shape[0]] = torch.from_numpy(nominal_masses).to(
                device=env.device
            )
            env.oracle_body_mass[env_id, : randomized_masses.shape[0]] = torch.from_numpy(randomized_masses).to(
                device=env.device
            )
            env.oracle_link_mass_scale[env_id, : link_mass_scales.shape[0]] = torch.from_numpy(link_mass_scales).to(
                device=env.device
            )
            env.oracle_base_mass_delta[env_id, 0] = base_mass_delta
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc

        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        link_body_names = list(env.robot_config.randomize_link_body_names or [])
        if replace_fixed_payload_dr:
            link_body_names = [name for name in link_body_names if name not in fixed_payload_names]
        apply_base_mass = bool(
            enable_base_mass and not (replace_fixed_payload_dr and env.robot_config.torso_name in fixed_payload_names)
        )

        if enable_link_mass:
            if link_body_names:
                asset_cfg = SceneEntityCfg("robot", body_names=link_body_names)
                asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
                _isaacsim_randomize_rigid_body_mass(
                    simulator,
                    env_ids_cpu,
                    asset_cfg,
                    (link_mass_range[0], link_mass_range[1]),
                    operation="scale",
                )

        if apply_base_mass:
            asset_cfg = SceneEntityCfg("robot", body_names=[env.robot_config.torso_name])
            asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
            _isaacsim_randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                (added_mass_range[0], added_mass_range[1]),
                operation="add",
            )
        for body_name, added_mass in fixed_payloads.items():
            asset_cfg = SceneEntityCfg("robot", body_names=[body_name])
            asset_cfg.resolve(simulator.scene)
            _isaacsim_randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                (float(added_mass), float(added_mass)),
                operation="add",
            )
        randomized_masses_isaacsim = _read_isaacsim_body_masses(simulator)
        _write_isaacsim_mass_oracle_cache(
            env,
            idx,
            nominal_masses=nominal_masses_isaacsim,
            randomized_masses=randomized_masses_isaacsim,
        )
    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        # randomize over the range (scale and/or shift)
        if idx.numel() == 0:
            return

        link_body_names = list(env.robot_config.randomize_link_body_names or [])
        if replace_fixed_payload_dr:
            link_body_names = [name for name in link_body_names if name not in fixed_payload_names]
        apply_base_mass = bool(
            enable_base_mass and not (replace_fixed_payload_dr and env.robot_config.torso_name in fixed_payload_names)
        )

        if enable_link_mass:
            assert len(link_mass_range) == 2, (
                f"link_mass_range must have exactly 2 elements, got {len(link_mass_range)}"
            )
            if link_body_names:
                randomize_field(
                    simulator,
                    field=getattr(randomize_mass_startup, MUJOCO_FIELD_ATTR),
                    ranges=(link_mass_range[0], link_mass_range[1]),
                    env_ids=idx,
                    entity_names=link_body_names,
                    entity_type="body",
                    operation="scale",
                )

        if apply_base_mass:
            assert len(added_mass_range) == 2, (
                f"added_mass_range must have exactly 2 elements, got {len(added_mass_range)}"
            )
            randomize_field(
                simulator,
                field=getattr(randomize_mass_startup, MUJOCO_FIELD_ATTR),
                ranges=(added_mass_range[0], added_mass_range[1]),
                env_ids=idx,
                entity_names=[env.robot_config.torso_name],
                entity_type="body",
                operation="add",
            )
        for body_name, added_mass in fixed_payloads.items():
            randomize_field(
                simulator,
                field=getattr(randomize_mass_startup, MUJOCO_FIELD_ATTR),
                ranges=(float(added_mass), float(added_mass)),
                env_ids=idx,
                entity_names=[body_name],
                entity_type="body",
                operation="add",
            )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Mass randomization not supported for simulator type '{type(simulator).__name__}'."
        )


@mujoco_required_field("geom_friction")
def randomize_friction_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    friction_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize contact friction coefficients for robot rigid shapes.

    Note: Uses ABSOLUTE operation to set friction values (e.g., [0.5, 1.5]).
    """
    env._randomize_friction = bool(enabled)
    env._friction_range = list(friction_range)
    _ensure_oracle_randomization_buffers(env)
    env.oracle_friction_coeff[:] = 1.0
    if not enabled:
        return

    logger.info(f"[Randomization] Friction: range={friction_range} (operation=abs)")

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator

    num_buckets = 64
    buckets = torch_rand_float(
        friction_range[0],
        friction_range[1],
        (num_buckets, 1),
        device="cpu",
    )

    idx_cpu = idx.to(device="cpu", dtype=torch.long)
    bucket_ids = torch.randint(0, num_buckets, (idx_cpu.shape[0],), device="cpu")
    friction_samples_cpu = buckets[bucket_ids]

    if hasattr(simulator, "gym"):
        gym = simulator.gym
        for offset, env_id in enumerate(idx_cpu.tolist()):
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor)
            friction_value = friction_samples_cpu[offset].item()
            for prop in shape_props:
                prop.friction = friction_value
            gym.set_actor_rigid_shape_properties(env_ptr, actor, shape_props)
            env.oracle_friction_coeff[env_id, 0] = float(friction_value)
    elif simulator.__class__.__name__ == "IsaacSim":
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim friction randomization requires isaaclab.") from exc
        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        asset_cfg = SceneEntityCfg("robot", body_names=".*")
        asset_cfg.resolve(simulator.scene)  # Not stricly required, but a good practice

        _isaacsim_randomize_rigid_body_material(
            simulator,
            env_ids_cpu,
            asset_cfg,
            static_friction_range=(friction_range[0], friction_range[1]),
            dynamic_friction_range=(friction_range[0], friction_range[1]),
            restitution_range=(0.0, 0.0),
            num_buckets=num_buckets,
        )
        friction_obs = _read_isaacsim_static_friction(simulator)
        if friction_obs is None:
            logger.warning("IsaacSim friction oracle write-back skipped because material properties were unavailable.")
        else:
            idx_friction = idx.to(device=friction_obs.device, dtype=torch.long)
            if friction_obs.shape[0] > int(idx_friction.max().item()):
                env.oracle_friction_coeff[idx, 0] = friction_obs[idx_friction, 0].to(device=env.device)
            else:
                logger.warning("IsaacSim friction oracle write-back skipped due to unexpected friction tensor shape.")

    elif simulator.simulator_config.mujoco_backend == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        assert len(friction_range) == 2, f"friction_range must have exactly 2 elements, got {len(friction_range)}"
        randomize_field(
            simulator,
            field=getattr(randomize_friction_startup, MUJOCO_FIELD_ATTR),
            ranges={0: (friction_range[0], friction_range[1])},
            env_ids=idx,
            operation="abs",
        )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for friction randomization."
        )


def randomize_robot_rigid_body_material_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    static_friction_range: Sequence[float],
    dynamic_friction_range: Sequence[float],
    restitution_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize robot rigid body material properties (friction, restitution)."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_robot_rigid_body_material_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("robot", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    num_buckets = 64
    _isaacsim_randomize_rigid_body_material(
        simulator,
        env_ids_cpu,
        asset_cfg,
        static_friction_range=(static_friction_range[0], static_friction_range[1]),
        dynamic_friction_range=(dynamic_friction_range[0], dynamic_friction_range[1]),
        restitution_range=(restitution_range[0], restitution_range[1]),
        num_buckets=num_buckets,
    )


def randomize_object_rigid_body_material_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    static_friction_range: Sequence[float],
    dynamic_friction_range: Sequence[float],
    restitution_range: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body material properties (friction, restitution)."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_material_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("object", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    num_buckets = 64
    _isaacsim_randomize_rigid_body_material(
        simulator,
        env_ids_cpu,
        asset_cfg,
        static_friction_range=(static_friction_range[0], static_friction_range[1]),
        dynamic_friction_range=(dynamic_friction_range[0], dynamic_friction_range[1]),
        restitution_range=(restitution_range[0], restitution_range[1]),
        num_buckets=num_buckets,
    )


def randomize_object_rigid_body_mass_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    mass_distribution_params: Sequence[float],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body mass."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_mass_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg

    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("object", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    _isaacsim_randomize_rigid_body_mass(
        simulator,
        env_ids_cpu,
        asset_cfg,
        (mass_distribution_params[0], mass_distribution_params[1]),
        operation="add",
    )


def randomize_object_rigid_body_inertia_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    inertia_distribution_params_dict: dict[str, tuple[float, float]],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize object rigid body inertia."""
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.__class__.__name__ != "IsaacSim":
        raise RandomizerNotSupportedError(
            f"randomize_object_rigid_body_inertia_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim inertia randomization requires isaaclab.") from exc

    from holosoma.simulator.isaacsim.events import randomize_rigid_body_inertia

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    asset_cfg = SceneEntityCfg("object", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    ordering = ["Ixx", "Iyy", "Izz", "Ixy", "Iyz", "Ixz"]
    lower_bounds = [inertia_distribution_params_dict[key][0] for key in ordering]
    upper_bounds = [inertia_distribution_params_dict[key][1] for key in ordering]
    inertia_distribution_params = (torch.tensor(lower_bounds, device="cpu"), torch.tensor(upper_bounds, device="cpu"))

    randomize_rigid_body_inertia(
        simulator,
        env_ids_cpu,
        asset_cfg,
        inertia_distribution_params,
        operation="scale",
        distribution="uniform",
    )


def configure_torque_rfi(
    env,
    env_ids,
    *,
    enabled: bool | None = None,
    rfi_lim: float | None = None,
    **_,
) -> None:
    """Toggle torque RFI injection flag."""
    prev_enabled, prev_lim = env._pending_torque_rfi
    enabled_flag = prev_enabled if enabled is None else bool(enabled)
    rfi_limit = prev_lim if rfi_lim is None else float(rfi_lim)
    env._pending_torque_rfi = (enabled_flag, rfi_limit)

    state = env.randomization_manager.get_state("actuator_randomizer_state")
    if state is not None:
        state.enable_rfi_lim = enabled_flag
    term = _get_joint_action_term(env)
    if term is not None:
        term.configure_torque_rfi(enabled=enabled_flag, rfi_lim=rfi_limit)


def apply_pushes(
    env,
    *,
    enabled: bool | None = None,
    push_interval_s: Sequence[float] | None = None,
    max_push_vel: Sequence[float] | None = None,
    **_,
) -> None:
    """Apply random pushes based on the current schedule."""
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("PushRandomizerState is not registered with the randomization manager.")

    state.configure(enabled=enabled, push_interval_s=push_interval_s, max_push_vel=max_push_vel)
    env._push_robots_enabled = state.enabled

    if env.is_evaluating or not state.enabled:
        return

    push_robot_env_ids = state.due_envs(env.dt)
    if push_robot_env_ids.numel() == 0:
        return

    state.zero_counters(push_robot_env_ids)
    state.resample(push_robot_env_ids)
    env._max_push_vel = state.max_push_vel.clone()
    env._push_robots(push_robot_env_ids)
