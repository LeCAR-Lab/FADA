from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float


class LeggedRobotLocomotionManager(BaseTask):
    BASE_NUM_ENVS = 4096

    def __init__(
        self,
        tyro_config,
        *,
        device,
    ):
        self.init_done = False
        super().__init__(
            tyro_config,
            device=device,
        )
        self.init_done = True

    def _get_task_name(self) -> str:
        return "locomotion"

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        self.base_quat = self.simulator.base_quat

        # initialize some data used later on
        self._init_counters()
        # joint positions offsets
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self._init_domain_rand_buffers()

        self.lidar_height_offset = getattr(self.robot_config, "lidar_height_offset", 0.5)

        # Foot-mesh footprint sampler. No-op when the active reward config has no
        # ``penalty_foothold`` term; only then do we import trimesh/shapely/scipy.
        self._init_foothold_sampling()

    def _foothold_params_from_reward_cfg(self) -> dict:
        """Return ``penalty_foothold`` term params (grid counts + epsilon), or empty dict."""
        cfg = getattr(self.reward_manager, "cfg", None)
        terms = getattr(cfg, "terms", None) if cfg is not None else None
        if not terms or "penalty_foothold" not in terms:
            return {}
        return dict(terms["penalty_foothold"].params)

    def _init_foothold_sampling(self) -> None:
        """Sole sample grids in each foot frame (mesh footprint sampler).

        Skipped when no ``penalty_foothold`` term is configured, so vanilla locomotion
        presets (and robots without ankle STLs in the expected location) stay unaffected.
        """
        p = self._foothold_params_from_reward_cfg()
        if not p:
            return

        import trimesh
        from scipy.spatial import ConvexHull
        from shapely.geometry import LineString, Polygon

        self._foothold_LineString = LineString
        self._foothold_Polygon = Polygon
        self._foothold_ConvexHull = ConvexHull
        self._foothold_trimesh = trimesh

        n_long = max(int(p.get("foothold_num_samples_long", 10)), 2)
        n_wide = max(int(p.get("foothold_num_samples_wide", 5)), 2)

        mesh_dir = self._resolve_foot_mesh_dir()
        left_mesh = self._find_stl(mesh_dir, self.robot_config.asset.left_foot_mesh)
        right_mesh = self._find_stl(mesh_dir, self.robot_config.asset.right_foot_mesh)
        if left_mesh is None or right_mesh is None:
            raise FileNotFoundError(
                f"Could not find foot STL meshes in {mesh_dir}. "
                f"Expected stems: {self.robot_config.asset.left_foot_mesh}, "
                f"{self.robot_config.asset.right_foot_mesh}"
            )

        left_xy = self._footprint_sampler_grid(left_mesh, n_long, n_wide)
        right_xy = self._footprint_sampler_grid(right_mesh, n_long, n_wide)
        num_samples = min(left_xy.shape[0], right_xy.shape[0])

        left_local = torch.as_tensor(left_xy[:num_samples], device=self.device, dtype=torch.float32)
        right_local = torch.as_tensor(right_xy[:num_samples], device=self.device, dtype=torch.float32)
        left_local = torch.cat([left_local, torch.zeros(num_samples, 1, device=self.device)], dim=1)
        right_local = torch.cat([right_local, torch.zeros(num_samples, 1, device=self.device)], dim=1)
        self.foot_samples_local = torch.stack([left_local, right_local], dim=0)
        self.num_foot_samples = num_samples

    def _resolve_foot_mesh_dir(self) -> Path:
        mesh_root = self.robot_config.asset.mesh_root
        if mesh_root is not None:
            return Path(mesh_root)
        asset_root = self.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root(), 1)
        robot_subdir = self.robot_config.asset.robot_type.split("_")[0]
        return Path(asset_root) / robot_subdir / "meshes"

    def _find_stl(self, mesh_dir: Path, stem: str) -> Path | None:
        for suffix in (".STL", ".stl"):
            p = mesh_dir / f"{stem}{suffix}"
            if p.is_file():
                return p
        return None

    def _footprint_sampler_grid(self, mesh_path: Path, num_x_points: int, num_y_points: int) -> np.ndarray:
        """``footprint_sampler_grid``: convex hull in XY + in-polygon grid sampling."""
        foot_mesh = self._foothold_trimesh.load_mesh(str(mesh_path))
        points_2d = np.asarray(foot_mesh.vertices[:, :2], dtype=np.float64)
        hull = self._foothold_ConvexHull(points_2d)
        footprint_polygon = self._foothold_Polygon(points_2d[hull.vertices])

        min_x, min_y, max_x, max_y = footprint_polygon.bounds
        sampled_points: list[tuple[float, float]] = []

        x_cell_width = (max_x - min_x) / (num_x_points * 2)
        x_coords = np.linspace(min_x + x_cell_width, max_x - x_cell_width, num_x_points)
        for x in x_coords:
            vertical_line = self._foothold_LineString([(x, min_y - 1.0), (x, max_y + 1.0)])
            intersection = footprint_polygon.intersection(vertical_line)
            if intersection.is_empty:
                continue
            slice_min_y, slice_max_y = intersection.bounds[1], intersection.bounds[3]
            y_cell_width = (slice_max_y - slice_min_y) / (num_y_points * 2)
            y_coords = np.linspace(slice_min_y + y_cell_width, slice_max_y - y_cell_width, num_y_points)
            sampled_points.extend((float(x), float(y)) for y in y_coords)

        if not sampled_points:
            raise RuntimeError(f"footprint_sampler_grid produced no points for {mesh_path}")
        return np.asarray(sampled_points, dtype=np.float32)

    def _get_terrain_heights_at_points_world(self, pts_world: torch.Tensor) -> torch.Tensor:
        """Terrain surface Z at each world XY (``pts_world`` shape ``[E, S, 3]``)."""
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        if terrain_state is None or not hasattr(terrain_state, "query_terrain_heights"):
            raise RuntimeError(
                "penalty_foothold requires terrain state 'locomotion_terrain' with query_terrain_heights."
            )
        num_envs, num_pts, _ = pts_world.shape
        xy = pts_world[..., :2].reshape(num_envs * num_pts, 2)
        h = terrain_state.query_terrain_heights(xy)
        return h.view(num_envs, num_pts)

    def _init_counters(self):
        self.common_step_counter = 0

    def _update_counters_each_step(self):
        self.common_step_counter += 1

    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(2, dtype=torch.float32, device=self.device)

    def _setup_robot_body_indices(self):
        foot_body_names = [s for s in self.body_names if self.robot_config.foot_body_name in s]
        foot_height_names = [s for s in self.body_names if self.robot_config.foot_height_name in s]

        termination_contact_names = []
        for name in self.robot_config.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.feet_indices = torch.zeros(len(foot_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(foot_body_names):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.feet_height_indices = torch.zeros(
            len(foot_height_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(foot_height_names):
            self.feet_height_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.simulator.find_rigid_body_indice(termination_contact_names[i])

        if self.robot_config.has_torso:
            self.torso_name = self.robot_config.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

    def set_is_evaluating(self, command=None):
        logger.info("Setting Env is evaluating")
        super().set_is_evaluating()
        commands = self.command_manager.commands
        commands.zero_()
        if command is not None:
            command_tensor = torch.as_tensor(command, device=self.device, dtype=commands.dtype)
            commands[:] = command_tensor.view(1, -1).expand_as(commands)
        gait_state = self.command_manager.get_state("locomotion_gait")
        gait_state.set_eval_mode(True)

    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        self.simulator.commands = self.command_manager.commands

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _pre_compute_observations_callback(self):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.terrain_manager.update_heights()

    def _update_tasks_callback(self):
        super()._update_tasks_callback()

        # Assign commands to simulator for headless recording
        if hasattr(self.simulator, "headless_recording") and self.simulator.headless_recording:
            if hasattr(self.command_manager, "commands"):
                self.simulator.commands = self.command_manager.commands

    def _post_compute_observations_callback(self):
        return

    def reset_all(self):
        self._init_buffers()
        return super().reset_all()

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            self._reset_dofs(env_ids, target_states["dof_states"])
            self._reset_root_states(env_ids, target_states["root_states"])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        # Observation manager reset is now handled in base_task.py
        self.need_to_refresh_envs[env_ids] = True

        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.base_quat[env_ids] = target_buf["base_quat"].to(self.base_quat.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self._pending_episode_update_mask[env_ids] = False
            self._pending_episode_lengths[env_ids] = 0
        else:
            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1
            self._pending_episode_update_mask[env_ids] = True

    def _update_log_dict(self):
        avg = self._get_average_episode_tracker().get_average()
        self.log_dict["average_episode_length"] = avg.detach().cpu()

    ################ Curriculum #################

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    @property
    def average_episode_length(self) -> float:
        avg = self._get_average_episode_tracker().get_average()
        return float(avg.detach().cpu().item())

    # ------------------------------------------------------------------
    # Checkpoint helpers

    def get_checkpoint_state(self) -> dict[str, torch.Tensor | float]:
        state: dict[str, torch.Tensor | float] = {}
        state["average_episode_tracker"] = self._get_average_episode_tracker().state_dict()
        if hasattr(self, "reward_penalty_scale"):
            state["reward_penalty_scale"] = float(self.reward_penalty_scale)
        terrain_curr = self.curriculum_manager.get_term("terrain_level_curriculum")
        if terrain_curr is not None:
            terrain_state = terrain_curr.state_dict()
            if terrain_state:
                state["terrain_level_curriculum"] = terrain_state
        return state

    def load_checkpoint_state(self, state: dict[str, torch.Tensor | float] | None) -> None:
        if not state:
            return

        tracker_state = state.get("average_episode_tracker")
        if tracker_state is not None:
            tracker = self._get_average_episode_tracker()
            tracker.load_state_dict(tracker_state)
            tracker.suppress_next_update()

        penalty_state = state.get("reward_penalty_scale")
        if penalty_state is not None:
            if isinstance(penalty_state, torch.Tensor):
                self.reward_penalty_scale = float(penalty_state.item())
            else:
                self.reward_penalty_scale = float(penalty_state)

        terrain_state = state.get("terrain_level_curriculum")
        if terrain_state is not None:
            terrain_curr = self.curriculum_manager.get_term("terrain_level_curriculum")
            if terrain_curr is not None:
                terrain_curr.load_state_dict(terrain_state)

    def synchronize_curriculum_state(self, *, device: str, world_size: int) -> None:
        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        tracker = self._get_average_episode_tracker()
        avg_tensor = tracker.get_average().clone().detach().to(device)
        torch.distributed.broadcast(avg_tensor, src=0)
        tracker.set_average(avg_tensor.to(self.device), suppress_update=False)

        if hasattr(self, "reward_penalty_scale"):
            penalty_tensor = torch.tensor(float(self.reward_penalty_scale), device=device, dtype=torch.float)
            torch.distributed.broadcast(penalty_tensor, src=0)
            self.reward_penalty_scale = float(penalty_tensor.item())

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 2:
            raise ValueError("Locomotion push velocity vector must have exactly 2 components.")

        rand = torch.rand(len(env_ids), 2, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
        # Push impulses only take effect in the simulator once we write the mutated root state tensor back.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    ################ ENV CALLBACKS #################

    def _reset_dofs(self, env_ids, target_state=None):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        If target_state is not None, reset to target_state

        Args:
            env_ids (List[int]): Environemnt ids
            target_state (Tensor): Target state
        """
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos[env_ids] * torch_rand_float(
                0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device)
            )
            self.simulator.dof_vel[env_ids] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        """Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.terrain_manager.get_state(
                "locomotion_terrain"
            ).env_origins[env_ids]

        else:
            # base position
            self.simulator.robot_root_states[env_ids] = self.base_init_state
            self.simulator.robot_root_states[env_ids, :3] += self.terrain_manager.get_state(
                "locomotion_terrain"
            ).env_origins[env_ids]

            # Apply randomized XY offset if custom_origins
            if self.terrain_manager.get_state("locomotion_terrain").custom_origins:
                # Get spawn config from terrain term
                spawn_cfg = self.terrain_manager.cfg.terrain_term.spawn

                # Generate random XY offsets
                xy_offsets = torch_rand_float(
                    -spawn_cfg.xy_offset_range, spawn_cfg.xy_offset_range, (len(env_ids), 2), device=str(self.device)
                )

                if spawn_cfg.query_terrain_height:
                    # ACCURATE: Query terrain height at new XY position (slower, for rough terrain)
                    current_xy = self.simulator.robot_root_states[env_ids, :2]
                    new_xy = current_xy + xy_offsets

                    terrain_state = self.terrain_manager.get_state("locomotion_terrain")
                    terrain_heights = terrain_state.query_terrain_heights(
                        new_xy,
                        use_grid_sampling=spawn_cfg.use_grid_sampling,
                        grid_size=spawn_cfg.grid_size,
                        grid_spacing=spawn_cfg.grid_spacing,
                    )
                    robot_base_height = self.robot_config.init_state.pos[2]  # Robot base height above ground
                    new_z = terrain_heights + robot_base_height

                    # Write new XYZ position all at once
                    new_xyz = torch.cat([new_xy, new_z.unsqueeze(1)], dim=1)
                    self.simulator.robot_root_states[env_ids, :3] = new_xyz
                else:
                    # FAST: Original simple spawning - just apply XY offset, keep original Z (faster, for flat terrain)
                    self.simulator.robot_root_states[env_ids, :2] += xy_offsets

            # base velocities
            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
            )  # [7:10]: lin vel, [10:13]: ang vel
