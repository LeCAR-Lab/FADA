from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from holosoma.fada.common.current_command import CURRENT_COMMAND_DIM
from holosoma.fada.planner_idm.trainer import FADATrainer
from holosoma.utils.safe_torch_import import torch


class _DummyStudentModel:
    def eval(self) -> "_DummyStudentModel":
        return self


class _DummyExpertPolicy:
    def __init__(self, *, act_dim: int) -> None:
        self.act_dim = int(act_dim)

    def act(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if not obs_dict:
            raise RuntimeError("obs_dict must not be empty")
        first = next(iter(obs_dict.values()))
        return torch.zeros((int(first.shape[0]), self.act_dim), dtype=torch.float32)


class _DummyControlCfg:
    clip_actions = False
    action_clip_value = 1.0


class _DummyRobotCfg:
    control = _DummyControlCfg()


class _DummyCommandCfg:
    locomotion_command_resampling_time = 0.0


class _DummyGaitState:
    """Mirrors managers.command.terms.locomotion.LocomotionGait's public surface:
    .phase is a (num_envs, 2) tensor (see LocomotionGait.__init__/_step)."""

    def __init__(self, *, num_envs: int) -> None:
        self.phase = torch.zeros((num_envs, 2), dtype=torch.float32)


class _DummyCommandManager:
    def __init__(self, *, num_envs: int, cmd_dim: int) -> None:
        self.commands = torch.zeros((num_envs, cmd_dim), dtype=torch.float32)
        self.command_cfg = _DummyCommandCfg()
        self._gait_state = _DummyGaitState(num_envs=num_envs)

    def get_state(self, term_name: str) -> _DummyGaitState | None:
        # Mirrors CommandManager.get_state (managers/command/manager.py): current_command.py's
        # _phase_features() reads command_manager.get_state("locomotion_gait").phase
        # rather than calling obs_terms.sin_phase/cos_phase(env) directly.
        if term_name == "locomotion_gait":
            return self._gait_state
        return None


class _DummyEnv:
    def __init__(self, *, num_envs: int, obs_dim: int, cmd_dim: int) -> None:
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.robot_config = _DummyRobotCfg()
        self.command_manager = _DummyCommandManager(num_envs=self.num_envs, cmd_dim=cmd_dim)
        self.dt = 0.02
        self.reset_all_calls = 0
        self.step_calls = 0
        self.current_obs = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32)

    def reset_all(self) -> dict[str, torch.Tensor]:
        self.reset_all_calls += 1
        marker = float(self.reset_all_calls)
        self.current_obs = torch.full((self.num_envs, self.obs_dim), marker, dtype=torch.float32)
        return {"obs": self.current_obs.clone()}

    def step(self, _: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.step_calls += 1
        marker = float(100 + self.step_calls)
        self.current_obs = torch.full((self.num_envs, self.obs_dim), marker, dtype=torch.float32)
        rewards = torch.zeros((self.num_envs,), dtype=torch.float32)
        dones = torch.zeros((self.num_envs,), dtype=torch.bool)
        return {"obs": self.current_obs.clone()}, rewards, dones, {}


class _DummyEnvWithDoneSchedule(_DummyEnv):
    def __init__(self, *, num_envs: int, obs_dim: int, cmd_dim: int, done_schedule: list[list[bool]]) -> None:
        super().__init__(num_envs=num_envs, obs_dim=obs_dim, cmd_dim=cmd_dim)
        self.done_schedule = done_schedule

    def step(self, _: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.step_calls += 1
        marker = float(100 + self.step_calls)
        self.current_obs = torch.full((self.num_envs, self.obs_dim), marker, dtype=torch.float32)
        rewards = torch.zeros((self.num_envs,), dtype=torch.float32)
        idx = self.step_calls - 1
        if idx < len(self.done_schedule):
            dones = torch.tensor(self.done_schedule[idx], dtype=torch.bool)
        else:
            dones = torch.zeros((self.num_envs,), dtype=torch.bool)
        return {"obs": self.current_obs.clone()}, rewards, dones, {}


def test_collect_offline_by_episodes_resets_on_chunk_end(monkeypatch: pytest.MonkeyPatch):
    trainer = object.__new__(FADATrainer)
    trainer.cfg = SimpleNamespace(
        command_resample_interval=5,
        show_progress=False,
        history_len=2,
    )
    trainer.device = torch.device("cpu")
    trainer.obs_dim = 2
    trainer.act_dim = 1
    trainer.cmd_dim = CURRENT_COMMAND_DIM
    trainer.student_model = _DummyStudentModel()
    trainer.expert_policy = _DummyExpertPolicy(act_dim=trainer.act_dim)
    trainer.offline_buffer = object()

    def _extract_compact_obs_for_env(env: _DummyEnv) -> torch.Tensor:
        return env.current_obs.clone()

    trainer._extract_compact_obs_for_env = _extract_compact_obs_for_env  # type: ignore[assignment]
    trainer._use_teacher_aligned_labels = False

    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.get_base_lin_vel",
        lambda env: torch.zeros((env.num_envs, 3), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.get_base_ang_vel",
        lambda env: torch.zeros((env.num_envs, 3), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.sin_phase",
        lambda env: torch.zeros((env.num_envs, 2), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.cos_phase",
        lambda env: torch.zeros((env.num_envs, 2), dtype=torch.float32),
    )

    observed_markers: list[float] = []
    pushed_dones: list[np.ndarray] = []

    def _push_step_to_buffer(**kwargs: Any) -> None:
        obs_curr = kwargs["obs_curr"]
        dones = kwargs["dones"]
        observed_markers.append(float(obs_curr[0, 0].item()))
        pushed_dones.append(dones.detach().cpu().numpy().copy())

    trainer._push_step_to_buffer = _push_step_to_buffer  # type: ignore[assignment]

    env = _DummyEnv(num_envs=2, obs_dim=trainer.obs_dim, cmd_dim=trainer.cmd_dim)
    stats = trainer._collect_offline_by_episodes(
        env,
        num_episodes=3,
        max_steps_per_episode=2,
    )

    assert int(stats["rollout_steps"]) == 6
    assert int(stats["collected_steps"]) == 12
    assert int(stats["completed_episodes"]) == 6
    assert int(stats["env_done_events"]) == 0

    # One reset at the start of each per-episode rollout call.
    assert env.reset_all_calls == 3

    # After each chunk_end, next step should observe reset markers (2.0, then 3.0).
    assert observed_markers[2] == 2.0
    assert observed_markers[4] == 3.0

    # timeout_dones must mark every env done at chunk boundaries.
    assert np.all(pushed_dones[1])
    assert np.all(pushed_dones[3])
    assert np.all(pushed_dones[5])


class _EmptyLabelWorker:
    def compute(
        self,
        *,
        snapshot: dict[str, Any],
        anchor_obs: dict[str, torch.Tensor],
    ) -> tuple[None, None, None]:
        del snapshot, anchor_obs
        return None, None, None


def test_compute_strict_expert_chunk_labels_raises_on_empty_worker_labels():
    trainer = object.__new__(FADATrainer)
    trainer._use_teacher_aligned_labels = True
    trainer.strict_label_worker = _EmptyLabelWorker()
    trainer.strict_label_env = None
    trainer.device = torch.device("cpu")
    trainer._capture_env_snapshot = lambda env: {}  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="empty labels"):
        trainer._compute_strict_expert_chunk_labels(
            env=SimpleNamespace(),
            anchor_obs_dict={"obs": torch.zeros((1, 1), dtype=torch.float32)},
        )


def test_collect_with_policy_marks_rollout_boundary_done_and_persists_episode_ids(monkeypatch: pytest.MonkeyPatch):
    trainer = object.__new__(FADATrainer)
    trainer.cfg = SimpleNamespace(
        command_resample_interval=5,
        show_progress=False,
        history_len=2,
    )
    trainer.device = torch.device("cpu")
    trainer.obs_dim = 2
    trainer.act_dim = 1
    trainer.cmd_dim = CURRENT_COMMAND_DIM
    trainer.student_model = _DummyStudentModel()
    trainer.expert_policy = _DummyExpertPolicy(act_dim=trainer.act_dim)
    trainer._use_teacher_aligned_labels = False
    trainer._expert_collect_episode_ids = None
    trainer._online_rollout_episode_ids = None

    def _extract_compact_obs_for_env(env: _DummyEnv) -> torch.Tensor:
        return env.current_obs.clone()

    trainer._extract_compact_obs_for_env = _extract_compact_obs_for_env  # type: ignore[assignment]

    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.get_base_lin_vel",
        lambda env: torch.zeros((env.num_envs, 3), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.get_base_ang_vel",
        lambda env: torch.zeros((env.num_envs, 3), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.sin_phase",
        lambda env: torch.zeros((env.num_envs, 2), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.cos_phase",
        lambda env: torch.zeros((env.num_envs, 2), dtype=torch.float32),
    )

    pushed_dones: list[np.ndarray] = []
    pushed_episode_ids: list[np.ndarray] = []

    def _push_step_to_buffer(**kwargs: Any) -> None:
        dones = kwargs["dones"]
        episode_ids = kwargs["episode_ids"]
        pushed_dones.append(dones.detach().cpu().numpy().copy())
        pushed_episode_ids.append(episode_ids.detach().cpu().numpy().copy())

    trainer._push_step_to_buffer = _push_step_to_buffer  # type: ignore[assignment]

    env = _DummyEnv(num_envs=2, obs_dim=trainer.obs_dim, cmd_dim=trainer.cmd_dim)
    trainer._collect_with_policy(
        env,
        num_steps=3,
        sigma=0.0,
        execute_expert=True,
        target_buffer=object(),
    )

    assert len(pushed_dones) == 3
    assert not np.any(pushed_dones[0])
    assert not np.any(pushed_dones[1])
    assert np.all(pushed_dones[2])
    assert np.all(pushed_episode_ids[0] == 0)
    assert np.all(pushed_episode_ids[1] == 0)
    assert np.all(pushed_episode_ids[2] == 0)

    persisted_episode_ids = trainer._expert_collect_episode_ids
    assert isinstance(persisted_episode_ids, torch.Tensor)
    assert torch.all(persisted_episode_ids == 1)

    pushed_dones.clear()
    pushed_episode_ids.clear()
    trainer._collect_with_policy(
        env,
        num_steps=1,
        sigma=0.0,
        execute_expert=True,
        target_buffer=object(),
    )
    assert len(pushed_dones) == 1
    assert np.all(pushed_dones[0])
    assert np.all(pushed_episode_ids[0] == 1)


def test_collect_with_policy_resets_history_and_continues_after_failure(monkeypatch: pytest.MonkeyPatch):
    trainer = object.__new__(FADATrainer)
    trainer.cfg = SimpleNamespace(
        command_resample_interval=5,
        show_progress=False,
        history_len=2,
    )
    trainer.device = torch.device("cpu")
    trainer.obs_dim = 2
    trainer.act_dim = 1
    trainer.cmd_dim = CURRENT_COMMAND_DIM
    trainer.student_model = _DummyStudentModel()
    trainer.expert_policy = _DummyExpertPolicy(act_dim=trainer.act_dim)
    trainer._use_teacher_aligned_labels = False
    trainer._expert_collect_episode_ids = None
    trainer._online_rollout_episode_ids = None

    def _extract_compact_obs_for_env(env: _DummyEnv) -> torch.Tensor:
        return env.current_obs.clone()

    trainer._extract_compact_obs_for_env = _extract_compact_obs_for_env  # type: ignore[assignment]

    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.get_base_lin_vel",
        lambda env: torch.zeros((env.num_envs, 3), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.get_base_ang_vel",
        lambda env: torch.zeros((env.num_envs, 3), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.sin_phase",
        lambda env: torch.zeros((env.num_envs, 2), dtype=torch.float32),
    )
    monkeypatch.setattr(
        "holosoma.fada.planner_idm.trainer_rollout.obs_terms.cos_phase",
        lambda env: torch.zeros((env.num_envs, 2), dtype=torch.float32),
    )

    pushed_dones: list[np.ndarray] = []
    pushed_episode_ids: list[np.ndarray] = []
    pushed_valid: list[np.ndarray] = []

    def _push_step_to_buffer(**kwargs: Any) -> None:
        dones = kwargs["dones"]
        episode_ids = kwargs["episode_ids"]
        valid = kwargs["strict_label_valid"]
        pushed_dones.append(dones.detach().cpu().numpy().copy())
        pushed_episode_ids.append(episode_ids.detach().cpu().numpy().copy())
        pushed_valid.append(valid.detach().cpu().numpy().copy())

    trainer._push_step_to_buffer = _push_step_to_buffer  # type: ignore[assignment]

    env = _DummyEnvWithDoneSchedule(
        num_envs=2,
        obs_dim=trainer.obs_dim,
        cmd_dim=trainer.cmd_dim,
        done_schedule=[
            [True, False],   # env0 fails early
            [False, False],  # should be ignored for env0
            [False, False],  # boundary step
        ],
    )
    trainer._collect_with_policy(
        env,
        num_steps=3,
        sigma=0.0,
        execute_expert=True,
        target_buffer=object(),
    )

    # Distinguish failure reset vs boundary reset:
    # step1: failure of env0 only
    # step2: env0 has reset and continues collecting valid data
    # step3: boundary closes both envs so all episodes are closed in replay
    assert np.array_equal(pushed_dones[0], np.asarray([True, False], dtype=np.bool_))
    assert np.array_equal(pushed_dones[1], np.asarray([False, False], dtype=np.bool_))
    assert np.array_equal(pushed_dones[2], np.asarray([True, True], dtype=np.bool_))

    # Post-failure rows stay train-valid after reset; no freeze-until-boundary masking.
    assert np.array_equal(pushed_valid[0], np.asarray([True, True], dtype=np.bool_))
    assert np.array_equal(pushed_valid[1], np.asarray([True, True], dtype=np.bool_))
    assert np.array_equal(pushed_valid[2], np.asarray([True, True], dtype=np.bool_))

    # Episode ids advance on every done event (including boundary close).
    assert np.array_equal(pushed_episode_ids[0], np.asarray([0, 0], dtype=np.int64))
    assert np.array_equal(pushed_episode_ids[1], np.asarray([1, 0], dtype=np.int64))
    assert np.array_equal(pushed_episode_ids[2], np.asarray([1, 0], dtype=np.int64))

    persisted_episode_ids = trainer._expert_collect_episode_ids
    assert isinstance(persisted_episode_ids, torch.Tensor)
    assert torch.equal(persisted_episode_ids, torch.tensor([2, 1], dtype=torch.long))
