from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from holosoma.fada.common.dataset import ReplayBuffer
from holosoma.fada.planner_idm.config import FADAConfig
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.fada.planner_idm.trainer import ExpertPolicyWrapper, FADATrainer
from holosoma.utils.safe_torch_import import torch


class _DummyAlgo:
    def __init__(self, act_dim: int) -> None:
        self.act_dim = int(act_dim)

    def load(self, path: str) -> None:
        self.last_loaded = path

    def _eval_mode(self) -> None:
        return None

    def get_inference_policy(self):
        def _policy(policy_input):
            actor_obs = policy_input["actor_obs"]
            return torch.zeros((int(actor_obs.shape[0]), self.act_dim), dtype=torch.float32)

        return _policy


def _compact_terms(scale: float = 1.0) -> dict[str, float]:
    return {
        "base_ang_vel": scale,
        "dof_pos": scale,
        "dof_vel": scale,
        "projected_gravity": scale,
    }


def _build_buffer(*, base: float, growable: bool = True) -> ReplayBuffer:
    buffer = ReplayBuffer(
        capacity=32,
        obs_dim=2,
        act_dim=1,
        cmd_dim=1,
        history_len=2,
        pred_horizon=1,
        require_future_obs_targets=True,
        growable=growable,
    )
    num_steps = 6
    obs = np.full((num_steps, 2), base, dtype=np.float32)
    current_command = np.full((num_steps, 1), base + 0.1, dtype=np.float32)
    executed_act = np.full((num_steps, 1), base + 0.2, dtype=np.float32)
    expert_act = np.full((num_steps, 1), base + 0.3, dtype=np.float32)
    expert_chunk = np.full((num_steps, 1, 1), base + 0.3, dtype=np.float32)
    expert_future_obs_chunk = np.full((num_steps, 1, 2), base + 0.4, dtype=np.float32)
    strict_label_valid = np.ones((num_steps,), dtype=np.bool_)
    reward = np.full((num_steps,), base + 0.5, dtype=np.float32)
    done = np.zeros((num_steps,), dtype=np.bool_)
    done[-1] = True
    env_id = np.zeros((num_steps,), dtype=np.int32)
    episode_id = np.zeros((num_steps,), dtype=np.int64)
    buffer.add_batch(
        obs=obs,
        current_command=current_command,
        executed_act=executed_act,
        expert_act=expert_act,
        expert_chunk=expert_chunk,
        expert_future_obs_chunk=expert_future_obs_chunk,
        strict_label_valid=strict_label_valid,
        reward=reward,
        done=done,
        env_id=env_id,
        episode_id=episode_id,
    )
    return buffer


def _build_expert_wrapper(act_dim: int) -> ExpertPolicyWrapper:
    wrapper = object.__new__(ExpertPolicyWrapper)
    wrapper.algo = _DummyAlgo(act_dim=act_dim)
    wrapper.policy = wrapper.algo.get_inference_policy()
    wrapper.actor_obs_keys = ["actor_obs"]
    return wrapper


def _build_episode_buffer_for_trainer(
    trainer: FADATrainer,
    *,
    episodes: list[tuple[int, int, int, float]],
) -> ReplayBuffer:
    buffer = ReplayBuffer(
        capacity=max(1, sum(length for _, _, length, _ in episodes)),
        obs_dim=trainer.offline_buffer.obs_dim,
        act_dim=trainer.offline_buffer.act_dim,
        cmd_dim=trainer.offline_buffer.cmd_dim,
        history_len=trainer.offline_buffer.history_len,
        pred_horizon=trainer.offline_buffer.pred_horizon,
        require_future_obs_targets=trainer.offline_buffer.require_future_obs_targets,
        growable=True,
    )
    for env_id, episode_id, length, base in episodes:
        obs = np.full((length, trainer.obs_dim), base, dtype=np.float32)
        current_command = np.full((length, trainer.cmd_dim), base + 0.1, dtype=np.float32)
        executed_act = np.full((length, trainer.act_dim), base + 0.2, dtype=np.float32)
        expert_act = np.full((length, trainer.act_dim), base + 0.3, dtype=np.float32)
        expert_chunk = np.full(
            (length, trainer.cfg.pred_horizon, trainer.act_dim),
            base + 0.7,
            dtype=np.float32,
        )
        expert_future_obs_chunk = np.full(
            (length, trainer.cfg.pred_horizon, trainer.obs_dim),
            base + 0.8,
            dtype=np.float32,
        )
        strict_label_valid = np.ones((length,), dtype=np.bool_)
        strict_label_valid[-1] = False
        reward = np.full((length,), base + 0.5, dtype=np.float32)
        done = np.zeros((length,), dtype=np.bool_)
        done[-1] = True
        env_ids = np.full((length,), env_id, dtype=np.int32)
        episode_ids = np.full((length,), episode_id, dtype=np.int64)
        buffer.add_batch(
            obs=obs,
            current_command=current_command,
            executed_act=executed_act,
            expert_act=expert_act,
            expert_chunk=expert_chunk,
            expert_future_obs_chunk=expert_future_obs_chunk,
            strict_label_valid=strict_label_valid,
            reward=reward,
            done=done,
            env_id=env_ids,
            episode_id=episode_ids,
        )
    return buffer


def _build_trainer(
    tmp_path: Path,
    *,
    use_generalized_idm: bool = True,
    idm_use_teacher_forcing: bool = True,
    idm_teacher_forcing_ratio: float = 1.0,
    idm_detach_planner_future_obs: bool = True,
    **extra_cfg,
) -> FADATrainer:
    cfg = FADAConfig(
        history_len=2,
        pred_horizon=1,
        use_generalized_idm=use_generalized_idm,
        idm_use_teacher_forcing=idm_use_teacher_forcing,
        idm_teacher_forcing_ratio=idm_teacher_forcing_ratio,
        idm_detach_planner_future_obs=idm_detach_planner_future_obs,
        planner_d_model=8,
        planner_nhead=1,
        planner_num_layers=1,
        planner_dim_feedforward=16,
        idm_d_model=8,
        idm_nhead=1,
        idm_encoder_num_layers=1,
        idm_decoder_num_layers=1,
        idm_dim_feedforward=16,
        batch_size=4,
        mixed_batch_offline_ratio=0.5,
        idm_suboptimal_batch_ratio=0.5,
        validation_batches=2,
        compact_obs_term_scale=_compact_terms(1.0),
        compact_obs_term_noise=_compact_terms(0.0),
        expert_checkpoint=str(tmp_path / "model_9999.pt"),
        offline_cache_path=str(tmp_path / "optimal_cache.npz"),
        offline_suboptimal_cache_path=str(tmp_path / "suboptimal_cache.npz"),
        warmup_ckpt_path=str(tmp_path / "warmup.pt"),
        shared_norm_source="idm_visible_union",
        wandb_enable=False,
        wandb_mode="disabled",
        **extra_cfg,
    ).validate()
    model = PlannerIDMPolicy(
        obs_dim=2,
        act_dim=1,
        cmd_dim=1,
        history_len=cfg.history_len,
        pred_horizon=cfg.pred_horizon,
        planner_d_model=cfg.planner_d_model,
        planner_nhead=cfg.planner_nhead,
        planner_num_layers=cfg.planner_num_layers,
        planner_dim_feedforward=cfg.planner_dim_feedforward,
        planner_dropout=0.0,
        idm_d_model=cfg.idm_d_model,
        idm_nhead=cfg.idm_nhead,
        idm_encoder_num_layers=cfg.idm_encoder_num_layers,
        idm_decoder_num_layers=cfg.idm_decoder_num_layers,
        idm_dim_feedforward=cfg.idm_dim_feedforward,
        idm_dropout=0.0,
    )
    optimizer = torch.optim.AdamW(list(model.planner_parameters()), lr=1e-3)
    idm_optimizer = torch.optim.AdamW(list(model.idm_parameters()), lr=1e-3)
    return FADATrainer(
        cfg=cfg,
        student_model=model,
        expert_policy=_build_expert_wrapper(act_dim=1),
        offline_buffer=_build_buffer(base=1.0),
        suboptimal_offline_buffer=_build_buffer(base=3.0),
        online_buffer=_build_buffer(base=5.0, growable=False),
        optimizer=optimizer,
        lr_scheduler=None,
        idm_optimizer=idm_optimizer,
        idm_lr_scheduler=None,
        device=torch.device("cpu"),
        obs_dim=2,
        act_dim=1,
        cmd_dim=1,
        resolved_expert_checkpoint=str(tmp_path / "model_9999.pt"),
        compact_obs_source_checkpoint=str(tmp_path / "model_9999.pt"),
        strict_label_env=object(),
    )


def test_split_batch_sampling_counts(tmp_path: Path) -> None:
    # Expected counts below follow _planner_batch_counts/_idm_batch_counts's arithmetic
    # at the FADAConfig defaults (planner_suboptimal_batch_ratio 0.375,
    # suboptimal_expert_batch_ratio 0.5, suboptimal_data_ratio 1.0,
    # warmup_{idm,planner}_suboptimal_batch_ratio 0.75), under which
    # planner_suboptimal_enabled and suboptimal_enabled are both True. The aggregate
    # rows are sums of the per-module ones (optimal_batch_size == planner_optimal +
    # idm_optimal, etc.; see _build_sample_stats/_merge_source_counts).
    trainer = _build_trainer(tmp_path)

    _, sample_stats_offline, _ = trainer.train_step(mode="offline")
    assert sample_stats_offline["planner_optimal_batch_size"] == 1
    assert sample_stats_offline["planner_suboptimal_batch_size"] == 3
    assert sample_stats_offline["planner_online_batch_size"] == 0
    assert sample_stats_offline["idm_optimal_batch_size"] == 1
    assert sample_stats_offline["idm_suboptimal_batch_size"] == 1
    assert sample_stats_offline["idm_suboptimal_expert_batch_size"] == 2
    assert sample_stats_offline["idm_online_batch_size"] == 0
    assert sample_stats_offline["optimal_batch_size"] == 2
    assert sample_stats_offline["suboptimal_batch_size"] == 4
    assert sample_stats_offline["online_batch_size"] == 0

    _, sample_stats_mixed, _ = trainer.train_step(mode="mixed")
    assert sample_stats_mixed["planner_optimal_batch_size"] == 1
    assert sample_stats_mixed["planner_suboptimal_batch_size"] == 2
    assert sample_stats_mixed["planner_online_batch_size"] == 1
    assert sample_stats_mixed["idm_optimal_batch_size"] == 1
    assert sample_stats_mixed["idm_suboptimal_batch_size"] == 1
    assert sample_stats_mixed["idm_suboptimal_expert_batch_size"] == 1
    assert sample_stats_mixed["idm_online_batch_size"] == 1
    assert sample_stats_mixed["optimal_batch_size"] == 2
    assert sample_stats_mixed["suboptimal_batch_size"] == 3
    assert sample_stats_mixed["online_batch_size"] == 2


def test_optimal_only_idm_mode_disables_suboptimal_sampling(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, use_generalized_idm=False)

    # The suboptimal data source is on by default: planner_suboptimal_batch_ratio
    # is 0.375, so planner_suboptimal_enabled -- and therefore suboptimal_enabled --
    # is True regardless of use_generalized_idm. Planner-level suboptimal sampling is
    # an independent knob from the IDM generalized/optimal-only switch this test
    # targets, so pin it off here; what is asserted below is that IDM optimal-only
    # mode disables suboptimal sampling for the IDM.
    trainer.cfg.planner_suboptimal_batch_ratio = 0.0
    trainer.planner_suboptimal_enabled = False
    trainer.suboptimal_enabled = False

    assert trainer.suboptimal_enabled is False

    _, sample_stats_offline, _ = trainer.train_step(mode="offline")
    assert sample_stats_offline["planner_optimal_batch_size"] == 4
    assert sample_stats_offline["planner_suboptimal_batch_size"] == 0
    assert sample_stats_offline["planner_online_batch_size"] == 0
    assert sample_stats_offline["idm_optimal_batch_size"] == 4
    assert sample_stats_offline["idm_suboptimal_batch_size"] == 0
    assert sample_stats_offline["idm_online_batch_size"] == 0

    _, sample_stats_mixed, _ = trainer.train_step(mode="mixed")
    assert sample_stats_mixed["planner_optimal_batch_size"] == 2
    assert sample_stats_mixed["planner_suboptimal_batch_size"] == 0
    assert sample_stats_mixed["planner_online_batch_size"] == 2
    assert sample_stats_mixed["idm_optimal_batch_size"] == 2
    assert sample_stats_mixed["idm_suboptimal_batch_size"] == 0
    # use_generalized_idm=False routes the full online IDM budget through
    # online_trajectory (real executed dynamics) instead of online (oracle-shadow
    # targets) -- see _idm_batch_counts's "skip oracle-shadow online data so IDM
    # only sees real dynamics" branch.
    assert sample_stats_mixed["idm_online_batch_size"] == 0
    assert sample_stats_mixed["idm_online_trajectory_batch_size"] == 2


def test_non_teacher_forcing_uses_detached_planner_future_obs(tmp_path: Path) -> None:
    trainer = _build_trainer(
        tmp_path,
        idm_use_teacher_forcing=False,
        idm_teacher_forcing_ratio=1.0,
    )
    planner_future_obs = torch.randn((2, 1, 2), dtype=torch.float32, requires_grad=True)
    teacher_future_obs = torch.randn((2, 1, 2), dtype=torch.float32)

    resolved = trainer._resolve_idm_future_obs(
        planner_future_obs=planner_future_obs,
        teacher_future_obs=teacher_future_obs,
    )

    assert torch.allclose(resolved, planner_future_obs.detach())
    assert resolved.requires_grad is False


def test_teacher_forcing_source_policy_matches_expected_semantics(tmp_path: Path) -> None:
    trainer = _build_trainer(
        tmp_path,
        use_generalized_idm=True,
        idm_use_teacher_forcing=True,
        idm_teacher_forcing_ratio=1.0,
    )

    assert trainer._idm_source_uses_teacher_forcing("optimal") is True
    assert trainer._idm_source_uses_teacher_forcing("suboptimal") is True
    assert trainer._idm_source_uses_teacher_forcing("online") is True

    trainer_no_tf = _build_trainer(
        tmp_path,
        use_generalized_idm=True,
        idm_use_teacher_forcing=False,
        idm_teacher_forcing_ratio=1.0,
    )
    assert trainer_no_tf._idm_source_uses_teacher_forcing("optimal") is False
    assert trainer_no_tf._idm_source_uses_teacher_forcing("suboptimal") is True
    assert trainer_no_tf._idm_source_uses_teacher_forcing("online") is False


def test_idm_branch_does_not_backprop_into_planner(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    batch, _ = trainer._sample_idm_batch(mode="offline")

    trainer.optimizer.zero_grad(set_to_none=True)
    # _compute_idm_batch_loss_from_batch returns
    # (loss_total, idm_loss, idm_loss_unweighted).
    idm_loss_total, _, _ = trainer._compute_idm_batch_loss_from_batch(batch)
    idm_loss_total.backward()

    planner_grads = [parameter.grad for parameter in trainer.student_policy.planner.parameters()]
    idm_grads = [parameter.grad for parameter in trainer.student_policy.idm.parameters()]

    assert all(grad is None or torch.allclose(grad, torch.zeros_like(grad)) for grad in planner_grads)
    assert any(grad is not None and torch.any(grad != 0) for grad in idm_grads)


def test_shared_norm_source_uses_optimal_suboptimal_then_adds_online(tmp_path: Path) -> None:
    # normalize_io (and FADATrainer._use_io_normalization) is constant-folded to False,
    # so _refresh_shared_norm_stats does not refresh action/command norm stats itself.
    # Call the three underlying _refresh_*_norm_stats methods directly to cover the
    # shared_norm_source buffer-selection logic, which stays live via the independent
    # predict_future_obs gate.
    trainer = _build_trainer(tmp_path)

    trainer._refresh_obs_norm_stats(include_online=False)
    trainer._refresh_action_norm_stats(include_online=False)
    trainer._refresh_command_norm_stats(include_online=False)
    np.testing.assert_allclose(
        trainer.obs_norm_mean.detach().cpu().numpy(),
        np.asarray([2.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        trainer.action_norm_mean.detach().cpu().numpy(),
        np.asarray([2.3], dtype=np.float32),
    )
    np.testing.assert_allclose(
        trainer.command_norm_mean.detach().cpu().numpy(),
        np.asarray([2.1], dtype=np.float32),
    )

    trainer._refresh_obs_norm_stats(include_online=True)
    trainer._refresh_action_norm_stats(include_online=True)
    trainer._refresh_command_norm_stats(include_online=True)
    np.testing.assert_allclose(
        trainer.obs_norm_mean.detach().cpu().numpy(),
        np.asarray([3.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        trainer.action_norm_mean.detach().cpu().numpy(),
        np.asarray([3.3], dtype=np.float32),
    )
    np.testing.assert_allclose(
        trainer.command_norm_mean.detach().cpu().numpy(),
        np.asarray([3.1], dtype=np.float32),
    )


def test_optimal_offline_collection_stops_after_configured_rollout_count(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.cfg.warmup_episodes = 2
    trainer.cfg.warmup_max_steps_per_episode = 4
    checkpoint_path = tmp_path / "model_9999.pt"
    checkpoint_path.write_text("x", encoding="utf-8")

    class _DummyEnv:
        num_envs = 4

    rollout_stats = iter(
        [
            {"kept_steps": 12, "kept_episodes": 3, "dropped_episodes": 1},
            {"kept_steps": 4, "kept_episodes": 1, "dropped_episodes": 3},
            {"kept_steps": 16, "kept_episodes": 4, "dropped_episodes": 0},
        ]
    )
    trainer._collect_strict_optimal_rollout_buffer = lambda env, *, horizon: object()  # type: ignore[assignment]
    trainer._append_complete_episodes_from_buffer = lambda **kwargs: next(rollout_stats)  # type: ignore[assignment]

    stats = trainer._collect_optimal_offline_data(_DummyEnv())

    assert stats["collect_mode"] == "episodes"
    assert stats["kept_episodes"] == 4
    assert stats["dropped_episodes"] == 4
    assert stats["rollouts"] == 2
    assert stats["target_steps"] == 32


def test_suboptimal_target_steps_use_optimal_target_not_actual_kept(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.cfg.warmup_episodes = 5
    trainer.cfg.warmup_max_steps_per_episode = 4
    trainer.cfg.suboptimal_data_ratio = 2.0
    # This test exercises the planner_suboptimal-disabled offline-collection branch via
    # the _collect_expert_policy_rollout/_append_filtered_rollout_to_buffer mocks below;
    # it covers target-step bookkeeping, not the oracle-relabeling use_loaded_policy=True
    # path. planner_suboptimal_batch_ratio defaults to 0.375, so pin it to 0.0 here:
    # otherwise trainer.planner_suboptimal_enabled routes through
    # _collect_with_policy(use_loaded_policy=True), which requires a real
    # strict_label_worker (covered in test_dagger_trainer_offline_collection.py) rather
    # than these mocks.
    trainer.cfg.planner_suboptimal_batch_ratio = 0.0
    trainer.planner_suboptimal_enabled = False
    checkpoint_dir = tmp_path
    for step in (0, 100, 9999):
        (checkpoint_dir / f"model_{step:05d}.pt").write_text("x", encoding="utf-8")

    class _DummyEnv:
        num_envs = 4

    rollout_stats = iter([{"kept_steps": 170, "kept_episodes": 10, "dropped_episodes": 0}])
    trainer._load_reward_curve = lambda checkpoint_dir: {}  # type: ignore[assignment]
    trainer._collect_expert_policy_rollout = lambda env, *, horizon, progress_desc="Offline Collect (Steps)": object()  # type: ignore[assignment]
    trainer._append_filtered_rollout_to_buffer = lambda **kwargs: next(rollout_stats)  # type: ignore[assignment]

    stats = trainer._collect_suboptimal_offline_data(_DummyEnv(), optimal_target_steps=80)

    assert stats["target_optimal_steps"] == 80
    assert stats["target_steps"] == 160
    assert stats["kept_steps"] == 170
    assert stats["rollouts"] == 1


def test_warmup_reuses_saved_optimal_cache_and_saves_suboptimal_immediately(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.cfg.warmup_train_steps = 0
    trainer.cfg.validation_batches = 0
    trainer.cfg.load_warmup_ckpt = False
    trainer.offline_buffer.save_npz(
        trainer.cfg.offline_cache_path,
        compact_obs_metadata=trainer._offline_cache_compact_obs_metadata,
    )
    trainer.offline_buffer.clear()
    trainer.suboptimal_offline_buffer.clear()

    class _DummyEnv:
        num_envs = 4

    def _collect_suboptimal(env, *, optimal_target_steps: int):
        payload = _build_buffer(base=7.0)._as_chronological_dict()
        trainer.suboptimal_offline_buffer.add_batch(
            obs=payload["obs"],
            current_command=payload["current_command"],
            executed_act=payload["executed_act"],
            expert_act=payload["expert_act"],
            expert_chunk=payload["expert_chunk"],
            expert_future_obs_chunk=payload["expert_future_obs_chunk"],
            strict_label_valid=payload["strict_label_valid"],
            reward=payload["reward"],
            done=payload["done"],
            env_id=payload["env_id"],
            episode_id=payload["episode_id"],
        )
        return {
            "enabled": True,
            "target_optimal_steps": int(optimal_target_steps),
            "target_steps": int(optimal_target_steps * trainer.cfg.suboptimal_data_ratio),
            "kept_steps": int(len(trainer.suboptimal_offline_buffer)),
            "kept_episodes": 1,
            "dropped_episodes": 0,
            "rollouts": 1,
            "selected_checkpoints": ["dummy"],
            "checkpoint_rollouts": {"dummy": 1},
            "checkpoint_steps": {"dummy": int(len(trainer.suboptimal_offline_buffer))},
            "sampling_mode": "reward_aware",
        }

    trainer._collect_optimal_offline_data = lambda env: (_ for _ in ()).throw(AssertionError("optimal should not be recollected"))  # type: ignore[assignment]
    trainer._collect_suboptimal_offline_data = _collect_suboptimal  # type: ignore[assignment]

    stats = trainer.warmup(_DummyEnv())

    assert stats["optimal_offline_cache_hit"] is True
    assert stats["suboptimal_offline_cache_hit"] is False
    assert stats["optimal_offline_cache_saved"] is False
    assert stats["suboptimal_offline_cache_saved"] is True
    assert Path(trainer.cfg.offline_suboptimal_cache_path).exists()


def test_optimal_strict_filter_keeps_only_full_horizon_teacher_aligned_episodes(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.offline_buffer.clear()
    strict_buffer = _build_episode_buffer_for_trainer(
        trainer,
        episodes=[
            (0, 10, 4, 10.0),
            (1, 20, 2, 20.0),
            (1, 21, 2, 30.0),
        ],
    )

    stats = trainer._append_complete_episodes_from_buffer(
        source_buffer=strict_buffer,
        target_buffer=trainer.offline_buffer,
        required_length=4,
    )

    assert stats["kept_episodes"] == 1
    assert stats["dropped_episodes"] == 2
    assert stats["kept_steps"] == 4

    kept = trainer.offline_buffer._as_chronological_dict()
    assert kept["obs"].shape[0] == 4
    np.testing.assert_array_equal(np.unique(kept["episode_id"]), np.asarray([10], dtype=np.int64))
    np.testing.assert_allclose(
        kept["current_command"],
        np.full((4, 1), 10.1, dtype=np.float32),
    )
    np.testing.assert_allclose(
        kept["expert_chunk"],
        np.full((4, 1, 1), 10.7, dtype=np.float32),
    )
    np.testing.assert_allclose(
        kept["expert_future_obs_chunk"],
        np.full((4, 1, 2), 10.8, dtype=np.float32),
    )


def test_checkpoint_and_cache_artifacts_include_split_buffer_metadata(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.offline_buffer.save_npz(
        trainer.cfg.offline_cache_path,
        compact_obs_metadata=trainer._offline_cache_compact_obs_metadata,
    )
    trainer.suboptimal_offline_buffer.save_npz(
        trainer.cfg.offline_suboptimal_cache_path,
        compact_obs_metadata=trainer._offline_cache_compact_obs_metadata,
    )
    checkpoint_path = trainer.save_student_checkpoint(tmp_path / "student.pt", extra={"phase": "smoke"})
    training_log_path = trainer.dump_training_log(tmp_path / "training_log.json", logs=[{"iter": 1, "status": "ok"}])

    assert Path(trainer.cfg.offline_cache_path).exists()
    assert Path(trainer.cfg.offline_suboptimal_cache_path).exists()
    assert checkpoint_path.exists()
    assert training_log_path.exists()

    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["optimal_offline_buffer_size"] == len(trainer.offline_buffer)
    assert payload["suboptimal_offline_buffer_size"] == len(trainer.suboptimal_offline_buffer)
    assert payload["online_buffer_size"] == len(trainer.online_buffer)

    log_payload = json.loads(training_log_path.read_text(encoding="utf-8"))
    assert log_payload == [{"iter": 1, "status": "ok"}]


def test_the_planner_pass_does_not_detach_and_is_the_planner_s_only_gradient_source(tmp_path: Path) -> None:
    """The planner pass runs with `detach_override=False`, so the IDM action loss carries its gradient.

    Detaching is a property of *one of the two passes*: `_train_step_separate_pass` runs
    the IDM update at the config default (`idm_detach_planner_future_obs=True`, so no
    gradient reaches the planner -- `test_idm_branch_does_not_backprop_into_planner`),
    and the planner update with an explicit `detach_override=False`.
    `planner_obs_loss_coef` defaults to `0.0`, so the planner's own observation loss
    contributes nothing and the IDM path is its only gradient source.

    Asserted as a comparison rather than a threshold: same trainer, same batch, one
    argument flipped.
    """
    trainer = _build_trainer(tmp_path)
    assert trainer.cfg.idm_detach_planner_future_obs is True, "the default this test contrasts against"
    assert float(trainer.cfg.planner_obs_loss_coef) == 0.0, (
        "if the planner's own obs loss were weighted, the claim below would not be load-bearing"
    )

    batch, _ = trainer._sample_planner_batch(mode="offline")

    def planner_grad_magnitude(*, detach_override: bool) -> tuple[int, float]:
        trainer.optimizer.zero_grad(set_to_none=True)
        _total, loss, _uw = trainer._compute_idm_batch_loss_from_batch(
            batch,
            use_teacher_forcing=False,
            teacher_forcing_ratio=0.0,
            detach_override=detach_override,
            augment=False,
        )
        loss.backward()
        grads = [p.grad for p in trainer.student_model.planner_parameters() if p.grad is not None]
        return len(grads), float(sum(g.abs().sum().item() for g in grads))

    # What trainer.py's planner pass passes.
    count_no_detach, magnitude_no_detach = planner_grad_magnitude(detach_override=False)
    # The same call with the IDM pass's setting instead.
    count_detached, magnitude_detached = planner_grad_magnitude(detach_override=True)

    assert count_no_detach > 0 and magnitude_no_detach > 0.0, (
        "the planner pass must reach the planner through the IDM; with planner_obs_loss_coef=0 "
        "it is the planner's only gradient source"
    )
    assert count_detached == 0 and magnitude_detached == 0.0, (
        "detaching must cut that path entirely -- which is why the planner pass overrides it"
    )


def test_the_two_passes_pass_the_detach_flag_they_are_documented_to_pass(tmp_path: Path) -> None:
    """Pins which detach/teacher-forcing arguments each of the two passes reaches the batching layer with.

    Recorded by wrapping `_compute_idm_batch_loss_from_batch` for one real `train_step`
    rather than by reading the source.
    """
    trainer = _build_trainer(tmp_path)

    seen: list[dict] = []
    original = FADATrainer._compute_idm_batch_loss_from_batch

    def recording(self, batch, **kwargs):  # noqa: ANN001
        seen.append(dict(kwargs))
        return original(self, batch, **kwargs)

    trainer._compute_idm_batch_loss_from_batch = recording.__get__(trainer, FADATrainer)
    trainer.train_step(mode="offline")

    # The IDM pass leaves detach at the config default (None == "use cfg"), and forces TF=1.0.
    idm_pass = [k for k in seen if k.get("detach_override") is None]
    assert idm_pass, f"no IDM-pass call recorded: {seen}"
    assert all(k.get("teacher_forcing_ratio") == 1.0 for k in idm_pass), seen

    # The planner pass overrides both: no detach, no teacher forcing.
    planner_pass = [k for k in seen if k.get("detach_override") is False]
    assert len(planner_pass) == 1, f"expected exactly one planner-pass call, got {planner_pass}"
    assert planner_pass[0].get("teacher_forcing_ratio") == 0.0
    assert planner_pass[0].get("use_teacher_forcing") is False

    # And nothing ever asks for detach=True explicitly; that is the config default's job.
    assert not [k for k in seen if k.get("detach_override") is True], seen


# ---------------------------------------------------------------------------
# The three teacher-forcing / detach config fields are inert on the supported
# training step, which is why they are not CLI flags.
#
# The test above shows *structurally* that both passes pass explicit overrides.
# This one shows the consequence: changing the fields changes nothing at all --
# not one loss value, not one weight. It lives with the trainer rather than with
# the CLI surface tests that assert the flags are absent.
# ---------------------------------------------------------------------------


def _train_and_fingerprint(tmp_path: Path, *, steps: int = 3, **cfg_overrides) -> dict[str, object]:
    """Run `train_step` from a fixed seed and return every number it produced.

    Seeding immediately before construction makes the model init, the batch
    sampling and the augmentation noise identical across configurations, so any
    difference in the result is attributable to `cfg_overrides` and nothing else.
    """
    torch.manual_seed(20260828)
    np.random.seed(20260828)
    trainer = _build_trainer(tmp_path, **cfg_overrides)

    # `float.hex()` rather than the float: some per-term planner metrics are NaN for this
    # two-dimensional fixture, and `nan != nan` would make every comparison below fail.
    # Hex compares exactly, with no tolerance.
    losses: list[dict[str, str]] = []
    for _ in range(steps):
        _total, _stats, metrics = trainer.train_step(mode="offline")
        losses.append({k: float(v).hex() for k, v in metrics.items()})
    # Raw bytes, for the same reason and with the same exactness (`torch.equal` is also
    # False for NaN).
    state = {k: v.detach().cpu().numpy().tobytes() for k, v in trainer.student_model.state_dict().items()}
    return {"losses": losses, "state": state}


def _assert_identical(label: str, a: dict[str, object], b: dict[str, object]) -> None:
    losses_a, losses_b = a["losses"], b["losses"]
    assert losses_a == losses_b, f"{label}: loss sequence differs\n{losses_a}\n{losses_b}"
    state_a, state_b = a["state"], b["state"]
    assert state_a.keys() == state_b.keys(), label
    for key in state_a:
        assert state_a[key] == state_b[key], f"{label}: model tensor {key!r} differs"


def test_the_teacher_forcing_and_detach_fields_change_nothing_on_the_supported_step(
    tmp_path: Path,
) -> None:
    """Four configurations produce bit-identical losses and bit-identical weights.

    `_train_step_separate_pass` forces TF=1.0 for the IDM pass and
    `use_teacher_forcing=False, teacher_forcing_ratio=0.0, detach_override=False`
    for the planner pass. `_resolve_idm_future_obs` consults
    `cfg.idm_use_teacher_forcing` / `cfg.idm_teacher_forcing_ratio` /
    `cfg.idm_detach_planner_future_obs` only when the corresponding argument is
    `None`, which never happens there, so the fields have readers but no reachable
    one. Fails the moment the supported step starts honouring any of them.
    """
    baseline = _train_and_fingerprint(tmp_path)

    _assert_identical(
        "idm_use_teacher_forcing=False",
        baseline,
        _train_and_fingerprint(tmp_path, idm_use_teacher_forcing=False),
    )
    _assert_identical(
        "idm_teacher_forcing_ratio=0.25",
        baseline,
        _train_and_fingerprint(tmp_path, idm_teacher_forcing_ratio=0.25),
    )
    _assert_identical(
        "idm_detach_planner_future_obs=False",
        baseline,
        _train_and_fingerprint(tmp_path, idm_detach_planner_future_obs=False),
    )

    # Control for the fingerprint's sensitivity: `idm_action_loss_coef` (default 1.0) is
    # read by both passes on this step, so changing it must move the numbers.
    control = _train_and_fingerprint(tmp_path, idm_action_loss_coef=2.0)
    assert control["losses"] != baseline["losses"], (
        "the fingerprint did not move for a config change the trainer does honour, so the "
        "four comparisons above prove nothing"
    )
