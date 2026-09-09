"""Bit-exactness probe: the training step.

Runs N `train_step` calls on CPU only, with no IsaacSim and no expert checkpoint, and
emits the loss sequence at full float64 precision plus the sha256 of the final
`state_dict`.

It never emits a hash of the whole checkpoint payload: constant-folding a config field
changes the key set of the `cfg` dict inside that payload, so comparing the payload
would report spurious failures.

Two probe variants:

- `run_train_probe` / `build_probe_config`: `train_step(mode="offline")` under the
  separate IDM/planner pass -- the warmup phase, which back-propagates and steps the
  IDM first, then runs the planner pass with the IDM frozen.
- `run_train_probe_mixed` / `build_probe_config_mixed`: `train_step(mode="mixed")`
  under the same separate pass, with all four of the suboptimal / suboptimal_expert /
  online / online_trajectory batch ratios non-zero -- the mode DAgger uses in its
  online phase, covering the data routing in `_idm_batch_counts`,
  `_planner_batch_counts`, `_sample_idm_batches_by_source` and `_resolve_buffers`.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
from holosoma.fada.common.dataset import ReplayBuffer
from holosoma.fada.planner_idm.config import FADAConfig
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.fada.planner_idm.trainer import ExpertPolicyWrapper, FADATrainer
from holosoma.utils.safe_torch_import import torch

PROBE_SEED = 20260827
PROBE_OBS_DIM = 45
PROBE_ACT_DIM = 23
PROBE_CMD_DIM = 5
PROBE_EPISODES = 8
PROBE_EPISODE_LEN = 120

# batch_size for the mixed-mode probe. Must divide exactly by every ratio in
# build_probe_config_mixed() so _idm_batch_counts()/_planner_batch_counts() produce
# integers rather than round()-ed values. Ratios used: 0.375 (=3/8), 0.5 (=1/2),
# 0.75 (=3/4), 0.2 (=1/5); LCM(8, 2, 4, 5) = 40, giving
#   40*0.375=15, 40*0.5=20, 40*0.75=30, 40*0.2=8.
PROBE_MIXED_BATCH_SIZE = 40

# Fixed placeholder, never dereferenced. FADATrainer.__init__ only resolves this path
# string (Path.expanduser().resolve()) into a config bookkeeping field and never opens
# it, so the file need not exist. validate() rejects an empty string for this field.
_PROBE_EXPERT_CHECKPOINT = "unused_probe_expert_checkpoint/model_9999.pt"

# compact_obs term_scale/term_noise must be non-None dicts covering exactly
# COMPACT_TERM_ORDER (see holosoma.fada.common.compact_obs). Their numeric
# content is irrelevant here because the probe buffer is pre-filled with synthetic
# obs directly -- no env-side compact-obs extraction ever runs.
_PROBE_COMPACT_TERMS = {
    "base_ang_vel": 1.0,
    "dof_pos": 1.0,
    "dof_vel": 1.0,
    "projected_gravity": 1.0,
}
_PROBE_COMPACT_TERM_NOISE = {
    "base_ang_vel": 0.0,
    "dof_pos": 0.0,
    "dof_vel": 0.0,
    "projected_gravity": 0.0,
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _probe_config_kwargs() -> dict[str, Any]:
    """Shared base kwargs for both probe configs.

    These select `FADATrainer._train_step_separate_pass`, the path
    `holosoma.fada.planner_idm.train` runs; the merged single-pass branch is
    constant-folded away in this release.
    """
    return {
        "expert_checkpoint": _PROBE_EXPERT_CHECKPOINT,
        # FADAConfig.run_name defaults to a factory embedding dt.datetime.now(), the one
        # non-deterministic default in FADAConfig. It is a str, so it passes the
        # isinstance(v, (int, float, str, bool)) filter in _run_probe and would vary
        # config_fingerprint between runs. Pinned here.
        "run_name": "bitexact_probe_fixed_run_name",
        "output_dir": "",
        "history_len": 30,
        "pred_horizon": 6,
        "planner_use_action_history": False,
        "planner_predict_delta": True,
        "planner_obs_loss_coef": 0.0,
        "planner_d_model": 32,
        "planner_nhead": 2,
        "planner_num_layers": 1,
        "planner_dim_feedforward": 64,
        "planner_dropout": 0.0,
        "idm_d_model": 32,
        "idm_nhead": 2,
        "idm_encoder_num_layers": 1,
        "idm_decoder_num_layers": 1,
        "idm_dim_feedforward": 64,
        "idm_dropout": 0.0,
        "idm_use_teacher_forcing": True,
        "idm_teacher_forcing_ratio": 1.0,
        "idm_detach_planner_future_obs": True,
        "idm_action_loss_coef": 1.0,
        "idm_action_loss_horizon_gamma": 0.0,
        "idm_use_current_command_for_history": False,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "augment_obs_noise": False,
        "augment_action_noise_level": 0.0,
        "augment_history_mask_ratio": 0.0,
        "augment_future_mask_ratio": 0.0,
        "mixed_batch_offline_ratio": 0.5,
        "online_trajectory_batch_ratio": 0.5,
        "suboptimal_data_ratio": 0.0,
        "idm_suboptimal_batch_ratio": 0.0,
        "planner_suboptimal_batch_ratio": 0.0,
        "suboptimal_expert_batch_ratio": 0.0,
        "warmup_idm_suboptimal_batch_ratio": 0.0,
        "warmup_planner_suboptimal_batch_ratio": 0.0,
        "replay_capacity": 100_000,
        "seed": PROBE_SEED,
        "device": "cpu",
        "wandb_enable": False,
        "compact_obs_term_scale": dict(_PROBE_COMPACT_TERMS),
        "compact_obs_term_noise": dict(_PROBE_COMPACT_TERM_NOISE),
    }


def build_probe_config() -> FADAConfig:
    """The fixed, deterministic config for the offline/warmup path.

    The values mirror the shipped recipe, scaled down. Every suboptimal ratio is
    passed explicitly as 0.0 rather than left to the shipped defaults, so the probe
    stays pinned to the golden if those defaults change. Under
    `train_step(mode="offline")` the ratios in effect are
    `warmup_*_suboptimal_batch_ratio`, also 0.0, so this variant exercises none of
    the suboptimal/online routing -- see `build_probe_config_mixed()` /
    `run_train_probe_mixed()` for that.
    """
    return FADAConfig(**_probe_config_kwargs()).validate()


def build_probe_config_mixed() -> FADAConfig:
    """The fixed, deterministic config for the mixed/DAgger-online path, with every
    suboptimal/online ratio non-zero.

    Covers the full routing through `_idm_batch_counts` / `_planner_batch_counts` under
    `mode="mixed"`: all five IDM sources -- optimal, suboptimal, suboptimal_expert,
    online and online_trajectory -- produce a non-zero batch_size (see the derivation
    in `PROBE_MIXED_BATCH_SIZE`'s comment).
    """
    overrides = {
        "batch_size": PROBE_MIXED_BATCH_SIZE,
        "mixed_batch_offline_ratio": 0.2,
        "online_trajectory_batch_ratio": 0.5,
        "suboptimal_data_ratio": 1.0,
        "idm_suboptimal_batch_ratio": 0.375,
        "planner_suboptimal_batch_ratio": 0.375,
        "suboptimal_expert_batch_ratio": 0.5,
        "warmup_idm_suboptimal_batch_ratio": 0.75,
        "warmup_planner_suboptimal_batch_ratio": 0.75,
    }
    kwargs = {**_probe_config_kwargs(), **overrides}
    return FADAConfig(**kwargs).validate()


def _fill_buffer(buffer: ReplayBuffer, cfg: FADAConfig, rng: np.random.Generator) -> None:
    """Fill the buffer with deterministic synthetic data.

    The fields and shapes follow how
    `src/holosoma/tests/fada/test_planner_idm_generalized_trainer.py` builds them,
    via `ReplayBuffer.add_batch` (`ReplayBuffer` has no `add_episode`). Values are
    `rng` draws rather than constants, so samples differ from one another. Each
    episode is `PROBE_EPISODE_LEN` steps with `done` set only on the last step.
    """
    for episode in range(PROBE_EPISODES):
        n = PROBE_EPISODE_LEN
        done = np.zeros((n,), dtype=bool)
        done[-1] = True
        buffer.add_batch(
            obs=rng.standard_normal((n, PROBE_OBS_DIM)).astype(np.float32),
            current_command=rng.standard_normal((n, PROBE_CMD_DIM)).astype(np.float32),
            executed_act=rng.standard_normal((n, PROBE_ACT_DIM)).astype(np.float32),
            expert_act=rng.standard_normal((n, PROBE_ACT_DIM)).astype(np.float32),
            expert_chunk=rng.standard_normal((n, cfg.pred_horizon, PROBE_ACT_DIM)).astype(np.float32),
            expert_future_obs_chunk=rng.standard_normal((n, cfg.pred_horizon, PROBE_OBS_DIM)).astype(np.float32),
            strict_label_valid=np.ones((n,), dtype=bool),
            reward=rng.standard_normal((n,)).astype(np.float32),
            done=done,
            env_id=np.full((n,), episode, dtype=np.int32),
            episode_id=np.full((n,), episode, dtype=np.int64),
        )


class _ProbeDummyAlgo:
    """Minimal stand-in for a loaded PPO algo, mirroring _DummyAlgo in
    test_planner_idm_generalized_trainer.py. Used only to build an ExpertPolicyWrapper
    whose .policy is never invoked: train_step reads pre-filled buffer chunks and does
    not call the expert policy."""

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


def _build_probe_expert_wrapper(act_dim: int) -> ExpertPolicyWrapper:
    # ExpertPolicyWrapper.__init__ expects a loaded algo with a checkpoint on disk, so
    # it is bypassed via object.__new__ and the attributes it would set are assigned
    # directly; the probe loads no expert checkpoint.
    wrapper = object.__new__(ExpertPolicyWrapper)
    wrapper.algo = _ProbeDummyAlgo(act_dim=act_dim)
    wrapper.policy = wrapper.algo.get_inference_policy()
    wrapper.actor_obs_keys = ["actor_obs"]
    return wrapper


def _build_probe_trainer(
    cfg: FADAConfig,
    seed: int,
    *,
    fill_suboptimal_and_online: bool = False,
) -> FADATrainer:
    """Build a CPU-only `FADATrainer` whose buffers are already filled.

    Construction follows `_build_trainer` / `_build_buffer` / `_build_expert_wrapper`
    in `test_planner_idm_generalized_trainer.py`. The optimizers are built for the
    separate IDM/planner pass: `optimizer` holds only `planner_parameters()`,
    `idm_optimizer` only `idm_parameters()`, and the two are stepped separately --
    matching the two-stage update in `FADATrainer._train_step_separate_pass`.

    With `fill_suboptimal_and_online=False` (the offline probe), `offline_buffer` is the
    only buffer that needs real synthetic data: the cfg sets both
    `warmup_idm_suboptimal_batch_ratio` and `warmup_planner_suboptimal_batch_ratio` to
    0.0, so under `train_step(mode="offline")` the sample counts drawn from
    `suboptimal_offline_buffer` and `online_buffer` are always 0 and those two can stay
    empty (they exist only to satisfy `ReplayBuffer`'s construction constraints).

    With `fill_suboptimal_and_online=True` (the mixed probe) all three buffers are
    filled, each from its own rng seed (`seed` / `seed+1` / `seed+2`), so their
    contents differ.
    """
    total_steps = PROBE_EPISODES * PROBE_EPISODE_LEN
    online_capacity = total_steps if fill_suboptimal_and_online else 1
    suboptimal_capacity = total_steps if fill_suboptimal_and_online else 1

    offline_buffer = ReplayBuffer(
        capacity=total_steps,
        obs_dim=PROBE_OBS_DIM,
        act_dim=PROBE_ACT_DIM,
        cmd_dim=PROBE_CMD_DIM,
        history_len=cfg.history_len,
        pred_horizon=cfg.pred_horizon,
        require_future_obs_targets=True,
        growable=True,
    )
    _fill_buffer(offline_buffer, cfg, np.random.default_rng(seed))

    suboptimal_offline_buffer = ReplayBuffer(
        capacity=suboptimal_capacity,
        obs_dim=PROBE_OBS_DIM,
        act_dim=PROBE_ACT_DIM,
        cmd_dim=PROBE_CMD_DIM,
        history_len=cfg.history_len,
        pred_horizon=cfg.pred_horizon,
        require_future_obs_targets=True,
        growable=True,
    )
    online_buffer = ReplayBuffer(
        capacity=online_capacity,
        obs_dim=PROBE_OBS_DIM,
        act_dim=PROBE_ACT_DIM,
        cmd_dim=PROBE_CMD_DIM,
        history_len=cfg.history_len,
        pred_horizon=cfg.pred_horizon,
        require_future_obs_targets=True,
        growable=True,
    )
    if fill_suboptimal_and_online:
        _fill_buffer(suboptimal_offline_buffer, cfg, np.random.default_rng(seed + 1))
        _fill_buffer(online_buffer, cfg, np.random.default_rng(seed + 2))

    model = PlannerIDMPolicy(
        obs_dim=PROBE_OBS_DIM,
        act_dim=PROBE_ACT_DIM,
        cmd_dim=PROBE_CMD_DIM,
        history_len=cfg.history_len,
        pred_horizon=cfg.pred_horizon,
        planner_d_model=cfg.planner_d_model,
        planner_nhead=cfg.planner_nhead,
        planner_num_layers=cfg.planner_num_layers,
        planner_dim_feedforward=cfg.planner_dim_feedforward,
        planner_dropout=cfg.planner_dropout,
        planner_use_action_history=cfg.planner_use_action_history,
        planner_predict_delta=cfg.planner_predict_delta,
        idm_d_model=cfg.idm_d_model,
        idm_nhead=cfg.idm_nhead,
        idm_encoder_num_layers=cfg.idm_encoder_num_layers,
        idm_decoder_num_layers=cfg.idm_decoder_num_layers,
        idm_dim_feedforward=cfg.idm_dim_feedforward,
        idm_dropout=cfg.idm_dropout,
        idm_use_current_command_for_history=cfg.idm_use_current_command_for_history,
    )

    # Separate-pass mode takes two disjoint AdamW optimizers (planner-only /
    # idm-only).
    optimizer = torch.optim.AdamW(list(model.planner_parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    idm_optimizer = torch.optim.AdamW(list(model.idm_parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    return FADATrainer(
        cfg=cfg,
        student_model=model,
        expert_policy=_build_probe_expert_wrapper(act_dim=PROBE_ACT_DIM),
        offline_buffer=offline_buffer,
        suboptimal_offline_buffer=suboptimal_offline_buffer,
        online_buffer=online_buffer,
        optimizer=optimizer,
        lr_scheduler=None,
        idm_optimizer=idm_optimizer,
        idm_lr_scheduler=None,
        device=torch.device("cpu"),
        obs_dim=PROBE_OBS_DIM,
        act_dim=PROBE_ACT_DIM,
        cmd_dim=PROBE_CMD_DIM,
        resolved_expert_checkpoint=_PROBE_EXPERT_CHECKPOINT,
        compact_obs_source_checkpoint=_PROBE_EXPERT_CHECKPOINT,
        strict_label_env=object(),
    )


def _state_dict_sha256(module: Any) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _run_probe(
    cfg_factory: Callable[[], FADAConfig],
    out_path: Path,
    steps: int,
    *,
    mode: str,
    fill_suboptimal_and_online: bool,
) -> dict[str, Any]:
    _seed_everything(PROBE_SEED)
    cfg = cfg_factory()

    trainer = _build_probe_trainer(cfg, PROBE_SEED, fill_suboptimal_and_online=fill_suboptimal_and_online)

    losses: list[dict[str, float]] = []
    for _ in range(steps):
        # train_step returns (loss_total: float, sample_stats: dict[str, int],
        # loss_metrics: dict[str, float]). The per-step scalar breakdown logged here
        # (loss_total, planner/idm sub-losses, ...) is the third element.
        _loss_total, _sample_stats, loss_metrics = trainer.train_step(mode=mode)
        losses.append({k: float(v) for k, v in sorted(loss_metrics.items())})

    result = {
        "loss_sequence": losses,
        "state_dict_sha256": _state_dict_sha256(trainer.student_model),
        "config_fingerprint": hashlib.sha256(
            json.dumps(
                {k: v for k, v in sorted(vars(cfg).items()) if isinstance(v, (int, float, str, bool))},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_train_probe(out_path: Path, steps: int = 20) -> dict[str, Any]:
    """Offline/warmup-path probe: train_step(mode="offline") under idm_planner_separate_pass=True."""
    return _run_probe(build_probe_config, out_path, steps, mode="offline", fill_suboptimal_and_online=False)


def run_train_probe_mixed(out_path: Path, steps: int = 20) -> dict[str, Any]:
    """Mixed/DAgger-online-path probe: train_step(mode="mixed") with all five IDM
    sources (optimal/suboptimal/suboptimal_expert/online/online_trajectory) active."""
    return _run_probe(build_probe_config_mixed, out_path, steps, mode="mixed", fill_suboptimal_and_online=True)
