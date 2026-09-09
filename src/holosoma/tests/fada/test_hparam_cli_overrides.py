"""Override-path tests for the FADA entry points' hyperparameter CLI flags.

`FADAConfig` / `_FinetuneDefaults` / `_EvalDefaults` hold the defaults; the flags on top
of those classes are an *override* path: passing nothing leaves those defaults untouched,
and passing a flag changes the value the code consumes -- not merely the value argparse
parsed into a Namespace.

The tests come in pairs: a "no flags -> class default" assertion, and a "flag -> the
consumer saw it" assertion that goes through a real consumer (`build_student_policy`,
`FADATrainer.train_step`, `finetune_idm_lora.main`) rather than re-reading the Namespace.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from holosoma.fada.common.dataset import ReplayBuffer
from holosoma.fada.planner_idm import config as fada_config
from holosoma.fada.planner_idm import eval_checkpoint as fada_eval_checkpoint
from holosoma.fada.planner_idm import finetune_idm_lora as fada_finetune
from holosoma.fada.planner_idm.config import (
    FADAConfig,
    build_arg_parser,
    config_from_args,
)
from holosoma.fada.planner_idm.model import PlannerIDMPolicy, build_student_policy
from holosoma.fada.planner_idm.trainer import ExpertPolicyWrapper, FADATrainer
from holosoma.utils.safe_torch_import import torch

_EXPERT = "/nonexistent/oracle_run/20250101_120000_t1_oracle/model_24999.pt"


# ---------------------------------------------------------------------------
# Training entry point (config.py / train.py)
# ---------------------------------------------------------------------------


def _train_cfg(*extra_args: str) -> FADAConfig:
    args = build_arg_parser().parse_args(["--expert-checkpoint", _EXPERT, *extra_args])
    return config_from_args(args)


def test_no_flags_reproduces_every_fadaconfig_release_default() -> None:
    """A no-flag invocation equals `FADAConfig` built from only the run-identity fields.

    The `default=None` sentinel on every hyperparameter flag is what makes that hold: an
    argparse default carrying the literal release value would pin it in a second place.
    """
    cfg = _train_cfg()
    reference = FADAConfig(
        expert_checkpoint=_EXPERT,
        run_name=cfg.run_name,
        output_dir=cfg.output_dir,
    )
    fada_config._derive_path_fields(reference)

    resolved = dataclasses.asdict(cfg)
    expected = dataclasses.asdict(reference)
    differing = {key: (resolved[key], expected[key]) for key in expected if resolved[key] != expected[key]}
    assert not differing, f"no-flag invocation drifted from the FADAConfig defaults: {differing}"


def test_overridable_fields_are_real_fadaconfig_fields_and_exclude_derived_dimensions() -> None:
    """`_OVERRIDABLE_FIELDS` names only real `FADAConfig` fields, and no runtime-derived one.

    train.py overwrites `obs_dim`/`act_dim`/`cmd_dim` and the two compact-obs dicts at
    startup from the expert checkpoint and its environment, so a flag for any of them
    would have no effect.
    """
    field_names = {f.name for f in dataclasses.fields(FADAConfig)}
    exposed = {name for name, _t, _c, _h in fada_config._OVERRIDABLE_FIELDS}

    assert exposed <= field_names, f"flags for non-existent FADAConfig fields: {sorted(exposed - field_names)}"
    assert not exposed & {
        "obs_dim",
        "act_dim",
        "cmd_dim",
        "compact_obs_term_scale",
        "compact_obs_term_noise",
    }, "runtime-derived fields must not get flags -- train.py overwrites them at startup"
    # Already declared explicitly by build_arg_parser(); a second flag would collide.
    assert not exposed & {"expert_checkpoint", "run_name", "output_dir", "device", "wandb_mode"}
    # Not read on the release path (see _OVERRIDABLE_FIELDS' comment for each).
    assert not exposed & {
        "planner_mlp_hidden_dims",
        "idm_encoder_mlp_hidden_dims",
        "idm_decoder_mlp_hidden_dims",
        "idm_mlp_hidden_dims",
        "idm_decoder_hidden_dims",
        "idm_decoder_activation",
        "obs_loss_coef",
        "save_online_buffer",
        "eval_num_episodes",
        "eval_command_resample_interval",
        "eval_exp_name",
        "wandb_run_name",
    }
    # Overridden per pass by `_train_step_separate_pass`, so the cfg value is not read
    # on the supported path.
    assert not exposed & {
        "idm_use_teacher_forcing",
        "idm_teacher_forcing_ratio",
        "idm_detach_planner_future_obs",
    }
    # Every reader spells it `strict_chunk_labels or predict_future_obs`, and
    # predict_future_obs is mandatory-true, so `--strict-chunk-labels false` could not
    # disable anything.
    assert "strict_chunk_labels" not in exposed


@pytest.mark.parametrize(
    ("flag", "value", "field", "expected"),
    [
        ("--lr", "1.5e-5", "lr", 1.5e-5),
        ("--batch-size", "8", "batch_size", 8),
        ("--weight-decay", "0.25", "weight_decay", 0.25),
        ("--dagger-iters", "3", "dagger_iters", 3),
        ("--num-envs", "7", "num_envs", 7),
        ("--seed", "1234", "seed", 1234),
        ("--replay-capacity", "4096", "replay_capacity", 4096),
        ("--suboptimal-checkpoint-sampling", "logspace", "suboptimal_checkpoint_sampling", "logspace"),
        ("--online-buffer-eviction-policy", "fifo", "online_buffer_eviction_policy", "fifo"),
        ("--headless", "false", "headless", False),
        ("--augment-obs-noise", "false", "augment_obs_noise", False),
        ("--planner-obs-loss-coef", "0.25", "planner_obs_loss_coef", 0.25),
    ],
)
def test_single_flag_changes_resolved_config_and_nothing_else(
    flag: str, value: str, field: str, expected: object
) -> None:
    baseline = dataclasses.asdict(_train_cfg())
    overridden = dataclasses.asdict(_train_cfg(flag, value))

    assert overridden[field] == expected
    changed = {key for key in baseline if baseline[key] != overridden[key]}
    assert changed == {field}, f"{flag} changed more than {field}: {sorted(changed)}"


def test_dict_valued_flag_uses_key_equals_value_spelling() -> None:
    cfg = _train_cfg("--augment-obs-noise-overrides", "dof_pos=0.5, projected_gravity=0.03")
    assert cfg.augment_obs_noise_overrides == {"dof_pos": 0.5, "projected_gravity": 0.03}

    with pytest.raises(SystemExit):
        _train_cfg("--augment-obs-noise-overrides", "dof_pos")


def test_boolean_flags_require_an_explicit_value() -> None:
    """`--headless` with no value is an error; `true`/`false` set it.

    `store_true` cannot express "leave the (already-True) default alone", which the
    `default=None` sentinel scheme depends on.
    """
    with pytest.raises(SystemExit):
        _train_cfg("--headless")
    assert _train_cfg("--headless", "true").headless is True
    assert _train_cfg("--headless", "false").headless is False


def test_derived_path_flags_win_over_the_output_dir_derivation() -> None:
    """`offline_cache_path`/`warmup_ckpt_path`/`wandb_project` are recomputed by
    `_derive_path_fields()` from --output-dir/--run-name, and an explicit flag is
    applied *after* that derivation.
    """
    default_cfg = _train_cfg("--run-name", "t1_loco", "--output-dir", "/tmp/out")
    assert default_cfg.offline_cache_path == "/tmp/out/dataset/t1_loco_planner_idm_optimal.npz"
    assert default_cfg.wandb_project == "t1_loco_dagger"

    cfg = _train_cfg(
        "--run-name",
        "t1_loco",
        "--output-dir",
        "/tmp/out",
        "--offline-cache-path",
        "/tmp/reuse/optimal.npz",
        "--offline-suboptimal-cache-path",
        "/tmp/reuse/suboptimal.npz",
        "--warmup-ckpt-path",
        "/tmp/reuse/warmup.pt",
        "--wandb-project",
        "my_project",
    )
    assert cfg.offline_cache_path == "/tmp/reuse/optimal.npz"
    assert cfg.offline_suboptimal_cache_path == "/tmp/reuse/suboptimal.npz"
    assert cfg.warmup_ckpt_path == "/tmp/reuse/warmup.pt"
    assert cfg.wandb_project == "my_project"


def test_invalid_override_is_rejected_by_fadaconfig_validate() -> None:
    with pytest.raises(ValueError, match="mixed_batch_offline_ratio"):
        _train_cfg("--mixed-batch-offline-ratio", "1.5")


def test_model_architecture_flags_reach_build_student_policy() -> None:
    """Architecture flags reach `build_student_policy(cfg, ...)`, the call train.py makes
    to turn the config into weights, and are read back off the built model rather than
    off the Namespace.
    """
    cfg = _train_cfg(
        "--planner-d-model",
        "16",
        "--planner-nhead",
        "2",
        "--planner-num-layers",
        "1",
        "--idm-d-model",
        "16",
        "--idm-nhead",
        "2",
        "--idm-encoder-num-layers",
        "1",
        "--idm-decoder-num-layers",
        "1",
        "--history-len",
        "4",
        "--pred-horizon",
        "2",
    )
    model = build_student_policy(cfg, obs_dim=6, act_dim=3, cmd_dim=2)

    assert model.history_len == 4
    assert model.pred_horizon == 2
    widths = {int(p.shape[0]) for name, p in model.named_parameters() if name.endswith("self_attn.in_proj_bias")}
    # in_proj_bias is 3 * d_model for a torch MultiheadAttention; default d_model is 128.
    assert widths == {3 * 16}, f"unexpected attention widths {widths} -- --*-d-model did not reach the model"


# ---------------------------------------------------------------------------
# Training entry point: --batch-size reaching the real batch sampler
# ---------------------------------------------------------------------------


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


def _build_expert_wrapper(act_dim: int) -> ExpertPolicyWrapper:
    wrapper = object.__new__(ExpertPolicyWrapper)
    wrapper.algo = _DummyAlgo(act_dim=act_dim)
    wrapper.policy = wrapper.algo.get_inference_policy()
    wrapper.actor_obs_keys = ["actor_obs"]
    return wrapper


def _build_buffer(*, base: float, growable: bool = True) -> ReplayBuffer:
    buffer = ReplayBuffer(
        capacity=64,
        obs_dim=2,
        act_dim=1,
        cmd_dim=1,
        history_len=2,
        pred_horizon=1,
        require_future_obs_targets=True,
        growable=growable,
    )
    num_steps = 8
    buffer.add_batch(
        obs=np.full((num_steps, 2), base, dtype=np.float32),
        current_command=np.full((num_steps, 1), base + 0.1, dtype=np.float32),
        executed_act=np.full((num_steps, 1), base + 0.2, dtype=np.float32),
        expert_act=np.full((num_steps, 1), base + 0.3, dtype=np.float32),
        expert_chunk=np.full((num_steps, 1, 1), base + 0.3, dtype=np.float32),
        expert_future_obs_chunk=np.full((num_steps, 1, 2), base + 0.4, dtype=np.float32),
        strict_label_valid=np.ones((num_steps,), dtype=np.bool_),
        reward=np.full((num_steps,), base + 0.5, dtype=np.float32),
        done=np.array([False] * (num_steps - 1) + [True], dtype=np.bool_),
        env_id=np.zeros((num_steps,), dtype=np.int32),
        episode_id=np.zeros((num_steps,), dtype=np.int64),
    )
    return buffer


def _trainer_from_cli(tmp_path: Path, *extra_args: str) -> FADATrainer:
    cfg = _train_cfg(
        "--history-len",
        "2",
        "--pred-horizon",
        "1",
        "--planner-d-model",
        "8",
        "--planner-nhead",
        "1",
        "--planner-num-layers",
        "1",
        "--planner-dim-feedforward",
        "16",
        "--planner-dropout",
        "0.0",
        "--idm-d-model",
        "8",
        "--idm-nhead",
        "1",
        "--idm-encoder-num-layers",
        "1",
        "--idm-decoder-num-layers",
        "1",
        "--idm-dim-feedforward",
        "16",
        "--idm-dropout",
        "0.0",
        "--augment-obs-noise",
        "false",
        "--mixed-batch-offline-ratio",
        "0.5",
        "--idm-suboptimal-batch-ratio",
        "0.5",
        "--validation-batches",
        "2",
        "--wandb-enable",
        "false",
        "--offline-cache-path",
        str(tmp_path / "optimal_cache.npz"),
        "--offline-suboptimal-cache-path",
        str(tmp_path / "suboptimal_cache.npz"),
        "--warmup-ckpt-path",
        str(tmp_path / "warmup.pt"),
        *extra_args,
    )
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
        planner_dropout=cfg.planner_dropout,
        idm_d_model=cfg.idm_d_model,
        idm_nhead=cfg.idm_nhead,
        idm_encoder_num_layers=cfg.idm_encoder_num_layers,
        idm_decoder_num_layers=cfg.idm_decoder_num_layers,
        idm_dim_feedforward=cfg.idm_dim_feedforward,
        idm_dropout=cfg.idm_dropout,
    )
    # Exactly train.py's optimizer construction, so cfg.lr/cfg.weight_decay are read
    # from the config the CLI produced.
    optimizer = torch.optim.AdamW(list(model.planner_parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    idm_optimizer = torch.optim.AdamW(list(model.idm_parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
        resolved_expert_checkpoint=_EXPERT,
        compact_obs_source_checkpoint=_EXPERT,
        strict_label_env=object(),
    )


def test_batch_size_flag_reaches_the_dagger_batch_sampler(tmp_path: Path) -> None:
    """`--batch-size 8` reaches `trainer_batching._idm_batch_counts` /
    `_planner_batch_counts`, where the number is spent. The per-source batch sizes
    `train_step` reports come out of that arithmetic, so their sum is the CLI value.
    """
    trainer = _trainer_from_cli(tmp_path, "--batch-size", "8", "--lr", "0.0125")
    assert trainer.cfg.batch_size == 8
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0125)

    _, stats, _ = trainer.train_step(mode="offline")
    planner_total = (
        stats["planner_optimal_batch_size"]
        + stats["planner_suboptimal_batch_size"]
        + stats["planner_online_batch_size"]
    )
    idm_total = (
        stats["idm_optimal_batch_size"]
        + stats["idm_suboptimal_batch_size"]
        + stats["idm_suboptimal_expert_batch_size"]
        + stats["idm_online_batch_size"]
    )
    assert planner_total == 8, f"planner batch composed to {planner_total}, not the requested 8"
    assert idm_total == 8, f"IDM batch composed to {idm_total}, not the requested 8"


# ---------------------------------------------------------------------------
# Finetune entry point (finetune_idm_lora.py)
# ---------------------------------------------------------------------------


def _finetune_args(*extra_args: str):
    parser = fada_finetune._build_arg_parser()
    return parser.parse_args(["--checkpoint", "c.pt", "--target-datasets", "d.h5", *extra_args])


def test_finetune_no_flags_returns_the_release_defaults_class_itself() -> None:
    """Identity, not equality: with no flags the resolver returns the `_FinetuneDefaults`
    class object itself.
    """
    assert fada_finetune.resolve_finetune_defaults(_finetune_args()) is fada_finetune._FinetuneDefaults


def test_finetune_overrides_do_not_mutate_the_release_defaults_class() -> None:
    overridden = fada_finetune.resolve_finetune_defaults(_finetune_args("--lr", "5e-5", "--lora-r", "32"))

    assert overridden is not fada_finetune._FinetuneDefaults
    assert overridden.lr == pytest.approx(5e-5)
    assert overridden.lora_r == 32
    # Untouched fields still fall through to the release defaults...
    assert overridden.lora_alpha == pytest.approx(16.0)
    assert overridden.max_train_steps == 100
    # ...and the shipped class itself is unmodified.
    assert fada_finetune._FinetuneDefaults.lr == pytest.approx(1e-4)
    assert fada_finetune._FinetuneDefaults.lora_r == 8


def test_finetune_overrides_flow_into_the_resolved_recipe() -> None:
    """`resolve_finetune_recipe(defaults=...)` is what `main()` consumes for
    lr/weight_decay/grad_clip/LoRA; the "cli" tier wins over a checkpoint cfg carrying
    its own diverging values.
    """
    realistic_ckpt_cfg = {"lr": 0.09, "weight_decay": 0.01, "grad_clip": 1.0}
    defaults = fada_finetune.resolve_finetune_defaults(
        _finetune_args("--lr", "5e-5", "--grad-clip", "2.5", "--lora-r", "4", "--lora-target-scope", "decoder_only")
    )
    recipe = fada_finetune.resolve_finetune_recipe(train_cfg=realistic_ckpt_cfg, defaults=defaults)

    assert recipe.lr == pytest.approx(5e-5)
    assert recipe.lr_source == "cli"
    assert recipe.grad_clip == pytest.approx(2.5)
    assert recipe.grad_clip_source == "cli"
    assert recipe.lora_r == 4
    assert recipe.lora_target_scope == "decoder_only"
    # Not overridden -> still the non-CLI tier.
    assert recipe.weight_decay == pytest.approx(1e-3)


def test_finetune_inert_fields_have_no_flags() -> None:
    """The `npz_*`/budget/ratio fields belong to the removed NPZ buffer-mixing feature and
    `idm_loss_target`/`idm_anchor_lambda` are unread on the release path, so none of them
    has a flag.
    """
    exposed = {name for name, _t, _c, _h in fada_finetune._OVERRIDABLE_DEFAULTS}
    assert not exposed & {
        "npz_optimal",
        "npz_suboptimal",
        "npz_online",
        "buffer_step_budget",
        "target_step_budget",
        "idm_suboptimal_ratio",
        "idm_suboptimal_expert_ratio",
        "idm_online_trajectory_ratio",
        "idm_mixed_offline_ratio",
        "idm_loss_target",
        "idm_anchor_lambda",
        "finetune_stage",
    }
    # ...and every flag that does exist names a real _FinetuneDefaults field.
    known = set(vars(fada_finetune._FinetuneDefaults)) | set(
        getattr(fada_finetune._FinetuneDefaults, "__annotations__", {})
    )
    assert exposed <= known, f"flags for non-existent fields: {sorted(exposed - known)}"


def _write_tiny_finetune_fixture(tmp_path: Path, *, timesteps: int = 24) -> tuple[Path, Path]:
    """A minimal planner-IDM checkpoint + one-episode H5, enough for a real `main()` run.

    The checkpoint's `cfg` carries lr/weight_decay/grad_clip values that differ from the
    CLI values used below, so the CLI tier is distinguishable from the checkpoint tier.
    """
    model = PlannerIDMPolicy(
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=3,
        pred_horizon=2,
        planner_d_model=8,
        planner_nhead=1,
        planner_num_layers=1,
        planner_dim_feedforward=16,
        planner_dropout=0.0,
        planner_use_action_history=True,
        idm_d_model=8,
        idm_nhead=1,
        idm_encoder_num_layers=1,
        idm_decoder_num_layers=1,
        idm_dim_feedforward=16,
        idm_dropout=0.0,
        idm_use_current_command_for_history=True,
    )
    checkpoint_path = tmp_path / "planner_idm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "planner_state_dict": model.planner_state_dict(),
            "idm_state_dict": model.idm_state_dict(),
            "cfg": {
                "history_len": 3,
                "pred_horizon": 2,
                "planner_d_model": 8,
                "planner_nhead": 1,
                "planner_num_layers": 1,
                "planner_dim_feedforward": 16,
                "planner_dropout": 0.0,
                "planner_use_learned_positional_encoding": False,
                "planner_use_action_history": True,
                "idm_d_model": 8,
                "idm_nhead": 1,
                "idm_encoder_num_layers": 1,
                "idm_decoder_num_layers": 1,
                "idm_dim_feedforward": 16,
                "idm_dropout": 0.0,
                "idm_use_learned_positional_encoding": False,
                "idm_use_current_command_for_history": True,
                "fdm_enabled": False,
                "lr": 0.09,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
            },
            "obs_dim": 4,
            "act_dim": 2,
            "cmd_dim": 3,
            "compact_obs": {
                "term_order": ["base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
                "term_scale": dict.fromkeys(("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 1.0),
                "term_noise": dict.fromkeys(("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 0.0),
                "add_noise": False,
            },
        },
        checkpoint_path,
    )

    h5py = pytest.importorskip("h5py")
    dataset_path = tmp_path / "dataset.h5"
    with h5py.File(dataset_path, "w") as handle:
        episode = handle.create_group("episodes").create_group("episode_0000")
        episode.create_dataset("dynamics_obs", data=np.arange(timesteps * 4, dtype=np.float32).reshape(timesteps, 1, 4))
        episode.create_dataset(
            "actions", data=np.arange(timesteps * 2, dtype=np.float32).reshape(timesteps, 1, 2) + 1.0
        )
        episode.create_dataset(
            "current_command", data=np.arange(timesteps * 3, dtype=np.float32).reshape(timesteps, 1, 3) + 10.0
        )
        episode.create_dataset("dones", data=np.zeros((timesteps, 1, 1), dtype=np.bool_))
    return checkpoint_path, dataset_path


def _run_finetune_cli(checkpoint_path: Path, dataset_path: Path, output_root: Path, run_name: str, *extra: str) -> dict:
    """Run `finetune_idm_lora.main()` through argv, with no monkeypatching.

    Every knob that keeps the run tiny and offline (`--device cpu`, `--use-wandb false`,
    `--obs-key dynamics_obs`, ...) is passed as a CLI flag.  Returns the parsed
    ``summary.json``.
    """
    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_root),
        "--obs-key",
        "dynamics_obs",
        "--device",
        "cpu",
        "--use-wandb",
        "false",
        "--show-progress",
        "false",
        "--export-onnx",
        "false",
        "--save-checkpoints",
        "true",
        "--save-adapter-only",
        "true",
        "--max-train-steps",
        "2",
        *extra,
    ]
    try:
        fada_finetune.main()
    finally:
        sys.argv = old_argv
    return json.loads((output_root / run_name / "summary.json").read_text(encoding="utf-8"))


def test_finetune_cli_flags_reach_the_training_run(tmp_path: Path) -> None:
    """`--lr`, `--batch-size`, `--lora-r` and `--trim-head-steps` are read back out of the
    artifacts `main()` wrote and the adapter tensors it produced, not out of the argparse
    Namespace.
    """
    checkpoint_path, dataset_path = _write_tiny_finetune_fixture(tmp_path)
    output_root = tmp_path / "finetune_out"

    summary = _run_finetune_cli(
        checkpoint_path,
        dataset_path,
        output_root,
        "20260101_000000_cli_overrides",
        "--lr",
        "1e-5",
        "--weight-decay",
        "0.02",
        "--grad-clip",
        "0.25",
        "--batch-size",
        "4",
        "--lora-r",
        "4",
        "--lora-alpha",
        "9.0",
        "--seed",
        "7",
        "--trim-head-steps",
        "6",
    )
    run_dir = output_root / "20260101_000000_cli_overrides"

    # --lr / --weight-decay / --grad-clip: the summary's lr is the value handed to
    # optim.AdamW; `lr_source == "cli"` means it took precedence over the checkpoint cfg.
    assert summary["lr"] == pytest.approx(1e-5)
    assert summary["lr_source"] == "cli"
    assert summary["weight_decay"] == pytest.approx(0.02)
    assert summary["grad_clip"] == pytest.approx(0.25)

    # --seed / --trim-head-steps: recorded in the data provenance and used to load H5.
    assert summary["trim_head_steps"] == 6
    provenance = json.loads((run_dir / "training_data_provenance.json").read_text(encoding="utf-8"))
    assert provenance["seed"] == 7

    # --batch-size: read back off the checkpoint `main()` wrote, not off the Namespace.
    finetuned = torch.load(run_dir / "model_finetuned_idm_lora.pt", map_location="cpu", weights_only=False)
    assert finetuned["extra"]["finetune_idm_lora"]["batch_size"] == 4
    assert finetuned["extra"]["finetune_idm_lora"]["lora_r"] == 4

    # --lora-r: the injected adapter tensors themselves must have rank 4 (default 8).
    adapter = torch.load(run_dir / "idm_lora_adapters.pt", map_location="cpu", weights_only=False)
    ranks = {int(t.shape[0]) for name, t in adapter["adapter_state_dict"].items() if name.endswith(".A.weight")}
    assert ranks == {4}, f"LoRA A matrices have ranks {ranks}; --lora-r 4 did not reach the injector"


def test_finetune_trim_head_steps_actually_removes_windows(tmp_path: Path) -> None:
    """`--trim-head-steps` changes *what data is loaded*, not just what is reported.

    Two otherwise identical runs differ in window count by exactly the number of
    trimmed steps.
    """
    checkpoint_path, dataset_path = _write_tiny_finetune_fixture(tmp_path)
    output_root = tmp_path / "finetune_out"

    untrimmed = _run_finetune_cli(
        checkpoint_path, dataset_path, output_root, "20260101_000000_trim0", "--batch-size", "4"
    )
    trimmed = _run_finetune_cli(
        checkpoint_path,
        dataset_path,
        output_root,
        "20260101_000000_trim6",
        "--batch-size",
        "4",
        "--trim-head-steps",
        "6",
    )

    assert untrimmed["trim_head_steps"] == 0
    assert trimmed["trim_head_steps"] == 6
    assert trimmed["train_samples"] == untrimmed["train_samples"] - 6


# ---------------------------------------------------------------------------
# Eval / export entry point (eval_checkpoint.py)
# ---------------------------------------------------------------------------


def _eval_defaults(*extra_args: str):
    parser = fada_eval_checkpoint._build_arg_parser()
    args, _remaining = parser.parse_known_args(["--checkpoint", "c.pt", *extra_args])
    return args, fada_eval_checkpoint.resolve_eval_defaults(args)


def test_eval_no_flags_returns_the_release_defaults_class_itself() -> None:
    _args, resolved = _eval_defaults()
    assert resolved is fada_eval_checkpoint._EvalDefaults


def test_eval_overrides_are_applied_without_mutating_the_defaults_class() -> None:
    _args, resolved = _eval_defaults(
        "--num-envs",
        "8",
        "--max-steps",
        "250",
        "--num-episodes",
        "3",
        "--seed",
        "7",
        "--simulator-override",
        "mjwarp",
        "--export-onnx",
        "false",
        "--onnx-output-path",
        "/tmp/policy.onnx",
        "--fixed-ee-force-left",
        "0,0,-30",
    )

    assert (resolved.num_envs, resolved.max_steps, resolved.num_episodes, resolved.seed) == (8, 250, 3, 7)
    assert resolved.simulator_override == "mjwarp"
    assert resolved.export_onnx is False
    assert resolved.onnx_output_path == "/tmp/policy.onnx"
    assert resolved.fixed_ee_force_left == (0.0, 0.0, -30.0)
    # Untouched fields fall through; the shipped class is unmodified.
    assert resolved.command_resample_interval == 200
    assert fada_eval_checkpoint._EvalDefaults.num_envs == 1024
    assert fada_eval_checkpoint._EvalDefaults.export_onnx is True


def test_eval_flags_do_not_swallow_tyro_experiment_overrides() -> None:
    """`main()` forwards unparsed args to `tyro.cli(ExperimentConfig, ...)`; the entry
    point's own flags do not intercept them.
    """
    parser = fada_eval_checkpoint._build_arg_parser()
    _args, remaining = parser.parse_known_args(
        ["--checkpoint", "c.pt", "--num-envs", "8", "--training.num-envs", "4", "--robot.name", "t1"]
    )
    assert remaining == ["--training.num-envs", "4", "--robot.name", "t1"]


def test_eval_covers_every_eval_default_field() -> None:
    exposed = {name for name, _t, _c, _h in fada_eval_checkpoint._OVERRIDABLE_DEFAULTS}
    known = {
        name
        for name in set(vars(fada_eval_checkpoint._EvalDefaults))
        | set(getattr(fada_eval_checkpoint._EvalDefaults, "__annotations__", {}))
        if not name.startswith("_") and not callable(getattr(fada_eval_checkpoint._EvalDefaults, name, None))
    }
    assert exposed == known, f"uncovered _EvalDefaults fields: {sorted(known - exposed)}"
