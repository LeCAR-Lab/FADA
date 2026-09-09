"""Invalid-input tests for the three FADA entry points' numeric flags.

`argparse`'s `type=int` / `type=float` only checks that the token parses, so on its own a
numeric override flag accepts whatever a numeric literal can express: `nan`, negative
probabilities, dropout above 1, `--lora-r 0`, zero-valued intervals. `nan` passes every
ordinary `<` / `>` comparison, so a NaN learning rate trains to NaN weights and still
writes checkpoints, ONNX and a summary.json.

`fada/common/cli_validation.py` classifies each flag into a numeric domain and the resolver
that turns the parsed namespace into the runtime config enforces it. These tests pin:

* every value that parses as a numeric literal but has no defined behavior is rejected,
  with the flag named;
* every value the reading code's own guard treats as "off" is still accepted -- see
  `_DISABLE_SENTINEL_CASES`, which pins the whole far side of each guard, not one value;
* the domain map covers *every* numeric flag, so a new flag cannot be added unvalidated;
* no shipped default is rejected by its own domain;
* an accepted value reaches the resolved config unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from holosoma.fada.common import cli_validation
from holosoma.fada.planner_idm import config as fada_config
from holosoma.fada.planner_idm import eval_checkpoint as fada_eval
from holosoma.fada.planner_idm import finetune_idm_lora as fada_finetune

_EXPERT = "/nonexistent/oracle_run/20250101_120000_t1_oracle/model_24999.pt"
_CHECKPOINT = "/nonexistent/dagger_run/model_final.pt"
_DATASET = "/nonexistent/target.h5"


def _train(*extra: str):
    args = fada_config.build_arg_parser().parse_args(["--expert-checkpoint", _EXPERT, *extra])
    return fada_config.config_from_args(args)


def _finetune(*extra: str):
    args = fada_finetune._build_arg_parser().parse_args(
        ["--checkpoint", _CHECKPOINT, "--target-datasets", _DATASET, *extra]
    )
    return fada_finetune.resolve_finetune_defaults(args)


def _eval(*extra: str):
    args = fada_eval._build_arg_parser().parse_args(["--checkpoint", _CHECKPOINT, *extra])
    return fada_eval.resolve_eval_defaults(args)


_ENTRY_POINTS = {"train": _train, "finetune": _finetune, "eval": _eval}


# ---------------------------------------------------------------------------
# Concrete accepted-but-meaningless inputs, one per numeric domain
# ---------------------------------------------------------------------------

_REVIEWER_CASES = [
    ("train", "--lr", "nan"),
    ("train", "--planner-dropout", "1.5"),
    ("finetune", "--lora-dropout", "1.5"),
    ("finetune", "--lora-r", "0"),
    ("finetune", "--idm-action-loss-horizon-gamma", "nan"),
    ("eval", "--num-envs", "0"),
    ("eval", "--max-steps", "0"),
    ("eval", "--future-obs-mask-step", "-99"),
]


@pytest.mark.parametrize(("entry_point", "flag", "value"), _REVIEWER_CASES)
def test_reviewer_reported_invalid_values_are_rejected(entry_point: str, flag: str, value: str) -> None:
    with pytest.raises(ValueError, match=r"--") as excinfo:
        _ENTRY_POINTS[entry_point](flag, value)
    message = str(excinfo.value)
    assert flag in message, f"the error must name the flag: {message}"
    assert value in message or value.lstrip("-") in message, f"the error must show the value: {message}"
    assert "expected" in message, f"the error must state the allowed range: {message}"


# ---------------------------------------------------------------------------
# Non-finite values everywhere, and the other domain classes
# ---------------------------------------------------------------------------

_NON_FINITE_CASES = [
    ("train", "--lr", "inf"),
    ("train", "--weight-decay", "nan"),
    ("train", "--grad-clip", "nan"),
    ("train", "--strict-label-worker-timeout-s", "inf"),
    ("finetune", "--lr", "nan"),
    ("finetune", "--grad-clip", "inf"),
    ("finetune", "--lora-alpha", "nan"),
]


@pytest.mark.parametrize(("entry_point", "flag", "value"), _NON_FINITE_CASES)
def test_non_finite_values_are_rejected_even_where_the_sign_is_free(entry_point: str, flag: str, value: str) -> None:
    """`--grad-clip` accepts any sign (`<=0` disables clipping), but never nan/inf."""
    with pytest.raises(ValueError, match="finite"):
        _ENTRY_POINTS[entry_point](flag, value)


_OUT_OF_RANGE_CASES = [
    ("train", "--idm-dropout", "1.0"),
    ("train", "--augment-history-mask-ratio", "1.0"),
    ("train", "--mixed-batch-offline-ratio", "0"),
    ("train", "--mixed-batch-offline-ratio", "1"),
    ("train", "--online-rollout-sample-ratio", "0"),
    ("train", "--batch-size", "0"),
    ("train", "--history-len", "0"),
    ("train", "--pred-horizon", "-1"),
    ("train", "--num-envs", "-4"),
    ("train", "--seed", "-1"),
    ("train", "--replay-capacity", "0"),
    ("train", "--planner-obs-loss-coef", "-0.5"),
    ("train", "--suboptimal-num-checkpoints", "0"),
    # 1.0 is degenerate (everything held out), not a documented "off" -- unlike the
    # non-positive side of the same flag, which _DISABLE_SENTINEL_CASES pins as valid.
    ("train", "--val-trajectory-ratio", "1.0"),
    ("finetune", "--batch-size", "0"),
    ("finetune", "--max-train-steps", "0"),
    ("finetune", "--pred-horizon-k", "0"),
    # resolve_finetune_recipe() raises on a negative resolved grad clip, so the flag
    # rejects it as well.
    ("finetune", "--grad-clip", "-1"),
    ("finetune", "--trim-head-steps", "-1"),
    ("finetune", "--trim-tail-steps", "-5"),
    ("eval", "--num-episodes", "0"),
    ("eval", "--open-loop-deploy-steps", "0"),
    ("eval", "--seed", "-1"),
    # One-based horizon index: `_apply_future_obs_mask` rejects anything below 1, so 0 is
    # not an "off" sentinel.
    ("eval", "--future-obs-mask-step", "0"),
]


@pytest.mark.parametrize(("entry_point", "flag", "value"), _OUT_OF_RANGE_CASES)
def test_out_of_range_values_are_rejected(entry_point: str, flag: str, value: str) -> None:
    with pytest.raises(ValueError, match=r"--"):
        _ENTRY_POINTS[entry_point](flag, value)


_ACCEPTED_EDGE_CASES = [
    # 0 is the documented "off"/"auto" value for these, so it must stay accepted.
    ("train", "--dagger-iters", "0", "dagger_iters", 0),
    ("train", "--train-steps-per-iter", "0", "train_steps_per_iter", 0),
    ("train", "--warmup-train-steps", "0", "warmup_train_steps", 0),
    ("train", "--suboptimal-data-ratio", "0", "suboptimal_data_ratio", 0.0),
    ("train", "--dagger-sigma", "0", "dagger_sigma", 0.0),
    ("train", "--augment-action-noise-level", "0", "augment_action_noise_level", 0.0),
    ("train", "--planner-obs-loss-coef", "0", "planner_obs_loss_coef", 0.0),
    ("train", "--grad-clip", "-1", "grad_clip", -1.0),
    ("train", "--seed", "0", "seed", 0),
    ("finetune", "--grad-clip", "0", "grad_clip", 0.0),
    ("eval", "--force-scale-eval-level", "-1", "force_scale_eval_level", -1),
]


@pytest.mark.parametrize(("entry_point", "flag", "value", "field", "expected"), _ACCEPTED_EDGE_CASES)
def test_documented_disabled_and_boundary_values_still_work(
    entry_point: str, flag: str, value: str, field: str, expected: object
) -> None:
    """Validation must not narrow the accepted set: "0 means off" stays usable."""
    resolved = _ENTRY_POINTS[entry_point](flag, value)
    assert getattr(resolved, field) == expected


# ---------------------------------------------------------------------------
# Disable sentinels: the *whole* side of each guard, not one tidy value
# ---------------------------------------------------------------------------
#
# Each field below is read behind a single guard -- `if level <= 0.0: return`, `if sigma >
# 0.0:`, `if interval_steps <= 0: return`, `if force_level >= 0:`, `if val_ratio > 0.0:`,
# `x if x > 0 else <fallback>` -- so every value on the far side of that guard behaves
# identically to the sentinel a reader would naturally type, and all of them are accepted.
# The `(file, guard)` comment on each row names the guard that defines it.
_DISABLE_SENTINEL_CASES = [
    # trainer_batching._apply_action_noise: `if level <= 0.0: return actions`
    ("train", "--augment-action-noise-level", "-2", "augment_action_noise_level", -2.0),
    # trainer_rollout: `if sigma > 0.0:`
    ("train", "--dagger-sigma", "-1", "dagger_sigma", -1.0),
    # trainer._set_command_resampling_interval_from_steps: `if interval_steps <= 0: return`
    ("train", "--command-resample-interval", "0", "command_resample_interval", 0),
    ("train", "--command-resample-interval", "-5", "command_resample_interval", -5),
    ("eval", "--command-resample-interval", "0", "command_resample_interval", 0),
    # trainer: `if val_ratio > 0.0:` / `if val_ratio <= 0.0:`
    ("train", "--val-trajectory-ratio", "-0.25", "val_trajectory_ratio", -0.25),
    # eval_checkpoint: `if int(D.force_scale_eval_level) >= 0:` -- the flag's own help says
    # "<0 disables it", so any negative value behaves like -1.
    ("eval", "--force-scale-eval-level", "-2", "force_scale_eval_level", -2),
    ("eval", "--force-scale-eval-level", "-99", "force_scale_eval_level", -99),
]


@pytest.mark.parametrize(("entry_point", "flag", "value", "field", "expected"), _DISABLE_SENTINEL_CASES)
def test_every_value_the_original_guard_treats_as_off_is_still_accepted(
    entry_point: str, flag: str, value: str, field: str, expected: object
) -> None:
    resolved = _ENTRY_POINTS[entry_point](flag, value)
    assert getattr(resolved, field) == expected


def test_disable_sentinel_domains_carry_a_note_instead_of_a_lower_bound() -> None:
    """These domains declare no lower bound and carry a `disabled_note` documenting the sentinel."""
    domains = {
        ("train", "augment_action_noise_level"): fada_config._NUMERIC_DOMAINS,
        ("train", "dagger_sigma"): fada_config._NUMERIC_DOMAINS,
        ("train", "command_resample_interval"): fada_config._NUMERIC_DOMAINS,
        ("train", "val_trajectory_ratio"): fada_config._NUMERIC_DOMAINS,
        ("eval", "command_resample_interval"): fada_eval._NUMERIC_DOMAINS,
        ("eval", "force_scale_eval_level"): fada_eval._NUMERIC_DOMAINS,
    }
    for (label, field), table in domains.items():
        domain = table[field]
        assert domain.lo is None, f"{label}.{field} regained a lower bound"
        assert domain.disabled_note, f"{label}.{field} has no disabled_note to document the sentinel"


def test_the_command_resample_interval_note_denies_the_obvious_misreading() -> None:
    """0 leaves the env's own resampling interval in place rather than disabling resampling."""
    for table in (fada_config._NUMERIC_DOMAINS, fada_eval._NUMERIC_DOMAINS):
        note = table["command_resample_interval"].disabled_note
        assert "does not mean 'never resample'" in note


# ---------------------------------------------------------------------------
# Coverage and default-compatibility of the domain maps
# ---------------------------------------------------------------------------

_SPECS = [
    ("train", fada_config._OVERRIDABLE_FIELDS, fada_config._NUMERIC_DOMAINS),
    ("finetune", fada_finetune._OVERRIDABLE_DEFAULTS, fada_finetune._NUMERIC_DOMAINS),
    ("eval", fada_eval._OVERRIDABLE_DEFAULTS, fada_eval._NUMERIC_DOMAINS),
]


@pytest.mark.parametrize(("label", "spec", "domains"), _SPECS)
def test_every_numeric_flag_has_a_declared_domain(label: str, spec: tuple, domains: dict) -> None:
    """Every `int`/`float` flag in the spec table has an entry in the domain map."""
    numeric = {name for name, arg_type, _choices, _help in spec if arg_type in (int, float)}
    missing = sorted(numeric - set(domains))
    assert not missing, f"{label}: numeric flags without a validation domain: {missing}"


@pytest.mark.parametrize(("label", "spec", "domains"), _SPECS)
def test_domain_map_has_no_entries_for_flags_that_do_not_exist(label: str, spec: tuple, domains: dict) -> None:
    declared = {name for name, _t, _c, _h in spec}
    if label == "finetune":
        # These two are declared explicitly rather than generated from the spec table.
        declared |= {"trim_head_steps", "trim_tail_steps"}
    stale = sorted(set(domains) - declared)
    assert not stale, f"{label}: domains for non-existent flags: {stale}"


def test_no_shipped_training_default_is_rejected_by_its_own_domain() -> None:
    defaults = {
        field.name: field.default
        for field in dataclasses.fields(fada_config.FADAConfig)
        if field.default is not dataclasses.MISSING
    }
    for name, domain in fada_config._NUMERIC_DOMAINS.items():
        value = defaults.get(name)
        if value is None:
            continue
        domain.check(value, field=name)


@pytest.mark.parametrize(
    ("defaults_class", "domains"),
    [
        (fada_finetune._FinetuneDefaults, fada_finetune._NUMERIC_DOMAINS),
        (fada_eval._EvalDefaults, fada_eval._NUMERIC_DOMAINS),
    ],
)
def test_no_shipped_default_is_rejected_by_its_own_domain(defaults_class: type, domains: dict) -> None:
    for name, domain in domains.items():
        value = getattr(defaults_class, name, None)
        if value is None:
            continue
        domain.check(value, field=name)


def test_help_text_documents_the_range_and_the_disabled_sentinel() -> None:
    # argparse hard-wraps help text, so compare on whitespace-normalized output.
    help_text = " ".join(fada_finetune._build_arg_parser().format_help().split())
    assert "0 disables gradient clipping" in help_text
    assert "an integer >= 1" in help_text
    train_help = " ".join(fada_config.build_arg_parser().format_help().split())
    assert "a finite float in [0, 1)" in train_help
    assert "0 disables the weak-policy (suboptimal) data source" in train_help


# ---------------------------------------------------------------------------
# Valid values are untouched
# ---------------------------------------------------------------------------


def test_a_valid_run_resolves_exactly_as_before() -> None:
    """Validation is a gate, not a transform.

    An accepted value reaches the config unchanged, and a no-flag invocation equals the
    shipped defaults.
    """
    baseline = dataclasses.asdict(_train())
    reference = dataclasses.asdict(
        fada_config.config_from_args(fada_config.build_arg_parser().parse_args(["--expert-checkpoint", _EXPERT]))
    )
    assert baseline == reference

    overridden = dataclasses.asdict(_train("--lr", "1.5e-5"))
    changed = {key for key in baseline if baseline[key] != overridden[key]}
    assert changed == {"lr"}
    assert overridden["lr"] == 1.5e-5


def test_augment_obs_noise_overrides_rejects_non_finite_and_negative_levels() -> None:
    for bad in ("dof_pos=nan", "dof_pos=inf", "dof_pos=-0.1"):
        with pytest.raises(SystemExit):
            _train("--augment-obs-noise-overrides", bad)
    cfg = _train("--augment-obs-noise-overrides", "dof_pos=0.02,projected_gravity=0")
    assert cfg.augment_obs_noise_overrides == {"dof_pos": 0.02, "projected_gravity": 0.0}


def test_fixed_ee_force_triplets_reject_non_finite_components() -> None:
    with pytest.raises(SystemExit):
        _eval("--fixed-ee-force-left", "0,0,nan")
    resolved = _eval("--fixed-ee-force-left", "0,0,-30")
    assert resolved.fixed_ee_force_left == (0.0, 0.0, -30.0)


# ---------------------------------------------------------------------------
# The shared validator itself
# ---------------------------------------------------------------------------


def test_numeric_range_rejects_booleans_and_non_integral_ints() -> None:
    with pytest.raises(ValueError, match="not a boolean"):
        cli_validation.POSITIVE_INT.check(True, field="num_envs")
    with pytest.raises(ValueError, match="an integer"):
        cli_validation.POSITIVE_INT.check(2.5, field="num_envs")


def test_numeric_range_describe_covers_every_bound_shape() -> None:
    assert cli_validation.FINITE_FLOAT.describe() == "a finite float"
    assert cli_validation.POSITIVE_FLOAT.describe() == "a finite float > 0"
    assert cli_validation.PROBABILITY.describe() == "a finite float in [0, 1]"
    assert cli_validation.UNIT_OPEN.describe() == "a finite float in (0, 1)"
    assert cli_validation.POSITIVE_INT.describe() == "an integer >= 1"
    assert cli_validation.NumericRange("float", hi=1.0).describe() == "a finite float <= 1"


def test_validate_numeric_values_skips_unset_and_unknown_fields() -> None:
    spec = {"a": cli_validation.POSITIVE_INT}
    cli_validation.validate_numeric_values(spec, {"a": None, "b": -5})
    with pytest.raises(ValueError, match="--a"):
        cli_validation.validate_numeric_values(spec, {"a": 0})


def test_numeric_range_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        cli_validation.NumericRange("decimal")


def test_flag_spelling_matches_the_generated_parser_flags() -> None:
    parser_flags = set()
    for action in fada_config.build_arg_parser()._actions:
        parser_flags.update(action.option_strings)
    for field in fada_config._NUMERIC_DOMAINS:
        assert cli_validation.flag_spelling(field) in parser_flags, field


def test_a_hand_built_namespace_is_validated_too() -> None:
    """The check lives in the resolver, not in `argparse`'s `type=`.

    Callers that build a namespace directly (tests, tooling) go through it too.
    """
    namespace = argparse.Namespace(checkpoint=_CHECKPOINT, target_datasets=[_DATASET], lora_r=0)
    for name, _t, _c, _h in fada_finetune._OVERRIDABLE_DEFAULTS:
        if not hasattr(namespace, name):
            setattr(namespace, name, None)
    with pytest.raises(ValueError, match="--lora-r"):
        fada_finetune.resolve_finetune_defaults(namespace)
