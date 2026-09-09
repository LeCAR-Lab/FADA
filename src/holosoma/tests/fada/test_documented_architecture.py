"""Pins the documented FADA architecture to the code that implements it.

Covers:

* the two optional switches (`planner_use_action_history`,
  `idm_use_current_command_for_history`), checked at the config default *and* at the
  model structure that default produces;
* the ``--help`` text for those fields;
* ``pred_horizon`` versus the horizon actually supervised, computed rather than
  asserted from the formula.

The two-pass gradient behaviour is covered in
``test_planner_idm_generalized_trainer.py``.
"""

from __future__ import annotations

import dataclasses

import torch
from holosoma.fada.planner_idm.config import _OVERRIDABLE_FIELDS, FADAConfig
from holosoma.fada.planner_idm.loss_utils import weighted_horizon_mse
from holosoma.fada.planner_idm.model import PlannerIDMPolicy

#: Values read out of the stored `cfg` of both released student checkpoints.
RELEASED_STUDENT_CFG = {
    "history_len": 30,
    "pred_horizon": 6,
    "planner_use_action_history": False,
    "idm_use_current_command_for_history": False,
    "idm_detach_planner_future_obs": True,
    "idm_teacher_forcing_ratio": 1.0,
    "idm_action_loss_horizon_gamma": 0.0,
}


def _help_for(field: str) -> str:
    matches = [entry[3] for entry in _OVERRIDABLE_FIELDS if entry[0] == field]
    assert len(matches) == 1, f"{field}: expected exactly one CLI entry, got {len(matches)}"
    return matches[0]


def _tiny_policy(**overrides) -> PlannerIDMPolicy:
    kwargs = {
        "obs_dim": 3,
        "act_dim": 2,
        "cmd_dim": 4,
        "history_len": 2,
        "pred_horizon": 2,
        "planner_d_model": 8,
        "planner_nhead": 1,
        "planner_num_layers": 1,
        "planner_dim_feedforward": 16,
        "planner_dropout": 0.0,
        "idm_d_model": 8,
        "idm_nhead": 1,
        "idm_encoder_num_layers": 1,
        "idm_decoder_num_layers": 1,
        "idm_dim_feedforward": 16,
        "idm_dropout": 0.0,
    }
    kwargs.update(overrides)
    return PlannerIDMPolicy(**kwargs)


def test_the_defaults_are_the_values_the_released_students_were_trained_with() -> None:
    """Asserts the `FADAConfig` defaults equal the released students' stored `cfg` values."""
    cfg = FADAConfig()
    actual = {name: getattr(cfg, name) for name in RELEASED_STUDENT_CFG}
    assert actual == RELEASED_STUDENT_CFG


# ---------------------------------------------------------------------------
# Claim 1: the planner does not consume action history.
# ---------------------------------------------------------------------------


def test_the_planner_ignores_action_history_at_the_default() -> None:
    """Asserts that at the default the planner has no action embedding at all.

    Checked at the model structure, not only at the flag.
    """
    policy = _tiny_policy()
    assert policy.planner_use_action_history is False
    assert policy.planner.use_action_history is False
    assert policy.planner.act_embed is None

    # Enabling the switch adds the action embedding.
    enabled = _tiny_policy(planner_use_action_history=True)
    assert enabled.planner_use_action_history is True
    assert enabled.planner.act_embed is not None


def test_the_action_history_help_states_the_default() -> None:
    """Asserts the `planner_use_action_history` help text states its default.

    Flags use `default=None` as an unset sentinel, so argparse prints no default;
    the help string is where the default is stated.
    """
    help_text = _help_for("planner_use_action_history").lower()
    assert "default false" in help_text

    # The `history_len` help must name the switch that gates the action-history input.
    history_help = _help_for("history_len").lower()
    assert "planner-use-action-history" in history_help


# ---------------------------------------------------------------------------
# Claim 2: the IDM does not consume command context.
# ---------------------------------------------------------------------------


def test_the_idm_history_carries_no_command_at_the_default() -> None:
    """Asserts the IDM history tokens carry no command at the default.

    At the default, `_resolve_idm_current_command` returns None and the IDM is built
    with `use_command_history=False`, so there is no command embedding in the history
    tokens.
    """
    policy = _tiny_policy()
    assert policy.idm_use_current_command_for_history is False

    command = torch.zeros((1, 4), dtype=torch.float32)
    assert (
        PlannerIDMPolicy._resolve_idm_current_command(command, idm_use_current_command_for_history=False)
        is None
    )
    assert policy.idm.use_command_history is False
    assert policy.idm.history_cmd_embed is None

    enabled = _tiny_policy(idm_use_current_command_for_history=True)
    assert enabled.idm.history_cmd_embed is not None
    assert (
        PlannerIDMPolicy._resolve_idm_current_command(command, idm_use_current_command_for_history=True)
        is command
    )


def test_the_idm_command_help_states_the_default() -> None:
    help_text = _help_for("idm_use_current_command_for_history").lower()
    assert "default false" in help_text


# ---------------------------------------------------------------------------
# Claim 3: pred_horizon is not what is supervised.
# ---------------------------------------------------------------------------


def test_only_the_first_action_is_supervised_at_the_shipped_gamma() -> None:
    """Asserts `idm_action_loss_horizon_gamma=0.0` supervises horizon step 0 alone.

    Computed rather than asserted from the formula: the loss is invariant to the
    contents of steps 1..K-1, is not invariant to step 0, and equals the plain MSE of
    step 0.
    """
    gamma = FADAConfig().idm_action_loss_horizon_gamma
    assert gamma == 0.0

    torch.manual_seed(0)
    pred = torch.randn((3, 6, 2))
    target = torch.randn((3, 6, 2))
    baseline = weighted_horizon_mse(pred, target, gamma)

    perturbed_tail = target.clone()
    perturbed_tail[:, 1:, :] += 100.0
    assert torch.allclose(weighted_horizon_mse(pred, perturbed_tail, gamma), baseline), (
        "steps 1..5 must carry zero weight"
    )

    perturbed_head = target.clone()
    perturbed_head[:, 0, :] += 100.0
    assert not torch.allclose(weighted_horizon_mse(pred, perturbed_head, gamma), baseline), (
        "step 0 must carry all of it"
    )

    # Equals the plain MSE of step 0 alone.
    step_zero = torch.nn.functional.mse_loss(pred[:, :1, :], target[:, :1, :])
    assert torch.allclose(baseline, step_zero)


def test_the_idm_still_emits_every_horizon_step() -> None:
    """Asserts the IDM emits an action for every horizon step: emitted != supervised."""
    policy = _tiny_policy(pred_horizon=2)
    history_obs = torch.zeros((1, 2, 3))
    history_act = torch.zeros((1, 2, 2))
    future_obs = torch.zeros((1, 2, 3))
    actions = policy.idm(history_obs, history_act, None, future_obs)
    assert actions.shape == (1, 2, 2)


def test_the_pred_horizon_help_no_longer_claims_it_is_the_supervised_horizon() -> None:
    """Asserts the `pred_horizon` help separates emitted from supervised horizon and
    names `idm-action-loss-horizon-gamma` as the field deciding what is supervised.
    """
    help_text = _help_for("pred_horizon")
    assert "supervised" in help_text.lower(), "the distinction must be stated, not dropped"
    assert "idm-action-loss-horizon-gamma" in help_text.lower(), (
        "the help must name the field that actually decides what is supervised"
    )

    gamma_help = _help_for("idm_action_loss_horizon_gamma").lower()
    assert "0.0" in gamma_help and "single-step" in gamma_help, (
        'calling 0.0 a "per-horizon discount" understates it: it is single-step supervision'
    )


# ---------------------------------------------------------------------------
# Claim 4: the teacher-forcing / detach fields are FADAConfig fields but not CLI
# flags. `_train_step_separate_pass` passes explicit per-pass overrides for all
# three, so the cfg values are never read.
# ---------------------------------------------------------------------------

#: Overridden per pass by `_train_step_separate_pass`, therefore not flags.
PASS_OVERRIDDEN_FIELDS = (
    "idm_use_teacher_forcing",
    "idm_teacher_forcing_ratio",
    "idm_detach_planner_future_obs",
)


def test_the_pass_overridden_fields_are_not_cli_flags() -> None:
    exposed = {entry[0] for entry in _OVERRIDABLE_FIELDS}
    offenders = sorted(exposed & set(PASS_OVERRIDDEN_FIELDS))
    assert not offenders, (
        f"{offenders} are exposed as flags again, but `_train_step_separate_pass` overrides "
        "all three per pass, so passing them changes nothing a user can observe"
    )


def test_the_pass_overridden_fields_are_still_fadaconfig_fields() -> None:
    """Asserts the three fields are still `FADAConfig` fields.

    They are serialized into the checkpoint `cfg` payload and the exported ONNX
    metadata, which `eval_checkpoint.py` reads back, and they are the defaults for
    callers driving the loss helpers directly.
    """
    field_names = {f.name for f in dataclasses.fields(FADAConfig)}
    missing = sorted(set(PASS_OVERRIDDEN_FIELDS) - field_names)
    assert not missing, f"{missing} were removed from FADAConfig, which breaks the export metadata"


def test_the_planner_obs_loss_help_states_that_it_is_zero() -> None:
    """Asserts the `planner_obs_loss_coef` help states 0.0, and that the default is 0.0."""
    help_text = _help_for("planner_obs_loss_coef").lower()
    assert "0.0" in help_text
    assert FADAConfig().planner_obs_loss_coef == 0.0
