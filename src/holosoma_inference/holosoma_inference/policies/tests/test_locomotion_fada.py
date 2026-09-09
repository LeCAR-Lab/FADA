from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from holosoma.fada.common.current_command import DEFAULT_COMMAND_PROFILE

from holosoma_inference.policies.base import (
    TRANSFORMER_FINETUNE_DYNAMICS_TERMS,
    BasePolicy,
)
from holosoma_inference.policies.locomotion_fada import (
    LocomotionPolicy_FADA,
)


def test_collect_current_command_includes_phase_terms() -> None:
    policy = object.__new__(LocomotionPolicy_FADA)
    policy.cmd_dim = 7
    policy.lin_vel_command = np.asarray([[0.2, -0.1]], dtype=np.float32)
    policy.ang_vel_command = np.asarray([[0.3]], dtype=np.float32)
    policy.stand_command = np.asarray([[1.0]], dtype=np.float32)
    policy.command_profile = DEFAULT_COMMAND_PROFILE
    policy.phase = np.asarray([[0.0, np.pi / 2.0]], dtype=np.float32)

    current_command = policy._collect_current_command()

    assert current_command.shape == (1, 7)
    assert np.allclose(
        current_command,
        np.asarray([[0.2, -0.1, 0.3, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# A --task.collect-data init failure is not downgraded to data_collector=None:
# _init_data_collector raises, so a failed init propagates out of __init__ ->
# run_policy.py's top-level except -> sys.exit(1).
# See locomotion_fada.py::_init_data_collector.
# ---------------------------------------------------------------------------


def _minimal_fada_policy_for_collector(data_collection_output_dir: Path) -> LocomotionPolicy_FADA:
    policy = object.__new__(LocomotionPolicy_FADA)
    policy.robot_config = SimpleNamespace(robot_type="t1_23dof")
    policy.compact_term_order = ("dof_pos", "dof_vel")
    policy.command_profile = DEFAULT_COMMAND_PROFILE
    policy.logger = logging.getLogger("test_locomotion_fada")
    policy.config = SimpleNamespace(
        task=SimpleNamespace(
            interface="lo",
            data_collection_output_dir=str(data_collection_output_dir),
            data_collection_skip_obs_keys=("actor_obs", "critic_obs"),
            data_collection_dataset_name="dataset",
            data_collection_compress=False,
            model_path="dummy.onnx",
        )
    )
    return policy


def test_init_data_collector_raises_when_output_dir_unwritable(tmp_path: Path) -> None:
    """An init failure (a parent path component is a file, so Path.mkdir(parents=True)
    cannot create the output dir) raises instead of downgrading to data_collector=None."""
    blocked = tmp_path / "blocked_by_a_file"
    blocked.write_text("not a directory")
    unwritable_output_dir = blocked / "sub" / "dataset_dir"

    policy = _minimal_fada_policy_for_collector(unwritable_output_dir)

    with pytest.raises(RuntimeError, match="collect-data was requested but data collector initialization failed"):
        policy._init_data_collector()

    assert policy.data_collector is None


def test_init_data_collector_succeeds_for_writable_output_dir(tmp_path: Path) -> None:
    """A writable output dir does not raise and produces a DataCollector pointed at the
    requested H5 path."""
    output_dir = tmp_path / "collected"
    policy = _minimal_fada_policy_for_collector(output_dir)

    policy._init_data_collector()  # must not raise

    assert policy.data_collector is not None
    assert (output_dir / "dataset.h5").exists()
    policy.data_collector.close()


def test_base_policy_emits_transformer_collection_payload() -> None:
    policy = object.__new__(BasePolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(collect_data=True)
    )
    policy.num_dofs = 2
    policy.last_policy_action = np.asarray([[0.4, -0.2]], dtype=np.float32)
    policy.obs_buf_dict = {}

    current_obs = {
        "base_ang_vel": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        "dof_pos": np.asarray([[4.0, 5.0]], dtype=np.float32),
        "dof_vel": np.asarray([[6.0, 7.0]], dtype=np.float32),
        "projected_gravity": np.asarray([[8.0, 9.0, 10.0]], dtype=np.float32),
        "command_lin_vel": np.asarray([[0.1, 0.2]], dtype=np.float32),
        "command_ang_vel": np.asarray([[0.3]], dtype=np.float32),
        "sin_phase": np.asarray([[0.4, 0.5]], dtype=np.float32),
        "cos_phase": np.asarray([[0.6, 0.7]], dtype=np.float32),
    }

    policy._add_transformer_collection_payload(current_obs)

    assert list(policy._transformer_collection_obs_dict()["raw_dynamics_obs"]) == list(
        TRANSFORMER_FINETUNE_DYNAMICS_TERMS
    )
    assert np.allclose(
        policy.obs_buf_dict["raw_dynamics_obs"],
        np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
    )
    assert np.allclose(
        policy.obs_buf_dict["current_command"],
        np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float32),
    )
    assert sorted(policy.obs_buf_dict.keys()) == ["current_command", "raw_dynamics_obs"]


def test_base_policy_without_collect_data_does_not_emit_transformer_payload() -> None:
    policy = object.__new__(BasePolicy)
    policy.config = SimpleNamespace(task=SimpleNamespace(collect_data=False))
    policy.num_dofs = 2
    policy.last_policy_action = np.zeros((1, 2), dtype=np.float32)
    policy.obs_buf_dict = {}

    policy._add_transformer_collection_payload(
        {
            "base_ang_vel": np.zeros((1, 3), dtype=np.float32),
            "dof_pos": np.zeros((1, 2), dtype=np.float32),
            "dof_vel": np.zeros((1, 2), dtype=np.float32),
            "projected_gravity": np.zeros((1, 3), dtype=np.float32),
            "command_lin_vel": np.zeros((1, 2), dtype=np.float32),
            "command_ang_vel": np.zeros((1, 1), dtype=np.float32),
            "sin_phase": np.zeros((1, 2), dtype=np.float32),
            "cos_phase": np.zeros((1, 2), dtype=np.float32),
        }
    )

    assert policy.obs_buf_dict == {}


def test_base_policy_collection_obs_dict_augments_raw_trajectory_keys() -> None:
    policy = object.__new__(BasePolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(collect_data=True),
        observation=SimpleNamespace(
            obs_dict={
                "actor_obs": ["base_ang_vel"],
                "obs_history": ["obs_history"],
                "actions_history": ["actions_history"],
            }
        ),
    )

    obs_dict = policy._build_data_collection_obs_dict()

    assert obs_dict["obs_history"] == ["obs_history"]
    assert obs_dict["actions_history"] == ["actions_history"]
    assert obs_dict["raw_dynamics_obs"] == list(TRANSFORMER_FINETUNE_DYNAMICS_TERMS)
    assert "dynamics_obs" not in obs_dict
    assert obs_dict["current_command"] == [
        "command_lin_vel",
        "command_ang_vel",
        "sin_phase",
        "cos_phase",
    ]


def test_validate_io_contract_requires_current_command() -> None:
    policy = object.__new__(LocomotionPolicy_FADA)
    policy._validate_io_contract(
        ["history_obs", "history_act", "current_command", "history_valid_mask"],
        ["actions", "pred_next_obs"],
    )
    assert policy._command_input_name == "current_command"


def test_validate_io_contract_accepts_pred_future_obs() -> None:
    policy = object.__new__(LocomotionPolicy_FADA)
    policy._validate_io_contract(
        ["history_obs", "history_act", "current_command", "history_valid_mask"],
        ["actions", "pred_future_obs"],
    )


def test_validate_io_contract_requires_actions() -> None:
    policy = object.__new__(LocomotionPolicy_FADA)
    with pytest.raises(ValueError, match="output 'actions'"):
        policy._validate_io_contract(
            ["history_obs", "history_act", "current_command", "history_valid_mask"],
            ["pred_future_obs"],
        )


def test_validate_io_contract_accepts_optional_fdm_teacher_io() -> None:
    policy = object.__new__(LocomotionPolicy_FADA)
    policy._validate_io_contract(
        [
            "history_obs",
            "history_act",
            "current_command",
            "history_valid_mask",
            "teacher_future_obs",
            "teacher_future_actions",
        ],
        ["actions", "pred_future_obs", "idm_teacher_actions", "fdm_teacher_future_obs"],
    )


def test_validate_io_contract_requires_matching_fdm_teacher_input() -> None:
    policy = object.__new__(LocomotionPolicy_FADA)
    with pytest.raises(ValueError, match="teacher_future_actions"):
        policy._validate_io_contract(
            ["history_obs", "history_act", "current_command", "history_valid_mask", "teacher_future_obs"],
            ["actions", "pred_future_obs", "fdm_teacher_future_obs"],
        )


def test_capture_restore_policy_state_round_trips_idm_command_flag() -> None:
    policy = object.__new__(LocomotionPolicy_FADA)
    policy.onnx_policy_session = SimpleNamespace(name="session")
    policy.onnx_input_names = ["history_obs"]
    policy.onnx_output_names = ["actions"]
    policy.policy = SimpleNamespace(name="callable")
    policy.onnx_kp = [1.0]
    policy.onnx_kd = [2.0]
    policy.idm_use_current_command_for_history = True
    policy._idm_teacher_supported = True
    policy._fdm_teacher_supported = True
    policy.fdm_enabled = True

    state = policy._capture_policy_state()
    restored = object.__new__(LocomotionPolicy_FADA)
    restored._restore_policy_state(state)

    assert restored.idm_use_current_command_for_history is True
    assert restored._idm_teacher_supported is True
    assert restored._fdm_teacher_supported is True
    assert restored.fdm_enabled is True


# ---------------------------------------------------------------------------
# How many inputs an export from this release has.
#
# `_export_planner_idm_onnx` declares five graph inputs: `history_obs`, `history_act`,
# `current_command`, `history_valid_mask` and -- unconditionally -- `teacher_future_obs`.
# ONNX Runtime demands a feed for every graph input on every run, so a caller that feeds
# only the first four gets a `RUNTIME_EXCEPTION`.
#
# `_validate_io_contract`'s smaller required set is a compatibility floor for older
# plain-transformer exports, not the exporter's contract; the two are pinned to each
# other below.
# ---------------------------------------------------------------------------

EXPORTED_ONNX_INPUT_NAMES = [
    "history_obs",
    "history_act",
    "current_command",
    "history_valid_mask",
    "teacher_future_obs",
]
EXPORTED_ONNX_OUTPUT_NAMES = ["actions", "pred_future_obs", "idm_teacher_actions"]


def _exporter_io_literals() -> tuple[list[str], list[str]]:
    """The `input_names` / `output_names` lists `_export_planner_idm_onnx` passes.

    Parsed out of the source rather than imported: importing `eval_checkpoint` drags in
    the training-side stack, and this test lives in the inference suite. The literals are
    read with `ast`, so reformatting does not affect the result.
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[4]
        / "holosoma"
        / "holosoma"
        / "fada"
        / "planner_idm"
        / "eval_checkpoint.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in ("input_names", "output_names"):
            continue
        if not isinstance(node.value, ast.List):
            continue
        found[target.id] = [ast.literal_eval(element) for element in node.value.elts]
    assert set(found) == {"input_names", "output_names"}, f"exporter I/O literals not found: {found}"
    return found["input_names"], found["output_names"]


def test_the_exporter_declares_five_inputs_not_four() -> None:
    input_names, output_names = _exporter_io_literals()
    assert input_names == EXPORTED_ONNX_INPUT_NAMES
    assert len(input_names) == 5
    assert output_names == EXPORTED_ONNX_OUTPUT_NAMES
    # The fifth input exists because the third output does; they must move together.
    assert "teacher_future_obs" in input_names
    assert "idm_teacher_actions" in output_names


def test_the_runtime_feeds_every_input_the_exporter_declares() -> None:
    """The policy path supplies a feed for every input the exporter declares.

    ORT raises on a missing feed, so `teacher_future_obs` is fed zeros: the planner heads
    do not read the tensor, and the IDM teacher head's output is used only for metrics.
    """
    input_names, _ = _exporter_io_literals()

    policy = object.__new__(LocomotionPolicy_FADA)
    policy._idm_teacher_input_name = "teacher_future_obs"
    policy._fdm_teacher_input_name = "teacher_future_actions"
    policy.onnx_input_names = list(input_names)
    policy.pred_horizon = 3
    policy.obs_dim = 4
    policy.act_dim = 2

    placeholders = policy._teacher_future_obs_placeholder(1)
    placeholders.update(policy._teacher_future_actions_placeholder(1))

    # The three the caller always builds by hand, plus whatever placeholders cover.
    hand_built = {"history_obs", "history_act", "history_valid_mask", "current_command"}
    covered = hand_built | set(placeholders)
    assert set(input_names) <= covered, f"no feed for {set(input_names) - covered}"

    assert placeholders["teacher_future_obs"].shape == (1, 3, 4)
    assert not placeholders["teacher_future_obs"].any(), "the placeholder must be zeros"
    # `teacher_future_actions` is genuinely optional: absent from this export, absent here.
    assert "teacher_future_actions" not in placeholders


def test_the_runtime_required_set_is_a_floor_not_the_exporter_s_contract() -> None:
    """`_validate_io_contract` accepts a four-input graph (legacy plain-transformer
    exports); every export from this release has five."""
    input_names, output_names = _exporter_io_literals()

    policy = object.__new__(LocomotionPolicy_FADA)
    policy._validate_io_contract(["history_obs", "history_act", "current_command", "history_valid_mask"], ["actions"])

    # ...but a graph that exports the teacher output and drops the teacher input is
    # refused.
    policy = object.__new__(LocomotionPolicy_FADA)
    with pytest.raises(ValueError, match="teacher_future_obs"):
        policy._validate_io_contract(
            ["history_obs", "history_act", "current_command", "history_valid_mask"],
            output_names,
        )

    # And this release's own export passes.
    policy = object.__new__(LocomotionPolicy_FADA)
    policy._validate_io_contract(list(input_names), list(output_names))
