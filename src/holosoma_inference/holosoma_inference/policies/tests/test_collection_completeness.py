"""An interrupted collection must not read back as a complete one.

`DataCollector` buffers the active episode until finalization, and the policy's teardown
path calls `close()` from a `finally`, so a SIGTERM FLUSHES and CLOSES cleanly. The
resulting H5 is structurally intact: `h5py` opens it, `_validate_data_collection_output`
passes it, and `load_trajectories_from_h5` builds training windows out of it. A SIGKILL
truncates the file physically, and both h5py and the loader reject it.

These tests pin that the provenance is recorded at collection time, the step-5 loader
restates it, and nothing about which windows get trained on changes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from holosoma.fada.common.lora_utils import (
    describe_collection_completeness,
    load_trajectories_from_h5,
)
from holosoma.utils.data_collector import DataCollector

from holosoma_inference.policies.locomotion import LocomotionPolicy_Deploy
from holosoma_inference.policies.locomotion_fada import LocomotionPolicy_FADA

_OBS_DIM = 3
_ACT_DIM = 2
_CMD_DIM = 1


def _collect(tmp_path: Path, *, steps: int, planned: int | None, close: bool = True) -> Path:
    collector = DataCollector(
        output_dir=str(tmp_path),
        dataset_name="dataset",
        compress=False,
        batch_size=1,
        num_envs=1,
        obs_dict={"dynamics_obs": ["dof_pos"], "current_command": ["cmd"]},
        planned_steps=planned,
        planned_steps_source="--task.max-steps",
    )
    collector.start_episode()
    for step in range(steps):
        collector.collect_step(
            {
                "dynamics_obs": np.full((1, _OBS_DIM), float(step), dtype=np.float32),
                "current_command": np.zeros((1, _CMD_DIM), dtype=np.float32),
            },
            np.zeros((1, _ACT_DIM), dtype=np.float32),
            np.zeros((1,), dtype=bool),
        )
    if close:
        collector.close()
    else:
        # Everything written is on disk, but close() -- and therefore the final status --
        # never ran.
        collector.flush()
        collector.h5_file.close()
        collector.h5_file = None
    return tmp_path / "dataset.h5"


def _load(h5_path: Path):
    return load_trajectories_from_h5(
        [str(h5_path)],
        obs_key="dynamics_obs",
        command_key="current_command",
        obs_dim=_OBS_DIM,
        act_dim=_ACT_DIM,
        cmd_dim=_CMD_DIM,
        raw_obs_preprocess=None,
        pred_horizon_k=2,
    )


def _status(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as handle:
        return {key: handle.attrs[key] for key in handle.attrs}


def test_a_clean_early_stop_is_distinguishable_from_a_complete_run(tmp_path: Path) -> None:
    """A short run and a complete run get different `collection_status` attributes."""
    interrupted = _collect(tmp_path / "short", steps=20, planned=5000)
    complete = _collect(tmp_path / "full", steps=40, planned=40)

    assert _status(interrupted)["collection_status"] == "truncated"
    assert _status(interrupted)["planned_steps"] == 5000
    assert _status(interrupted)["collected_steps"] == 20
    assert _status(complete)["collection_status"] == "complete"


def test_a_writer_that_never_reached_close_stays_in_progress(tmp_path: Path) -> None:
    """A writer that never reached `close()` leaves `collection_status == "in_progress"`."""
    killed = _collect(tmp_path / "killed", steps=20, planned=5000, close=False)
    assert _status(killed)["collection_status"] == "in_progress"
    # ...and the step count stamped on the way past is still present.
    assert _status(killed)["collected_steps"] == 20


def test_an_open_ended_run_says_so_rather_than_inventing_a_planned_count(tmp_path: Path) -> None:
    path = _collect(tmp_path / "open", steps=20, planned=None)
    assert _status(path)["collection_status"] == "closed_unknown_length"
    assert _status(path)["planned_steps"] == -1


def test_step_five_states_what_it_found_for_each_dataset(tmp_path: Path) -> None:
    """`describe_collection_completeness` restates each dataset's recorded provenance."""
    interrupted = _collect(tmp_path / "short", steps=20, planned=5000)
    _trajectories, stats = _load(interrupted)
    lines = describe_collection_completeness(stats)
    assert any("collected SHORT" in line and "20 of a planned 5000" in line for line in lines), lines

    complete = _collect(tmp_path / "full", steps=40, planned=40)
    _trajectories, stats = _load(complete)
    assert any("collection complete" in line for line in describe_collection_completeness(stats))


def test_step_five_says_when_a_dataset_cannot_answer_the_question(tmp_path: Path) -> None:
    """An H5 with no provenance attributes is reported as unanswerable."""
    path = _collect(tmp_path / "old", steps=20, planned=5000)
    with h5py.File(path, "a") as handle:
        for key in ("collection_status", "planned_steps", "collected_steps", "planned_steps_source"):
            del handle.attrs[key]

    _trajectories, stats = _load(path)
    lines = describe_collection_completeness(stats)
    assert any("no collection_status attribute" in line for line in lines), lines


def test_step_five_reports_a_killed_collection_without_refusing_it(tmp_path: Path) -> None:
    killed = _collect(tmp_path / "killed", steps=20, planned=5000, close=False)
    trajectories, stats = _load(killed)
    assert trajectories, "the surviving data is still used -- state the fact, do not refuse it"
    assert any("in_progress" in line for line in describe_collection_completeness(stats))


def test_reporting_does_not_change_which_windows_step_five_trains_on(tmp_path: Path) -> None:
    """Provenance is reporting only: the loaded windows are unaffected by it."""
    interrupted = _collect(tmp_path / "short", steps=20, planned=5000)
    stripped = _collect(tmp_path / "stripped", steps=20, planned=5000)
    with h5py.File(stripped, "a") as handle:
        for key in ("collection_status", "planned_steps", "collected_steps", "planned_steps_source"):
            del handle.attrs[key]

    with_provenance, _ = _load(interrupted)
    without_provenance, _ = _load(stripped)
    assert len(with_provenance) == len(without_provenance) == 1
    np.testing.assert_array_equal(with_provenance[0].obs, without_provenance[0].obs)
    np.testing.assert_array_equal(with_provenance[0].actions, without_provenance[0].actions)


def _policy_with(task: SimpleNamespace, **attrs) -> LocomotionPolicy_Deploy:
    policy = object.__new__(LocomotionPolicy_Deploy)
    policy.config = SimpleNamespace(task=task)
    for name, value in attrs.items():
        setattr(policy, name, value)
    return policy


@pytest.mark.parametrize(
    ("task", "attrs", "expected_steps", "expected_source_fragment"),
    [
        (
            SimpleNamespace(max_steps=5000, exit_after_max_eval_time=True, max_eval_time=-1.0),
            {"max_eval_time": -1.0, "rl_rate": 50.0},
            5000,
            "--task.max-steps",
        ),
        (
            SimpleNamespace(max_steps=-1, exit_after_max_eval_time=True, max_eval_time=20.0),
            {"max_eval_time": 20.0, "rl_rate": 50.0},
            1000,
            "--task.max-eval-time",
        ),
        (
            SimpleNamespace(max_steps=-1, exit_after_max_eval_time=False, max_eval_time=20.0),
            {"max_eval_time": 20.0, "rl_rate": 50.0},
            None,
            "open-ended",
        ),
    ],
)
def test_the_planned_step_count_comes_from_the_same_bounds_that_stop_the_loop(
    task, attrs, expected_steps, expected_source_fragment
) -> None:
    """The planned count comes from `_should_stop`'s two conditions, in its order.

    Third case: `exit_after_max_eval_time=False` means the time limit only zeroes the
    commands, so there is no planned step count and `None` is returned.
    """
    policy = _policy_with(task, **attrs)
    steps, source = policy._resolve_planned_collection_steps()
    assert steps == expected_steps
    assert expected_source_fragment in source


def test_the_fada_policys_own_collector_records_the_same_provenance() -> None:
    """`LocomotionPolicy_FADA` overrides `_init_data_collector`, and step 4 runs that one.

    Its override must pass the same `planned_steps` / `planned_steps_source` provenance.
    """
    source = Path(LocomotionPolicy_FADA.__module__.replace(".", "/"))
    module_file = Path(sys.modules[LocomotionPolicy_FADA.__module__].__file__)
    assert module_file.name == source.name + ".py"
    text = module_file.read_text()
    override = text.split("def _init_data_collector", 1)[1]
    assert "planned_steps=planned_steps" in override
    assert "planned_steps_source=planned_steps_source" in override
    assert "self._resolve_planned_collection_steps()" in override


# ---------------------------------------------------------------------------
# Multi-session files, and self-contradicting attributes.
#
# The four file-level attributes above are rewritten every time the H5 is reopened, so in a
# file that two sessions appended to they describe only the second, while the file still
# holds every episode of the first. `collection_sessions` plus each episode's
# `collection_session` tag is what carries the per-session record.
#
# Provenance is REPORTING: the windows the finetune trains on come out byte-identical.
# ---------------------------------------------------------------------------


def _append_sequence(tmp_path: Path) -> Path:
    """3-of-10 truncated, then reopen the same file and append 4-of-4 complete."""
    root = tmp_path / "appended"
    _collect(root, steps=3, planned=10)
    return _collect(root, steps=4, planned=4)


def _window_fingerprint(trajectories) -> str:
    """Hash of the arrays a trajectory carries.

    `Trajectory.source` is excluded: it is the dataset path, which differs between the two
    tmp directories being compared.
    """
    import hashlib

    digest = hashlib.sha256()
    for trajectory in trajectories:
        for field in ("obs", "actions", "current_command"):
            value = getattr(trajectory, field)
            digest.update(field.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def test_a_two_session_file_is_not_described_by_its_newest_session_alone(tmp_path: Path) -> None:
    path = _append_sequence(tmp_path)

    with h5py.File(path, "r") as handle:
        lengths = sorted(int(handle["episodes"][name].attrs["episode_length"]) for name in handle["episodes"])
    assert lengths == [3, 4], f"premise: the file holds both sessions' episodes, got {lengths}"
    assert _status(path)["collection_status"] == "complete", "premise: the file-level verdict is the newest one's"

    _trajectories, stats = _load(path)
    lines = describe_collection_completeness(stats)
    blob = "\n".join(lines)

    assert "2 collection sessions" in blob, lines
    # The 3-of-10 session must be visible.
    assert "3 of a planned 10" in blob, lines
    # And no unqualified "this file is complete" claim.
    assert not any(line.strip().startswith("[data] dataset.h5: collection complete") for line in lines), lines


def test_each_episode_records_which_session_wrote_it(tmp_path: Path) -> None:
    """Each episode carries a `collection_session` index naming the session that wrote it."""
    path = _append_sequence(tmp_path)
    with h5py.File(path, "r") as handle:
        by_length = {
            int(handle["episodes"][name].attrs["episode_length"]): int(
                handle["episodes"][name].attrs["collection_session"]
            )
            for name in handle["episodes"]
        }
    assert by_length == {3: 0, 4: 1}, by_length


def test_the_appended_file_still_yields_byte_identical_windows(tmp_path: Path) -> None:
    """Provenance is reporting only.

    A single 3-step session and a single 4-step session, loaded separately, produce exactly
    the arrays the appended file produces. Compared field-by-field rather than by count,
    because a count assertion passes when two arrays swap contents.
    """
    appended = _append_sequence(tmp_path)
    appended_trajectories, _stats = _load(appended)

    # The same two episodes, produced without any session history to record.
    reference_root = tmp_path / "reference"
    _collect(reference_root, steps=3, planned=10)
    reference = _collect(reference_root, steps=4, planned=4)
    with h5py.File(reference, "a") as handle:
        handle.attrs.pop("collection_sessions", None)
        for name in handle["episodes"]:
            handle["episodes"][name].attrs.pop("collection_session", None)
    reference_trajectories, _stats = _load(reference)

    assert len(appended_trajectories) == len(reference_trajectories) == 2
    for got, want in zip(appended_trajectories, reference_trajectories):
        for field in ("obs", "actions", "current_command"):
            np.testing.assert_array_equal(getattr(got, field), getattr(want, field))
            assert getattr(got, field).dtype == getattr(want, field).dtype
    assert _window_fingerprint(appended_trajectories) == _window_fingerprint(reference_trajectories)


def test_contradictory_attributes_are_not_restated_as_a_verdict(tmp_path: Path) -> None:
    """`complete` with `collected_steps` far below `planned_steps` reports as UNKNOWN.

    The status and the counts are cross-checked rather than restated.
    """
    path = _collect(tmp_path / "liar", steps=20, planned=5000)
    with h5py.File(path, "a") as handle:
        handle.attrs["collection_status"] = "complete"
        handle.attrs.pop("collection_sessions", None)

    _trajectories, stats = _load(path)
    lines = describe_collection_completeness(stats)
    blob = "\n".join(lines)
    assert "collection complete (20/5000" not in blob, lines
    assert "contradict" in blob, lines
    assert "UNKNOWN" in blob, lines


# ---------------------------------------------------------------------------
# Reconciling the session records against the episodes.
#
# `collection_sessions` is a claim each writer makes about itself; the episodes are the
# evidence, and each carries a `collection_session` tag naming its record. The cases below
# cover the ways the two can disagree. Reporting only --
# `test_reconciliation_leaves_the_trained_windows_byte_identical` pins that.
# ---------------------------------------------------------------------------


def _sessions_attr(path: Path) -> list[dict]:
    import json

    with h5py.File(path, "r") as handle:
        raw = handle.attrs["collection_sessions"]
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def _rewrite_sessions(path: Path, sessions: list[dict]) -> None:
    import json

    with h5py.File(path, "a") as handle:
        handle.attrs["collection_sessions"] = json.dumps(sessions)


def test_an_untagged_episode_beside_a_tagged_one_is_reported_as_unattributed(tmp_path: Path) -> None:
    """Case 1: an episode carrying no `collection_session` tag beside a tagged one.

    The report names it as unattributed rather than omitting it.
    """
    root = tmp_path / "mixed"
    _collect(root, steps=3, planned=3)
    path = _collect(root, steps=4, planned=4)
    with h5py.File(path, "a") as handle:  # age the first episode: strip its tag
        oldest = sorted(handle["episodes"].keys())[0]
        del handle["episodes"][oldest].attrs["collection_session"]

    _trajectories, stats = _load(path)
    blob = "\n".join(describe_collection_completeness(stats))
    assert "carry no collection_session tag" in blob, blob
    assert "1 of 2 episode(s)" in blob, blob
    assert "holds the episodes of all of them" not in blob, blob


def test_session_records_that_match_no_episode_tag_are_not_reported_as_counts(tmp_path: Path) -> None:
    """Case 2: records that disagree with every episode in the file.

    Both records claim episode counts no tag supports; the report states the disagreement
    instead of repeating the claimed counts.
    """
    path = _append_sequence(tmp_path)
    sessions = _sessions_attr(path)
    assert [s["episodes"] for s in sessions] == [1, 1], sessions
    for session in sessions:
        session["episodes"] = 7
    _rewrite_sessions(path, sessions)

    _trajectories, stats = _load(path)
    blob = "\n".join(describe_collection_completeness(stats))
    assert blob.count("records episodes=7 but 1 episode(s) in the file carry its tag") == 2, blob
    assert "neither number is evidence" in blob, blob
    assert "holds the episodes of all of them" not in blob, blob


def test_zero_episode_session_records_are_not_folded_into_a_holds_them_all_claim(tmp_path: Path) -> None:
    """Case 3: one episode, six records, five of them recording zero episodes.

    The "holds the episodes of all of them" claim is not made for such a file.
    """
    path = _collect(tmp_path / "mostly-empty", steps=3, planned=3)
    sessions = _sessions_attr(path)
    assert len(sessions) == 1
    for index in range(1, 6):
        sessions.append(
            {
                "session_index": index,
                "collection_status": "closed_unknown_length",
                "planned_steps": -1,
                "planned_steps_source": "--task.max-steps",
                "collected_steps": 0,
                "episodes": 0,
                "episode_ids": [],
            }
        )
    _rewrite_sessions(path, sessions)

    _trajectories, stats = _load(path)
    lines = describe_collection_completeness(stats)
    blob = "\n".join(lines)
    assert "6 collection sessions" in blob, blob
    assert "holds the episodes of all of them" not in blob, blob
    assert "holds the episodes of 1 of them (5 contributed no episodes at all)" in blob, blob
    assert blob.count("recorded zero episodes, and the file confirms it") == 5, blob


def test_a_genuine_zero_is_not_rendered_as_not_recorded(tmp_path: Path) -> None:
    """Case 4: a real 0 must not be rendered as the -1 "unknown" sentinel.

    `_as_int` maps None and unparseable values to -1 and leaves 0 as 0.
    """
    from holosoma.fada.common.lora_utils import _as_int

    assert _as_int(0) == 0
    assert _as_int(None) == -1
    assert _as_int("nonsense") == -1
    assert _as_int(None, default=None) is None

    # A second session that ran, collected nothing, and closed: both of its zeros
    # (`collected_steps` and `episodes`) must render as 0, not as "unknown".
    path = _collect(tmp_path / "zero", steps=3, planned=3)
    sessions = _sessions_attr(path)
    sessions.append(
        {
            "session_index": 1,
            "collection_status": "truncated",
            "planned_steps": 5,
            "planned_steps_source": "--task.max-steps",
            "collected_steps": 0,
            "episodes": 0,
            "episode_ids": [],
        }
    )
    _rewrite_sessions(path, sessions)

    _trajectories, stats = _load(path)
    blob = "\n".join(describe_collection_completeness(stats))
    assert "0 of a planned 5" in blob, blob
    assert "[0 episode(s)]" in blob, blob
    assert "-1" not in blob, blob
    assert "recorded zero episodes, and the file confirms it" in blob, blob


def test_two_records_claiming_one_session_index_do_not_reconcile(tmp_path: Path) -> None:
    """Case 5: two distinct records claiming the same `session_index`.

    An episode's `collection_session` tag names an *index*, not a record, so two records
    sharing index 0 are both satisfied by the single episode tagged 0. The report must not
    then claim the file "holds the episodes of all of them"; it names the duplicate index
    instead.
    """
    path = _collect(tmp_path / "duplicate-index", steps=3, planned=3)
    sessions = _sessions_attr(path)
    assert len(sessions) == 1 and sessions[0].get("session_index", 0) == 0, sessions
    duplicate = dict(sessions[0])
    duplicate["session_index"] = 0  # a second record claiming the FIRST record's slot
    sessions.append(duplicate)
    _rewrite_sessions(path, sessions)

    _trajectories, stats = _load(path)
    blob = "\n".join(describe_collection_completeness(stats))

    assert "holds the episodes of all of them" not in blob, blob
    assert "do NOT fully account for" in blob, blob
    # ...and the reason is named.
    assert "more than one record in collection_sessions claims the same session index" in blob, blob
    assert "session_index=0 claimed by 2 records" in blob, blob
    assert "1 episode(s) carry that tag" in blob, blob


def test_duplicate_indices_are_caught_even_when_every_other_check_passes(tmp_path: Path) -> None:
    """`_sessions_account_for_every_episode` alone, without the console rendering.

    The duplicated row satisfies every other term of the predicate.
    """
    from holosoma.fada.common.lora_utils import _sessions_account_for_every_episode

    duplicated = {
        "file_episodes": 1,
        "untagged_episodes": 0,
        "unparseable_session_tags": 0,
        "episodes_by_session": {0: 1},
        "sessions": [{"session_index": 0, "episodes": 1}, {"session_index": 0, "episodes": 1}],
    }
    assert _sessions_account_for_every_episode(duplicated) is False

    # Same row with the second record moved to its own slot, and an episode to match:
    # this shape reconciles.
    distinct = {
        **duplicated,
        "file_episodes": 2,
        "episodes_by_session": {0: 1, 1: 1},
        "sessions": [{"session_index": 0, "episodes": 1}, {"session_index": 1, "episodes": 1}],
    }
    assert _sessions_account_for_every_episode(distinct) is True


def test_a_fully_reconciled_multi_session_file_still_says_it_holds_them_all(tmp_path: Path) -> None:
    """A fully reconciled multi-session file still reports "holds the episodes of all of them"."""
    path = _append_sequence(tmp_path)
    _trajectories, stats = _load(path)
    blob = "\n".join(describe_collection_completeness(stats))
    assert "holds the episodes of all of them" in blob, blob
    assert "do NOT fully account for" not in blob, blob
    assert "carry no collection_session tag" not in blob, blob


def test_a_file_predating_session_tagging_gains_no_extra_noise(tmp_path: Path) -> None:
    """A file with no session tags at all produces no reconciliation lines."""
    path = _collect(tmp_path / "old", steps=3, planned=3)
    with h5py.File(path, "a") as handle:
        handle.attrs.pop("collection_sessions", None)
        for name in handle["episodes"]:
            handle["episodes"][name].attrs.pop("collection_session", None)

    _trajectories, stats = _load(path)
    blob = "\n".join(describe_collection_completeness(stats))
    assert "carry no collection_session tag" not in blob, blob


@pytest.mark.parametrize("mutation", ["untagged", "mismatched", "empty_records"])
def test_reconciliation_leaves_the_trained_windows_byte_identical(tmp_path: Path, mutation: str) -> None:
    """Reporting changes, the loaded windows do not.

    For each of the three broken-provenance shapes above, the trajectories loaded out of
    the mutated file are byte-identical to those loaded out of the untouched one -- same
    dtypes, same shapes, same bytes. Compared field-by-field and by fingerprint, because a
    count assertion passes when two arrays swap contents.
    """
    import json

    pristine = _append_sequence(tmp_path / "pristine")
    mutated = _append_sequence(tmp_path / "mutated")

    with h5py.File(mutated, "a") as handle:
        if mutation == "untagged":
            oldest = sorted(handle["episodes"].keys())[0]
            del handle["episodes"][oldest].attrs["collection_session"]
        elif mutation == "mismatched":
            sessions = json.loads(handle.attrs["collection_sessions"])
            for session in sessions:
                session["episodes"] = 7
            handle.attrs["collection_sessions"] = json.dumps(sessions)
        else:
            sessions = json.loads(handle.attrs["collection_sessions"])
            sessions.append({"session_index": 2, "collection_status": "complete", "episodes": 0})
            handle.attrs["collection_sessions"] = json.dumps(sessions)

    pristine_trajectories, pristine_stats = _load(pristine)
    mutated_trajectories, mutated_stats = _load(mutated)

    # Premise: the reports really do differ, so the equality below is not vacuous.
    assert describe_collection_completeness(pristine_stats) != describe_collection_completeness(mutated_stats)

    assert len(mutated_trajectories) == len(pristine_trajectories) == 2
    for got, want in zip(mutated_trajectories, pristine_trajectories):
        for field in ("obs", "actions", "current_command"):
            np.testing.assert_array_equal(getattr(got, field), getattr(want, field))
            assert getattr(got, field).dtype == getattr(want, field).dtype
            assert getattr(got, field).shape == getattr(want, field).shape
    assert _window_fingerprint(mutated_trajectories) == _window_fingerprint(pristine_trajectories)


def test_the_documented_recipe_is_not_reported_as_truncated() -> None:
    """`--task.max-steps N` records N-1 steps, and that still reads as complete.

    The policy loop's stop check fires before the last iteration is recorded, so a
    collection run lands one step short of the number the user typed.

    The tolerance is one step and no more: two steps short is still `truncated`.
    """
    from holosoma.utils.data_collector import (
        COLLECTION_STATUS_CLOSED_UNKNOWN_LENGTH,
        COLLECTION_STATUS_COMPLETE,
        COLLECTION_STATUS_TRUNCATED,
        DataCollector,
    )

    def verdict(planned: int | None, collected: int) -> str:
        stub = DataCollector.__new__(DataCollector)
        stub.planned_steps = planned
        stub.new_step_count = collected
        return DataCollector._final_collection_status(stub)

    assert verdict(5000, 4999) == COLLECTION_STATUS_COMPLETE
    assert verdict(1000, 999) == COLLECTION_STATUS_COMPLETE
    assert verdict(5000, 5000) == COLLECTION_STATUS_COMPLETE

    assert verdict(5000, 4998) == COLLECTION_STATUS_TRUNCATED
    assert verdict(5000, 20) == COLLECTION_STATUS_TRUNCATED
    assert verdict(None, 999) == COLLECTION_STATUS_CLOSED_UNKNOWN_LENGTH
