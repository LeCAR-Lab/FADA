"""Tests for :mod:`tools.upstream_ref`, which backs every "is this file inherited
from upstream Holosoma?" answer in this repository.

The validation is an identity test on the recursive tree object, so the cases here
are mutations: each takes a tree the module accepts and changes one thing about it
-- adds a file, deletes one, edits a byte, flips a mode bit, renames a path -- and
asserts the result is refused. None names a FADA path or the string "fada"; the
rule does not look for either.

Most tests build a throwaway repository and point the module's pin at that
repository's own tree. Two run against the real repository with the shipped pin
unmonkeypatched: one asserts it accepts the real baseline, one that it refuses
``HEAD``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parent / "upstream_ref.py"
REPO_ROOT = MODULE_PATH.parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location("_upstream_ref_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upstream_ref = _load_module()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    ).stdout.strip()


def _make_repo(tmp_path: Path, name: str, paths: list[str]) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for rel in paths:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {rel}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", name)
    return repo


def _upstream_shaped() -> list[str]:
    return [*upstream_ref.UPSTREAM_ANCHOR_PATHS, "src/holosoma/holosoma/envs/base.py"]


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)


@pytest.fixture
def baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An upstream-shaped repo, with `UPSTREAM_BASELINE_TREE` monkeypatched to its
    own tree so `main` validates until a test mutates the tree."""
    repo = _make_repo(tmp_path, "upstream", _upstream_shaped())
    monkeypatch.setattr(upstream_ref, "UPSTREAM_BASELINE_TREE", _git(repo, "rev-parse", "main^{tree}"))
    return repo


def test_the_pinned_baseline_is_accepted(baseline: Path) -> None:
    ok, lines = upstream_ref.check_upstream_ref("main", baseline)
    assert ok, lines
    assert "IS the pinned upstream baseline tree" in " ".join(lines)
    assert upstream_ref.upstream_baseline_ref(baseline) == "main"


def test_adding_release_code_no_enumerated_path_or_marker_covers_is_rejected(baseline: Path) -> None:
    """Adding modules whose paths and contents never mention ``fada`` is still
    refused: the tree no longer matches the pin."""
    added = {
        "src/holosoma/holosoma/utils/sync_rendezvous.py": "def wait_for_peer():\n    return None\n",
        "src/holosoma_inference/holosoma_inference/utils/sync_rendezvous.py": "def ack():\n    return None\n",
    }
    for rel, text in added.items():
        (baseline / rel).parent.mkdir(parents=True, exist_ok=True)
        (baseline / rel).write_text(text)
    _commit_all(baseline, "add two modules")

    joined = "\n".join(added) + "\n".join(added.values())
    assert "fada" not in joined.lower(), "premise: the construction never says the method's name"

    ok, lines = upstream_ref.check_upstream_ref("main", baseline)
    assert not ok, lines
    assert "is not the upstream baseline tree" in " ".join(lines)
    assert upstream_ref.upstream_baseline_ref(baseline) is None


def test_deleting_a_file_is_rejected(baseline: Path) -> None:
    """Removing a file from the baseline tree is refused."""
    _git(baseline, "rm", "-q", "src/holosoma/holosoma/envs/base.py")
    _git(baseline, "commit", "-qm", "delete one file")
    assert not upstream_ref.looks_like_upstream("main", baseline)


def test_editing_one_byte_of_one_file_is_rejected(baseline: Path) -> None:
    target = baseline / "src/holosoma/holosoma/envs/base.py"
    target.write_text(target.read_text().replace("#", ";", 1))
    _commit_all(baseline, "one byte")
    assert not upstream_ref.looks_like_upstream("main", baseline)


def test_renaming_a_file_is_rejected(baseline: Path) -> None:
    _git(baseline, "mv", "src/holosoma/holosoma/envs/base.py", "src/holosoma/holosoma/envs/base2.py")
    _git(baseline, "commit", "-qm", "rename")
    assert not upstream_ref.looks_like_upstream("main", baseline)


def test_flipping_an_executable_bit_is_rejected(baseline: Path) -> None:
    """A mode-bit change, which leaves file contents identical, is refused."""
    _git(baseline, "update-index", "--chmod=+x", "src/holosoma/holosoma/envs/base.py")
    _git(baseline, "commit", "-qm", "mode change")
    assert not upstream_ref.looks_like_upstream("main", baseline)


def test_a_tree_that_is_not_holosoma_at_all_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pin naming an unrelated tree satisfies the identity test by construction;
    the anchor-path check is what refuses it."""
    repo = _make_repo(tmp_path, "unrelated", ["README.md"])
    monkeypatch.setattr(upstream_ref, "UPSTREAM_BASELINE_TREE", _git(repo, "rev-parse", "main^{tree}"))
    ok, lines = upstream_ref.check_upstream_ref("main", repo)
    assert not ok
    assert "not a Holosoma tree" in " ".join(lines)


def test_the_reported_evidence_names_the_trees_it_compared(baseline: Path) -> None:
    """The returned lines carry the pinned tree id and every anchor path."""
    _, lines = upstream_ref.check_upstream_ref("main", baseline)
    joined = " ".join(lines)
    assert upstream_ref.UPSTREAM_BASELINE_TREE in joined
    for path in upstream_ref.UPSTREAM_ANCHOR_PATHS:
        assert path in joined


def test_a_nonexistent_ref_is_reported_as_such(baseline: Path) -> None:
    ok, lines = upstream_ref.check_upstream_ref("no-such-ref", baseline)
    assert not ok
    assert "does not resolve" in " ".join(lines)


def test_an_env_override_that_fails_validation_does_not_fall_back(
    baseline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override that does not resolve, or resolves to a non-baseline tree, yields
    None rather than falling back to `main`; a valid one is returned."""
    _git(baseline, "branch", "same-commit")
    monkeypatch.setenv(upstream_ref.UPSTREAM_REF_ENV, "typo-ref")
    assert upstream_ref.upstream_baseline_ref(baseline) is None
    monkeypatch.setenv(upstream_ref.UPSTREAM_REF_ENV, "same-commit")
    assert upstream_ref.upstream_baseline_ref(baseline) == "same-commit"

    (baseline / "extra.py").write_text("x = 1\n")
    _commit_all(baseline, "diverge")
    _git(baseline, "branch", "-f", "same-commit", "main")
    assert upstream_ref.upstream_baseline_ref(baseline) is None


def test_the_unresolved_message_names_both_halves_of_the_rule() -> None:
    message = upstream_ref.unresolved_baseline_message()
    assert upstream_ref.UPSTREAM_BASELINE_TREE in message
    for path in upstream_ref.UPSTREAM_ANCHOR_PATHS:
        assert path in message


# ---------------------------------------------------------------------------
# Against the real repository: the shipped pin, unmonkeypatched.
# ---------------------------------------------------------------------------


PIN_FILE = REPO_ROOT / "tools" / "release" / "UPSTREAM_COMMIT"


@pytest.fixture(scope="module")
def real_baseline_ref() -> str:
    """The ref naming the baseline in whichever tree these tests run in.

    A development checkout has ``main`` at the pinned upstream snapshot; the
    published tree has ``main`` at the release and the annotated tag on commit 1.
    The presence of ``tools/release/UPSTREAM_COMMIT`` distinguishes the two. If the
    chosen ref does not exist (e.g. a clone that fetched no tags), the test skips.
    """
    ref = "main" if PIN_FILE.exists() else upstream_ref.UPSTREAM_BASELINE_TAG
    if not upstream_ref.ref_exists(ref, REPO_ROOT):
        pytest.skip(f"{ref!r} does not exist in this checkout")
    return ref


def test_the_shipped_pin_accepts_this_repositorys_real_baseline(real_baseline_ref: str) -> None:
    """The shipped pin validates this checkout's real baseline ref, and
    `upstream_baseline_ref` resolves to it."""
    ok, lines = upstream_ref.check_upstream_ref(real_baseline_ref, REPO_ROOT)
    assert ok, lines
    assert upstream_ref.upstream_baseline_ref(REPO_ROOT) == real_baseline_ref


def test_the_shipped_pin_refuses_this_repositorys_own_head() -> None:
    """`HEAD` is not accepted as the upstream baseline."""
    ok, lines = upstream_ref.check_upstream_ref("HEAD", REPO_ROOT)
    assert not ok, lines


@pytest.mark.skipif(not PIN_FILE.exists(), reason="tools/release/ is withheld from the published tree")
def test_the_shipped_tree_pin_matches_the_release_builds_commit_pin() -> None:
    """`UPSTREAM_BASELINE_TREE` equals the tree of the commit pinned in
    ``tools/release/UPSTREAM_COMMIT``.

    That pin file is withheld from the published tree, so this test only runs in a
    development checkout. ``build_release_tree.sh`` asserts the same equality during
    a build.
    """
    payload = [
        line for line in PIN_FILE.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(payload) == 1, payload
    pinned_sha = payload[0].split()[1]
    pinned_tree = subprocess.run(
        ["git", "rev-parse", "--verify", f"{pinned_sha}^{{tree}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert pinned_tree == upstream_ref.UPSTREAM_BASELINE_TREE
