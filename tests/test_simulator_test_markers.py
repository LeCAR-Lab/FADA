"""Every test that needs a simulator or a GPU must be deselectable by the documented command.

The suite is meant to run as

    pytest -m "not isaacsim" src/ tests/

on a checkout that has neither IsaacGym/IsaacSim nor necessarily a CUDA device. A test that
builds real envs or hardcodes ``device="cuda"`` but carries no marker is not skipped by that
command -- it is *run*, and it fails with ``ModuleNotFoundError: isaacgym``.

The candidate set is re-derived from source on every run, so a simulator/GPU test added
without a marker fails here too.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories whose tests are explicitly out of scope for the documented command.
# `tests/e2e/` is run with `--ignore=tests/e2e` everywhere it appears, and is the one
# place where "this needs a real simulator" is the entire point of the directory.
EXCLUDED_DIRS = (REPO_ROOT / "tests" / "e2e",)

SEARCH_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "tests")

# What makes a module "needs hardware/simulator". Deliberately textual and broad: a false
# positive costs one marker, a false negative costs a stranger a failing suite.
_NEEDS_HARDWARE = re.compile(
    r"""
    import\s+isaacgym          # the simulator module itself
    | \bisaacgym\b
    | device\s*=\s*["']cuda    # a hardcoded CUDA device (no cpu fallback)
    | ["']cuda:0["']
    """,
    re.VERBOSE,
)

# Markers that make a test disappear under the documented invocation. `skip` is included
# because an unconditionally skipped test also never runs, which is what the reader needs;
# it is the convention already used by test_push_randomization.py in the same directory.
DESELECTING_MARKERS = frozenset({"isaacsim", "multi_gpu", "skip"})

# The exact deselect expression a reader is told to use, plus the multi_gpu half that the
# repo's own CI adds. Kept as a literal so a change to the documented command has to be
# made here too.
DESELECT_EXPR = "not isaacsim and not multi_gpu"


def _candidate_test_files() -> list[Path]:
    files: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("test_*.py")):
            if any(excluded in path.parents for excluded in EXCLUDED_DIRS):
                continue
            # This file names the patterns it looks for, so it matches its own detector.
            if path == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
                continue
            if _NEEDS_HARDWARE.search(text):
                files.append(path)
    return files


def _marker_names(decorator: ast.expr) -> list[str]:
    """Marker names reachable from one decorator expression.

    Handles `@pytest.mark.foo`, `@pytest.mark.foo(...)`, and bare `@mark.foo`.
    """
    node = decorator
    if isinstance(node, ast.Call):
        node = node.func
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    return names


def _unmarked_tests(path: Path) -> list[str]:
    """Test functions in `path` that no deselecting marker covers (module-level or decorator)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    module_marked = False
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        sources = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
        if any(set(_marker_names(src)) & DESELECTING_MARKERS for src in sources):
            module_marked = True
    if module_marked:
        return []

    unmarked: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        marked = any(set(_marker_names(dec)) & DESELECTING_MARKERS for dec in node.decorator_list)
        if not marked:
            unmarked.append(node.name)
    return unmarked


def test_candidate_set_is_not_empty() -> None:
    """Guard the guard: a detector that finds nothing passes vacuously forever."""
    candidates = _candidate_test_files()
    assert candidates, "the simulator/GPU test detector matched no files -- its heuristic has rotted"


@pytest.mark.parametrize("path", _candidate_test_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_hardware_tests_are_deselectable(path: Path) -> None:
    unmarked = _unmarked_tests(path)
    assert not unmarked, (
        f"{path.relative_to(REPO_ROOT)} needs a simulator and/or a CUDA device, but "
        f"{unmarked} carry none of {sorted(DESELECTING_MARKERS)}. "
        f'`pytest -m "{DESELECT_EXPR}"` therefore RUNS them on a checkout that has neither, '
        "and they fail with ModuleNotFoundError. Mark them."
    )


def test_e2e_env_test_is_actually_deselected() -> None:
    """The static check above, confirmed against real pytest collection.

    A marker that pytest does not honour (misspelled, unregistered under
    `--strict-markers`, shadowed) would still satisfy the AST scan. This runs the
    documented deselect expression for real over the module the reviewer found failing.
    """
    target = REPO_ROOT / "src" / "holosoma" / "holosoma" / "envs" / "tests" / "test_e2e.py"
    assert target.is_file(), target
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", DESELECT_EXPR, str(target)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert "test_e2e_step" not in combined, (
        "test_e2e.py::test_e2e_step survives the documented deselect expression "
        f'`-m "{DESELECT_EXPR}"`, so a stranger running the documented command runs it '
        f"and hits ModuleNotFoundError: isaacgym.\n{combined[-4000:]}"
    )


# ---------------------------------------------------------------------------
# A test that never runs is documentation, not verification.
# ---------------------------------------------------------------------------
# `envs/tests/test_push_randomization.py` carried `@pytest.mark.skip(reason="Cannot run
# multiple Isaac Gym instances in a single process")` on both of its tests, in a module
# that defines a module-scoped `shared_env` fixture precisely so that it creates one
# instance. The reason was stale; the effect was that a file whose docstring said it
# "verifies" two behaviours verified neither, under every invocation, forever. Neither the
# marker check above nor the pytest gate in `tools/release/build_release_tree.sh` could see
# it: a skipped test is deselected-shaped to one and a green record to the other.
#
# So an unconditional skip is now a build-visible error, and the only accepted way to say
# "this needs hardware" is a marker the documented command deselects.
#
# Deliberately narrow: `@pytest.mark.skipif(...)` is untouched (it names a condition, and
# runs when the condition is false), and so is `pytest.param(..., marks=pytest.mark.skip(
# ...))` built at collection time from the environment -- `test_finetune_data_path_parity`
# uses that to include real H5 rollouts when `FADA_PARITY_REAL_H5` points at some. What is
# forbidden is the decorator form on a test function or class, which no invocation can
# turn back on.


def _unconditionally_skipped_tests(path: Path) -> list[str]:
    """`@pytest.mark.skip`-decorated tests and classes in `path` (not `skipif`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not node.name.startswith(("test", "Test")):
            continue
        for decorator in node.decorator_list:
            names = _marker_names(decorator)
            if names and names[0] == "skip" and "mark" in names:
                found.append(node.name)
    return found


def _all_shipped_test_files() -> list[Path]:
    files: list[Path] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("test_*.py")):
            if any(excluded in path.parents for excluded in EXCLUDED_DIRS):
                continue
            files.append(path)
    return files


def test_the_shipped_test_file_set_is_not_empty() -> None:
    """Guard the guard below: an empty file set passes it vacuously forever."""
    assert len(_all_shipped_test_files()) > 20, "the shipped test file scan found almost nothing"


def test_no_shipped_test_is_unconditionally_skipped() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _all_shipped_test_files():
        skipped = _unconditionally_skipped_tests(path)
        if skipped:
            offenders[str(path.relative_to(REPO_ROOT))] = skipped
    assert not offenders, (
        f"unconditionally skipped test(s): {offenders}. A `@pytest.mark.skip` decorator "
        "cannot be turned on by any invocation, so these never run while their file's "
        "docstring claims they verify something. Use `@pytest.mark.isaacsim` (or "
        "`@pytest.mark.skipif` with a real condition) if the test needs hardware, or delete "
        "the test and the claim together."
    )
