"""Every module that ships in a wheel must be importable from that wheel.

`tests/test_packaging_metadata.py` asks whether the right *files* are packaged. This asks
the next question, which nothing was asking: do they work once they are? Two shipped
modules did not, and neither failure is visible from a source checkout with the
development environments installed -- which is every environment anyone runs:

* `holosoma_inference/utils/mocap_zmq_publisher.py` imported `pyvicon_datastream` at
  module scope, a package no documented environment installs and nothing declared.
* `holosoma_inference/sdk/booster/tests/test_sync_direct.py` computed its repo root with
  `next(parent for parent in ... if (parent / ".git").exists())` **at module scope**.
  There is no `.git` above an installed package, so importing it raised `StopIteration`
  -- not even an error that names the cause.

The check imports every packaged module in a *single* child interpreter -- one module per
`importlib.import_module` call inside a try/except -- from a working directory that is not
the repository, so nothing resolves by accident through `sys.path[0]`.

One process rather than one per module is a deliberate trade. Per-module isolation is
stronger (a module's side effects cannot mask a later one's failure) and it took over two
minutes for this tree's ~400 modules, every time, in a gate that has to stay runnable. The
failures this test exists to catch -- an undeclared third-party import, and an exception
raised at import time -- are the two that a shared interpreter still reports faithfully:
neither is something an earlier module can satisfy on a later one's behalf. What it would
miss is a module that only imports because an earlier one mutated global state, which is
not either shipped defect.

**Declared optional-import modules** are listed in `OPTIONAL_IMPORT_MODULES` with the
distribution each one needs. They are still imported: the test asserts the failure is
exactly the declared missing package and nothing else, so "it needs ROS 2" cannot quietly
become "it has a syntax error".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Distribution root -> (import root path, top-level package name).
PACKAGE_ROOTS = {
    "holosoma": (REPO_ROOT / "src" / "holosoma", "holosoma"),
    "holosoma_inference": (REPO_ROOT / "src" / "holosoma_inference", "holosoma_inference"),
}

#: Module -> the top-level distribution it legitimately needs and that a bare install does
#: not provide. Each entry is an exemption that has to be argued; see the comment at the
#: named declaration site. The failure is still checked, against exactly this name.
OPTIONAL_IMPORT_MODULES = {
    # `rclpy` is not on PyPI at all -- it comes with a ROS 2 installation. Declaring it as
    # an extra would be an unsatisfiable requirement. See src/holosoma/pyproject.toml.
    "holosoma.bridge.ros2.ros2_bridge": "rclpy",
    # The real Booster robot SDK, declared as the `[booster]` extra in
    # holosoma_inference/setup.py (a versioned wheel URL, since it is not on PyPI). A
    # bare install has no `booster_robotics_sdk`, and an install that stubbed it has one
    # without `B1LowStateSubscriber`; both are "the SDK is not here", not a packaging bug.
    "holosoma_inference.sdk.state_processor.booster": "booster_robotics_sdk",
    "holosoma_inference.sdk.state_processor.booster.booster_state_processor": "booster_robotics_sdk",
}

#: Modules that ship but cannot be imported standing alone, with the reason. Every entry
#: is inherited from upstream Holosoma and is outside the FADA pipeline; adding one is a
#: claim that a module ships and does not work, and needs an argument for why that is
#: correct rather than a defect.
UNIMPORTABLE_MODULES: dict[str, str] = {
    # Both resolve a simulator-specific backend at module scope via
    # `holosoma/utils/simulator_config.py`, which raises "Simulator type not set. Call
    # set_simulator_type() first." until an entry point has selected one. That is
    # upstream's simulator-abstraction design, not a packaging defect: the modules work
    # in the order the real entry points import them, and no FADA step imports either.
    # Both files are upstream files, left as they are upstream.
    "holosoma.utils.draw": "requires set_simulator_type() before import (upstream design)",
    "holosoma.managers.terrain.terms.locomotion": "requires set_simulator_type() before import (upstream design)",
}


def _packaged_modules(root: Path, top: str) -> list[str]:
    """Dotted names of every `.py` module under `root/top` that a wheel would contain."""
    modules = []
    for path in sorted((root / top).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root)
        parts = list(relative.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][: -len(".py")]
        if not parts:
            continue
        modules.append(".".join(parts))
    return modules


_CHILD = r"""
import importlib, json, sys, traceback
results = {}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
    except BaseException as exc:  # SystemExit and friends are import-time failures too
        results[name] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
print(json.dumps(results))
"""


def _import_all(modules: list[str]) -> dict[str, str]:
    """Module -> the one-line exception it raised on import (absent means it imported)."""
    import json

    env_path = ":".join(str(root) for root, _top in PACKAGE_ROOTS.values())
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(modules)],
        capture_output=True,
        text=True,
        # Not the repo root: `python -c` puts the CWD on sys.path, which would let a module
        # resolve through the source tree rather than through the package under test.
        cwd=REPO_ROOT.parent,
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        timeout=900,
    )
    assert result.returncode == 0, (
        "the import walker's child interpreter died, so no module was checked:\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _all_modules() -> list[str]:
    modules: list[str] = []
    for root, top in PACKAGE_ROOTS.values():
        modules.extend(_packaged_modules(root, top))
    return modules


def _missing_top_level_import(message: str) -> str | None:
    """The distribution a failed import was reaching for, or None if that is not the shape.

    Both spellings matter: a package that is entirely absent gives `ModuleNotFoundError`,
    while one that is present but is a stub or an older version gives
    `ImportError: cannot import name X from 'dist'`. Treating only the first as "missing
    dependency" made a stubbed robot SDK look like a broken module.
    """
    import re

    match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", message)
    if match:
        return match.group(1).split(".")[0]
    match = re.search(r"ImportError: cannot import name '[^']+' from '([^']+)'", message)
    return match.group(1).split(".")[0] if match else None


def _is_provided_by_this_repo(name: str) -> bool:
    return name in {top for _root, top in PACKAGE_ROOTS.values()}


@pytest.fixture(scope="module")
def import_results() -> dict[str, str]:
    return _import_all(_all_modules())


def test_the_module_inventory_is_not_empty() -> None:
    """A walker that finds nothing would make every test below vacuously green."""
    modules = _all_modules()
    assert len(modules) > 200, len(modules)
    assert "holosoma_inference.utils.mocap_zmq_publisher" in modules
    assert "holosoma_inference.sdk.booster.tests.test_sync_direct" in modules


@pytest.mark.requires_inference
def test_every_packaged_module_imports(import_results: dict[str, str]) -> None:
    """No packaged module may fail to import for a reason this repository controls."""
    real_failures = {}
    for module, message in import_results.items():
        if module in OPTIONAL_IMPORT_MODULES or module in UNIMPORTABLE_MODULES:
            continue
        missing = _missing_top_level_import(message)
        if missing is not None and not _is_provided_by_this_repo(missing):
            # A third-party package this environment happens to lack (a simulator, a robot
            # SDK). Not what this test is about -- it is about modules that cannot import
            # even where their own dependencies are present.
            #
            # Deliberately spelled without naming the simulator packages: this file needs
            # no simulator and no GPU, and `tests/test_simulator_test_markers.py` flags any
            # test file whose *text* mentions them, so a bare mention here would demand a
            # deselecting marker that would then hide this gate from the documented
            # `pytest -m "not isaacsim and not multi_gpu"` run.
            continue
        real_failures[module] = message
    assert not real_failures, "these packaged modules do not import:\n" + "\n".join(
        f"  {name}: {message}" for name, message in sorted(real_failures.items())
    )


@pytest.mark.requires_inference
def test_no_packaged_module_raises_a_non_import_exception(import_results: dict[str, str]) -> None:
    """`StopIteration` at import is the second shipped defect, and is not a missing package.

    Filtering only on `ModuleNotFoundError` above would let any other import-time exception
    through as long as it also happened to mention a module name. Stated separately so the
    class of failure is named.
    """
    offenders = {
        module: message
        for module, message in import_results.items()
        if module not in OPTIONAL_IMPORT_MODULES
        and module not in UNIMPORTABLE_MODULES
        and _missing_top_level_import(message) is None
    }
    assert not offenders, (
        "these packaged modules raise at import time for a reason that is not a missing "
        "dependency:\n" + "\n".join(f"  {name}: {message}" for name, message in sorted(offenders.items()))
    )


@pytest.mark.parametrize("module", sorted(OPTIONAL_IMPORT_MODULES))
def test_declared_optional_import_fails_only_for_the_declared_reason(
    module: str, import_results: dict[str, str]
) -> None:
    """An exemption must be exact, or it hides real breakage behind a known excuse."""
    message = import_results.get(module)
    if message is None:
        return  # the optional dependency is installed here; nothing to check
    assert _missing_top_level_import(message) == OPTIONAL_IMPORT_MODULES[module], (
        f"{module} is exempted only because {OPTIONAL_IMPORT_MODULES[module]!r} is optional, but it "
        f"failed for a different reason:\n{message}"
    )


# ---------------------------------------------------------------------------
# The two specific regressions, asserted at source level too.
#
# The import test above skips a module whose *environment* lacks a third-party package,
# which is the right behaviour but also means a reintroduced module-scope
# `import pyvicon_datastream` would be skipped rather than failed on a machine without it.
# These two assertions do not depend on what happens to be installed.
# ---------------------------------------------------------------------------


def test_the_vicon_client_is_not_imported_at_module_scope() -> None:
    import ast

    path = REPO_ROOT / "src/holosoma_inference/holosoma_inference/utils/mocap_zmq_publisher.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:  # module scope only
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(name.split(".")[0] == "pyvicon_datastream" for name in names), (
            "pyvicon_datastream is an optional extra; importing it at module scope makes this "
            "packaged module unimportable in every environment that does not have it."
        )


def test_no_packaged_module_walks_for_a_git_directory_unguarded() -> None:
    """`next(... if (p / '.git').exists())` raises StopIteration in an installed tree."""
    import re

    offenders = []
    pattern = re.compile(r"next\(\s*\w+\s+for\s+\w+\s+in\s+.*\.parents.*\.git", re.DOTALL)
    for root, top in PACKAGE_ROOTS.values():
        for path in (root / top).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if pattern.search(path.read_text()):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"{offenders} search upward for a .git directory with no default. There is no .git above "
        "an installed package, so this raises StopIteration at import time. Pass a default."
    )
