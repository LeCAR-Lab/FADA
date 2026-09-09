"""Contract for :mod:`holosoma.utils.safe_torch_load` and its call sites.

The documented pipeline tells a user to download an oracle checkpoint *directory*
from a third-party host and hand it to ``--expert-checkpoint``, where
``--suboptimal-data-ratio`` reads twenty of the files in it.  ``torch.load`` is a
pickle reader, so the mode it runs in is the difference between "parse a tensor
file" and "execute whatever the file says".  These tests pin three things:

1. the helper asks for the restricted reader first,
2. when the restricted reader refuses a payload, the helper **fails closed** --
   it raises and executes nothing, and the permissive reader is reachable only
   through an explicit per-call opt-in that no release call site passes, and
3. every checkpoint load site in the release goes through the helper rather than
   calling ``torch.load`` directly.
"""

from __future__ import annotations

import contextlib
import io
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
from holosoma.utils.safe_torch_load import (
    UNRESTRICTED_LOAD_BANNER_MARKER,
    RestrictedLoadRejected,
    load_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


@contextlib.contextmanager
def no_user_warning():
    """Assert no ``UserWarning`` escapes the block (``pytest.warns(None)`` is gone in pytest 8)."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        yield
        offenders = [str(r.message) for r in records if issubclass(r.category, UserWarning)]
    assert not offenders, f"unexpected UserWarning(s): {offenders}"


class _NotATensor:
    """A plain object the restricted unpickler must refuse to reconstruct."""

    def __init__(self, value: int = 7) -> None:
        self.value = value


# ---------------------------------------------------------------------------
# 1. The restricted reader is the default
# ---------------------------------------------------------------------------


def test_plain_tensor_payload_loads_without_any_warning(tmp_path: Path, capsys) -> None:
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.arange(4, dtype=torch.float32), "iter": 3}, path)

    with no_user_warning():
        payload = load_checkpoint(path, map_location="cpu")

    assert torch.equal(payload["w"], torch.arange(4, dtype=torch.float32))
    assert payload["iter"] == 3
    assert UNRESTRICTED_LOAD_BANNER_MARKER not in capsys.readouterr().err


def test_restricted_mode_is_actually_requested(tmp_path: Path, monkeypatch) -> None:
    """The first attempt passes ``weights_only=True`` explicitly rather than relying on a default.

    torch's own default flipped to True only in 2.6, and this release supports
    older versions.
    """
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.zeros(2)}, path)

    seen: list[dict] = []
    real_load = torch.load

    def _spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy)
    load_checkpoint(path, map_location="cpu")

    assert seen, "torch.load was never called"
    assert seen[0].get("weights_only") is True
    assert len(seen) == 1, "a clean payload must not trigger a second, permissive load"


def test_map_location_is_forwarded_and_omitted_when_unset(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.zeros(2)}, path)

    seen: list[dict] = []
    real_load = torch.load

    def _spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy)
    load_checkpoint(path, map_location="cpu")
    load_checkpoint(path)

    assert seen[0]["map_location"] == "cpu"
    assert "map_location" not in seen[1]


# ---------------------------------------------------------------------------
# 2. A rejected payload fails closed
# ---------------------------------------------------------------------------


class _MarkerWritingReducer:
    """A ``torch.save``-able object whose *unpickling* runs a shell command.

    The pickle stream's ``REDUCE`` opcode names ``os.system``, so any reader that
    reconstructs it runs the command before the caller sees a single byte of the
    payload. The restricted unpickler refuses to resolve ``os.system`` at all; the
    permissive one calls it.
    """

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def _write_malicious_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    marker = tmp_path / "PWNED"
    path = tmp_path / "evil.pt"
    torch.save({"model": _MarkerWritingReducer(marker)}, path)
    assert not marker.exists(), "torch.save must not have run it -- only loading does"
    return path, marker


def test_malicious_reducer_payload_is_rejected_without_executing(tmp_path: Path, capsys) -> None:
    """A hostile checkpoint raises ``RestrictedLoadRejected`` and its payload never executes."""
    path, marker = _write_malicious_checkpoint(tmp_path)

    with pytest.raises(RestrictedLoadRejected) as excinfo:
        load_checkpoint(path, map_location="cpu")

    assert not marker.exists(), (
        "the malicious payload executed -- a rejected checkpoint must never reach the "
        "permissive reader without an explicit opt-in"
    )
    assert str(path) in str(excinfo.value), "the error must name the file it refused"
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)
    assert UNRESTRICTED_LOAD_BANNER_MARKER not in capsys.readouterr().err


def test_rejection_is_not_softened_by_a_stderr_capturing_logger(tmp_path: Path) -> None:
    """Redirecting stderr and ignoring ``UserWarning`` does not change the fail-closed behaviour."""
    path, marker = _write_malicious_checkpoint(tmp_path)

    sink = io.StringIO()
    with contextlib.redirect_stderr(sink), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RestrictedLoadRejected):
            load_checkpoint(path, map_location="cpu")

    assert not marker.exists()
    assert sink.getvalue() == "", "nothing is printed on the fail-closed path"


def test_benign_payload_the_restricted_reader_refuses_also_fails_closed(tmp_path: Path) -> None:
    """A benign payload the restricted reader refuses takes the same path as a hostile one.

    Nothing at the rejection point distinguishes "an old checkpoint holding a
    plain object" from "an attack": both arrive as ``UnpicklingError``.
    """
    path = tmp_path / "rich.pt"
    torch.save({"w": torch.zeros(2), "obj": _NotATensor(11)}, path)

    with pytest.raises(pickle.UnpicklingError):
        torch.load(path, map_location="cpu", weights_only=True)
    with pytest.raises(RestrictedLoadRejected):
        load_checkpoint(path, map_location="cpu")


def test_the_rejection_message_names_the_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "rich.pt"
    torch.save({"w": torch.zeros(2), "obj": _NotATensor()}, path)

    with pytest.raises(RestrictedLoadRejected) as excinfo:
        load_checkpoint(path, map_location="cpu")

    message = str(excinfo.value)
    assert "allow_unrestricted=True" in message, "a dead end without a documented way out is a bad error"
    assert "does NOT retry" in message


def test_explicit_opt_in_loads_and_still_prints_the_banner(tmp_path: Path, capsys) -> None:
    """``allow_unrestricted=True`` loads the payload and prints the banner to stderr."""
    path = tmp_path / "rich.pt"
    torch.save({"w": torch.zeros(2), "obj": _NotATensor(11)}, path)

    with pytest.warns(UserWarning, match=UNRESTRICTED_LOAD_BANNER_MARKER):
        payload = load_checkpoint(path, map_location="cpu", allow_unrestricted=True)
    assert payload["obj"].value == 11

    stderr = capsys.readouterr().err
    assert UNRESTRICTED_LOAD_BANNER_MARKER in stderr
    assert str(path) in stderr
    assert "trust decision" in stderr
    assert "weights_only=False" in stderr


def test_opt_in_is_keyword_only_and_defaults_to_false() -> None:
    """``allow_unrestricted`` is keyword-only and defaults to False.

    ``load_checkpoint(path, True)`` therefore does not reach the permissive reader.
    """
    import inspect

    parameter = inspect.signature(load_checkpoint).parameters["allow_unrestricted"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


def test_opt_in_does_not_change_a_clean_payload(tmp_path: Path, capsys) -> None:
    """The opt-in only ever applies *after* a rejection; it is not a mode switch."""
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.zeros(2)}, path)

    seen: list[dict] = []
    real_load = torch.load

    def _spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(torch, "load", _spy)
    try:
        with no_user_warning():
            load_checkpoint(path, map_location="cpu", allow_unrestricted=True)
    finally:
        monkeypatch.undo()

    assert [call.get("weights_only") for call in seen] == [True]
    assert UNRESTRICTED_LOAD_BANNER_MARKER not in capsys.readouterr().err


def test_missing_file_is_not_turned_into_a_permissive_load(tmp_path: Path, capsys) -> None:
    """Only a restricted-mode *rejection* is reinterpreted at all."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does-not-exist.pt", map_location="cpu")
    assert UNRESTRICTED_LOAD_BANNER_MARKER not in capsys.readouterr().err


def test_permission_error_propagates_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.zeros(2)}, path)
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):  # pragma: no cover - running as root
            pytest.skip("cannot make a file unreadable as this user")
        with pytest.raises(PermissionError):
            load_checkpoint(path, map_location="cpu")
    finally:
        path.chmod(0o600)


# ---------------------------------------------------------------------------
# 2b. No "old torch" downgrade path exists to be tricked
# ---------------------------------------------------------------------------


def test_a_wrapped_modern_torch_load_is_not_downgraded(tmp_path: Path, monkeypatch, capsys) -> None:
    """A ``TypeError`` naming ``weights_only`` surfaces as itself, with no permissive retry.

    A wrapper declared ``wrapper(*args, **kwargs)`` -- what a profiler, tracer or
    monkeypatch looks like -- hides ``weights_only`` from ``inspect.signature``, so
    a signature-based "is this an old torch?" test would misread a modern install.
    """
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.zeros(2)}, path)

    calls: list[dict] = []

    def _opaque_wrapper(*args, **kwargs):
        calls.append(dict(kwargs))
        raise TypeError("load() got an unexpected keyword argument 'weights_only'")

    import inspect

    # The wrapper hides ``weights_only`` from signature introspection.
    assert "weights_only" not in inspect.signature(_opaque_wrapper).parameters

    monkeypatch.setattr(torch, "load", _opaque_wrapper)

    with pytest.raises(TypeError, match="weights_only"):
        load_checkpoint(path, map_location="cpu")

    assert len(calls) == 1, "there must be no retry at all"
    assert calls[0].get("weights_only") is True
    assert UNRESTRICTED_LOAD_BANNER_MARKER not in capsys.readouterr().err


def test_typeerror_always_surfaces_as_itself(tmp_path: Path, monkeypatch, capsys) -> None:
    """A malformed call must never become a downgrade to permissive mode."""
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.zeros(2)}, path)

    def _modern_but_broken(f, map_location=None, weights_only=None, **kwargs):
        raise TypeError("something else entirely")

    monkeypatch.setattr(torch, "load", _modern_but_broken)

    with pytest.raises(TypeError, match="something else entirely"):
        load_checkpoint(path, map_location="cpu")
    assert UNRESTRICTED_LOAD_BANNER_MARKER not in capsys.readouterr().err


def test_no_signature_introspection_remains_in_the_helper() -> None:
    """``safe_torch_load.py`` does not import ``inspect``.

    ``inspect.signature(torch.load)`` cannot answer "does this torch support
    ``weights_only``" for a wrapped callable, so the helper does not ask it. This
    is a source assertion because it guards against a code path being added, not a
    behaviour reachable from the current one.
    """
    import ast

    tree = ast.parse((REPO_ROOT / "src/holosoma/holosoma/utils/safe_torch_load.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "inspect" not in imported, "signature introspection is back; see this test's docstring"


# Both distributions that import the helper. ``weights_only`` landed in torch 1.13,
# so a declared floor at or above it guarantees the argument exists.
_TORCH_FLOOR_DECLARATIONS = (
    "src/holosoma/pyproject.toml",
    "src/holosoma_inference/setup.py",
)


@pytest.mark.parametrize("relative_path", _TORCH_FLOOR_DECLARATIONS)
def test_declared_torch_floor_guarantees_weights_only_exists(relative_path: str) -> None:
    """Each distribution declares exactly one torch requirement, and its floor excludes 1.12."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    source = (REPO_ROOT / relative_path).read_text()
    requirements = re.findall(r'"(torch(?![-_a-zA-Z])[^"]*)"', source)
    assert len(requirements) == 1, f"{relative_path}: expected exactly one torch requirement, got {requirements}"

    specifier = SpecifierSet(requirements[0][len("torch") :])
    assert Version("1.12.1") not in specifier, (
        f"{relative_path} declares {requirements[0]!r}, which admits torch 1.12 -- a version with no "
        "weights_only argument. safe_torch_load.py has no fallback for that, by design; either raise "
        "the floor or reintroduce the branch knowingly."
    )
    # The range must also admit the versions the release runs on, not only its floor.
    assert Version("2.0.0") in specifier and Version("2.6.0") in specifier


# ---------------------------------------------------------------------------
# 3. No release code path bypasses the helper
# ---------------------------------------------------------------------------

# Every module that reads a checkpoint file. Each must route through the helper.
CHECKPOINT_LOAD_SITES = (
    "src/holosoma/holosoma/fada/planner_idm/eval_checkpoint.py",
    "src/holosoma/holosoma/fada/planner_idm/finetune_idm_lora.py",
    "src/holosoma/holosoma/fada/planner_idm/train.py",
    "src/holosoma/holosoma/fada/planner_idm/trainer.py",
    "src/holosoma/holosoma/agents/ppo/ppo.py",
    "src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py",
    "src/holosoma/holosoma/utils/eval_utils.py",
    "src/holosoma_inference/holosoma_inference/utils/compute_mocap_velocity_metrics.py",
)

_DIRECT_LOAD = re.compile(r"\btorch\.load\s*\(")


@pytest.mark.parametrize("relative_path", CHECKPOINT_LOAD_SITES)
def test_call_site_does_not_call_torch_load_directly(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text()
    offenders = [line.strip() for line in source.splitlines() if _DIRECT_LOAD.search(line)]
    # Docstring/comment mentions are fine; a bare call is not.
    offenders = [line for line in offenders if not line.startswith(("#", ">>>", "*"))]
    assert not offenders, f"{relative_path} still calls torch.load directly: {offenders}"
    assert "safe_torch_load" in source, f"{relative_path} does not import the restricted-load helper"


def test_no_release_module_passes_weights_only_false() -> None:
    """``weights_only=False`` must appear only inside the helper's own fallback."""
    hits: list[str] = []
    for package in ("src/holosoma/holosoma", "src/holosoma_inference/holosoma_inference"):
        for path in (REPO_ROOT / package).rglob("*.py"):
            if path.name == "safe_torch_load.py" or "/tests/" in path.as_posix():
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if "weights_only=False" in line and not line.lstrip().startswith("#"):
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not hits, f"permissive torch.load outside the helper: {hits}"


def test_no_release_module_opts_into_unrestricted_loading() -> None:
    """No release module passes ``allow_unrestricted``.

    The opt-in is for a caller with a file they trust, not for shipped code: a call
    site that passes it defeats the helper's default for every user of that entry
    point.
    """
    hits: list[str] = []
    for package in ("src/holosoma/holosoma", "src/holosoma_inference/holosoma_inference"):
        for path in (REPO_ROOT / package).rglob("*.py"):
            if path.name == "safe_torch_load.py" or "/tests/" in path.as_posix():
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if "allow_unrestricted" in line and not line.lstrip().startswith("#"):
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not hits, f"release code opts into the permissive reader: {hits}"


def test_helper_module_has_no_import_time_torch_dependency() -> None:
    """Importing the helper does not import torch; ``compute_mocap_velocity_metrics`` relies on this."""
    script = "import sys; import holosoma.utils.safe_torch_load; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    assert result.stdout.strip() == "False"
