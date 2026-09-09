"""Checkpoint loading through PyTorch's restricted (``weights_only``) unpickler.

``torch.load`` is a pickle reader: with ``weights_only=False`` it executes
whatever the file says to execute, before any of this project's code inspects
the payload. ``weights_only=True`` restricts the unpickler to tensors and plain
containers, so such a file fails to load instead of running.

**This module fails closed.** A payload the restricted unpickler refuses is
*not* retried permissively. :func:`load_checkpoint` raises
:class:`RestrictedLoadRejected` and nothing in the file is executed.

The permissive reader is reachable only as an explicit per-call opt-in:
``load_checkpoint(path, allow_unrestricted=True)``. No call site in this release
passes it, and ``src/holosoma/tests/utils/test_safe_torch_load.py`` asserts
that. When it is passed and used, a banner is printed to stderr and a
:class:`UserWarning` is raised.

Errors that mean "this file could not be read at all" --
:class:`FileNotFoundError`, :class:`PermissionError` and friends -- propagate
untouched and are never reinterpreted.

There is no "old torch" branch: both distributions that import this helper
declare ``torch>=2.0.0`` (``src/holosoma/pyproject.toml`` and
``src/holosoma_inference/setup.py``) and ``weights_only`` has existed since
torch 1.13, so a :class:`TypeError` from ``torch.load`` is a real API error and
surfaces as itself.

This module has no import-time dependency on torch: ``torch`` is imported inside
the function so that callers which are themselves lazy about torch (for example
``holosoma_inference.utils.compute_mocap_velocity_metrics``) keep their existing
import surface. That plain ``import torch`` respects
``holosoma.utils.safe_torch_import``'s isaacgym-before-torch ordering rule: every
caller has already imported torch (through that module or directly) by the time
it has a checkpoint path to hand here, so the import inside
:func:`load_checkpoint` only ever resolves an entry already in ``sys.modules``.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

__all__ = ["UNRESTRICTED_LOAD_BANNER_MARKER", "RestrictedLoadRejected", "load_checkpoint"]

UNRESTRICTED_LOAD_BANNER_MARKER = "UNRESTRICTED CHECKPOINT LOAD"
"""Stable string every opt-in banner contains, so tests and log greps can find it."""

# Errors that mean "the restricted unpickler refused this payload", as opposed to
# "this file could not be read at all".  torch raises ``pickle.UnpicklingError``
# for a disallowed global; ``AttributeError`` / ``ModuleNotFoundError`` /
# ``ImportError`` show up when the payload references a symbol the restricted
# allowlist resolves differently.  ``OSError`` (and therefore
# ``FileNotFoundError`` / ``PermissionError``) is absent, so those propagate.
_RESTRICTED_MODE_REJECTIONS = (pickle.UnpicklingError, AttributeError, ImportError)


class RestrictedLoadRejected(RuntimeError):
    """``torch.load(..., weights_only=True)`` refused a checkpoint.

    Raised instead of retrying permissively. The original rejection is attached
    as ``__cause__``, so the underlying :class:`pickle.UnpicklingError` (or the
    attribute/module lookup failure that stood in for it) is still available.
    """


def _emit_banner(lines: list[str]) -> None:
    """Print an unmissable banner to stderr *and* raise a warning."""
    width = max(len(line) for line in lines)
    rule = "=" * min(width, 100)
    print(rule, file=sys.stderr, flush=True)
    for line in lines:
        print(line, file=sys.stderr, flush=True)
    print(rule, file=sys.stderr, flush=True)
    warnings.warn(" ".join(lines), UserWarning, stacklevel=3)


def _warn_opt_in_unrestricted(path: Path | str) -> None:
    _emit_banner(
        [
            f"WARNING: {UNRESTRICTED_LOAD_BANNER_MARKER} (allow_unrestricted=True was passed)",
            f"  file:  {path}",
            "  torch.load(..., weights_only=True) rejected this checkpoint, and the caller",
            "  explicitly opted into re-reading it with weights_only=False, which EXECUTES",
            "  code embedded in the file. This is a trust decision that has now been made.",
            "  No released FADA checkpoint needs this path -- an oracle or student checkpoint",
            "  that does is either not one of ours, or was not produced by this codebase.",
        ]
    )


def _rejection_message(path: Path | str, exc: BaseException) -> str:
    cause = str(exc).splitlines()[0] if str(exc) else "<no message>"
    return "\n".join(
        [
            f"torch.load(..., weights_only=True) refused to read {path}",
            f"  cause: {type(exc).__name__}: {cause}",
            "",
            "The restricted unpickler rejects payloads that would reconstruct arbitrary",
            "objects, which is also exactly what a malicious checkpoint does. This loader",
            "does NOT retry with weights_only=False, because that would execute the file's",
            "pickle stream before anything could inspect it.",
            "",
            "No released FADA checkpoint needs the permissive reader; a file that does is",
            "either not one of ours, was not produced by this codebase, or is corrupt.",
            "",
            "If -- and only if -- you trust the origin of this specific file, a caller can",
            "opt in per call with load_checkpoint(path, allow_unrestricted=True).",
        ]
    )


def load_checkpoint(path: Path | str, *, map_location: Any = None, allow_unrestricted: bool = False) -> Any:
    """``torch.load`` ``path`` through the restricted (``weights_only=True``) unpickler.

    Args:
        path: Checkpoint file to read.
        map_location: Forwarded to ``torch.load`` when not ``None``.
        allow_unrestricted: Opt in to re-reading the file with
            ``weights_only=False`` if -- and only if -- the restricted reader
            refuses it. **This executes code embedded in the file.** Defaults to
            ``False``, and no call site in this release passes ``True``.

    Returns:
        The unpickled payload, exactly as ``torch.load`` would have returned it.

    Raises:
        RestrictedLoadRejected: The restricted unpickler refused the payload and
            ``allow_unrestricted`` was not set. Nothing in the file was executed.
        Exception: Anything else ``torch.load`` raises -- notably
            :class:`FileNotFoundError` -- is propagated unchanged.
    """
    import torch  # noqa: PLC0415 - lazy import; see this module's docstring

    kwargs: dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location

    try:
        return torch.load(path, weights_only=True, **kwargs)
    except _RESTRICTED_MODE_REJECTIONS as exc:
        if not allow_unrestricted:
            raise RestrictedLoadRejected(_rejection_message(path, exc)) from exc
        _warn_opt_in_unrestricted(path)
        return torch.load(path, weights_only=False, **kwargs)
