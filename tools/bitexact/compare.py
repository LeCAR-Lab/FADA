"""Bit-exactness comparison of the golden and current probe JSON.

Compares ``loss_sequence`` (full-precision floats) and the
state_dict/onnx/adapter ``*_sha256`` values, never a hash of a whole checkpoint
payload: the `cfg` dict inside such a payload changes its key set whenever a
config field is folded away.

Keys in ``EXCLUDED_KEYS`` are carried through the JSON verbatim as diagnostics but
do not affect pass/fail. The one entry, ``config_fingerprint``, is
``sha256(json.dumps(vars(cfg)))`` and so moves whenever `FADAConfig`'s field set
changes, independently of numerics.

The exclusion is an exact key-name match (``key in EXCLUDED_KEYS``), not a
prefix/substring match: only the literal ``"config_fingerprint"`` is skipped and
similarly named keys are compared as normal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_NAMES = ["train", "train_mixed", "export", "sft"]

# Keys excluded from pass/fail comparison; matched by exact key name only.
EXCLUDED_KEYS = frozenset({"config_fingerprint"})


def _diff(path: str, a: Any, b: Any) -> list[str]:
    if type(a) is not type(b):
        return [f"{path}: type mismatch golden={type(a).__name__} current={type(b).__name__}"]
    if isinstance(a, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            if key in EXCLUDED_KEYS:
                continue
            if key not in a:
                out.append(f"{path}.{key}: absent from golden")
            elif key not in b:
                out.append(f"{path}.{key}: absent from current")
            else:
                out += _diff(f"{path}.{key}", a[key], b[key])
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length mismatch golden={len(a)} current={len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(f"{path}[{i}]", x, y)
        return out
    if a != b:
        return [f"{path}: golden={a!r} current={b!r}"]
    return []


def compare_dirs(golden: Path, current: Path, names: list[str] | None = None) -> list[str]:
    names = names or DEFAULT_NAMES
    diffs: list[str] = []
    for name in names:
        g, c = golden / f"{name}.json", current / f"{name}.json"
        if not g.exists():
            diffs.append(f"{name}: golden missing at {g}")
            continue
        if not c.exists():
            diffs.append(f"{name}: current missing at {c}")
            continue
        diffs += _diff(name, json.loads(g.read_text()), json.loads(c.read_text()))
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path("tools/bitexact/golden"))
    parser.add_argument("--current", type=Path, default=Path("tools/bitexact/current"))
    args = parser.parse_args()
    diffs = compare_dirs(args.golden, args.current)
    if not diffs:
        for name in DEFAULT_NAMES:
            print(f"{name}: IDENTICAL")
        # Each probe runs a fixed numerical fixture (loss sequence +
        # state_dict/onnx/adapter sha256) through a slice of the real code rather
        # than the full CLI entrypoint, so the scope of "IDENTICAL" is those
        # fixtures, not every shipped path.
        print(f"{len(DEFAULT_NAMES)} numerical probes identical (not a claim that all release paths are drift-free)")
        return 0
    print(f"Found {len(diffs)} difference(s); first 20:")
    for line in diffs[:20]:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
