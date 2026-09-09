#!/usr/bin/env python3
"""Static check: flags any git-tracked file that mentions an identifier listed in
``STALE_IDENTIFIERS`` (names removed or renamed out of this release), except where
a ``WHITELIST`` entry or ``SKIPPED_PATHS`` excuses it. Exits 1 on any hit.

Matching method
---------------
Each identifier is compiled into a pattern that:

1. Splits the identifier on its own internal ``-``/``_``/whitespace into bare
   parts (e.g. ``"transformer_planner_idm"`` -> ``["transformer", "planner",
   "idm"]``), then rejoins them with a flexible separator class (``[-_\\s]+``),
   so ``transformer_planner_idm``, ``transformer-planner-idm`` and
   ``transformer planner idm`` all match, case-insensitively.
2. Wraps the result in alnum-boundary lookarounds
   (``(?<![A-Za-z0-9])...(?![A-Za-z0-9])``) rather than ``\\b``. ``\\b`` counts
   ``_`` as a word character, so ``\\btransformer_planner_idm\\b`` does not match
   inside ``locomotion_transformer_planner_idm.py``; the alnum boundary treats
   ``_`` as a boundary and does match there.

Consequences of the alnum boundary for the short entries in the list:

* ``dela`` matches a standalone ``DeLA`` token but not ``delay``, ``delattr``
  or ``Delaunay`` (the next character is alphanumeric in each).
* ``mip`` does not match inside ``mipmap``. It can still match byte sequences
  that decode to ``mip`` inside binary assets (``.npz``/``.STL``/``.onnx``);
  those files are skipped by the binary check instead (see
  ``_is_probably_binary``).

Not matched: ``holosoma.transformer_chunk_twin.v1``, a versioned wire-protocol
string that a ZMQ publisher and consumer must spell identically. The list holds
only compound names (``transformer_dagger``, ``transformer_planner_idm``), no
bare ``transformer`` entry.

Whitelist mechanism
-------------------
A whitelist entry is scoped to an identifier, a file, and optionally a line
substring; it excuses only matching lines. An entry whose line never occurs is
inert, not an error. ``SKIPPED_PATHS`` files are not scanned at all.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# This script's own path relative to REPO_ROOT. It spells out every stale
# identifier literally (in STALE_IDENTIFIERS, in the docstring, in the
# whitelist) and is excluded from the scan.
SELF_PATH = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()

# Files present in the working repository but absent from the published tree.
# They quote retired names by design.
#
# Invariant: every path here must be one the release withholds. A path listed
# here is never scanned, published or not, and nothing in this file detects a
# violation. Confirm the published tree does not contain a path before adding it.
SKIPPED_PATHS: frozenset[str] = frozenset(
    {
        "CLAUDE.md",
        "UPSTREAM_DEVIATIONS.md",
        "tools/MUTATION_MATRIX.md",
        "tools/bitexact/golden/CHANGELOG.md",
        "src/holosoma/tests/fada/test_finetune_data_path_parity.py",
    }
)


# ---------------------------------------------------------------------------
# The identifier list. Each entry carries the retired `name` that is matched
# and a `note` recording what it named and what replaced or removed it. Extend
# it from `git log --diff-filter=D --name-only` / `--diff-filter=R` when a
# rename or removal lands.
# ---------------------------------------------------------------------------
STALE_IDENTIFIERS: list[dict[str, str]] = [
    {
        "name": "transformer_dagger",
        "note": (
            "Former top-level package `holosoma.transformer_dagger` (+ "
            "`transformer_dagger_planner_idm`, and the wrapper scripts named after "
            "them). Folded into `holosoma.fada.common` / `holosoma.fada.planner_idm`."
        ),
    },
    {
        "name": "transformer_planner_idm",
        "note": (
            "Former top-level package name for what is now `holosoma.fada.planner_idm` "
            "(see the `transformer_dagger` entry above). The module "
            "`locomotion_transformer_planner_idm.py` was the one straggler left behind "
            "by that rename, and is why this checker exists."
        ),
    },
    {
        "name": "dela",
        "note": (
            "Former inference-policy-class name family (`BasePolicy_DeLA`, "
            "`LocomotionPolicy_DeLA`), renamed to `..._Deploy`; the planner-IDM "
            "inference policy became `LocomotionPolicy_FADA`. NOTE: `dela` is a 4-letter substring "
            "of common English words (delay, delayed, delattr, Delaunay) -- see the "
            "module docstring above for how the alnum-boundary match avoids false "
            "positives on those without resorting to `\\b`."
        ),
    },
    {
        "name": "denet",
        "note": (
            "Deleted DeNet module/config family (`PPO_DeNetConfig`, `ppo_denet.py`, "
            "`denet_latent`, `loco_*_DeNet` presets, and the scripts named after "
            "them). A pre-FADA latent-dynamics baseline, dropped because this "
            "release ships only the FADA pipeline; nothing in the six documented "
            "steps constructs or loads it."
        ),
    },
    {
        "name": "mip",
        "note": (
            "Deleted `policy_mode=\"mip\"` value and the script family named after it "
            "(motion-imitation-pretraining). Removed from "
            "`TaskConfig.policy_mode`'s accepted values alongside the DeNet family "
            "above, for the same reason: no dispatch target for it ships. "
            "NOTE: `mip` is short and appears as decoded byte-garbage "
            "inside binary asset files (`.npz`/`.STL`/`.onnx`) -- this checker skips "
            "binary files entirely (see `_is_probably_binary`) rather than trying to "
            "regex its way around that."
        ),
    },
    {
        "name": "mlp_baseline",
        "note": (
            "Deleted standalone MLP-DAgger baseline policy arch/pipeline "
            "(`policy_arch=mlp_baseline`, `policy_mode=mlp`, `mlp_baseline.onnx`). "
            "This release's FADA pipeline "
            "compares pre-finetune vs. post-finetune FADA on matched rollouts instead. "
            "A standing line-scoped whitelist entry below covers a 'no separate MLP "
            "baseline' sentence in `README.md`: such a sentence exists to state the "
            "identifier does NOT apply here, not to reference live code."
        ),
    },
    {
        "name": "loco_manip",
        "note": (
            "Deleted loco-manipulation/FALCON-force command-profile task family "
            "(`LegedRobotLocoManipForceManagerFar`, `UpperBodyRefPoseCommandFixedPose`, "
            "loco-manip curricula). Removed as part of narrowing this release to "
            "loco-only, T1/G1."
        ),
    },
    {
        "name": "falcon",
        "note": (
            "Deleted FALCON/FALCON-HM/FAR task family (FALCON-specific train/eval/deploy "
            "scripts, force-adaptive presets). Dropped because this release is "
            "loco-only, T1/G1 -- the historical `loco/slope/falcon` task split does "
            "not exist here, so a surviving mention of it names nothing."
        ),
    },
    {
        "name": "use_predefined_commands",
        "note": (
            "Deleted `TaskConfig.use_predefined_commands` / `predefined_command_file` "
            "command source (the `command_sequence_*.json` playback path). Removed for "
            "this release, leaving `randomize_commands` and live joystick/keyboard input "
            "as the only command sources. `DualModePolicy` kept a "
            "`hasattr(self.secondary, 'use_predefined_commands')` guard afterwards: "
            "unreachable, since no shipped policy class defines the attribute."
        ),
    },
    {
        "name": "predefined_command_file",
        "note": (
            "The removed `TaskConfig` field the guard above read its sequence from. "
            "Listed separately because the alnum-boundary pattern is built per "
            "identifier, so the `use_predefined_commands` entry does not match it."
        ),
    },
    {
        "name": "prepare_predefined_command_session",
        "note": (
            "Deleted policy hook that loaded a predefined command sequence at the start "
            "of a recorded dual-mode session. `DualModePolicy._enter_primary_test_session` "
            "kept a `hasattr` call to it; no shipped policy class defines it, so it could "
            "never fire."
        ),
    },
    {
        "name": "copred",
        "note": (
            "Deleted plain compact-obs-transformer baseline entry point and its "
            "wrapper scripts. Not part of "
            "this release; `policy_mode=\"transformer\"` today just means 'no "
            "planner/IDM teacher heads' and dispatches to `LocomotionPolicy_FADA`, "
            "which is not the same thing as the deleted copred baseline."
        ),
    },
]

# Extensions treated as binary without opening the file. Anything not listed
# here falls through to the null-byte sniff in _is_probably_binary.
_BINARY_EXTENSIONS = {
    ".npz", ".npy", ".onnx", ".stl", ".pt", ".pth", ".pkl", ".h5", ".hdf5",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf", ".ttf", ".woff",
    ".woff2", ".zip", ".gz", ".tar", ".whl", ".so", ".dylib", ".dll",
}


class UnscannableFileError(Exception):
    """A tracked file this gate could not read, as distinct from one it skipped.

    Raised instead of classifying an unreadable file as binary. `main()` collects
    these, reports them, and exits 1 so an incomplete scan is not read as clean.
    """


def _is_probably_binary(path: Path) -> bool:
    """True for binary content. Raises `UnscannableFile` if it cannot tell."""
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    try:
        with path.open("rb") as fh:
            chunk = fh.read(8192)
    except OSError as exc:
        raise UnscannableFileError(f"{path}: {exc}") from exc
    return b"\0" in chunk


def _build_pattern(identifier: str) -> re.Pattern[str]:
    parts = [p for p in re.split(r"[-_\s]+", identifier.strip()) if p]
    if not parts:
        raise ValueError(f"empty identifier: {identifier!r}")
    inner = r"[-_\s]+".join(re.escape(p) for p in parts)
    # Alnum boundary, not \b: \b treats '_' as a word char, so it would not match
    # transformer_planner_idm inside locomotion_transformer_planner_idm.
    return re.compile(r"(?<![A-Za-z0-9])" + inner + r"(?![A-Za-z0-9])", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Whitelist: mentions that are not flagged.
#
# Each entry is (identifier_name, file_path_relative_to_repo_root,
# optional_required_line_substring). With line_substring None, every mention of
# that identifier in that file is excused; no entry currently uses that form.
# With a line_substring, only lines also containing that substring are excused,
# so other mentions in the same file are still reported.
# ---------------------------------------------------------------------------
WHITELIST: list[tuple[str, str, str | None]] = [
    # Covers a README sentence stating this release has no MLP baseline. No such
    # sentence is present, so this entry currently matches nothing.
    ("mlp_baseline", "README.md", "MLP baseline in this release"),
    # `_LEGACY_ALGO_TARGETS` maps the pre-rename `PPO_DeLA` dotted path onto
    # `PPO_Deploy` so older oracle checkpoints still load; scoped to the lines
    # carrying the old path.
    ("dela", "src/holosoma/holosoma/fada/common/utils.py", "PPO_DeLA"),
    ("dela", "src/holosoma/tests/fada/test_legacy_algo_target.py", "PPO_DeLA"),
    # Tests asserting the removed attributes are absent must name them; scoped
    # to the tuple literal holding them.
    (
        "use_predefined_commands",
        "src/holosoma_inference/holosoma_inference/policies/tests/test_dual_mode.py",
        "_REMOVED_PREDEFINED_ATTRS = ",
    ),
    (
        "prepare_predefined_command_session",
        "src/holosoma_inference/holosoma_inference/policies/tests/test_dual_mode.py",
        "_REMOVED_PREDEFINED_ATTRS = ",
    ),
    # The README claim-check names the two removed command-source fields it
    # resolves against the shipped code; scoped to the assignment holding them.
    ("use_predefined_commands", "tests/test_release_doc_claims.py", "_REMOVED_COMMAND_FILE_FIELDS = "),
    ("predefined_command_file", "tests/test_release_doc_claims.py", "_REMOVED_COMMAND_FILE_FIELDS = "),
]


def _whitelisted(identifier: str, file_path: str, line: str) -> bool:
    for ident_name, wl_file, substring in WHITELIST:
        if ident_name != identifier or wl_file != file_path:
            continue
        if substring is None or substring in line:
            return True
    return False


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def main() -> int:
    patterns = [(ident["name"], _build_pattern(ident["name"])) for ident in STALE_IDENTIFIERS]
    hits: list[str] = []
    unscannable: list[str] = []

    for rel_path in _tracked_files():
        if rel_path == SELF_PATH or rel_path in SKIPPED_PATHS:
            continue
        path = REPO_ROOT / rel_path
        if not path.is_file():
            continue
        try:
            if _is_probably_binary(path):
                continue
        except UnscannableFileError as exc:
            unscannable.append(str(exc))
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except OSError as exc:
            # Readable enough to sniff, not readable enough to scan.
            unscannable.append(f"{path}: {exc}")
            continue
        except UnicodeDecodeError as exc:
            # No NUL in the first 8 KiB but not valid UTF-8 either: neither
            # scannable nor classifiable as binary, so it is reported.
            unscannable.append(f"{path}: not valid UTF-8 and not detected as binary ({exc})")
            continue

        # Match the path string too, catching a file named after a stale
        # identifier whose contents never mention it.
        for name, pattern in patterns:
            if pattern.search(rel_path) and not _whitelisted(name, rel_path, rel_path):
                hits.append(f"{rel_path}: (path itself matches '{name}')")

        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in patterns:
                if pattern.search(line) and not _whitelisted(name, rel_path, line):
                    hits.append(f"{rel_path}:{lineno}: [{name}] {line.strip()[:160]}")

    if unscannable:
        print(
            f"{len(unscannable)} tracked file(s) could not be scanned; the identifier sweep is "
            "incomplete and its result must not be read as clean:\n",
            file=sys.stderr,
        )
        for entry in unscannable:
            print(f"  {entry}", file=sys.stderr)
        print(file=sys.stderr)

    if hits:
        print(f"Found {len(hits)} stale-identifier mention(s):\n", file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        print(
            "\nIf a hit is a legitimate mention (e.g. a sentence whose point is that "
            "the identifier does NOT apply here, or a migration map that has to name "
            "what it migrates away from), add a narrowly-scoped WHITELIST entry in "
            "tools/check_stale_identifiers.py instead of ignoring this failure.",
            file=sys.stderr,
        )
        return 1

    if unscannable:
        return 1

    print(f"OK: 0 stale-identifier mentions across {len(STALE_IDENTIFIERS)} tracked identifiers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
