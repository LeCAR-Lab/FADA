#!/usr/bin/env python3
"""Static check: every git-tracked ``*.py`` file with a detectable
module-level ``__main__`` guard, and every console-script entry point
declared in this repo's packaging metadata, must be either a pytest test
file (for the former) or an explicitly documented, reasoned entry point in
``DOCUMENTED_ENTRYPOINTS`` / ``DOCUMENTED_CONSOLE_SCRIPTS`` below.

This is a static AST/text scan, not an import-and-introspect check. See
"What this gate does NOT detect" below for the patterns it cannot see.

What counts as "runnable"
--------------------------
Two independent signals, each found via static analysis rather than a plain
text/regex search over the whole file, so neither matches the string
appearing inside an unrelated docstring or comment:

1. A module-level ``if`` statement whose test expression contains the
   string constant ``"__main__"`` *anywhere* in its AST subtree, found via
   an ``ast.walk`` over the ``if``'s test. The match is permissive about
   both sides of the comparison: it covers the canonical
   ``__name__ == "__main__"`` (either operand order) as well as
   ``globals().get("__name__") == "__main__"``, ``globals()["__name__"] ==
   "__main__"``, ``__name__.endswith("__main__")``, ``"__main__" in
   __name__``, and any other spelling that literally contains the string
   ``"__main__"`` in a module-level ``if`` test. A module-level ``if``
   mentioning that string for an unrelated reason is flagged too.
2. A console-script entry declared in this repo's packaging metadata
   (``[project.scripts]`` in a tracked ``pyproject.toml``, or an
   ``entry_points={"console_scripts": [...]}`` / ``[options.entry_points]``
   ``console_scripts`` block in a tracked ``setup.py`` / ``setup.cfg``).
   These make a module runnable as ``$ the-command`` after an editable
   install, with no ``__main__`` guard in the source at all -- signal (1)
   cannot see them, so they are collected separately and checked against
   ``DOCUMENTED_CONSOLE_SCRIPTS``.

Exemptions
----------
Pytest test files are exempt from the ``__main__``-guard check (signal 1
above) without an allowlist entry, and *only* if both hold: the basename
matches ``test_*.py`` **and** the file's repo-relative path starts with one
of the directories in ``TEST_DIRECTORIES`` below. A basename match alone is
not enough, so a module named ``test_tool.py`` outside those directories is
still checked. A file named ``tests/nightly/nightly.py`` does not match
``test_*.py`` at all and needs an explicit ``DOCUMENTED_ENTRYPOINTS`` entry
despite living under ``tests/``.

Everything else -- including this script itself -- must have a
``DOCUMENTED_ENTRYPOINTS`` entry with a reason string. Reasons fall into a
few buckets, spelled out per-entry below:

  * "FADA step N" -- one of the six documented commands in README.md's FADA
    section.
  * "upstream, not part of the FADA contract" -- an entry point this release
    inherited from upstream Holosoma (``git cat-file -e
    "$(git rev-parse --verify -q upstream-baseline >/dev/null && echo
    upstream-baseline || echo main):<path>"`` succeeds) that the FADA six-step
    pipeline does not use, kept because this release tree is additive over
    upstream.

    That ref is not a bare ``main``. In this working repository ``main`` is
    the pinned upstream snapshot, but in the published tree ``main`` is the
    release commit itself, so ``main:<path>`` resolves for every path. The
    published tree carries an annotated ``upstream-baseline`` tag on its
    first commit (the upstream snapshot). ``tools/upstream_ref.py``
    implements the same resolution for the checks that do it in code
    (``tools/bitexact/reachability.py``): ``python -c 'from
    tools.upstream_ref import upstream_baseline_ref;
    print(upstream_baseline_ref())'``.
  * "release tooling" -- this gate, its two siblings, and the bitexact
    determinism harness (``tools/bitexact/run.sh``'s components).
  * "PENDING-DECISION" -- a runnable module found by this gate that is real,
    working code but is not documented anywhere.

Both directions are checked: an undocumented runnable module fails, and so
does a stale ``DOCUMENTED_ENTRYPOINTS`` entry whose path is no longer tracked
or no longer has a ``__main__`` block.

What this gate does NOT detect
-------------------------------
This is a static scan of two lexical patterns (a module-level ``if``
mentioning ``"__main__"``, and packaging-metadata console-script
declarations). It does not detect any of the following ways a module can
still run as a standalone program:

  * ``runpy.run_module(...)`` / ``runpy.run_path(...)`` invoked from
    elsewhere, or any other mechanism that executes a module's top-level
    code without a ``__main__`` guard at all.
  * Unconditional module-level side effects -- a call at module scope (e.g.
    ``main()`` or ``app.run()`` with no enclosing ``if``) that runs on
    every import, not just ``python module.py``.
  * A dynamically constructed guard whose ``"__main__"`` string is not a
    literal AST constant reachable by ``ast.walk`` on the ``if`` test --
    e.g. built via string concatenation/formatting, read from a variable,
    or assembled at runtime (``getattr(sys.modules[__name__], ...)``
    tricks, ``exec``, etc.).
  * A console-script declaration this gate's metadata scan does not
    recognize -- it only understands ``[project.scripts]`` tables in
    ``pyproject.toml`` and ``console_scripts`` entries in ``setup.py`` /
    ``setup.cfg``; other distribution mechanisms (e.g. a wheel's
    post-install hook, a Makefile ``install`` target that symlinks a
    script) are invisible to it.

A green run is therefore not sufficient on its own to establish that a module
is inert on import.
"""

from __future__ import annotations

import ast
import configparser
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF_PATH = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()


@dataclass(frozen=True)
class Entry:
    path: str
    reason: str


# ---------------------------------------------------------------------------
# The allowlist. Every git-tracked, non-test-file module with a module-level
# `if __name__ == "__main__":` must appear here.
# ---------------------------------------------------------------------------
DOCUMENTED_ENTRYPOINTS: list[Entry] = [
    # --- FADA six-step contract. README.md is the only place the six steps are
    # written down: steps 1-3 in its "FADA: Planner-IDM Deployment-Domain
    # Adaptation" section, steps 4-6 in the numbered "### 4."/"### 5."/"### 6."
    # sections that follow it. ---
    Entry(
        "src/holosoma/holosoma/train_agent.py",
        "FADA step 1: oracle PPO expert (`exp:<robot>_oracle`; the `*_deploy` presets "
        "are the deployment policy's config and carry no privileged observations). "
        "Upstream file, used as-is by the FADA contract.",
    ),
    Entry(
        "src/holosoma/holosoma/fada/planner_idm/train.py",
        "FADA step 2: `python -m holosoma.fada.planner_idm.train` (DAgger training).",
    ),
    Entry(
        "src/holosoma/holosoma/fada/planner_idm/eval_checkpoint.py",
        "FADA step 3: `python -m holosoma.fada.planner_idm.eval_checkpoint` "
        "(evaluate + export ONNX). The documented, supported checkpoint-eval entry "
        "point for FADA policies.",
    ),
    Entry(
        "src/holosoma/holosoma/run_sim.py",
        "FADA step 4 (MuJoCo sim launch half of 'Deploy in MuJoCo and collect "
        "target-domain data'). Upstream file, used as-is by the FADA contract.",
    ),
    Entry(
        "src/holosoma_inference/holosoma_inference/run_policy.py",
        "FADA step 4 (deploy + collect half): `run_policy.py ... --task.policy-mode "
        "fada --task.collect-data`. Upstream file, used as-is by the FADA contract.",
    ),
    Entry(
        "src/holosoma/holosoma/fada/planner_idm/finetune_idm_lora.py",
        "FADA step 5: `python -m holosoma.fada.planner_idm.finetune_idm_lora` "
        "(IDM LoRA finetune on target-domain H5 data).",
    ),
    # --- Upstream entry points, not part of the FADA contract ---
    # Inherited from upstream Holosoma; no documented FADA step invokes them.
    Entry(
        "src/holosoma/holosoma/eval_agent.py",
        "Upstream entry point (oracle PPO eval), not part of the FADA contract.",
    ),
    Entry(
        "src/holosoma/holosoma/replay.py",
        "Upstream entry point (checkpoint replay/visualization), not part of the "
        "FADA contract.",
    ),
    # 8 upstream `src/holosoma_retargeting/...` entries follow.
    Entry(
        "src/holosoma_retargeting/holosoma_retargeting/viser_player.py",
        "Upstream retargeting tooling (backs whole-body tracking, inherited from "
        "upstream). None of the six FADA steps in README.md invoke it.",
    ),
    Entry(
        "src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py",
        "Upstream retargeting tooling, not part of the FADA contract (see above).",
    ),
    Entry(
        "src/holosoma_retargeting/holosoma_retargeting/data_conversion/viser_body_vel_player.py",
        "Upstream retargeting tooling, not part of the FADA contract (see above).",
    ),
    Entry(
        "src/holosoma_retargeting/holosoma_retargeting/data_utils/extract_global_positions.py",
        "Upstream retargeting tooling, not part of the FADA contract (see above).",
    ),
    Entry(
        "src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_amass_smplx_for_rt.py",
        "Upstream retargeting tooling, not part of the FADA contract (see above).",
    ),
    Entry(
        "src/holosoma_retargeting/holosoma_retargeting/evaluation/eval_retargeting.py",
        "Upstream retargeting tooling, not part of the FADA contract (see above).",
    ),
    Entry(
        "src/holosoma_retargeting/holosoma_retargeting/examples/parallel_robot_retarget.py",
        "Upstream retargeting tooling, not part of the FADA contract (see above).",
    ),
    Entry(
        "src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py",
        "Upstream retargeting tooling, not part of the FADA contract (see above).",
    ),
    Entry(
        "tests/nightly/nightly.py",
        "Upstream nightly-CI driver. Lives under `tests/` but is not a pytest file "
        "(no `test_` prefix, has its own argparse CLI) -- not part of the FADA "
        "contract.",
    ),
    Entry(
        "tests/nightly/generate_report.py",
        "Upstream nightly-CI driver, same tests/nightly/ family as above.",
    ),
    Entry(
        "tests/nightly/run_summary.py",
        "Upstream nightly-CI driver, same tests/nightly/ family as above.",
    ),
    # --- Release tooling ---
    Entry(
        "tools/bitexact/compare.py",
        "Determinism harness component (`tools/bitexact/run.sh`); release tooling. "
        "Run it with `bash tools/bitexact/run.sh`; all four probes must report "
        "IDENTICAL.",
    ),
    Entry(
        "tools/bitexact/reachability.py",
        "Determinism harness component (`tools/bitexact/run.sh`); release tooling.",
    ),
    Entry(
        "tools/check_script_args.py",
        "Release gate #1 (documented-command argparse/tyro flag check).",
    ),
    Entry(
        "tools/check_stale_identifiers.py",
        "Release gate #2 (retired-identifier reappearance check).",
    ),
    Entry(
        SELF_PATH,
        "Release gate #3 (this file) -- the entrypoint-inventory check itself is, "
        "necessarily, a documented entry point.",
    ),
    # No PENDING-DECISION entries. The mocap/plotting utilities expose no
    # `main()`/argparse/`__main__` surface -- only functions called from other
    # modules (`compute_velocity_errors_from_mocap_unified`,
    # `_k_step_recon_loss_over_time`) -- so this gate does not see them and they
    # do not belong in this list.
]


# Repo-relative directory prefixes (trailing slash) holding this repo's pytest
# suites, enumerated against `git ls-files | grep -E '(^|/)tests/'`.
# A fixed path list, not a "basename starts with test_" rule and not "any
# directory named tests": test_*.py files under a newly added tests/ directory
# are not exempt until that directory is added here.
TEST_DIRECTORIES: tuple[str, ...] = (
    "src/holosoma/holosoma/agents/modules/tests/",
    "src/holosoma/holosoma/config_types/tests/",
    "src/holosoma/holosoma/config_values/tests/",
    "src/holosoma/holosoma/envs/tests/",
    "src/holosoma/holosoma/managers/reward/terms/tests/",
    "src/holosoma/holosoma/utils/tests/",
    "src/holosoma/tests/",
    "src/holosoma_inference/holosoma_inference/policies/tests/",
    "src/holosoma_inference/holosoma_inference/sdk/booster/tests/",
    "src/holosoma_inference/holosoma_inference/utils/tests/",
    "tests/",
    "tools/bitexact/tests/",
)


def _is_test_file(rel_path: str) -> bool:
    """True only for `test_*.py` files that also live under a directory in
    `TEST_DIRECTORIES`; a basename match alone returns False."""
    if not Path(rel_path).name.startswith("test_"):
        return False
    return any(rel_path.startswith(prefix) for prefix in TEST_DIRECTORIES)


def _tracked_py_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


# ---------------------------------------------------------------------------
# Console-script / [project.scripts] detection -- signal (2) in this module's
# docstring, "What counts as 'runnable'".
#
# A module can become a standalone CLI with no `__main__` guard in its source
# at all, via a `[project.scripts]` entry in `pyproject.toml` or a
# `console_scripts` entry_points declaration in `setup.py` / `setup.cfg`.
# These are collected across all tracked packaging-metadata files and checked
# against DOCUMENTED_CONSOLE_SCRIPTS the same way runnable modules are checked
# against DOCUMENTED_ENTRYPOINTS. The repo declares no such entries, so the
# list below is empty; adding one fails this gate until it is listed here.
# ---------------------------------------------------------------------------
DOCUMENTED_CONSOLE_SCRIPTS: list[str] = []


def _packaging_metadata_files() -> list[str]:
    names = {"pyproject.toml", "setup.py", "setup.cfg"}
    return [p for p in _tracked_files() if Path(p).name in names]


def _scripts_from_pyproject_text(text: str) -> list[str]:
    """Names declared under a `[project.scripts]` table.

    A line-based scan rather than a TOML parse, so it runs on Pythons predating
    `tomllib` (3.11+). It handles a flat `name = "module:func"` table only, which
    is the shape every pyproject.toml in this repo uses; nested tables under
    `[project.scripts]` are not understood.
    """
    scripts: list[str] = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped[1:-1].strip() == "project.scripts"
            continue
        if in_section and "=" in stripped and not stripped.startswith("#"):
            scripts.append(stripped.split("=", 1)[0].strip())
    return scripts


def _scripts_from_setup_py(text: str, rel_path: str) -> list[str]:
    """Names declared under `entry_points={"console_scripts": [...]}` in a
    `setup(...)` call, found via AST rather than text search, so the string
    appearing in a comment or an unrelated dict does not match."""
    try:
        tree = ast.parse(text, filename=rel_path)
    except SyntaxError:
        return []
    scripts: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.keyword) and node.arg == "entry_points"):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        for key, val in zip(value.keys, value.values):
            if isinstance(key, ast.Constant) and key.value == "console_scripts" and isinstance(val, ast.List):
                scripts.extend(
                    elt.value.split("=", 1)[0].strip()
                    for elt in val.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )
    return scripts


def _scripts_from_setup_cfg(text: str) -> list[str]:
    """Names declared under `[options.entry_points]`'s `console_scripts` key
    in a `setup.cfg`-style ini file."""
    cp = configparser.ConfigParser()
    try:
        cp.read_string(text)
    except configparser.Error:
        return []
    if not cp.has_option("options.entry_points", "console_scripts"):
        return []
    raw = cp.get("options.entry_points", "console_scripts")
    return [line.split("=", 1)[0].strip() for line in raw.splitlines() if line.strip()]


def _console_script_entries() -> list[str]:
    """All `(source_file, script_name)` console-script declarations found in
    tracked packaging metadata, formatted as `"source_file:script_name"` to
    match `DOCUMENTED_CONSOLE_SCRIPTS`'s format."""
    entries: list[str] = []
    for rel_path in _packaging_metadata_files():
        path = REPO_ROOT / rel_path
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError) as exc:
            print(f"FATAL: cannot read packaging metadata file {rel_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        name = Path(rel_path).name
        if name == "pyproject.toml":
            names = _scripts_from_pyproject_text(text)
        elif name == "setup.py":
            names = _scripts_from_setup_py(text, rel_path)
        else:
            names = _scripts_from_setup_cfg(text)
        entries.extend(f"{rel_path}:{n}" for n in names)
    return sorted(entries)


def _if_test_mentions_dunder_main(test: ast.expr) -> bool:
    """True if the string constant `"__main__"` appears anywhere in `test`'s
    AST subtree, however `__name__` is accessed and however the comparison
    is spelled.

    The canonical `ast.Compare(Name("__name__"), [Eq], [Constant("__main__")])`
    shape is not required. `globals().get("__name__") == "__main__"`,
    `globals()["__name__"] == "__main__"`, `__name__.endswith("__main__")`,
    `"__main__" in __name__` and anything else putting the literal string
    `"__main__"` inside a module-level `if` test all match, as does a
    module-level `if` mentioning that string for an unrelated reason.
    """
    return any(isinstance(node, ast.Constant) and node.value == "__main__" for node in ast.walk(test))


def _has_dunder_main_block(tree: ast.Module) -> bool:
    """True if `tree` contains a module-level `if` whose test references
    `"__main__"` in any form (see `_if_test_mentions_dunder_main`)."""
    return any(isinstance(node, ast.If) and _if_test_mentions_dunder_main(node.test) for node in tree.body)


def _runnable(rel_path: str) -> bool:
    """True if the tracked file at `rel_path` has a detected `__main__` guard.

    A read or parse failure exits the process with status 1 rather than
    returning False.
    """
    path = REPO_ROOT / rel_path
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError) as exc:
        print(f"FATAL: cannot read tracked file {rel_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        tree = ast.parse(text, filename=rel_path)
    except SyntaxError as exc:
        print(f"FATAL: cannot parse tracked file {rel_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    return _has_dunder_main_block(tree)


def main() -> int:
    tracked = set(_tracked_py_files())
    runnable = {p for p in tracked if _runnable(p)}
    documented = {e.path for e in DOCUMENTED_ENTRYPOINTS}

    if len(documented) != len(DOCUMENTED_ENTRYPOINTS):
        seen: set[str] = set()
        dupes = set()
        for e in DOCUMENTED_ENTRYPOINTS:
            if e.path in seen:
                dupes.add(e.path)
            seen.add(e.path)
        print(f"DOCUMENTED_ENTRYPOINTS has duplicate path(s): {sorted(dupes)}", file=sys.stderr)
        return 1

    undocumented = sorted(p for p in runnable if p not in documented and not _is_test_file(p))
    stale = sorted(p for p in documented if p not in tracked or p not in runnable)

    console_scripts = _console_script_entries()
    documented_scripts = set(DOCUMENTED_CONSOLE_SCRIPTS)
    if len(documented_scripts) != len(DOCUMENTED_CONSOLE_SCRIPTS):
        print("DOCUMENTED_CONSOLE_SCRIPTS has duplicate entries.", file=sys.stderr)
        return 1
    undocumented_scripts = sorted(e for e in console_scripts if e not in documented_scripts)
    stale_scripts = sorted(e for e in documented_scripts if e not in console_scripts)

    ok = True
    if undocumented:
        ok = False
        print(
            f"Found {len(undocumented)} runnable module(s) with no "
            "DOCUMENTED_ENTRYPOINTS entry:\n",
            file=sys.stderr,
        )
        for p in undocumented:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nEvery module with a module-level `if` block whose test references "
            '"__main__" (in any form -- see this file\'s module docstring) must '
            "either be a `test_*.py` file under one of TEST_DIRECTORIES (pytest "
            "convention, auto-exempt) or have a reasoned entry in "
            "tools/check_entrypoints.py's DOCUMENTED_ENTRYPOINTS list -- see that "
            "file's module docstring for the reasoning categories. Either add the "
            "entry (with a real reason) or strip the module's `main()`/`__main__` "
            "block if it should not be a standalone CLI.",
            file=sys.stderr,
        )

    if stale:
        ok = False
        print(
            f"\nFound {len(stale)} stale DOCUMENTED_ENTRYPOINTS entry(ies) (path "
            "no longer tracked, or no longer has a `__main__` block):\n",
            file=sys.stderr,
        )
        for p in stale:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nRemove the entry from DOCUMENTED_ENTRYPOINTS in tools/check_entrypoints.py.",
            file=sys.stderr,
        )

    if undocumented_scripts:
        ok = False
        print(
            f"\nFound {len(undocumented_scripts)} console-script entry point(s) declared "
            "in packaging metadata with no DOCUMENTED_CONSOLE_SCRIPTS entry:\n",
            file=sys.stderr,
        )
        for script_entry in undocumented_scripts:
            print(f"  {script_entry}", file=sys.stderr)
        print(
            "\nA `[project.scripts]` / `console_scripts` entry makes a module runnable "
            "as a standalone command with no `__main__` guard in its source, which the "
            "rest of this gate cannot see. Add a reasoned entry to "
            "DOCUMENTED_CONSOLE_SCRIPTS in tools/check_entrypoints.py.",
            file=sys.stderr,
        )

    if stale_scripts:
        ok = False
        print(
            f"\nFound {len(stale_scripts)} stale DOCUMENTED_CONSOLE_SCRIPTS entry(ies) "
            "(no longer declared in any tracked packaging metadata file):\n",
            file=sys.stderr,
        )
        for script_entry in stale_scripts:
            print(f"  {script_entry}", file=sys.stderr)
        print(
            "\nRemove the entry from DOCUMENTED_CONSOLE_SCRIPTS in tools/check_entrypoints.py.",
            file=sys.stderr,
        )

    if not ok:
        return 1

    print(
        f"OK: {len(runnable)} module(s) with a detected __main__ guard or declared "
        f"console script accounted for ({len(documented)} __main__-guard entries, "
        f"{sum(1 for p in runnable if _is_test_file(p))} test-file exemptions, "
        f"{len(console_scripts)} console-script entries, all documented). This gate "
        "does NOT detect runpy invocation, unconditional module-level execution, or "
        "dynamically constructed __main__ guards -- see this file's module docstring, "
        '"What this gate does NOT detect".'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
