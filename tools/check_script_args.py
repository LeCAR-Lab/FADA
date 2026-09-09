#!/usr/bin/env python3
r"""Static + live check: every ``--flag`` a documented entry point passes to a
Python CLI must be a flag that CLI actually recognizes.

Outcomes
--------
1. Every ``--flag``-bearing invocation must end up CHECKED (via live ``--help``
   introspection for tyro CLIs, see below, or the static argparse AST scan) or
   explicitly WHITELISTED with a reason. Anything else -- SKIPPED, UNRESOLVED,
   or a truncated/partial tyro surface -- makes the run exit non-zero.
2. The SKIPPED/UNRESOLVED/WHITELISTED *counts* are always printed, even
   without ``--verbose``; only the itemized detail for successfully-resolved
   entries is verbose-gated. Anything that fails the run is printed in full,
   verbose or not.

Parser behaviour this check relies on
-------------------------------------
* A recognized-flag surface read out of a *tyro* ``--help`` transcript lists
  only each option's canonical spelling, which tyro renders with hyphen word
  delimiters (``--input-dir``, ``--training.num-envs``). tyro itself is not that
  strict: before parsing it rewrites every ``--option`` token's name to the
  canonical delimiter (``tyro/_strings.py:swap_delimiters``, called from
  ``tyro/_cli.py``), so ``--input_dir``, ``--training.num_envs`` and mixed forms
  like ``--observation.groups.actor_obs.history-length`` are all accepted and
  all land in the same field. ``argparse`` does no such rewriting and rejects
  the underscored spelling with exit 2, so the two cannot be matched by one
  rule.

  Whether a program accepts a spelling its transcript does not list is a
  property of the parser, not of the help text, so it is asked of the program.
  A flag that is not in the ``--help`` surface verbatim is handed to a runtime
  probe (``probe_documented_arguments``) that runs the real entry point with the
  real documented argv and stops it the instant its outermost parse completes --
  whichever layer that is, argparse or tyro, reached however it is reached --
  raising out before a single line after parsing executes. The verdict is
  accepted, rejected with "unrecognized arguments/options", or -- fail-closed --
  undecided, which is whitelist-or-fail like every other unverifiable outcome.

  Out of scope: an entry point that fabricates its own ``--help`` output. No
  amount of runtime observation distinguishes a fabricated flag surface from a
  real one; the probe makes only the *acceptance* half real.
* Invocation-start detection shell-tokenizes each line (``scan_shell_command``)
  and walks past a leading run of ``VAR=value`` assignments / ``env`` /
  ``conda run [-n NAME]`` before checking for a bare ``python``/``python3``
  token. Any fenced-code line in a documented ``*.md`` that contains the word
  ``python`` but was *not* recognized as an invocation start is itself reported
  and fails the run unless explicitly ``LINE_WHITELIST``-ed.
* ``--help``-transcript parsing (``_parse_help_flags``) is indentation-aware:
  the first flag-shaped line in a transcript establishes the option column, and
  only lines at or above that same indentation are treated as new option
  definitions; anything indented deeper is a continuation, and is only mined for
  further flag tokens if it still looks like a wrapped alias list (nothing but
  flags/metavars/commas -- no prose).
* The static argparse AST scan (used for pure-argparse files, and as the
  hybrid-CLI fallback) special-cases a literal constant ``if`` test in
  ``_ArgumentCollector``: a falsy constant skips the ``body`` (only ``orelse``
  is visited), a truthy one skips ``orelse``. This is not a general
  unreachable-code analysis -- flags gated behind a non-constant condition, an
  indirect constant (a module-level ``bool`` variable, an environment check,
  ...), or any other dead-code shape are still collected.
* Continuation following tokenizes in bash's own states (unquoted, ``'...'``,
  ``"..."``, ``$'...'``, comments, and the ``| || && & ;`` operators that also
  continue a line) -- see ``scan_shell_command``, whose docstring lists the four
  constructs it approximates (here-documents, multi-line command substitution,
  compound-command bodies, and unexpanded ``$VAR``). A block bash would not run
  as written is reported, fails the run, and is *not* returned as an invocation.
  Unlike every other failure category here it is not whitelistable. What is
  reported is what bash does to a reader who pastes the block: the command ended
  and the next line -- a ``-``-leading flag line -- runs as its own command
  (exit 127); or an argument arrived that is nothing but whitespace; or ``\#``
  turned a comment into a positional argument.

Method
------
For every fenced code block in a git-tracked ``*.md`` (this release documents
its commands only in Markdown -- there is no wrapper-script layer), except the
few that the release does not publish (``UNPUBLISHED_COMMAND_SOURCES``):

1. Find every ``python``/``python3`` invocation (a "logical command": a line
   starting the call, plus any ``\``-continued follow-on lines).
2. Extract every literal ``--flag`` (or ``--flag=value``) token appearing in
   that invocation's text -- including dotted tyro nested-dataclass flags like
   ``--task.policy-mode``.
3. Resolve the invoked entry point (``-m pkg.mod`` or a literal ``foo.py``
   path) to a source file.
4. Determine that file's recognized-flag surface:
   a. If it has no ``tyro.cli(`` call anywhere: statically collect its
      ``argparse`` option strings from any ``add_argument(...)`` call (direct,
      or via the ``run_experiment.py`` -> ``config.py`` re-exported
      ``build_arg_parser`` pattern).
   b. If it does call ``tyro.cli(``: run the file with the invocation's
      leading positional tokens (its tyro subcommand selectors, e.g.
      ``simulator:isaacsim`` -- see ``_extract_positional_subcommands``) plus
      ``--help``, in each of this repo's own tracked conda envs in turn
      (``hsmujoco``, ``hssim``, ``hsinference``, resolved the same way
      ``scripts/source_common.sh`` resolves ``WORKSPACE_DIR``/``CONDA_ROOT``,
      so this works on any machine that ran this repo's own setup scripts).
      Parse the real recognized flags out of tyro's box-drawn ``--help``
      output (or plain argparse's, for hybrid argparse+tyro files like
      ``eval_checkpoint.py``, where ``--help`` is served by the argparse
      pre-parser before tyro ever sees the remaining args -- so this same
      live path covers "at least the argparse subset" for hybrids without a
      separate hybrid-specific code path).
   c. If (b) fails in every candidate env (package not installed anywhere, or
      the file's ``--help`` handling is itself broken/dynamic -- see
      ``eval_agent.py`` in ``WHITELIST``), and the file *does* have a
      directly-visible ``add_argument`` parser (true hybrid), fall back to
      checking just that static argparse subset: flags found in it are fully
      verified; unresolved *dotted* flags are treated as downstream tyro
      overrides forwarded via ``parse_known_args()`` and are NEEDS_REVIEW
      (whitelist-or-fail); unresolved *non-dotted* flags are HITS (the local
      argparse parser is the only thing that could recognize them, and it
      doesn't).
   d. Otherwise: NEEDS_REVIEW (whitelist-or-fail). No flags can be verified.
5. Report any flag that fails verification.

Scope and known limitations
---------------------------
* tyro's own ``--help`` renderer truncates very large "default subcommand
  options" boxes (e.g. an unselected ``algo:`` subcommand's ~500 nested
  fields) to 25 entries with an "and N more" trailer. When that trailer is
  seen, invocation flags that *aren't* found in the (truncated) recognized set
  are NOT treated as confirmed HITS -- they might just be past the cutoff --
  and land in NEEDS_REVIEW instead. ``hsmujoco`` is tried first because
  ``hssim``'s tyro build hits this truncation for ``train_agent.py``'s
  unselected ``algo:`` block.
* Positional tyro subcommand tokens are recognized via this codebase's own
  ``group:variant`` convention (``exp:``, ``simulator:``, ``logger:``,
  ``robot:``, ``terrain:``, ``inference:``, ...) -- see
  ``_extract_positional_subcommands``. A bare positional that *doesn't* use
  this convention is not picked up.
* Flags built up dynamically (``"${EXTRA_ARGS[@]}"``, ``"$@"``) are opaque to
  static analysis and are not checked -- only literal ``--foo`` tokens written
  directly in the script are.
* Resolution of ``python some/relative/$VAR/path.py`` invocations is by
  basename lookup against the git-tracked file list; if a basename is
  ambiguous (matches more than one tracked file, or -- as with
  ``human_body_prior/setup.py``, a third-party repo cloned by the docs
  themselves, not part of this repository -- matches none) the invocation is
  UNRESOLVED rather than guessed at.

Exit status: 0 iff there are zero HITS and every NEEDS_REVIEW entry is covered
by an explicit, reasoned ``WHITELIST``/``LINE_WHITELIST`` entry below.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

# Top-level installed package name -> the directory that is its import root
# (i.e. the directory that would need to be on PYTHONPATH for `import <pkg>`
# to work). Mirrors the PYTHONPATH export the documented setup uses.
PACKAGE_SRC_ROOTS = {
    "holosoma": REPO_ROOT / "src" / "holosoma",
    "holosoma_inference": REPO_ROOT / "src" / "holosoma_inference",
    "holosoma_retargeting": REPO_ROOT / "src" / "holosoma_retargeting",
}

# Matches a whole dotted tyro flag (e.g. --task.policy-mode), '.' included, so it
# is not truncated to its first path segment.
FLAG_RE = re.compile(r"(?<![\w.-])(--[a-zA-Z][a-zA-Z0-9_.-]*)")
MODULE_INVOCATION_RE = re.compile(r"python3?\s+-m\s+([A-Za-z0-9_.]+)")
PY_FILE_TOKEN_RE = re.compile(r"([A-Za-z0-9_./${}-]+\.py)\b")
PARSER_BUILDER_NAMES = {"build_arg_parser", "_build_arg_parser"}

# Recognizes a bare python interpreter token, optionally versioned
# (python3, python3.10, ...). Used only *after* walking past any
# env-assignment/env/conda-run prefix -- see _python_invocation_start_index.
_PYTHON_BIN_RE = re.compile(r"^python(3(\.\d+)?)?$")
# A line contains a python "mention" worth caring about if this word appears
# anywhere in it (including inside a human-facing placeholder like
# `<python>`) -- used to report invocation forms _python_invocation_start_index
# does not recognize. See module docstring.
PYTHON_WORD_RE = re.compile(r"(?<![\w.-])python3?(?![\w-])")
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Unquoted control operators that continue a command onto the next line the way a
# trailing `\` does: `cmd |` followed by an indented continuation is one command.
_LINE_CONTINUING_OPERATORS = frozenset({"|", "||", "&&", "&", ";"})
# Characters bash lexes as control operators WITHOUT requiring surrounding
# whitespace. See scan_shell_command for the one exception (`>&`/`<&`).
_OPERATOR_CHARS = "|&;"

# This repo's own `group:variant` tyro-subcommand convention (exp:, simulator:,
# logger:, robot:, terrain:, inference:, ...). See module docstring.
POSITIONAL_SUBCOMMAND_RE = re.compile(r"(?<!\S)([a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_.:-]+)(?!\S)")

# Conda env names this repo's own scripts/setup_*.sh / source_*_setup.sh
# create, in the order they are tried for live --help introspection. hsmujoco
# goes first: its tyro build produces untruncated help output for entry points
# where hssim's truncates.
CANDIDATE_CONDA_ENVS = ["hsmujoco", "hssim", "hsinference"]

LIVE_HELP_TIMEOUT_S = 120

# ---------------------------------------------------------------------------
# Whitelist: entry points whose flag surface cannot be verified by this tool,
# each with a reason. Keyed by repo-relative source-file path. Matches ONLY
# the "cannot verify at all" outcome (live --help failed in every candidate
# env, or -- for a hybrid file -- the unresolved dotted-flag remainder); a
# real flag mismatch (HIT) for one of these files is NEVER suppressed by this
# whitelist, because HITS are classified independently of it (see main()).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WhitelistEntry:
    """An entry point whose flag surface cannot be reached, and WHICH flags that excuses.

    The entry excuses only the flags listed in `flags`; anything else documented
    for this entry point lands in NEEDS REVIEW and fails the run. `flags` is a
    declaration, not a check -- nothing can verify these flags -- so it has to be
    edited to grow.
    """

    reason: str
    flags: frozenset[str]


def whitelist_unexcused_flags(rel_py: str, unverified: set[str]) -> tuple[str | None, set[str]]:
    """`(reason, the flags this entry does NOT excuse)`.

    A path with no entry excuses nothing, so every flag comes back unexcused --
    which is how a non-whitelisted entry point still reaches NEEDS REVIEW.
    """
    entry = WHITELIST.get(rel_py)
    if entry is None:
        return None, set(unverified)
    return entry.reason, {flag for flag in unverified if flag not in entry.flags}


WHITELIST: dict[str, WhitelistEntry] = {
    "src/holosoma/holosoma/eval_agent.py": WhitelistEntry(
        reason=(
            "Two-stage tyro CLI: main() calls tyro.cli(CheckpointConfig, add_help=False) "
            "first, then builds the *real* ExperimentConfig schema for the second "
            "tyro.cli() call from the checkpoint's own saved config. With add_help=False on "
            "the first pass, `--help` is never intercepted -- it falls through as an "
            "unknown token and the process dies on `ValueError: No checkpoint provided` "
            "before any help text exists to parse. The full flag surface is therefore "
            "checkpoint-dependent and cannot be obtained without a real checkpoint file on "
            "disk, live or static."
        ),
        # Empty: no documented invocation currently reaches this entry point
        # unverified. With no flags declared, the first documented flag that needs
        # excusing here fails the run.
        flags=frozenset(),
    ),
    "src/holosoma_retargeting/holosoma_retargeting/data_utils/extract_global_positions.py": WhitelistEntry(
        reason=(
            "Imports `lafan1`, which the docs have the reader `git clone` from the LAFAN "
            "repository into this script's own directory rather than install into any of "
            "this repo's tracked conda envs (hsmujoco/hssim/hsinference). On a machine that "
            "followed those steps the clone sits next to the script, lands on sys.path[0], "
            "and live --help succeeds -- so this entry only takes effect on a checkout that "
            "has not done the LAFAN setup, where --help cannot run at all "
            "(ModuleNotFoundError: lafan1). It excuses only 'unverifiable'; where the clone "
            "is present the flags are still checked for real."
        ),
        # `lafan1/` is gitignored, so it is absent from every published clone and
        # these flags are what the entry has to excuse there. They are the fields of
        # the script's own `Config` dataclass (`input_dir: str`, `output_dir: str`),
        # the same source tyro builds `--help` from. Where the clone IS present the
        # live check still runs and still governs.
        flags=frozenset({"--input_dir", "--output_dir"}),
    ),
    "src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_amass_smplx_for_rt.py": WhitelistEntry(
        reason=(
            "Requires `human_body_prior`, which the docs themselves instruct installing via "
            "`git clone` + `python setup.py develop` into a scratch checkout -- not a "
            "dependency of any of this repo's tracked conda envs (hsmujoco/hssim/"
            "hsinference), so --help cannot be obtained in any of them "
            "(ModuleNotFoundError: human_body_prior)."
        ),
        flags=frozenset({"--amass-root-folder", "--model-root-folder", "--output-folder"}),
    ),
    "src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py": WhitelistEntry(
        reason=(
            "Requires `cvxpy`, not installed in any of this repo's tracked conda envs "
            "(ModuleNotFoundError: cvxpy in hsmujoco/hssim/hsinference)."
        ),
        flags=frozenset(
            {
                "--data_format",
                "--data_path",
                "--retargeter.debug",
                "--retargeter.foot-sticking-tolerance",
                "--retargeter.visualize",
                "--robot",
                "--robot-config.robot-urdf-file",
                "--save_dir",
                "--task-config.ground-range",
                "--task-name",
                "--task-type",
            }
        ),
    ),
    "src/holosoma_retargeting/holosoma_retargeting/examples/parallel_robot_retarget.py": WhitelistEntry(
        reason=(
            "Requires `cvxpy`, not installed in any of this repo's tracked conda envs "
            "(ModuleNotFoundError: cvxpy in hsmujoco/hssim/hsinference)."
        ),
        flags=frozenset(
            {
                "--data-dir",
                "--data_format",
                "--retargeter.foot-sticking-tolerance",
                "--robot-config.robot-urdf-file",
                "--save_dir",
                "--task-config.ground-range",
                "--task-config.object-name",
                "--task-type",
            }
        ),
    ),
    "src/holosoma_retargeting/holosoma_retargeting/viser_player.py": WhitelistEntry(
        reason=(
            "Requires `viser`, not installed in any of this repo's tracked conda envs "
            "(ModuleNotFoundError: viser in hsmujoco/hssim/hsinference)."
        ),
        flags=frozenset({"--object_urdf", "--qpos_npz", "--robot_urdf"}),
    ),
    "src/holosoma_retargeting/holosoma_retargeting/evaluation/eval_retargeting.py": WhitelistEntry(
        reason=(
            "Requires `igl` (libigl python bindings), not installed in any of this repo's "
            "tracked conda envs (ModuleNotFoundError: igl in hsmujoco/hssim/hsinference)."
        ),
        flags=frozenset({"--data_dir", "--data_type", "--res_dir", "--robot-config.robot-urdf-file"}),
    ),
}

# Per-invocation whitelist. Used for things WHITELIST (keyed by entry-point
# file, above) cannot express because they are properties of one specific
# documented invocation, not of the target script as a whole:
#   1. Fully-UNRESOLVED invocations (entry point couldn't even be located).
#   2. A tyro entry point that live-checks fine everywhere else in this repo's
#      docs, but this one invocation's positional subcommand selector is a
#      human-facing <placeholder> (e.g. `inference:<task-config>`) rather
#      than a literal one, so live --help can't descend into a real
#      subcommand to see its flags.
#   3. A doc line that mentions the word "python" (e.g. inside a
#      `<python>`-style human-facing placeholder) without being a real
#      invocation this tool can resolve -- see PYTHON_WORD_RE / module
#      docstring.
#
# Keyed by (doc path, invocation content fingerprint), NOT by line number: the
# key is immune to line shifts elsewhere in the file, and any change to the
# invocation itself changes the fingerprint and drops the excuse.
LINE_WHITELIST: dict[tuple[str, str], str] = {
    (
        "src/holosoma_retargeting/holosoma_retargeting/ADD_MOTION_FORMAT_README.md",
        "3d7b5491b985872e",  # ~line 23
    ): (
        "`python setup.py develop` here runs inside a `git clone`d third-party repo "
        "(human_body_prior, cloned by the preceding doc lines), not this repository -- "
        "there is no in-repo setup.py this could resolve to."
    ),
    (
        "src/holosoma_retargeting/holosoma_retargeting/README.md",
        "3d7b5491b985872e",  # ~line 108
    ): (
        "Same third-party human_body_prior setup.py as "
        "ADD_MOTION_FORMAT_README.md -- see that entry."
    ),
    (
        "src/holosoma_inference/docs/workflows/real-robot-locomotion.md",
        "5fdc9ef8b44b35bf",  # ~line 218
    ): (
        "`inference:<task-config>` is a human-facing placeholder (the reader "
        "substitutes a real inference:<preset> choice), not a literal tyro subcommand "
        "selector, so live --help can only reach the bare subcommand-choice stub here. "
        "The three flags after it (--task.interface, --task.model-path, "
        "--task.use-joystick) are plain TaskConfig fields shared by every concrete "
        "inference:* preset -- confirmed present via this same run_policy.py's "
        "successful live checks elsewhere in this file and in README.md "
        "(e.g. `inference:t1-23dof-loco-deploy --help` lists all three)."
    ),
}


# Command sources that exist in this working repository but are absent from the
# published tree (declared in `RELEASE_EXCLUDE_PATHS` in
# tools/release/build_release_tree.sh). Paths listed here are not scanned.
#
# Only files the release withholds belong here. A path listed here that DOES
# ship stops being scanned while the run still exits 0; nothing in this module
# detects that -- tests/test_release_guard.py asserts containment against the
# build script's own exclusion list.
UNPUBLISHED_COMMAND_SOURCES: frozenset[str] = frozenset(
    {
        "UPSTREAM_DEVIATIONS.md",
    }
)


@dataclass
class Invocation:
    script: Path
    line_no: int
    text: str
    flags: set[str] = field(default_factory=set)


def command_sources() -> list[Path]:
    """Every git-tracked file that hands a reader a command to run verbatim.

    This release has no wrapper-script layer, so ``*.md`` fenced blocks are the
    whole source set. (`extract_invocations` still special-cases the ``.md``
    suffix, so a non-Markdown command source added here is read as raw lines
    rather than fence-filtered down to nothing.)

    ``UNPUBLISHED_COMMAND_SOURCES`` is subtracted here.
    """
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked = [line.strip() for line in out.splitlines() if line.strip()]
    return [REPO_ROOT / rel for rel in tracked if rel not in UNPUBLISHED_COMMAND_SOURCES]


def _fenced_code_only(lines: list[str]) -> list[str]:
    """Blank out every markdown line outside a ``` fence, preserving line
    numbers so reported positions still point at the real README line.

    Prose routinely contains the word ``python`` mid-sentence; only fenced
    blocks are commands a reader will paste."""
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")
            continue
        out.append(line if in_fence else "")
    return out


def git_tracked_all_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def _skip_env_prefix(tokens: list[str]) -> int:
    """Return the index into `tokens` at which the actual command begins,
    after walking past a leading run of `VAR=value` env-assignments, a bare
    `env` wrapper (plus its own assignments/flags), or `conda run [-n NAME]`.
    See module docstring."""
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if _ENV_ASSIGNMENT_RE.match(tok):
            i += 1
            continue
        if tok == "env":
            i += 1
            while i < n and (_ENV_ASSIGNMENT_RE.match(tokens[i]) or tokens[i].startswith("-")):
                i += 1
            continue
        if tok == "conda" and i + 1 < n and tokens[i + 1] == "run":
            i += 2
            while i < n and tokens[i].startswith("-"):
                if tokens[i] in ("-n", "--name", "-p", "--prefix") and i + 1 < n:
                    i += 2
                else:
                    i += 1
            continue
        break
    return i


_NORMAL, _SINGLE, _DOUBLE, _ANSI_C = "normal", "single", "double", "ansi-c"

_ANSI_C_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", "'": "'", '"': '"'}


@dataclass
class ScannedCommand:
    """One logical shell command, as bash would delimit it.

    `end_index` is the index of the LAST physical line bash includes in the
    command; everything downstream keys off that extent.
    """

    start_index: int
    end_index: int
    tokens: list[str] = field(default_factory=list)
    #: Tokens that are non-empty and entirely whitespace. bash produces one only
    #: from an escaped (or quoted) space at a word boundary.
    whitespace_tokens: list[str] = field(default_factory=list)
    #: Tokens whose leading `#` came from a backslash escape (`... \#comment`):
    #: the escaped comment marker makes the rest of the line one positional
    #: argument.
    escaped_comment_tokens: list[str] = field(default_factory=list)
    #: The input ended while still inside a quote.
    unterminated: bool = False
    #: The token stream ended on a control operator with nothing after it
    #: (`python x.py|` followed by a comment and EOF). bash rejects that block
    #: at parse time -- `bash -n` exits 2.
    dangling_operator: str | None = None


def scan_shell_command(lines: list[str], start: int) -> ScannedCommand:
    r"""Tokenize one logical command starting at ``lines[start]``, bash's way.

    Walks the text in bash's own states: unquoted, ``'...'``, ``"..."``,
    ``$'...'``, and comments. A ``\`` continues a line only when the newline is
    the character it escapes; a ``\`` inside ``'...'`` is data and continues
    nothing.

    Known approximations, each a place a documented command can be mis-delimited:

    * **Here-documents.** ``<<EOF`` bodies are scanned as ordinary command lines,
      so a body line ending in ``\`` is read as a continuation.
    * **Multi-line command substitution.** A ``$(`` or backtick left open at end of
      line ends the command here; bash would keep reading.
    * **Compound commands.** ``if``/``for``/``while``/``case`` bodies are scanned as
      a sequence of independent commands, which is right for tokenizing each one
      and wrong if a documented invocation straddles a ``do``/``then`` boundary.
    * **``$VAR`` / ``${...}`` / ``$(...)`` are not expanded.** They are kept
      literally in the token, which is what every other part of this tool assumes.
    * **Pipelines and ``&&`` lists are one command here, not several.** The extent
      is right (bash does keep reading after a trailing ``|``), but the tokens of
      every stage land in one list, so a flag written after a pipe would be
      attributed to the program before it.

    The token stream is cross-checked against real bash (a stub interpreter that
    prints its own argv) in ``tools/test_check_script_args.py``.
    """
    state = _NORMAL
    tokens: list[str] = []
    whitespace_tokens: list[str] = []
    escaped_comment_tokens: list[str] = []
    current: list[str] = []
    started = False
    current_is_escaped_comment = False

    def flush() -> None:
        nonlocal started, current_is_escaped_comment
        if not started:
            return
        token = "".join(current)
        tokens.append(token)
        if token and not token.strip():
            whitespace_tokens.append(token)
        if current_is_escaped_comment:
            escaped_comment_tokens.append(token)
        current.clear()
        started = False
        current_is_escaped_comment = False

    def begin() -> None:
        nonlocal started
        started = True

    index = start
    while index < len(lines):
        line = lines[index]
        position = 0
        length = len(line)
        continues = False
        while position < length:
            char = line[position]
            if state == _NORMAL:
                if char == "\\":
                    if position + 1 == length:
                        continues = True  # backslash-newline: a real continuation
                        position += 1
                        continue
                    nxt = line[position + 1]
                    if nxt == "#" and not started:
                        current_is_escaped_comment = True
                    begin()
                    current.append(nxt)
                    position += 2
                    continue
                if char == "'":
                    begin()
                    state = _SINGLE
                    position += 1
                    continue
                if char == '"':
                    begin()
                    state = _DOUBLE
                    position += 1
                    continue
                if char == "$" and position + 1 < length and line[position + 1] == "'":
                    begin()
                    state = _ANSI_C
                    position += 2
                    continue
                if char == "$" and position + 1 < length and line[position + 1] == '"':
                    begin()
                    state = _DOUBLE
                    position += 2
                    continue
                if char == "#" and not started:
                    position = length  # comment to end of line; `\` in it is inert
                    break
                if char in " \t":
                    flush()
                    position += 1
                    continue
                if char in _OPERATOR_CHARS:
                    # bash needs no whitespace around a control operator: it lexes
                    # `x.py|` as `x.py` `|`.
                    #
                    # `&` immediately after a redirection char is part of that
                    # redirection word (`2>&1`), not an operator.
                    if char == "&" and current and current[-1] in "<>":
                        begin()
                        current.append(char)
                        position += 1
                        continue
                    flush()
                    operator = char
                    if position + 1 < length and line[position + 1] == char:
                        operator = char * 2
                    begin()
                    current.append(operator)
                    flush()
                    position += len(operator)
                    continue
                begin()
                current.append(char)
                position += 1
                continue
            if state == _SINGLE:
                if char == "'":
                    state = _NORMAL
                else:
                    current.append(char)
                position += 1
                continue
            if state == _DOUBLE:
                if char == "\\":
                    if position + 1 == length:
                        continues = True
                        position += 1
                        continue
                    nxt = line[position + 1]
                    if nxt in '$`"\\':
                        current.append(nxt)
                    else:
                        current.append(char)
                        current.append(nxt)
                    position += 2
                    continue
                if char == '"':
                    state = _NORMAL
                else:
                    current.append(char)
                position += 1
                continue
            # _ANSI_C
            if char == "\\" and position + 1 < length:
                current.append(_ANSI_C_ESCAPES.get(line[position + 1], line[position + 1]))
                position += 2
                continue
            if char == "'":
                state = _NORMAL
            else:
                current.append(char)
            position += 1

        if continues:
            index += 1
            continue
        if state in (_SINGLE, _DOUBLE, _ANSI_C):
            # A newline inside a quote is literal and the command keeps going.
            current.append("\n")
            index += 1
            continue
        flush()
        if tokens and tokens[-1] in _LINE_CONTINUING_OPERATORS and index + 1 < len(lines):
            index += 1
            continue
        return ScannedCommand(
            start,
            index,
            tokens,
            whitespace_tokens,
            escaped_comment_tokens,
            dangling_operator=tokens[-1] if tokens and tokens[-1] in _LINE_CONTINUING_OPERATORS else None,
        )

    flush()
    return ScannedCommand(
        start,
        max(start, len(lines) - 1),
        tokens,
        whitespace_tokens,
        escaped_comment_tokens,
        unterminated=state != _NORMAL,
        dangling_operator=tokens[-1] if tokens and tokens[-1] in _LINE_CONTINUING_OPERATORS else None,
    )


def command_continuation_problems(lines: list[str], command: ScannedCommand) -> list[tuple[int, str]]:
    r"""Ways ``command`` does not run as the block it sits in is written.

    Each is stated as what bash does to a reader who pastes the block.

    1. The command ends, and the very next line begins with ``-``. bash runs that
       line as its own command, which is ``exit 127``; every flag on it, and on the
       lines after it, is never passed to the program. A trailing ``\`` followed by
       whitespace, ``\#``, and ``\`` + a non-ASCII space all land here: none of
       them ends the line with a backslash as far as bash is concerned, so none of
       them continues.
    2. An argument that is entirely whitespace: ``\`` + space at a word boundary
       passes the program a literal one-space positional.
    3. An argument whose ``#`` came from ``\#``: the comment marker was escaped, so
       the rest of the line becomes one positional argument instead of a comment.
    4. The token stream ends on an unquoted control operator (``python x.py|`` and
       then EOF or a comment). bash rejects that at parse time; ``bash -n`` exits 2
       and nothing in the block runs.

    A backslash that is *quoted* (``--pattern '\ x'``) reaches none of these: it is
    data and the command runs.
    """
    problems: list[tuple[int, str]] = []
    if command.dangling_operator is not None:
        problems.append(
            (
                command.start_index + 1,
                f"the block ends on a control operator ({command.dangling_operator!r}) with no command "
                "after it, so bash refuses the whole block with a syntax error (`bash -n` exits 2) and "
                "none of it runs",
            )
        )
    following = command.end_index + 1
    if following < len(lines):
        nxt = lines[following].strip()
        if nxt.startswith("-"):
            problems.append(
                (
                    command.end_index + 1,
                    "the command ends on this line as bash reads it (no line continuation), so the "
                    f"next line runs as its own command and exits 127: {lines[following].strip()!r}",
                )
            )
    problems.extend(
        (
            command.end_index + 1,
            "an escaped space/tab makes a standalone whitespace-only argument "
            f"({token!r}) -- the interpreter receives it as a positional",
        )
        for token in command.whitespace_tokens
    )
    problems.extend(
        (
            command.end_index + 1,
            rf"a `\#` escaped the comment marker, so {token!r} is passed as a positional argument",
        )
        for token in command.escaped_comment_tokens
    )
    return problems


def broken_shell_continuations(lines: list[str]) -> list[tuple[int, str]]:
    """Every continuation problem in `lines`, over every command in them.

    ``(1-based line number, explanation)``. Used by the README's own assertion.
    """
    problems: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        command = scan_shell_command(lines, index)
        problems.extend(command_continuation_problems(lines, command))
        index = command.end_index + 1
    return problems


def _python_invocation_start_index(line: str) -> int | None:
    """Token index at which a `python`/`python3[.x]` invocation begins on `line`,
    after skipping any recognized env-assignment/`env`/`conda run` prefix -- or
    None if this line doesn't start one. Shell-tokenizing makes
    `env FOO=bar python ...`, `PYTHONPATH=... python ...`, and
    `conda run -n env python ...` visible. See module docstring."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    command = scan_shell_command([line], 0)
    if command.unterminated:
        return None  # unbalanced quotes etc -- not a parseable command line
    tokens = command.tokens
    if not tokens:
        return None
    i = _skip_env_prefix(tokens)
    if i < len(tokens) and _PYTHON_BIN_RE.match(tokens[i]):
        return i
    return None


def extract_invocations(script: Path) -> tuple[list[Invocation], list[tuple[int, str]], list[tuple[int, str]]]:
    """Split a shell script into logical `python ...` commands (following
    backslash line continuations) and record their raw text + line number.

    Returns ``(invocations, unrecognized_python_mentions, broken_continuations)``.

    ``unrecognized_python_mentions`` is every line that mentions the word
    "python" but was NOT recognized as an invocation start and isn't part of an
    already-collected invocation's continuation lines. See module docstring.

    ``broken_continuations`` is ``(line number, explanation)`` for every command
    whose block does not run as it is written, decided by tokenizing it the way
    bash does (``scan_shell_command`` / ``command_continuation_problems``). Such a
    block is NOT returned as an invocation. See module docstring."""
    lines = script.read_text().splitlines()
    if script.suffix == ".md":
        lines = _fenced_code_only(lines)
    invocations: list[Invocation] = []
    unrecognized: list[tuple[int, str]] = []
    broken_continuations: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if _python_invocation_start_index(line) is not None:
            command = scan_shell_command(lines, i)
            problems = command_continuation_problems(lines, command)
            if problems:
                broken_continuations.extend(problems)
            else:
                block_lines = lines[command.start_index : command.end_index + 1]
                invocations.append(Invocation(script=script, line_no=i + 1, text="\n".join(block_lines)))
            i = command.end_index + 1
            continue
        if PYTHON_WORD_RE.search(stripped):
            unrecognized.append((i + 1, stripped))
        i += 1
    return invocations, unrecognized, broken_continuations


def extract_flags(invocation_text: str) -> set[str]:
    flags: set[str] = set()
    for m in FLAG_RE.finditer(invocation_text):
        flag = m.group(1)
        # normalize `--flag=value` -> `--flag`, and defensively drop a
        # trailing '.' in case a flag is ever matched at the end of a prose
        # sentence inside a fenced block (e.g. "...run with --foo.").
        flags.add(flag.split("=", 1)[0].rstrip("."))
    return flags


def _extract_positional_subcommands(invocation_text: str) -> list[str]:
    """Leading tyro subcommand-selector tokens (this repo's `group:variant`
    convention -- see module docstring) found anywhere in the invocation, in
    the order they appear. Passed back to the script ahead of ``--help`` when
    live-introspecting so we see the flag surface for the *actual* documented
    subcommand tree, not just the bare default."""
    return [m.group(1) for m in POSITIONAL_SUBCOMMAND_RE.finditer(invocation_text)]


def resolve_module_path(dotted: str) -> Path | None:
    parts = dotted.split(".")
    top = parts[0]
    root = PACKAGE_SRC_ROOTS.get(top)
    if root is None:
        return None
    # e.g. holosoma.fada.planner_idm.train
    #   -> src/holosoma/holosoma/fada/planner_idm/train.py
    candidate = root / top / Path(*parts[1:]).with_suffix(".py")
    if candidate.is_file():
        return candidate
    # fall back: package __init__ style module (dotted path is a package)
    candidate_init = root / top / Path(*parts[1:]) / "__init__.py"
    if candidate_init.is_file():
        return candidate_init
    return None


def resolve_py_file_by_basename(token: str, tracked_files: list[str]) -> tuple[Path | None, str]:
    if not token.endswith(".py"):
        return None, "not a .py token"
    # Prefer an exact literal-path match first (e.g. `src/holosoma/holosoma/run_sim.py`
    # written out in full in the script -- no shell variables to resolve).
    literal = token[2:] if token.startswith("./") else token
    if "$" not in literal and "{" not in literal:
        tracked_set = set(tracked_files)
        if literal in tracked_set:
            return REPO_ROOT / literal, ""
    basename = Path(token.split("/")[-1]).name
    matches = [f for f in tracked_files if f.endswith("/" + basename) or f == basename]
    if len(matches) == 1:
        return REPO_ROOT / matches[0], ""
    if len(matches) == 0:
        return None, f"no tracked file named {basename}"
    return None, f"ambiguous basename {basename} ({len(matches)} tracked matches)"


class _ArgumentCollector(ast.NodeVisitor):
    """Collect all string-literal option-strings passed to any
    `<something>.add_argument(...)` call anywhere in a module (covers both
    parser.add_argument and group.add_argument, regardless of which local
    variable name the parser/group was assigned to), and independently note
    whether the file also calls `tyro.cli(...)` anywhere (the two are not
    mutually exclusive -- see the hybrid-CLI handling in module docstring).

    Does NOT descend into a statically-unreachable `if False:` / `if 0:` branch
    (see `visit_If`). This is a narrow carve-out, not general dead-code analysis:
    a flag gated behind a non-constant condition, an indirect constant
    (module-level `bool`, env-var check, ...), or any other dead-code shape is
    still collected as if it were live. See module docstring."""

    def __init__(self) -> None:
        self.flags: set[str] = set()
        self.uses_tyro = False
        self.parser_builder_imports: list[tuple[str | None, int, str]] = []
        # (module, level, imported_name) for `from X import build_arg_parser`

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    self.flags.add(arg.value)
        if isinstance(func, ast.Attribute) and func.attr == "cli" and isinstance(func.value, ast.Name) and func.value.id == "tyro":
            self.uses_tyro = True
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        if isinstance(test, ast.Constant):
            # `if False:` / `if 0:` -> body is unreachable, only orelse runs.
            # `if True:` / `if 1:` -> orelse is unreachable, only body runs.
            live_branch = node.orelse if not test.value else node.body
            for stmt in live_branch:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name in PARSER_BUILDER_NAMES:
                self.parser_builder_imports.append((node.module, node.level, alias.name))
        self.generic_visit(node)


def _module_dir_to_dotted(py_file: Path) -> tuple[Path, str] | None:
    """Best-effort: given a resolved source file, find which PACKAGE_SRC_ROOTS
    entry it lives under and return (package_root, dotted_package_path) for
    the *directory* containing it (used to resolve relative imports)."""
    for _pkg, root in PACKAGE_SRC_ROOTS.items():
        try:
            rel = py_file.relative_to(root)
        except ValueError:
            continue
        return root, rel
    return None


def _collect_static_argparse_flags(py_file: Path, _depth: int = 0) -> tuple[set[str] | None, str]:
    """Static argparse AST scan only (no tyro handling) -- used both for pure
    argparse files and as the hybrid-CLI fallback when live introspection is
    unavailable. Returns (flag_set_or_None, note)."""
    if _depth > 3:
        return None, "import-follow depth exceeded"
    if not py_file.is_file():
        return None, f"source file not found: {py_file}"

    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    collector = _ArgumentCollector()
    collector.visit(tree)

    if collector.flags:
        return collector.flags, ""

    for module, level, _name in collector.parser_builder_imports:
        target: Path | None = None
        if level and level > 0:
            resolved = _module_dir_to_dotted(py_file)
            if resolved is not None:
                _root, rel_pkg_dir = resolved
                base_dir = py_file.parent
                for _ in range(level - 1):
                    base_dir = base_dir.parent
                if module:
                    target = base_dir / Path(*module.split(".")).with_suffix(".py")
                else:
                    target = base_dir
        elif module:
            target = resolve_module_path(module)
        if target is not None and target.is_file():
            sub_flags, sub_note = _collect_static_argparse_flags(target, _depth + 1)
            if sub_flags is not None:
                return sub_flags, f"(via re-exported parser from {target.relative_to(REPO_ROOT)})"
            return None, sub_note

    return None, "no argparse parser found in file (and no re-exported build_arg_parser import)"


def _candidate_python_bins() -> list[tuple[Path, str]]:
    """Resolve this repo's own tracked conda envs the same way
    scripts/source_common.sh resolves WORKSPACE_DIR/CONDA_ROOT, so live
    introspection works on any machine that ran this repo's own setup scripts."""
    workspace_dir_str = os.environ.get("WORKSPACE_DIR") or os.environ.get("HOLOSOMA_WORKSPACE_DIR")
    if workspace_dir_str:
        workspace_dir = Path(workspace_dir_str)
    else:
        usr0_base = Path(f"/usr0/{os.environ.get('USER', '')}")
        if os.environ.get("USER") and usr0_base.is_dir() and os.access(usr0_base, os.W_OK):
            workspace_dir = usr0_base / ".holosoma_deps"
        else:
            workspace_dir = Path.home() / ".holosoma_deps"
    conda_root = workspace_dir / "miniconda3"
    out: list[tuple[Path, str]] = []
    for env_name in CANDIDATE_CONDA_ENVS:
        py = conda_root / "envs" / env_name / "bin" / "python"
        if py.is_file():
            out.append((py, env_name))
    return out


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX_CHARS = "│|"  # U+2502 = tyro's box char; '|' covers a plain-ASCII fallback
_FLAG_LINE_START_RE = re.compile(r"^(?:--[A-Za-z][A-Za-z0-9_.-]*|-[A-Za-z])(?:[,\s]|$)")
_FLAG_TOKEN_RE = re.compile(r"--[A-Za-z][A-Za-z0-9_.-]*")
# A continuation line (deeper-indented than the established option column) is
# read as a wrapped *alias* list only if, once any same-line description is
# dropped, it consists of nothing but flag tokens / short flags / ALL-CAPS
# metavars separated by commas or whitespace. An ordinary lowercase word
# anywhere in it fails the match.
_ALIAS_CONTINUATION_RE = re.compile(
    r"^(?:--[A-Za-z][A-Za-z0-9_.-]*|-[A-Za-z])"
    r"(?:[,\s]+(?:--[A-Za-z][A-Za-z0-9_.-]*|-[A-Za-z]|[A-Z0-9_\[\]{}|.'\"-]+))*$"
)


def _dewrap_help_line(raw_line: str) -> tuple[int, str]:
    """Strip ANSI escape codes and tyro's box-drawing left/right border from
    one --help transcript line, WITHOUT collapsing interior indentation, which
    is what distinguishes a real option definition from a wrapped
    description/alias continuation at a deeper column. Returns (indent,
    content)."""
    line = _ANSI_RE.sub("", raw_line).rstrip("\n")
    if line[:1] in _BOX_CHARS:
        line = line[1:]
        if line[:1] == " ":
            line = line[1:]
    line = line.rstrip()
    if line[-1:] in _BOX_CHARS:
        line = line[:-1].rstrip()
    content = line.lstrip(" ")
    indent = len(line) - len(content)
    return indent, content


def _parse_help_flags(text: str) -> set[str]:
    """Extract recognized flags from a captured --help transcript. Handles
    both tyro's box-drawn ('│ --task.foo STR │') and plain argparse's
    ('  --foo, --bar FOO  help text') formats, structurally rather than by
    "does the stripped line start with '--'": the first flag-shaped line
    establishes the option column (its indentation); only lines at or above
    that same indentation are new option definitions, and a flag token is
    only ever read from the invocation portion of such a line (everything
    before the first run of 2+ spaces, which is where inline help text
    starts). Anything indented deeper is a continuation -- of the previous
    option's wrapped description (never mined for flags, even if it happens
    to start with a flag-shaped token) or, if it looks like nothing but a
    wrapped alias/metavar list, of the previous option's own alias set (is
    mined). See module docstring."""
    flags: set[str] = set()
    baseline: int | None = None
    for raw_line in text.splitlines():
        indent, content = _dewrap_help_line(raw_line)
        if not content or not _FLAG_LINE_START_RE.match(content):
            continue
        if baseline is None or indent <= baseline:
            baseline = indent if baseline is None else min(baseline, indent)
            segment = re.split(r"\s{2,}", content, maxsplit=1)[0]
            for m in _FLAG_TOKEN_RE.finditer(segment):
                flags.add(m.group(0).rstrip("."))
        else:
            segment = re.split(r"\s{2,}", content, maxsplit=1)[0]
            if _ALIAS_CONTINUATION_RE.match(segment):
                for m in _FLAG_TOKEN_RE.finditer(segment):
                    flags.add(m.group(0).rstrip("."))
    return flags


# ---------------------------------------------------------------------------
# Containment for the two places this tool EXECUTES a shipped entry point
# (`--help` capture and the runtime acceptance probe).
#
# Both of them import and run real release code, so every import-time side
# effect in the entry point's own import graph happens for real, with this
# tool's environment.
#
# `_contained_run` gives the child a scratch cwd on a filesystem this repository
# does not own, sets `PYTHONDONTWRITEBYTECODE=1` plus `-B`, and starts a fresh
# session per child so the whole process GROUP can be killed -- on timeout and
# again after a normal exit, which reaps a descendant that outlived the parse.
#
# WHAT IS NOT CONTAINED, and cannot be by this mechanism:
#   * writes to ABSOLUTE paths the entry point chooses (`$HOME`, `/tmp`, a
#     hardcoded log dir), and anything it does to files outside the cwd;
#   * network, DDS/ZMQ traffic, or anything sent to a robot or a licence server;
#   * a child that deliberately re-parents itself (`setsid`, a double fork, a
#     daemon) -- it leaves the group this kills;
#   * mutation of shared state outside the filesystem (databases, GPU state);
#   * resource use: an entry point that allocates the GPU or spins the CPU for
#     its whole timeout does so.
# ---------------------------------------------------------------------------

# A directory that is NOT under REPO_ROOT: the release build aborts on any
# untracked file in the repository, so a probe writing into REPO_ROOT would
# break the build.
_PROBE_CWD_PREFIX = "fada-arg-cwd-"


@dataclass
class _ContainedResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _kill_process_group(pgid: int | None, proc: subprocess.Popen) -> None:
    """Kill the child AND everything it started, best effort.

    `Popen.kill()` signals one pid; a grandchild holding the inherited stdout/
    stderr pipes survives it.
    """
    if pgid is not None and hasattr(os, "killpg") and pgid != os.getpgrp():
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _contained_run(cmd: list[str], *, env: dict[str, str], timeout: int) -> _ContainedResult:
    """Run `cmd` under the containment described above. Never raises on timeout."""
    env = dict(env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix=_PROBE_CWD_PREFIX) as scratch_cwd:
        env["TMPDIR"] = scratch_cwd
        out_path = Path(scratch_cwd) / "stdout.txt"
        err_path = Path(scratch_cwd) / "stderr.txt"
        popen_kwargs: dict[str, object] = {}
        if hasattr(os, "setsid"):
            popen_kwargs["start_new_session"] = True
        # FILES, not pipes. `communicate()` waits for end-of-file on the pipes,
        # and a grandchild that inherited them holds the write end open past its
        # parent's exit. A regular file has no such reader/writer rendezvous:
        # `wait()` returns as soon as the direct child exits, and the descendants
        # are then killed by group.
        with open(out_path, "w+") as out_fh, open(err_path, "w+") as err_fh:
            # argv is built from tracked paths, never a shell string.
            proc = subprocess.Popen(
                cmd,
                cwd=scratch_cwd,
                env=env,
                stdout=out_fh,
                stderr=err_fh,
                stdin=subprocess.DEVNULL,
                text=True,
                **popen_kwargs,  # type: ignore[arg-type]
            )
            # Captured immediately: after the child is reaped its pgid is no longer
            # readable, and the group may still hold descendants that must be killed.
            pgid = proc.pid if popen_kwargs.get("start_new_session") else None
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(pgid, proc)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=10)
            finally:
                # Unconditional: reaps a descendant that survived a successful run.
                _kill_process_group(pgid, proc)
        stdout = out_path.read_text(errors="replace")
        stderr = err_path.read_text(errors="replace")
        return _ContainedResult(proc.returncode, stdout, stderr, timed_out)


def _contained_python_argv(python_bin: Path, *rest: str) -> list[str]:
    """`-B` as well as PYTHONDONTWRITEBYTECODE: an entry point can re-enable
    bytecode writing for a subinterpreter it starts."""
    return [str(python_bin), "-B", *rest]


_LIVE_HELP_CACHE: dict[tuple[str, tuple[str, ...]], tuple[set[str] | None, str, bool]] = {}


def _live_help_flags(py_file: Path, positional_tokens: list[str]) -> tuple[set[str] | None, str, bool]:
    """Run `<python> <py_file> <positional_tokens> --help` in each candidate
    conda env until one succeeds. Returns (flags_or_None, note, truncated)."""
    cache_key = (str(py_file), tuple(positional_tokens))
    if cache_key in _LIVE_HELP_CACHE:
        return _LIVE_HELP_CACHE[cache_key]

    candidates = _candidate_python_bins()
    if not candidates:
        result = (
            None,
            "no candidate conda env found (looked for "
            f"{'/'.join(CANDIDATE_CONDA_ENVS)} under $WORKSPACE_DIR/miniconda3/envs per "
            "scripts/source_common.sh's own WORKSPACE_DIR resolution -- run this repo's "
            "setup scripts first)",
            False,
        )
        _LIVE_HELP_CACHE[cache_key] = result
        return result

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(root) for root in PACKAGE_SRC_ROOTS.values())
    errors: list[str] = []
    for python_bin, env_name in candidates:
        cmd = _contained_python_argv(python_bin, str(py_file), *positional_tokens, "--help")
        proc = _contained_run(cmd, env=env, timeout=LIVE_HELP_TIMEOUT_S)
        if proc.timed_out:
            errors.append(f"{env_name}: timed out after {LIVE_HELP_TIMEOUT_S}s")
            continue
        if proc.returncode != 0:
            tail_lines = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = tail_lines[-1] if tail_lines else f"exit code {proc.returncode}"
            errors.append(f"{env_name}: exit {proc.returncode}: {tail}")
            continue
        flags = _parse_help_flags(proc.stdout + "\n" + proc.stderr)
        if not flags:
            errors.append(f"{env_name}: --help exited 0 but no flags were parsed from its output")
            continue
        # A required subcommand tier (e.g. run_policy.py's `{inference:...}`)
        # that wasn't given a real selector -- either because none was
        # extracted (positional_tokens empty) or the doc used a
        # <placeholder> our group:variant regex correctly declined to
        # match -- makes tyro print *only* a subcommand-choice stub (just
        # -h/--help, no real flags) and still exit 0. Treated as a failure, so
        # the caller falls back or reports it as unverifiable.
        if flags <= {"--help"} and "subcommands" in proc.stdout.lower():
            errors.append(
                f"{env_name}: --help returned only a subcommand-choice stub "
                f"(no real subcommand selector in the invocation, e.g. a <placeholder>) "
                f"-- flags of the real subcommand's config can't be seen this way"
            )
            continue
        truncated = bool(re.search(r"and \d+ more", proc.stdout))
        note = f"(live --help via {env_name} env" + (", TRUNCATED -- see module docstring)" if truncated else ")")
        result = (flags, note, truncated)
        _LIVE_HELP_CACHE[cache_key] = result
        return result

    result = (None, "tyro --help failed in every candidate env: " + " | ".join(errors), False)
    _LIVE_HELP_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# The runtime acceptance probe (module docstring).
#
# A `--help` transcript lists each option's canonical spelling. Whether the
# program ALSO accepts some other spelling of the same option -- tyro rewrites
# `_` <-> `-` in every option name before parsing, argparse does not -- is a
# property of the parser, not of the transcript, so it is answered by handing the
# real entry point the real argv and watching what its real parser does.
#
# The driver below monkeypatches the three places a parse can complete --
# `argparse.ArgumentParser.parse_args`, `.parse_known_args`, and `tyro.cli` --
# and raises out of the program the instant the OUTERMOST one finishes, so
# nothing after argument parsing runs: no simulator, no policy, no training loop.
# Nested calls (tyro's own internal argparse) are passed through untouched by a
# depth counter, so the verdict comes from the layer that owns the arguments.
#
# Three outcomes:
#   accepted  -- a top-level parse consumed the argv, probe flags included.
#   rejected  -- a top-level parse died with "unrecognized arguments/options".
#   error     -- anything else (a missing import, a <placeholder> that fails type
#                validation, a required file that isn't there, a timeout). This
#                is NOT "accepted": the probe could not decide, and the caller
#                falls back to needs-review, which is whitelist-or-fail.
# ---------------------------------------------------------------------------

_PROBE_DRIVER = r'''
import contextlib
import io
import json
import os
import re
import runpy
import signal
import sys

# Seconds the program may keep running after a covering parse has accepted, while
# it is fenced (see `_install_fence`), so a LATER parse can be reached and become
# the authoritative verdict.
POST_PARSE_BUDGET_S = 20

# Captured before the fence replaces builtins.open: the verdict file is this
# driver's own output and must not be stopped by its own guard.
_ORIG_OPEN = open

RESULT = os.environ["FADA_PROBE_RESULT"]
ENTRY = os.environ["FADA_PROBE_ENTRY"]
PROBE_FLAGS = json.loads(os.environ["FADA_PROBE_FLAGS"])

state = {
    "verdict": None,
    "how": "",
    "stderr": "",
    "depth": 0,
    "fenced": False,
    "parses": 0,
    # Set by `_accept`: did the accepting parse hand tokens on to a later parser?
    "provisional": False,
    "deferred": 0,
}


class _Stop(BaseException):
    """BaseException, so the entry point's own `except Exception` does not
    swallow it."""


def _unrecognized(text):
    return bool(re.search(r"unrecognized (argument|option)", text, re.IGNORECASE))


def _finish(verdict, how, err=""):
    state["verdict"] = verdict
    state["how"] = how
    state["stderr"] = err[-4000:]
    raise _Stop()


def _accept(how, deferred_tokens=0):
    """Record an acceptance WITHOUT stopping, and fence the program from here on.

    The verdict comes from the LAST parse that consumes the argv -- a file whose
    argparse pre-parser accepts `--foo` and whose tyro layer then rejects it
    exits 2 -- so an acceptance lets the program continue to the next parse.

    "Continue" is bounded: from the first acceptance onward every door out of the
    process is fenced (see `_fence`), and the first one the program reaches stops
    it with the acceptance already recorded.

    `deferred_tokens` is how many argv tokens THIS parse handed on to a later
    parser instead of adjudicating (`parse_known_args`' extras, `tyro.cli`'s
    `return_unknown_args` tail). Non-zero means the parse said, in its own return
    value, that it is not the last one -- so its acceptance is provisional and
    `_fence` must not report it as a pass. See `_fence`.
    """
    state["verdict"] = "accepted"
    state["how"] = how
    state["parses"] += 1
    state["fenced"] = True
    state["provisional"] = int(deferred_tokens) > 0
    state["deferred"] = int(deferred_tokens)
    _arm_budget()


def _stop_after_accept(reason):
    """End the run at a fence/budget, choosing between `accepted` and undecided.

    Two cases:

    * The accepting parse consumed the whole argv (`parse_args`, a plain
      `tyro.cli`). Anything it did not recognize would have exited 2 there, so the
      verdict stands.

    * The accepting parse DEFERRED tokens to a later parser that was not reached.
      `eval_agent.py:main` has this shape: `tyro.cli(CheckpointConfig,
      return_unknown_args=True)`, then a checkpoint load and directory creation,
      then a second `tyro.cli` over the deferred tail, which is the parser that
      decides whether the documented argv is valid. The result is `undecided`,
      i.e. `needs_review` (whitelist-or-fail), not a pass.
    """
    if state.get("provisional"):
        _finish(
            "undecided",
            "%s, but that parse deferred %d token(s) to a later parser that was never "
            "reached (%s)" % (state["how"], state.get("deferred", 0), reason),
        )
    _finish(state["verdict"], "%s; %s" % (state["how"], reason))


def _fence(name):
    if state["fenced"]:
        _stop_after_accept("stopped at %s (first side effect after the parse)" % (name,))


def _install_fence():
    """Doors out of the process, closed once the argv has been consumed.

    Not a sandbox -- see the containment note in check_script_args.py for what
    this cannot cover. It covers the calls a simulator/policy/robot start passes
    through, so "keep running to the next parse" does not become "run the
    program".
    """
    import builtins
    import io
    import mmap as _mmap
    import pathlib
    import shutil as _shutil
    import socket
    import subprocess as _sp

    # Every door is patched on the object the *caller* reaches. `Path.open` goes
    # through `Path._accessor.open`, which is `io.open` captured at import time,
    # so neither `builtins.open` nor `io.open` nor a later `os.open` patch sees a
    # `Path.write_text`.
    for owner, attr, label in (
        (builtins, "open", "open"),
        (io, "open", "io.open"),
        (pathlib.Path, "open", "Path.open"),
    ):
        original = getattr(owner, attr, None)
        if original is None:  # pragma: no cover - defensive
            continue

        # `*a, **kw` rather than a named signature: the wrapper is installed for the
        # whole run (it only *acts* once fenced), so it has to be call-compatible
        # with every spelling the program uses, including `open(file=..., mode=...)`.
        def open_(*a, __orig=original, __label=label, **kw):
            mode = kw.get("mode")
            if mode is None:
                # Position 1 for builtins/io.open (file, mode); position 1 for the
                # bound `Path.open(self, mode)` too, since `self` is a[0] there.
                mode = a[1] if len(a) > 1 else "r"
            if any(ch in str(mode) for ch in "wax+"):
                _fence("%s(%r, %r)" % (__label, a[0] if a else kw.get("file"), mode))
            return __orig(*a, **kw)

        setattr(owner, attr, open_)

    # Every entry below is a door out of the process. The list is enumerated, not
    # derived, so a door that is not on it stays open. It covers metadata writes
    # (chmod/chown/utime), namespace mutations
    # (link/symlink/replace/rmdir/removedirs/renames), process launches not spelled
    # `fork`/`exec*` (posix_spawn/spawn*/kill), the `shutil` copy/move family
    # (which reaches the filesystem through C-level `os` calls that patching `open`
    # does not see), `os.chdir` (which relocates every relative path the rest of
    # the program touches), and writable `mmap`, whose stores go through no patched
    # call at all.
    for owner, attr in (
        (os, "system"),
        (os, "fork"),
        (os, "execv"),
        (os, "execve"),
        (os, "execvp"),
        (os, "execvpe"),
        (os, "execl"),
        (os, "execle"),
        (os, "execlp"),
        (os, "posix_spawn"),
        (os, "posix_spawnp"),
        (os, "spawnv"),
        (os, "spawnve"),
        (os, "kill"),
        (os, "killpg"),
        (os, "remove"),
        (os, "unlink"),
        (os, "rename"),
        (os, "renames"),
        (os, "replace"),
        (os, "rmdir"),
        (os, "removedirs"),
        (os, "mkdir"),
        (os, "makedirs"),
        (os, "chmod"),
        (os, "fchmod"),
        (os, "lchmod"),
        (os, "chown"),
        (os, "lchown"),
        (os, "fchown"),
        (os, "utime"),
        (os, "truncate"),
        (os, "ftruncate"),
        (os, "chdir"),
        (os, "fchdir"),
        (os, "chroot"),
        (os, "link"),
        (os, "symlink"),
        (os, "mkfifo"),
        (os, "mknod"),
        (os, "setxattr"),
        (os, "removexattr"),
        (_sp, "Popen"),
        (_sp, "run"),
        (_sp, "call"),
        (_sp, "check_call"),
        (_sp, "check_output"),
        (_shutil, "copy"),
        (_shutil, "copy2"),
        (_shutil, "copyfile"),
        (_shutil, "copytree"),
        (_shutil, "copymode"),
        (_shutil, "copystat"),
        (_shutil, "move"),
        (_shutil, "rmtree"),
        (_shutil, "make_archive"),
        (_shutil, "unpack_archive"),
        (_shutil, "chown"),
        (socket, "socket"),
        (socket, "create_connection"),
        (socket, "create_server"),
        (pathlib.Path, "mkdir"),
        (pathlib.Path, "touch"),
        (pathlib.Path, "unlink"),
        (pathlib.Path, "rmdir"),
        (pathlib.Path, "rename"),
        (pathlib.Path, "replace"),
        (pathlib.Path, "chmod"),
        (pathlib.Path, "lchmod"),
        (pathlib.Path, "symlink_to"),
        (pathlib.Path, "hardlink_to"),
        (pathlib.Path, "link_to"),
        (pathlib.Path, "write_text"),
        (pathlib.Path, "write_bytes"),
    ):
        original = getattr(owner, attr, None)
        if original is None:
            continue

        def guarded(*a, __orig=original, __name="%s.%s" % (getattr(owner, "__name__", owner), attr), **kw):
            _fence(__name)
            return __orig(*a, **kw)

        try:
            setattr(owner, attr, guarded)
        except (AttributeError, TypeError):  # pragma: no cover - read-only attribute
            pass

    _os_open = os.open

    def os_open(path, flags, *a, **kw):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            _fence("os.open(%r)" % (path,))
        return _os_open(path, flags, *a, **kw)

    os.open = os_open

    # After `mmap()` returns, `m[0:4] = b"...."` writes to the file through the MMU
    # with no Python-level call to intercept, so the fence is at map-creation time
    # and `access` says whether the map can write. `ACCESS_READ` is the only
    # read-only spelling; the default (`ACCESS_DEFAULT` / a plain `prot`) maps
    # read-write.
    _orig_mmap = _mmap.mmap

    class _GuardedMmap(_orig_mmap):  # type: ignore[misc, valid-type]
        def __new__(cls, *a, **kw):
            # Fail closed. Only two spellings are unambiguously read-only, and both are
            # keywords; anything else (including every positional form, whose argument
            # order differs between Unix and Windows) counts as writable.
            read_only = kw.get("access") == _mmap.ACCESS_READ or (
                "access" not in kw and kw.get("prot") == getattr(_mmap, "PROT_READ", object())
            )
            if not read_only:
                _fence("mmap.mmap(access=%r)" % (kw.get("access"),))
            return _orig_mmap.__new__(cls, *a, **kw)

    _mmap.mmap = _GuardedMmap

    # Backstop: a post-acceptance loop that touches none of the doors above still
    # has to end. The outer subprocess timeout would catch it, but only after
    # LIVE_HELP_TIMEOUT_S with no verdict written; this writes the verdict it has.
    if hasattr(signal, "SIGALRM"):
        def _expire(_signum, _frame):
            if state["fenced"]:
                _stop_after_accept("stopped by the post-parse time budget")

        signal.signal(signal.SIGALRM, _expire)


def _arm_budget():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(POST_PARSE_BUDGET_S)


def _exit_verdict(how, err):
    """A top-level parse died. Decide what that means, given what came before.

    An "unrecognized argument" is definitive whatever preceded it. Any other
    non-zero exit is not about the flags under test, so it downgrades to `error`
    only when there is no earlier acceptance to fall back on; with one, that
    acceptance stands.
    """
    if _unrecognized(err):
        _finish("rejected", how, err)
    if state["verdict"] == "accepted":
        _finish("accepted", "%s; a later %s exited without naming the flags" % (state["how"], how), err)
    _finish("error", how, err)


def _call(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        try:
            return "ok", fn(*a, **kw), buf.getvalue()
        except SystemExit as exc:
            return "exit", exc.code, buf.getvalue()


def _covers(explicit_args):
    """Did this parse see the flags under test at all?

    A parse handed an explicit `args=` list without them says nothing about them
    (eval_agent.py's two-stage CLI has this shape), so the program runs on to the
    parse that does see them.
    """
    if explicit_args is None:
        return True
    text = " ".join(str(t) for t in explicit_args)
    return all(flag in text for flag in PROBE_FLAGS)


def _leftover(extras):
    return [t for t in extras if any(str(t).split("=", 1)[0] == f for f in PROBE_FLAGS)]


import argparse

_orig_parse_args = argparse.ArgumentParser.parse_args
_orig_parse_known = argparse.ArgumentParser.parse_known_args


def _parse_args(self, args=None, namespace=None):
    if state["depth"]:
        return _orig_parse_args(self, args, namespace)
    state["depth"] += 1
    try:
        kind, value, err = _call(_orig_parse_args, self, args, namespace)
    finally:
        state["depth"] -= 1
    if kind == "exit":
        _exit_verdict("argparse.parse_args", err)
    if _covers(args):
        _accept("argparse.parse_args")
    return value


def _parse_known(self, args=None, namespace=None):
    if state["depth"]:
        return _orig_parse_known(self, args, namespace)
    state["depth"] += 1
    try:
        kind, value, err = _call(_orig_parse_known, self, args, namespace)
    finally:
        state["depth"] -= 1
    if kind == "exit":
        _exit_verdict("argparse.parse_known_args", err)
    _ns, extras = value
    if _covers(args) and not _leftover(extras):
        # `extras` are tokens this parser handed on to a later one. Passed through
        # so `_fence` does not report a provisional acceptance as a pass. See
        # `_stop_after_accept`.
        _accept("argparse.parse_known_args", deferred_tokens=len(extras))
    return value


argparse.ArgumentParser.parse_args = _parse_args
argparse.ArgumentParser.parse_known_args = _parse_known

try:
    import tyro
except Exception:
    pass
else:
    _orig_cli = tyro.cli

    def _cli(*a, **kw):
        if state["depth"]:
            return _orig_cli(*a, **kw)
        state["depth"] += 1
        try:
            kind, value, err = _call(_orig_cli, *a, **kw)
        finally:
            state["depth"] -= 1
        if kind == "exit":
            _exit_verdict("tyro.cli", err)
        if not _covers(kw.get("args")):
            return value
        deferred = 0
        if kw.get("return_unknown_args") and isinstance(value, tuple) and len(value) == 2:
            if _leftover(value[1]):
                return value
            # Same as parse_known_args' extras: a `return_unknown_args=True` tail
            # means a later parser still has to see these tokens.
            deferred = len(value[1])
        _accept("tyro.cli", deferred_tokens=deferred)
        return value

    tyro.cli = _cli

_install_fence()

sys.argv = [ENTRY, *json.loads(os.environ["FADA_PROBE_ARGV"])]
sys.path.insert(0, os.path.dirname(os.path.abspath(ENTRY)))
try:
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(ENTRY, run_name="__main__")
except _Stop:
    pass
except SystemExit as exc:
    if state["verdict"] is None:
        state["verdict"] = "error"
        state["how"] = "SystemExit(%r) before any top-level parse" % (exc.code,)
except BaseException as exc:
    if state["verdict"] is None:
        state["verdict"] = "error"
        state["how"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
if hasattr(signal, "SIGALRM"):
    signal.alarm(0)
if state["verdict"] is None:
    state["verdict"] = "error"
    state["how"] = "the entry point never reached a top-level argument parse"
state.pop("depth")
state.pop("fenced")
state.pop("provisional", None)
state.pop("deferred", None)
with _ORIG_OPEN(RESULT, "w") as fh:
    json.dump(state, fh)
'''

_PROBE_CACHE: dict[tuple[str, tuple[str, ...], tuple[str, ...]], tuple[str, str, str]] = {}


def invocation_argument_tokens(invocation_text: str) -> list[str] | None:
    """The argv a documented invocation passes to its entry point.

    Everything after the interpreter and the module/script selector, tokenized
    the way bash tokenizes it (so quoted values survive and comments do not).
    None if the line does not parse as a python invocation.
    """
    command = scan_shell_command(invocation_text.splitlines(), 0)
    if command.unterminated or not command.tokens:
        return None
    index = _skip_env_prefix(command.tokens)
    if index >= len(command.tokens) or not _PYTHON_BIN_RE.match(command.tokens[index]):
        return None
    index += 1
    while index < len(command.tokens) and command.tokens[index] in ("-u", "-B", "-O", "-E", "-s", "-I"):
        index += 1
    if index < len(command.tokens) and command.tokens[index] == "-m":
        index += 2
    elif index < len(command.tokens):
        index += 1
    return command.tokens[index:]


def probe_documented_arguments(
    py_file: Path, argv: list[str], probe_flags: set[str]
) -> tuple[str, str, str]:
    """Ask the real entry point whether it accepts `argv`.

    Returns ``(verdict, how, stderr)`` with verdict in
    ``{"accepted", "rejected", "error"}``. See the block comment above for what
    each verdict means.
    """
    cache_key = (str(py_file), tuple(argv), tuple(sorted(probe_flags)))
    if cache_key in _PROBE_CACHE:
        return _PROBE_CACHE[cache_key]

    candidates = _candidate_python_bins()
    if not candidates:
        result = ("error", "no candidate conda env found for the runtime probe", "")
        _PROBE_CACHE[cache_key] = result
        return result

    errors: list[str] = []
    result = ("error", "the runtime probe did not run", "")
    # A scratch directory, never the repository: the release build aborts on any
    # untracked file under REPO_ROOT.
    with tempfile.TemporaryDirectory(prefix="fada-arg-probe-") as scratch:
        driver = Path(scratch) / "probe_driver.py"
        driver.write_text(_PROBE_DRIVER)
        for python_bin, env_name in candidates:
            out_path = Path(scratch) / f"verdict.{env_name}.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(str(root) for root in PACKAGE_SRC_ROOTS.values())
            env["FADA_PROBE_ENTRY"] = str(py_file)
            env["FADA_PROBE_ARGV"] = json.dumps(argv)
            env["FADA_PROBE_FLAGS"] = json.dumps(sorted(probe_flags))
            env["FADA_PROBE_RESULT"] = str(out_path)
            run = _contained_run(
                _contained_python_argv(python_bin, str(driver)),
                env=env,
                timeout=LIVE_HELP_TIMEOUT_S,
            )
            payload = None
            if out_path.exists():
                try:
                    payload = json.loads(out_path.read_text())
                except ValueError:
                    payload = None
                out_path.unlink()
            if run.timed_out and payload is None:
                errors.append(f"{env_name}: timed out after {LIVE_HELP_TIMEOUT_S}s")
                continue
            if payload is None:
                errors.append(f"{env_name}: the probe produced no verdict")
                continue
            if payload["verdict"] in ("accepted", "rejected"):
                result = (
                    payload["verdict"],
                    f"live parse via {env_name} env ({payload['how']})",
                    payload.get("stderr", ""),
                )
                _PROBE_CACHE[cache_key] = result
                return result
            errors.append(f"{env_name}: {payload['how']}")

    result = ("error", "the live parser could not be reached: " + " | ".join(errors), "")
    _PROBE_CACHE[cache_key] = result
    return result


def _flags_named_in(text: str, candidates: set[str]) -> set[str]:
    """Which of `candidates` a rejection message names."""
    named = {m.group(0) for m in _FLAG_TOKEN_RE.finditer(text)}
    return {f for f in candidates if f in named}


def adjudicate_missing_flags(
    py_file: Path, invocation_text: str, missing: set[str]
) -> tuple[str, set[str], str]:
    """Decide what a set of not-in-the-``--help``-surface flags really is.

    Returns ``(verdict, offenders, evidence)`` with verdict in
    ``{"accepted", "rejected", "undecided"}``.
    """
    argv = invocation_argument_tokens(invocation_text)
    if argv is None:
        return "undecided", set(), "the invocation does not tokenize as a python command"
    verdict, how, stderr = probe_documented_arguments(py_file, argv, missing)
    if verdict == "accepted":
        return "accepted", set(), how
    if verdict == "rejected":
        named = _flags_named_in(stderr, missing) or missing
        return "rejected", named, how
    return "undecided", set(), how


@dataclass
class Resolution:
    status: str  # "checked" | "needs_review"
    recognized: set[str] | None
    note: str
    # Only set when status == "needs_review": a partial flag set we have some
    # information about (a truncated live --help, or a true hybrid file's
    # static argparse pre-parser subset).
    partial_recognized: set[str] | None = None
    # True only for the hybrid-CLI static-argparse-fallback path. There, and
    # only there, partial_recognized is a COMPLETE accounting of what the local
    # argparse pre-parser recognizes, so a non-dotted flag missing from it is a
    # HIT; dotted flags are presumed-forwarded tyro overrides and stay
    # unverified. For a truncated live --help, or a live --help that failed with
    # nothing to fall back on, partial_recognized is incomplete and this stays
    # False: every missing flag there, dotted or not, is unverified, never a
    # HIT.
    hybrid_fallback: bool = False


def _unaccepted_flags(passed: set[str], recognized: set[str]) -> set[str]:
    """Flags not present VERBATIM in a recognized-flag surface.

    Exact, with no delimiter tolerance: a ``--help`` transcript lists canonical
    spellings only. Whether the program also accepts another spelling is decided
    by :func:`probe_documented_arguments`, which every caller consults before
    calling anything a mismatch.
    """
    return {f for f in passed if f not in recognized}


def resolve_flags(py_file: Path, invocation_text: str) -> Resolution:
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    collector = _ArgumentCollector()
    collector.visit(tree)

    if not collector.uses_tyro:
        recognized, note = _collect_static_argparse_flags(py_file)
        if recognized is None:
            return Resolution("needs_review", None, note)
        return Resolution("checked", recognized, note or "(static argparse AST scan)")

    # tyro.cli(...) appears somewhere in this file -- try live introspection
    # first. This covers pure-tyro AND hybrid argparse+tyro files: for hybrids,
    # --help is served by the argparse pre-parser before tyro is reached, so this
    # path yields that subset too.
    positional_tokens = _extract_positional_subcommands(invocation_text)
    recognized, note, truncated = _live_help_flags(py_file, positional_tokens)
    if recognized is not None and not truncated:
        return Resolution("checked", recognized, note)
    if recognized is not None and truncated:
        # An invocation flag may exist past the cutoff. partial_recognized lets
        # main() still accept a *found* flag, while anything missing goes to the
        # runtime probe and then, if that cannot decide, to needs_review.
        return Resolution("needs_review", None, note, partial_recognized=recognized)

    # Live introspection failed in every candidate env. If this file also has
    # a directly-visible static argparse parser (true hybrid), fall back to
    # checking at least that subset, per module docstring Method step 4c.
    if collector.flags:
        return Resolution(
            "needs_review",
            None,
            f"live --help unavailable ({note}); falling back to static argparse subset only",
            partial_recognized=collector.flags,
            hybrid_fallback=True,
        )
    return Resolution("needs_review", None, note)


def _invocation_fingerprint(text: str) -> str:
    """Stable content fingerprint for one invocation's raw text: insensitive
    to per-line leading/trailing whitespace (so reformatting a `\\`-continued
    command does not change it) but sensitive to any content change -- an
    appended flag, a changed value, a removed line. Not keyed by line number, so
    inserting an unrelated line earlier in the same doc does not shift it."""
    normalized = "\n".join(line.strip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="also print detail for CHECKED and WHITELISTED entries")
    args = ap.parse_args()

    for key, entry in WHITELIST.items():
        assert entry.reason.strip(), f"WHITELIST entry {key!r} has an empty reason"
        assert isinstance(entry.flags, frozenset), f"WHITELIST entry {key!r} must declare a flag set"
    for key, reason in LINE_WHITELIST.items():
        assert reason.strip(), f"LINE_WHITELIST entry {key!r} has an empty reason"

    scripts = command_sources()
    tracked_files = git_tracked_all_files()

    hits: list[str] = []
    checked_detail: list[str] = []
    whitelisted_detail: list[str] = []
    needs_review: list[str] = []  # un-whitelisted -- these fail the run
    unrecognized_mentions: list[str] = []  # un-whitelisted -- these fail the run
    broken_continuation_lines: list[str] = []  # never whitelistable -- these fail the run
    checked = 0

    for script in sorted(scripts):
        rel_script = script.relative_to(REPO_ROOT)
        invocations, unrecognized_lines, broken_lines = extract_invocations(script)

        for line_no, explanation in broken_lines:
            broken_continuation_lines.append(f"{rel_script}:{line_no}: {explanation}")

        for line_no, raw in unrecognized_lines:
            wl_reason = LINE_WHITELIST.get((str(rel_script), _invocation_fingerprint(raw)))
            msg = (
                f"{rel_script}:{line_no}: mentions 'python'/'python3' but was not recognized as an "
                f"invocation start (would otherwise be silently skipped): {raw}"
            )
            if wl_reason:
                whitelisted_detail.append(f"{msg}\n      WHITELISTED: {wl_reason}")
            else:
                unrecognized_mentions.append(msg)

        for inv in invocations:
            m = MODULE_INVOCATION_RE.search(inv.text)
            py_file: Path | None = None
            note = ""
            if m:
                dotted = m.group(1)
                py_file = resolve_module_path(dotted)
                if py_file is None:
                    note = f"could not resolve module {dotted} to a source file"
            else:
                py_tok = None
                for tok in PY_FILE_TOKEN_RE.finditer(inv.text):
                    candidate = tok.group(1)
                    if "check_script_args.py" in candidate:
                        continue
                    py_tok = candidate
                    break
                if py_tok is not None:
                    py_file, note = resolve_py_file_by_basename(py_tok, tracked_files)

            if py_file is None:
                wl_reason = LINE_WHITELIST.get((str(rel_script), _invocation_fingerprint(inv.text)))
                line = f"{rel_script}:{inv.line_no}: cannot resolve entry point ({note or 'no -m/.py token found'})"
                if wl_reason:
                    whitelisted_detail.append(f"{line}\n      WHITELISTED: {wl_reason}")
                else:
                    needs_review.append(line)
                continue

            passed_flags = extract_flags(inv.text)
            rel_py = py_file.relative_to(REPO_ROOT)
            resolution = resolve_flags(py_file, inv.text)

            if resolution.status == "checked":
                checked += 1
                missing = _unaccepted_flags(passed_flags, resolution.recognized or set())
                if not missing:
                    checked_detail.append(f"{rel_script}:{inv.line_no}: -> {rel_py} {resolution.note}")
                    continue
                # The `--help` surface lists canonical spellings only, so ask
                # the real parser. See `probe_documented_arguments`.
                verdict, offenders, evidence = adjudicate_missing_flags(py_file, inv.text, missing)
                if verdict == "accepted":
                    checked_detail.append(
                        f"{rel_script}:{inv.line_no}: -> {rel_py} {resolution.note} "
                        f"[{', '.join(sorted(missing))} not in the --help surface but accepted by the "
                        f"live parser: {evidence}]"
                    )
                    continue
                if verdict == "rejected":
                    hits.append(
                        f"{rel_script}:{inv.line_no}: -> {rel_py} does not recognize: "
                        f"{', '.join(sorted(offenders))} ({evidence})"
                    )
                    continue
                detail = (
                    f"{rel_script}:{inv.line_no}: -> {rel_py}: not in the --help surface and the live "
                    f"parser could not adjudicate ({evidence}) [unverified flags: "
                    f"{', '.join(sorted(missing))}]"
                )
                wl_reason = LINE_WHITELIST.get((str(rel_script), _invocation_fingerprint(inv.text)))
                unexcused: set[str] = set()
                if not wl_reason:
                    wl_reason, unexcused = whitelist_unexcused_flags(str(rel_py), missing)
                if wl_reason and not unexcused:
                    whitelisted_detail.append(f"{detail}\n      WHITELISTED: {wl_reason}")
                elif wl_reason:
                    needs_review.append(
                        f"{detail}\n      NOT EXCUSED by {rel_py}'s WHITELIST entry, which declares "
                        f"only the flags documented when it was written: {', '.join(sorted(unexcused))}"
                    )
                else:
                    needs_review.append(detail)
                continue

            # needs_review. If this came from the hybrid-CLI static-argparse
            # fallback, partial_recognized is a complete accounting of the local
            # pre-parser: a missing *non-dotted* flag there is a HIT independent
            # of whitelist status (the whitelist covers "cannot verify", never
            # "verified wrong"), but only if the runtime probe agrees, since the
            # pre-parser may not be the layer that owns the flag. Dotted flags are
            # presumed-forwarded tyro overrides. For anything else
            # (truncated/failed live --help with no trusted fallback),
            # partial_recognized is incomplete, so nothing missing is promoted to
            # a HIT on the strength of the surface alone.
            partial = resolution.partial_recognized or set()
            missing = _unaccepted_flags(passed_flags, partial)
            probe_verdict, probe_offenders, probe_evidence = (
                adjudicate_missing_flags(py_file, inv.text, missing) if missing else ("accepted", set(), "")
            )
            if probe_verdict == "accepted":
                definite_bad = set()
                unverifiable = set()
                if missing:
                    checked += 1
                    checked_detail.append(
                        f"{rel_script}:{inv.line_no}: -> {rel_py} {resolution.note} "
                        f"[{', '.join(sorted(missing))} accepted by the live parser: {probe_evidence}]"
                    )
            elif probe_verdict == "rejected":
                definite_bad = probe_offenders
                unverifiable = missing - probe_offenders
            elif resolution.hybrid_fallback:
                definite_bad = {f for f in missing if "." not in f}
                unverifiable = {f for f in missing if "." in f}
            else:
                definite_bad = set()
                unverifiable = missing

            if definite_bad:
                hits.append(
                    f"{rel_script}:{inv.line_no}: -> {rel_py} does not recognize: {', '.join(sorted(definite_bad))}"
                )

            if unverifiable:
                # Per-invocation whitelist first (a specific doc line's
                # quirk, e.g. a placeholder subcommand selector);
                # per-entry-point whitelist second (this file can't be
                # verified anywhere at all).
                detail = (
                    f"{rel_script}:{inv.line_no}: -> {rel_py}: {resolution.note} "
                    f"[unverified flags: {', '.join(sorted(unverifiable))}]"
                )
                wl_reason = LINE_WHITELIST.get((str(rel_script), _invocation_fingerprint(inv.text)))
                unexcused: set[str] = set()
                if not wl_reason:
                    wl_reason, unexcused = whitelist_unexcused_flags(str(rel_py), unverifiable)
                if wl_reason and not unexcused:
                    whitelisted_detail.append(f"{detail}\n      WHITELISTED: {wl_reason}")
                elif wl_reason:
                    needs_review.append(
                        f"{detail}\n      NOT EXCUSED by {rel_py}'s WHITELIST entry, which declares "
                        f"only the flags documented when it was written: {', '.join(sorted(unexcused))}"
                    )
                else:
                    needs_review.append(detail)

    print(
        f"# checked {checked} python invocation(s) across {len(scripts)} tracked command source(s) "
        f"(*.md)"
    )
    print(
        f"# {len(needs_review)} needing review (unresolved/uncheckable, not whitelisted), "
        f"{len(unrecognized_mentions)} unrecognized 'python' mention(s) (not whitelisted), "
        f"{len(broken_continuation_lines)} broken line-continuation(s), "
        f"{len(whitelisted_detail)} whitelisted, {len(hits)} hit(s)"
    )

    if args.verbose and checked_detail:
        print(f"\n# CHECKED detail ({len(checked_detail)}):")
        for line in checked_detail:
            print(f"  {line}")

    if args.verbose and whitelisted_detail:
        print(f"\n# WHITELISTED ({len(whitelisted_detail)}) -- explicitly excused, see WHITELIST/LINE_WHITELIST:")
        for line in whitelisted_detail:
            print(f"  {line}")

    if needs_review:
        print(
            f"\n# NEEDS REVIEW ({len(needs_review)}) -- entry point could not be verified and is NOT "
            f"whitelisted; this fails the run:"
        )
        for line in needs_review:
            print(f"  {line}")

    if unrecognized_mentions:
        print(
            f"\n# UNRECOGNIZED 'python' MENTION ({len(unrecognized_mentions)}) -- looks like it might be an "
            f"invocation this tool doesn't understand yet, and is NOT whitelisted; this fails the run:"
        )
        for line in unrecognized_mentions:
            print(f"  {line}")

    if broken_continuation_lines:
        print(
            f"\n# BROKEN LINE CONTINUATION ({len(broken_continuation_lines)}) -- the documented command "
            f"does not run as written; it is never whitelistable, and its flags were NOT checked:"
        )
        for line in broken_continuation_lines:
            print(f"  {line}")

    if hits:
        print(f"\n# HITS ({len(hits)}) -- script passes a flag its parser does not recognize:")
        for line in hits:
            print(f"  {line}")

    if hits or needs_review or unrecognized_mentions or broken_continuation_lines:
        return 1

    print("\n# 0 hits, 0 unwhitelisted needs-review entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
