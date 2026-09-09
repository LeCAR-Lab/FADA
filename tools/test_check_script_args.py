"""Unit tests for tools/check_script_args.py's parsing and adjudication logic.

Mostly pure parsing unit tests -- no network, no GPU/sim environment required.
Two groups are not pure:

* ``TestScannerAgreesWithBash`` runs real ``bash`` against a stub interpreter
  that prints its own argv, and compares that with the scanner's tokens.
* ``TestRuntimeAcceptanceProbe*`` run real entry points through the acceptance
  probe (skipped if none of this repo's conda envs is present).

Coverage: help-transcript parsing, unreachable-branch AST handling, invocation
discovery, line-continuation scanning in both directions, the runtime acceptance
probe, and whitelist fingerprinting.

Run with:

    pytest tools/test_check_script_args.py -v

(needs `holosoma` importable for the repo root conftest.py, e.g.
`PYTHONPATH=src/holosoma:src/holosoma_inference`, same as the rest of this
repo's test suite.)
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib.util
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_TOOL_PATH = Path(__file__).resolve().parent / "check_script_args.py"
_spec = importlib.util.spec_from_file_location("check_script_args", _TOOL_PATH)
csa = importlib.util.module_from_spec(_spec)
sys.modules["check_script_args"] = csa  # dataclass field resolution needs this registered
_spec.loader.exec_module(csa)


# ---------------------------------------------------------------------------
# --help transcript parsing must not treat a wrapped description continuation as
# a new option definition, must survive ANSI codes, must still capture a wrapped
# alias continuation, and must handle box-drawing variants.
# ---------------------------------------------------------------------------


class TestParseHelpFlags:
    def test_real_option_is_captured(self):
        text = (
            "╭─ options ──────────────────────────────────────────╮\n"
            "│ --robot.robot-type STR                              │\n"
            "│       Robot identifier. (default: t1_23dof)         │\n"
            "╰──────────────────────────────────────────────────────╯\n"
        )
        assert csa._parse_help_flags(text) == {"--robot.robot-type"}

    def test_description_continuation_starting_with_flag_is_not_a_flag(self):
        """A wrapped description line beginning with a flag-shaped token
        ('--ghost') is not parsed as an option; only the genuine option line
        ('--robot.robot-type') is."""
        text = (
            "╭─ options ──────────────────────────────────────────────────╮\n"
            "│ --robot.robot-type STR                                      │\n"
            "│       Robot identifier. See                                 │\n"
            "│       --ghost for legacy override behavior, deprecated.     │\n"
            "╰───────────────────────────────────────────────────────────────╯\n"
        )
        flags = csa._parse_help_flags(text)
        assert "--robot.robot-type" in flags
        assert "--ghost" not in flags

    def test_ansi_colored_output_still_parses(self):
        """ANSI color/style escape codes around option text do not break flag
        extraction or the indentation measurement that separates option lines
        from continuations."""
        esc = "\x1b[1m"
        reset = "\x1b[0m"
        text = (
            f"│ {esc}--robot.robot-type{reset} STR                          │\n"
            f"│       {esc}Robot identifier.{reset}                         │\n"
        )
        assert csa._parse_help_flags(text) == {"--robot.robot-type"}

    def test_wrapped_alias_continuation_is_captured(self):
        """An alias list wrapping onto a deeper-indented line that holds only
        flags/metavars and no prose continues the same option's alias set, and
        its flags are extracted."""
        text = (
            "  -o ARG, --output ARG,\n"
            "      --out ARG           write results here\n"
            "  -h, --help              show this help message and exit\n"
        )
        flags = csa._parse_help_flags(text)
        assert {"--output", "--out", "--help"} <= flags

    def test_box_drawing_ascii_fallback_variant(self):
        """Transcripts rendering tyro's U+2502 box character as ASCII '|' parse
        the same as the U+2502 form."""
        text = (
            "| --robot.robot-type STR                                       |\n"
            "|       Robot identifier. (default: t1_23dof)                  |\n"
        )
        assert csa._parse_help_flags(text) == {"--robot.robot-type"}

    def test_plain_argparse_style_still_works(self):
        text = (
            "usage: eval_checkpoint.py [-h] --checkpoint CHECKPOINT\n"
            "\n"
            "options:\n"
            "  -h, --help            show this help message and exit\n"
            "  --checkpoint CHECKPOINT\n"
            "  --output-dir OUTPUT_DIR\n"
        )
        assert csa._parse_help_flags(text) == {"--help", "--checkpoint", "--output-dir"}

    def test_same_line_description_mentioning_another_flag_is_not_captured(self):
        """A flag name mentioned in another flag's single-line description is not
        added to the recognized set: only the invocation portion, before the
        2+-space gap to the help text, is scanned."""
        text = "  --foo FOO   deprecated, use --bar instead\n  --bar BAR\n"
        assert csa._parse_help_flags(text) == {"--foo", "--bar"}


# ---------------------------------------------------------------------------
# The static AST scan must not count add_argument calls inside a
# statically-unreachable `if False:`/`if 0:` branch.
# ---------------------------------------------------------------------------


class TestArgumentCollectorUnreachableBranches:
    def _collect(self, src: str) -> set[str]:
        tree = ast.parse(src)
        collector = csa._ArgumentCollector()
        collector.visit(tree)
        return collector.flags

    def test_if_false_branch_is_excluded(self):
        src = (
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--real')\n"
            "if False:\n"
            "    parser.add_argument('--ghost')\n"
        )
        flags = self._collect(src)
        assert "--real" in flags
        assert "--ghost" not in flags

    def test_if_zero_branch_is_excluded(self):
        src = (
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "if 0:\n"
            "    parser.add_argument('--ghost')\n"
            "else:\n"
            "    parser.add_argument('--real')\n"
        )
        flags = self._collect(src)
        assert flags == {"--real"}

    def test_if_true_branch_is_included_orelse_excluded(self):
        src = (
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "if True:\n"
            "    parser.add_argument('--real')\n"
            "else:\n"
            "    parser.add_argument('--ghost')\n"
        )
        flags = self._collect(src)
        assert flags == {"--real"}

    def test_non_constant_condition_is_still_collected(self):
        """Only a literal constant `if` test is treated as dead-code-eliminable;
        a flag behind any other condition is collected as live."""
        src = (
            "import argparse, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "if sys.platform == 'win32':\n"
            "    parser.add_argument('--windows-only')\n"
        )
        flags = self._collect(src)
        assert "--windows-only" in flags


# ---------------------------------------------------------------------------
# Invocation-start detection must recognize env-assignment / env / conda-run
# wrapped python invocations, and must report an unrecognized 'python' mention
# rather than drop it.
# ---------------------------------------------------------------------------


class TestPythonInvocationStartIndex:
    @pytest.mark.parametrize(
        "line",
        [
            "python foo.py --bar",
            "python3 foo.py --bar",
            "CUDA_VISIBLE_DEVICES=0 python foo.py --bar",
            "env CUDA_VISIBLE_DEVICES=0 python foo.py --bar",
            "PYTHONPATH=src/holosoma python -m holosoma.fada.planner_idm.train --bar",
            "FOO=1 BAR=2 python foo.py",
            "conda run -n hsmujoco python foo.py --bar",
            "conda run --no-capture-output -n hsmujoco python3 foo.py",
        ],
    )
    def test_recognized_forms(self, line):
        assert csa._python_invocation_start_index(line) is not None

    @pytest.mark.parametrize(
        "line",
        [
            "# python foo.py --bar",
            "",
            "this line just mentions python in prose",
            "PYTHONPATH=src/holosoma <python> -m pytest",  # placeholder, not literal python
        ],
    )
    def test_unrecognized_forms(self, line):
        assert csa._python_invocation_start_index(line) is None

    def test_trailing_line_continuation_backslash_does_not_crash(self):
        # shlex.split raises ValueError("No escaped character") on a bare
        # trailing backslash unless it is stripped first.
        assert csa._python_invocation_start_index("python foo.py \\") is not None


class TestExtractInvocationsUnrecognizedMentions:
    def test_env_prefixed_invocation_is_found_and_flags_extracted(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text(
            "```bash\n"
            "env CUDA_VISIBLE_DEVICES=0 python -m holosoma.fada.planner_idm.train "
            "--this-flag-does-not-exist-xyz\n"
            "```\n"
        )
        invocations, unrecognized, broken = csa.extract_invocations(doc)
        assert len(invocations) == 1
        assert "--this-flag-does-not-exist-xyz" in csa.extract_flags(invocations[0].text)
        assert unrecognized == []
        assert broken == []

    def test_unrecognized_python_mention_is_reported(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text(
            "```bash\n"
            "SOME_WRAPPER_THIS_TOOL_DOESNT_KNOW python-ish-thing foo.py\n"
            "```\n"
        )
        invocations, unrecognized, broken = csa.extract_invocations(doc)
        assert invocations == []
        assert broken == []
        # "python-ish-thing" contains "python" only as a substring of a longer
        # identifier, and PYTHON_WORD_RE is word-boundary-aware, so it does not
        # match.
        assert unrecognized == []

    def test_unresolvable_wrapper_form_is_flagged(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text(
            "```bash\n"
            "some-other-wrapper.sh -- python foo.py --bar\n"
            "```\n"
        )
        invocations, unrecognized, broken = csa.extract_invocations(doc)
        assert broken == []
        # `python` is not the first post-prefix token here (an opaque wrapper
        # script and '--' precede it), so the line is reported as unrecognized
        # rather than parsed or skipped.
        assert invocations == []
        assert len(unrecognized) == 1
        assert unrecognized[0][1] == "some-other-wrapper.sh -- python foo.py --bar"


# ---------------------------------------------------------------------------
# LINE_WHITELIST matching is content-fingerprinted, so an edited command no
# longer matches the entry written for its earlier text.
# ---------------------------------------------------------------------------


class TestInvocationFingerprint:
    def test_stable_across_incidental_whitespace(self):
        a = "python foo.py \\\n    --bar baz"
        b = "python foo.py \\\n      --bar baz  "
        assert csa._invocation_fingerprint(a) == csa._invocation_fingerprint(b)

    def test_changes_with_added_flag(self):
        """Appending a flag to a registered command changes its fingerprint, so
        the new flag is not covered by the LINE_WHITELIST entry written for the
        original command."""
        a = "python foo.py --known-bad-flag"
        b = "python foo.py --known-bad-flag --probe-definitely-invalid"
        assert csa._invocation_fingerprint(a) != csa._invocation_fingerprint(b)


# ---------------------------------------------------------------------------
# A --help transcript lists only each option's canonical (hyphenated) spelling.
# tyro rewrites `_` <-> `-` in option names before parsing and accepts the
# underscored form; argparse does not and exits 2. The transcript does not say
# which applies, so `_unaccepted_flags` matches exactly and callers resolve the
# respelt case with the runtime probe below.
# ---------------------------------------------------------------------------


class TestUnacceptedFlagsIsExact:
    RECOGNIZED = {"--input-dir", "--output-dir", "--task.model-path"}

    def test_exact_matches_are_accepted(self):
        assert csa._unaccepted_flags({"--input-dir", "--output-dir"}, self.RECOGNIZED) == set()

    def test_a_respelt_flag_is_not_silently_accepted_by_the_surface(self):
        """An underscored respelling of a recognized flag is returned as
        unaccepted; the transcript does not decide it."""
        assert csa._unaccepted_flags({"--input_dir"}, self.RECOGNIZED) == {"--input_dir"}

    def test_a_genuinely_unknown_flag_is_reported(self):
        assert csa._unaccepted_flags({"--input_dirx", "--nope"}, self.RECOGNIZED) == {
            "--input_dirx",
            "--nope",
        }


# ---------------------------------------------------------------------------
# There is no "reported but tolerated" outcome: every documented invocation is
# either verified or whitelisted as unverifiable, and every whitelist entry
# carries a reason.
# ---------------------------------------------------------------------------


class TestNoTolerateOnlyRegistry:
    def test_known_upstream_doc_bugs_registry_is_gone(self):
        assert not hasattr(csa, "KNOWN_UPSTREAM_DOC_BUGS")
        assert not hasattr(csa, "KnownUpstreamDocBug")

    def test_every_whitelist_entry_has_a_reason(self):
        for key, reason in csa.LINE_WHITELIST.items():
            assert reason.strip(), key
        for key, entry in csa.WHITELIST.items():
            assert entry.reason.strip(), key
            assert isinstance(entry.flags, frozenset), key




# ---------------------------------------------------------------------------
# Whether a documented block runs as written is a question about bash
# tokenization, so the tests below are paired around the two directions:
#
#   not a continuation, though the line ends in a backslash: `--flag v \` plus
#     trailing whitespace (bash ends the command there, passes a one-space
#     positional, and runs the remaining flag lines as commands, exit 127); `\#`
#     and `\` + U+00A0 behave the same.
#   a continuation despite containing a backslash: `--pattern '\ x'`, where the
#     backslash is inside quotes and is data, so the command runs.
#
# `TestScannerAgreesWithBash` compares the tokenizer against real bash.
# ---------------------------------------------------------------------------


NBSP = " "  # noqa: RUF001 -- the character under test; a plain space is a different case


def _md(lines):
    return "```bash\n" + "\n".join(lines) + "\n```\n"


class TestBrokenLineContinuationFalseNegatives:
    """Blocks bash does not run as written: reported as broken, and no invocation
    is collected from them."""

    @pytest.mark.parametrize(
        ("name", "second_line"),
        [
            ("trailing space after the backslash", "    --task.interface eno1 \\ "),
            ("trailing tab after the backslash", "    --task.interface eno1 \\\t"),
            ("escaped comment marker", "    --task.interface eno1 \\#your interface"),
            ("non-breaking space after the backslash", f"    --task.interface eno1 \\{NBSP}"),
            ("comment after the backslash", "    --task.interface eno1 \\   # your interface"),
        ],
    )
    def test_the_block_is_reported_and_not_collected(self, tmp_path, name, second_line):
        doc = tmp_path / "doc.md"
        doc.write_text(
            _md(
                [
                    "python3 run_policy.py inference:t1-23dof-loco-fada \\",
                    second_line,
                    "    --task.no-randomize-commands \\",
                    "    --task.max-steps 5000",
                ]
            )
        )

        invocations, unrecognized, broken = csa.extract_invocations(doc)

        assert invocations == [], f"{name}: collected a command that does not run"
        assert unrecognized == []
        assert broken, name

    def test_the_old_rstrip_rule_would_have_accepted_these(self):
        """`rstrip()` reads each of these lines as a continuation, while
        `scan_shell_command` ends the command at that line."""
        for line in ("--a 1 \\ ", "--a 1 \\\t", f"--a 1 \\{NBSP}"):
            assert line.rstrip().endswith("\\"), "premise: rstrip() reads this as a continuation"
            command = csa.scan_shell_command([line, "--b 2"], 0)
            assert command.end_index == 0, "bash does not continue it"

    def test_the_old_regex_would_not_have_flagged_these(self):
        """`\\\\[ \\t]+\\S` matches none of these lines."""
        old = re.compile(r"\\[ \t]+\S")
        for line in ("--a 1 \\ ", "--a 1 \\#note", f"--a 1 \\{NBSP}"):
            assert not old.search(line)


class TestBrokenLineContinuationFalsePositives:
    """Blocks bash runs as written: not reported as broken, and collected."""

    def test_a_quoted_backslash_space_is_data_not_a_continuation(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text(
            _md(
                [
                    "python foo.py \\",
                    "    --pattern '\\ x' \\",
                    "    --b 2",
                ]
            )
        )

        invocations, _unrecognized, broken = csa.extract_invocations(doc)

        assert broken == []
        assert len(invocations) == 1
        assert csa.extract_flags(invocations[0].text) == {"--pattern", "--b"}

    def test_an_escaped_space_inside_a_word_is_data_too(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text(_md(["python foo.py \\", "    --path /my\\ dir \\", "    --b 2"]))

        invocations, _unrecognized, broken = csa.extract_invocations(doc)

        assert broken == []
        assert len(invocations) == 1

    def test_the_old_regex_would_have_rejected_the_quoted_form(self):
        assert re.compile(r"\\[ \t]+\S").search("--pattern '\\ x' \\")


class TestScannerAgreesWithBash:
    """`scan_shell_command`'s tokens must equal the argv bash passes.

    A stub `python` on PATH prints its own argv one token per line, so the
    comparison is between what bash passed and what the scanner reports. Cases
    needing tokens that contain newlines are excluded, since the stub's
    line-oriented output cannot represent them.
    """

    CASES = {
        "trailing_ws": ["python foo.py \\", "    --a 1 \\ ", "    --b 2"],
        "escaped_hash": ["python foo.py \\", "    --a 1 \\#note", "    --b 2"],
        "nbsp": ["python foo.py \\", f"    --a 1 \\{NBSP}", "    --b 2"],
        "space_then_comment": ["python foo.py \\", "    --a 1 \\  # note", "    --b 2"],
        "quoted_backslash_space": ["python foo.py \\", "    --pattern '\\ x' \\", "    --b 2"],
        "escaped_path_space": ["python foo.py \\", "    --path /my\\ dir \\", "    --b 2"],
        "plain": ["python foo.py \\", "    --a 1 \\", "    --b 2"],
        "double_quoted_continuation": ['python foo.py --msg "a \\', 'b" --c 3'],
        "ansi_c_quoting": ["python foo.py --msg $'a\\tb' --c 3"],
        "double_quoted_hash": ['python foo.py --msg "not # a comment" --c 3'],
        "inline_comment": ["python foo.py --a 1  # trailing comment"],
    }

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_tokens_match_what_bash_passes(self, tmp_path, name):
        lines = self.CASES[name]
        stub = tmp_path / "python"
        stub.write_text('#!/bin/sh\nfor a in "$@"; do printf "<%s>\\n" "$a"; done\n')
        stub.chmod(0o755)
        script = tmp_path / "block.sh"
        script.write_text("\n".join(lines) + "\n")

        proc = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
        )
        bash_argv = [ln[1:-1] for ln in proc.stdout.splitlines() if ln.startswith("<") and ln.endswith(">")]

        scanned = csa.scan_shell_command(lines, 0)
        ours = scanned.tokens[1:]
        assert "\n" not in "".join(ours), "this case must not need multi-line tokens"
        assert ours == bash_argv, f"{name}: scanner {ours!r} vs bash {bash_argv!r}"


class TestBrokenLineContinuationOnTheRealDocs:
    def test_no_tracked_command_source_carries_a_broken_continuation(self):
        """No tracked command source carries a broken continuation. Needs no conda env."""
        offenders: list[str] = []
        for script in sorted(csa.command_sources()):
            _invocations, _unrecognized, broken = csa.extract_invocations(script)
            offenders.extend(
                f"{script.relative_to(csa.REPO_ROOT)}:{line_no}: {text}" for line_no, text in broken
            )
        assert offenders == [], offenders

    def test_the_repaired_block_is_collected_whole(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text(
            _md(
                [
                    "python3 run_policy.py inference:t1-23dof-loco-fada \\",
                    "    --task.model-path <onnx> \\",
                    "    --task.interface eno1 \\",
                    "    --task.no-randomize-commands \\",
                    "    --task.onnx-provider cuda \\",
                    "    --task.collect-data \\",
                    "    --task.max-steps 5000",
                ]
            )
        )

        invocations, _unrecognized, broken = csa.extract_invocations(doc)

        assert broken == []
        assert len(invocations) == 1
        assert csa.extract_flags(invocations[0].text) == {
            "--task.model-path",
            "--task.interface",
            "--task.no-randomize-commands",
            "--task.onnx-provider",
            "--task.collect-data",
            "--task.max-steps",
        }


# ---------------------------------------------------------------------------
# Whether a program accepts a documented flag is decided by running it, not by
# walking its source for `ArgumentParser(...)` calls.
#
# The four constructions below mention tyro but are adjudicated by an argparse
# parser that rejects `--foo_bar` with exit 2. Each test first runs the file to
# establish the outcome, then asks the probe, and fails on any disagreement in
# either direction.
#
# Refactorings that do not change observable behaviour -- e.g. writing
# run_policy.py's `add_help=False` as a named constant -- must not change the
# probe's verdict either.
# ---------------------------------------------------------------------------


_PROBE_ENVS = csa._candidate_python_bins()
requires_env = pytest.mark.skipif(
    not _PROBE_ENVS, reason="the runtime probe needs one of this repo's conda envs"
)


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return path


def _real_exit_code(tmp_path, entry, argv):
    """Run `entry` with `argv` under a probe env and return (returncode, stdout+stderr)."""
    python_bin = _PROBE_ENVS[0][0]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), *(str(root) for root in csa.PACKAGE_SRC_ROOTS.values())]
    )
    proc = subprocess.run(
        [str(python_bin), str(entry), *argv],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=csa.LIVE_HELP_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout + proc.stderr


#: Each value is (extra_module_name, extra_module_source, entry_source), with
#: extra_module_name/source None when no helper module is needed. Every entry
#: point mentions tyro, which is what routes it down the live path, but is
#: parsed by a strict argparse parser.
_ARGPARSE_IN_DISGUISE = {
    "imported factory": (
        "parser_factory.py",
        "import argparse\n"
        "\n"
        "def make_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--foo-bar')\n"
        "    return parser\n",
        "import tyro  # noqa: F401\n"
        "from parser_factory import make_parser\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    make_parser().parse_args()\n",
    ),
    "aliased import": (
        None,
        None,
        "from argparse import ArgumentParser as AP\n"
        "import tyro  # noqa: F401\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    parser = AP()\n"
        "    parser.add_argument('--foo-bar')\n"
        "    parser.parse_args()\n",
    ),
    "ArgumentParser subclass": (
        None,
        None,
        "import argparse\n"
        "import tyro  # noqa: F401\n"
        "\n"
        "class Parser(argparse.ArgumentParser):\n"
        "    pass\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    parser = Parser()\n"
        "    parser.add_argument('--foo-bar')\n"
        "    parser.parse_args()\n",
    ),
    "add_help via a named constant": (
        None,
        None,
        "import argparse\n"
        "import tyro  # noqa: F401\n"
        "\n"
        "NO_HELP = False\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    parser = argparse.ArgumentParser(add_help=NO_HELP)\n"
        "    parser.add_argument('--foo-bar')\n"
        "    parser.parse_args()\n",
    ),
}


@requires_env
class TestRuntimeAcceptanceProbeRejects:
    @pytest.mark.parametrize("name", sorted(_ARGPARSE_IN_DISGUISE))
    def test_a_parser_the_ast_cannot_see_is_still_asked(self, tmp_path, name):
        helper_name, helper_source, entry_source = _ARGPARSE_IN_DISGUISE[name]
        if helper_name:
            _write(tmp_path, helper_name, helper_source)
        entry = _write(tmp_path, "entry.py", entry_source)

        rc, output = _real_exit_code(tmp_path, entry, ["--foo_bar", "1"])
        assert rc == 2 and "unrecognized" in output.lower(), f"{name}: premise -- {rc} {output!r}"

        verdict, how, stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
        assert verdict == "rejected", f"{name}: {how} {stderr!r}"

        decision, offenders, _evidence = csa.adjudicate_missing_flags(
            entry, f"python {entry} --foo_bar 1", {"--foo_bar"}
        )
        assert decision == "rejected"
        assert offenders == {"--foo_bar"}

    @pytest.mark.parametrize("name", sorted(_ARGPARSE_IN_DISGUISE))
    def test_the_hyphenated_spelling_those_same_files_document_is_accepted(self, tmp_path, name):
        """The canonical hyphenated spelling of the same flag probes as accepted."""
        helper_name, helper_source, entry_source = _ARGPARSE_IN_DISGUISE[name]
        if helper_name:
            _write(tmp_path, helper_name, helper_source)
        entry = _write(tmp_path, "entry.py", entry_source)

        verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo-bar", "1"], {"--foo-bar"})
        assert verdict == "accepted", f"{name}: {how}"


_PURE_TYRO = (
    "import dataclasses\n"
    "import tyro\n"
    "\n"
    "@dataclasses.dataclass\n"
    "class Config:\n"
    "    foo_bar: str = 'x'\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    cfg = tyro.cli(Config)\n"
    "    raise SystemExit('the probe must stop before this line')\n"
)

#: run_policy.py's shape: an `add_help=False` pre-parser forwarding the remaining
#: argv to tyro. `{add_help}` is formatted with either `False` or a named constant.
_FORWARDING_PREPARSER = (
    "import argparse\n"
    "import dataclasses\n"
    "import sys\n"
    "import tyro\n"
    "\n"
    "@dataclasses.dataclass\n"
    "class Config:\n"
    "    foo_bar: str = 'x'\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    pre = argparse.ArgumentParser(add_help={add_help}, allow_abbrev=False)\n"
    "    pre.add_argument('--secondary', default=None)\n"
    "    known, remaining = pre.parse_known_args()\n"
    "    cfg = tyro.cli(Config, args=remaining)\n"
    "    raise SystemExit('the probe must stop before this line')\n"
)


@requires_env
class TestRuntimeAcceptanceProbeAccepts:
    def test_a_tyro_entry_point_accepts_the_underscored_spelling(self, tmp_path):
        """tyro rewrites `_` to `-`, so a pure-tyro entry point accepts `--foo_bar`."""
        entry = _write(tmp_path, "pure.py", _PURE_TYRO)

        rc, output = _real_exit_code(tmp_path, entry, ["--foo_bar", "1"])
        assert "the probe must stop before this line" in output, f"premise: {rc} {output!r}"

        verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
        assert verdict == "accepted", how

    def test_the_probe_stops_before_the_program_does_anything(self, tmp_path):
        """The probe stops at the parse: the statement after `tyro.cli` does not
        run, so the probe can be pointed at entry points that would otherwise
        launch a simulator or a policy.
        """
        entry = _write(tmp_path, "pure.py", _PURE_TYRO)
        _verdict, _how, stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
        assert "the probe must stop before this line" not in stderr

    @pytest.mark.parametrize("add_help", ["False", "NO_HELP"])
    def test_spelling_add_help_as_a_constant_changes_nothing(self, tmp_path, add_help):
        """Writing `add_help=False` as a named constant yields the same verdict as
        the literal, since the probe runs the program rather than reading its
        source.
        """
        prefix = "" if add_help == "False" else "NO_HELP = False\n"
        entry = _write(
            tmp_path, f"fwd_{add_help}.py", prefix + _FORWARDING_PREPARSER.format(add_help=add_help)
        )

        verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
        assert verdict == "accepted", how


@requires_env
class TestRuntimeAcceptanceProbeFailsClosed:
    def test_an_entry_point_that_never_parses_is_undecided_not_accepted(self, tmp_path):
        entry = _write(tmp_path, "broken.py", "import no_such_module_xyz  # noqa: F401\n")

        verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo_bar"], {"--foo_bar"})
        assert verdict == "error", how

        decision, offenders, _evidence = csa.adjudicate_missing_flags(
            entry, f"python {entry} --foo_bar", {"--foo_bar"}
        )
        assert decision == "undecided"
        assert offenders == set()

    def test_a_parse_error_that_is_not_about_the_flag_is_undecided(self, tmp_path):
        """A `<placeholder>` value that fails type validation yields `error`, not a
        verdict on the flag."""
        entry = _write(
            tmp_path,
            "typed.py",
            "import dataclasses\n"
            "import tyro\n"
            "\n"
            "@dataclasses.dataclass\n"
            "class Config:\n"
            "    count: int = 1\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    tyro.cli(Config)\n",
        )

        verdict, _how, _stderr = csa.probe_documented_arguments(entry, ["--count", "<n>"], {"--count"})
        assert verdict == "error"


class TestNoParserProvenanceClassifierRemains:
    def test_the_source_walk_is_gone(self):
        """The parser-provenance helpers and the `delimiter_insensitive` field on
        `Resolution` are absent from the module."""
        assert not hasattr(csa, "_argparse_serves_help")
        assert not hasattr(csa, "_delimiter_variants")
        assert not hasattr(csa, "_flag_accepted")
        assert "delimiter_insensitive" not in {f.name for f in dataclasses.fields(csa.Resolution)}

    def test_the_collector_no_longer_records_add_help(self):
        collector = csa._ArgumentCollector()
        assert not hasattr(collector, "argument_parsers_serving_help")


# ---------------------------------------------------------------------------
# Containment. Both live paths execute shipped entry points, so the tests below
# pin three closures:
#
#   * the child's cwd is not the repository, so a relative-path write at import
#     time does not land in REPO_ROOT;
#   * bytecode writing is disabled, so a run leaves no `.pyc` files under the
#     shipped package trees;
#   * the whole process group is killed on timeout, so a grandchild holding the
#     captured pipes cannot outlive the driver.
#
# Not asserted, because not contained: writes to absolute paths, network
# traffic, and a child that re-parents itself out of the killed process group.
# See the containment note in check_script_args.py.
# ---------------------------------------------------------------------------


_SIDE_EFFECT_ENTRY = (
    "import dataclasses\n"
    "import pathlib\n"
    "import tyro\n"
    "\n"
    "pathlib.Path('PROBE_SIDE_EFFECT.txt').write_text('written at import time')\n"
    "\n"
    "@dataclasses.dataclass\n"
    "class Config:\n"
    "    foo_bar: str = 'x'\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    tyro.cli(Config)\n"
)


@requires_env
class TestTheProbeIsContained:
    def test_a_relative_write_does_not_land_in_the_repository(self, tmp_path):
        """The child's cwd is not REPO_ROOT, so a relative-path write performed at
        import time lands outside the tree being checked. `probe_documented_arguments`
        imports the entry point and so cannot prevent the write itself.
        """
        entry = _write(tmp_path, "writer.py", _SIDE_EFFECT_ENTRY)
        marker = csa.REPO_ROOT / "PROBE_SIDE_EFFECT.txt"
        assert not marker.exists(), "premise: the marker is not already there"
        csa._PROBE_CACHE.clear()
        try:
            verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
            assert verdict == "accepted", how
            assert not marker.exists(), "the probe wrote into the repository being checked"
        finally:
            if marker.exists():
                marker.unlink()

    def test_the_probe_writes_no_bytecode(self, tmp_path):
        """The probe leaves no `.pyc` files under a package it imports."""
        package = tmp_path / "probe_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "helper.py").write_text("VALUE = 1\n")
        entry = _write(
            tmp_path,
            "bcode.py",
            "import sys\n"
            f"sys.path.insert(0, {str(tmp_path)!r})\n"
            "import dataclasses\n"
            "import tyro\n"
            "from probe_pkg import helper  # noqa: F401\n"
            "\n"
            "@dataclasses.dataclass\n"
            "class Config:\n"
            "    foo_bar: str = 'x'\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    tyro.cli(Config)\n",
        )
        csa._PROBE_CACHE.clear()
        verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
        assert verdict == "accepted", how
        assert list(package.rglob("*.pyc")) == [], "the probe cached bytecode into an imported package"

    def test_a_descendant_spawned_before_the_parse_does_not_outlive_the_run(self, tmp_path):
        """The process group is killed, not just the child, so a grandchild holding
        the inherited pipes neither delays the probe past its timeout nor survives
        it.
        """
        stamp = tmp_path / "orphan_ran"
        entry = _write(
            tmp_path,
            "spawner.py",
            "import dataclasses\n"
            "import subprocess\n"
            "import tyro\n"
            "\n"
            f"subprocess.Popen(['/bin/sh', '-c', 'sleep 120; touch {stamp}'])\n"
            "\n"
            "@dataclasses.dataclass\n"
            "class Config:\n"
            "    foo_bar: str = 'x'\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    tyro.cli(Config)\n",
        )
        csa._PROBE_CACHE.clear()
        started = time.monotonic()
        verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
        elapsed = time.monotonic() - started
        assert verdict == "accepted", how
        assert elapsed < 60, f"the run was held open by a descendant for {elapsed:.0f}s"
        survivors = subprocess.run(
            ["pgrep", "-f", f"sleep 120; touch {stamp}"], capture_output=True, text=True, check=False
        ).stdout.split()
        for pid in survivors:  # only pids matching this test's own command line
            with contextlib.suppress(ProcessLookupError, ValueError):
                os.kill(int(pid), signal.SIGKILL)
        assert survivors == [], f"descendants outlived the probe: {survivors}"


# ---------------------------------------------------------------------------
# The authoritative parse is the LAST one that consumes the argv: a file whose
# argparse pre-parser accepts `--foo` and whose second parser rejects it exits 2,
# and the probe must report `rejected`.
# ---------------------------------------------------------------------------

_TWO_PARSERS = (
    "import argparse\n"
    "import tyro  # noqa: F401\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    first = argparse.ArgumentParser(add_help=False, allow_abbrev=False)\n"
    "    first.add_argument('--foo')\n"
    "    known, remaining = first.parse_known_args()\n"
    "    second = argparse.ArgumentParser()\n"
    "    second.add_argument('--bar')\n"
    "    second.parse_args()\n"
)


@requires_env
class TestTheLastParseDecides:
    def test_a_second_parser_that_rejects_overrides_the_first_that_accepted(self, tmp_path):
        entry = _write(tmp_path, "two.py", _TWO_PARSERS)

        rc, output = _real_exit_code(tmp_path, entry, ["--foo", "1"])
        assert rc == 2 and "unrecognized" in output.lower(), f"premise: {rc} {output!r}"

        csa._PROBE_CACHE.clear()
        verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo", "1"], {"--foo"})
        assert verdict == "rejected", how

    def test_the_program_is_still_stopped_at_its_first_side_effect(self, tmp_path):
        """Continuing past a parse reaches only another parse: the first side
        effect after `tyro.cli` does not run.
        """
        stamp = tmp_path / "ran_after_the_parse"
        entry = _write(
            tmp_path,
            "after.py",
            "import dataclasses\n"
            "import pathlib\n"
            "import tyro\n"
            "\n"
            "@dataclasses.dataclass\n"
            "class Config:\n"
            "    foo_bar: str = 'x'\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    tyro.cli(Config)\n"
            f"    pathlib.Path({str(stamp)!r}).write_text('the simulator started')\n"
            "    raise SystemExit('the probe must stop before this line')\n",
        )
        csa._PROBE_CACHE.clear()
        verdict, how, stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
        assert verdict == "accepted", how
        assert not stamp.exists(), "the program ran past the parse and wrote a file"
        assert "the probe must stop before this line" not in stderr


# ---------------------------------------------------------------------------
# WHITELIST entries excuse the flags they declare, not every future one.
# ---------------------------------------------------------------------------


class TestTheWhitelistIsNarrow:
    def test_a_declared_flag_is_excused(self):
        path = "src/holosoma_retargeting/holosoma_retargeting/viser_player.py"
        reason, unexcused = csa.whitelist_unexcused_flags(path, {"--robot_urdf"})
        assert reason and unexcused == set()

    def test_a_flag_the_entry_never_declared_is_not_excused(self):
        """A flag absent from a whitelisted entry's declared set is returned as
        unexcused."""
        path = "src/holosoma_retargeting/holosoma_retargeting/viser_player.py"
        reason, unexcused = csa.whitelist_unexcused_flags(path, {"--robot_urdf", "--not-a-real-flag"})
        assert reason
        assert unexcused == {"--not-a-real-flag"}

    def test_a_path_with_no_entry_excuses_nothing(self):
        reason, unexcused = csa.whitelist_unexcused_flags("src/holosoma/holosoma/train_agent.py", {"--x"})
        assert reason is None and unexcused == {"--x"}

    def test_every_entry_declares_a_flag_set(self):
        for key, entry in csa.WHITELIST.items():
            assert isinstance(entry, csa.WhitelistEntry), key
            assert isinstance(entry.flags, frozenset), key


# ---------------------------------------------------------------------------
# bash lexes control operators without needing whitespace.
# ---------------------------------------------------------------------------


class TestOperatorsWithoutWhitespace:
    def test_a_pipe_glued_to_the_previous_word_is_its_own_token(self):
        command = csa.scan_shell_command(["python x.py|", "# comment"], 0)
        assert command.tokens == ["python", "x.py", "|"]

    def test_a_block_ending_on_a_dangling_operator_is_reported(self, tmp_path):
        lines = ["python x.py|", "# comment"]
        script = tmp_path / "dangling.sh"
        script.write_text("\n".join(lines) + "\n")
        ground_truth = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        assert ground_truth.returncode == 2, "premise: bash rejects this block"

        problems = csa.broken_shell_continuations(lines)
        assert problems, "the scanner reported clean a block bash refuses to parse"
        assert "control operator" in problems[0][1]

    def test_a_redirection_is_not_split_into_a_trailing_background_operator(self):
        """`2>&1` scans as one token, leaving no trailing `&` operator."""
        command = csa.scan_shell_command(["python x.py --a 1 2>&1"], 0)
        assert command.tokens[-1] == "2>&1"
        assert command.dangling_operator is None
        assert csa.broken_shell_continuations(["python x.py --a 1 2>&1"]) == []

    def test_a_real_pipeline_still_scans_as_one_command(self):
        lines = ["python x.py --a 1 |", "  tee log"]
        command = csa.scan_shell_command(lines, 0)
        assert command.tokens == ["python", "x.py", "--a", "1", "|", "tee", "log"]
        assert csa.broken_shell_continuations(lines) == []


# ---------------------------------------------------------------------------
# The effect fence, and what an early stop may claim.
#
# Two properties, tested separately below:
#
#   * The fence stops the probe at `os.chmod`, `os.chdir`, `os.link`/`os.symlink`,
#     `os.replace`, the `shutil` copy/move family and writable `mmap` (whose
#     stores go through the MMU and reach no Python-level call), leaving the
#     filesystem unchanged.
#
#   * When the fence stops the probe between a deferring first parse and the
#     parser that adjudicates the rest of the argv, the verdict is undecided
#     rather than `accepted`. `eval_agent.py:main` has that shape: `tyro.cli(...,
#     return_unknown_args=True)`, then a checkpoint load and `os.makedirs`, then a
#     second `tyro.cli` over the deferred tail. Undecided maps to `needs_review`,
#     which is whitelist-or-fail.
# ---------------------------------------------------------------------------


def _fence_escape_source(setup: str, effect: str) -> str:
    """An entry point that parses, performs `effect`, then parses again and rejects.

    `setup` runs before the first parse, i.e. before the fence is armed. The mmap
    case needs that: `os.open(..., O_RDWR)` is itself fenced, so opening the file
    after the parse would stop the probe before the mapping.
    """
    return (
        "import argparse\n"
        "import os\n"
        "import pathlib\n"
        "import shutil\n"
        "import mmap\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    target = pathlib.Path(os.environ['FENCE_TARGET'])\n"
        f"    {setup}\n"
        "    first = argparse.ArgumentParser(allow_abbrev=False)\n"
        "    first.add_argument('--foo_bar')\n"
        "    first.parse_args()\n"
        f"    {effect}\n"
        "    second = argparse.ArgumentParser(allow_abbrev=False)\n"
        "    second.parse_args()\n"
    )


#: name -> (pre-parse setup, post-parse effect). `name` is also the substring the
#: probe's `how` must contain, i.e. the door the fence stops at.
_FENCE_ESCAPES = {
    "os.chmod": ("pass", "os.chmod(target, 0o600)"),
    "os.link": ("pass", "os.link(target, target.with_suffix('.hardlink'))"),
    "os.symlink": ("pass", "os.symlink(target, target.with_suffix('.symlink'))"),
    "os.replace": ("pass", "os.replace(target, target.with_suffix('.moved'))"),
    # `chdir` leaves no filesystem trace; it relocates every relative path the rest
    # of the program touches, so the observable here is the probe's `how` trail.
    "os.chdir": ("pass", "os.chdir(str(target.parent))"),
    # shutil reaches the filesystem through `open`, which is fenced as well; these
    # entries pin the fence naming shutil directly.
    "shutil.copy": ("pass", "shutil.copy(str(target), str(target.with_suffix('.copy')))"),
    "shutil.move": ("pass", "shutil.move(str(target), str(target.with_suffix('.moved')))"),
    # An mmap store reaches no Python-level call, so the fence sits at map-creation
    # time. The fd is acquired before the fence arms, since `os.open` is fenced too.
    "mmap.mmap": (
        "fd = os.open(str(target), os.O_RDWR)",
        "m = mmap.mmap(fd, 0); m[0:1] = b'X'; m.flush()",
    ),
}


def _fence_scene(tmp_path):
    """A directory holding one 0o644 file, for comparing state before and after the
    probe. Returns (scene_dir, target_file)."""
    scene = tmp_path / "scene"
    scene.mkdir()
    target = scene / "target.txt"
    target.write_text("original\n")
    target.chmod(0o644)
    return scene, target


def _scene_state(scene):
    return sorted(
        (p.name, p.stat().st_mode & 0o777, p.read_bytes() if p.is_file() and not p.is_symlink() else b"")
        for p in scene.iterdir()
    )


@requires_env
class TestEffectFenceIsNotEscapable:
    @pytest.mark.parametrize("name", sorted(_FENCE_ESCAPES))
    def test_no_effect_survives_the_fence(self, tmp_path, name):
        """The probe leaves the scene directory byte- and mode-identical, and its
        `how` names the fenced call it stopped at.
        """
        setup, effect = _FENCE_ESCAPES[name]
        entry = _write(tmp_path, "entry.py", _fence_escape_source(setup, effect))
        scene, target = _fence_scene(tmp_path)
        before = _scene_state(scene)

        with _env_var("FENCE_TARGET", str(target)):
            _verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})

        assert _scene_state(scene) == before, (
            f"{name} escaped the effect fence: the probe changed the filesystem. "
            f"before={before} after={_scene_state(scene)}"
        )
        assert name in how, (
            f"the probe did not stop at {name}; it ran past it and its verdict came from "
            f"somewhere else: {how!r}"
        )


import contextlib as _contextlib  # noqa: E402


@_contextlib.contextmanager
def _env_var(key, value):
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


#: eval_agent.py:main's shape, reduced: a deferring first parse, a fenced side
#: effect, then the parser that adjudicates the rest of the argv.
_DEFERRING_THEN_EFFECT_THEN_REJECT = (
    "import argparse\n"
    "import os\n"
    "import sys\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)\n"
    "    pre.add_argument('--foo_bar')\n"
    "    known, remaining = pre.parse_known_args()\n"
    "    os.makedirs(os.path.join(os.getcwd(), 'run_dir'), exist_ok=True)\n"
    "    final = argparse.ArgumentParser(allow_abbrev=False)\n"
    "    final.parse_args(remaining)\n"
)


@requires_env
class TestEarlyStopIsNotAPass:
    def test_a_deferring_parse_stopped_at_a_side_effect_is_undecided(self, tmp_path):
        entry = _write(tmp_path, "entry.py", _DEFERRING_THEN_EFFECT_THEN_REJECT)

        rc, output = _real_exit_code(tmp_path, entry, ["--foo_bar", "1", "--not-a-flag", "2"])
        assert rc == 2 and "unrecognized" in output.lower(), f"premise: {rc} {output!r}"

        verdict, how, _stderr = csa.probe_documented_arguments(
            entry, ["--foo_bar", "1", "--not-a-flag", "2"], {"--foo_bar"}
        )
        assert verdict != "accepted", (
            "the probe passed an argv the real program exits 2 on: it stopped at the "
            f"os.makedirs between the two parses and kept the provisional verdict. {how}"
        )

        decision, _offenders, _evidence = csa.adjudicate_missing_flags(
            entry, f"python {entry} --foo_bar 1 --not-a-flag 2", {"--foo_bar"}
        )
        assert decision == "undecided", decision

    def test_a_terminal_parse_stopped_at_a_side_effect_is_still_accepted(self, tmp_path):
        """A `parse_args` that consumed the whole argv would have exited 2 on an
        unrecognized flag, so a side effect after it still yields `accepted`.
        """
        entry = _write(
            tmp_path,
            "entry.py",
            "import argparse\n"
            "import os\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    p = argparse.ArgumentParser(allow_abbrev=False)\n"
            "    p.add_argument('--foo_bar')\n"
            "    p.parse_args()\n"
            "    os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)\n"
            "    raise SystemExit('the probe must stop before this line')\n",
        )
        verdict, how, _stderr = csa.probe_documented_arguments(entry, ["--foo_bar", "1"], {"--foo_bar"})
        assert verdict == "accepted", how


class TestLafanWhitelistEntryDeclaresItsFlags:
    """The LAFAN whitelist entry must declare the flags it excuses.

    `data_utils/lafan1/` is gitignored, so it is present in a working copy (where
    live `--help` succeeds and the entry is inert) and absent from a published
    clone. On a checkout without it, only the entry's declared flags excuse
    `--input_dir` / `--output_dir`.
    """

    ENTRY = "src/holosoma_retargeting/holosoma_retargeting/data_utils/extract_global_positions.py"

    def test_the_documented_flags_are_declared(self):
        entry = csa.WHITELIST[self.ENTRY]
        assert entry.flags >= {"--input_dir", "--output_dir"}, (
            "the LAFAN whitelist entry excuses no flags, so on a checkout without the "
            f"gitignored lafan1/ clone -- i.e. every published one -- {sorted(entry.flags)} "
            "leaves the documented --input_dir/--output_dir unexcused and the gate fails."
        )

    def test_the_declared_flags_are_the_scripts_real_fields(self):
        """Every declared flag corresponds to a field on the script's own tyro config
        dataclass, the same source `--help` is generated from.
        """
        source = (csa.REPO_ROOT / self.ENTRY).read_text()
        tree = ast.parse(source)
        fields = {
            target.id if isinstance(target := node.target, ast.Name) else ""
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for node in cls.body
            if isinstance(node, ast.AnnAssign)
        }
        entry = csa.WHITELIST[self.ENTRY]
        for flag in entry.flags:
            name = flag.lstrip("-").replace("-", "_")
            assert name in fields, f"{flag} is not a field of the script's config ({sorted(fields)})"
