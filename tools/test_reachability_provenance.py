"""Tests for `tools/bitexact/reachability.py`'s upstream/added partition.

Covers three contracts: a baseline that cannot be listed raises
`BaselineListingError` rather than yielding an empty upstream set, membership in
that set agrees with `git cat-file -e <baseline>:<path>`, and `main()` still
produces both partitions on a readable baseline.

Run with: pytest tools/test_reachability_provenance.py -v
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("reachability", REPO_ROOT / "tools" / "bitexact" / "reachability.py")
reach = importlib.util.module_from_spec(_spec)
sys.modules["reachability"] = reach
_spec.loader.exec_module(reach)


def _baseline_ref() -> str:
    ref = reach.upstream_baseline_ref()
    if ref is None:  # pragma: no cover - only in a checkout with no resolvable baseline
        pytest.skip("no upstream baseline ref resolves in this checkout")
    return ref


class TestTheBaselineListingFailsClosed:
    def test_an_unreadable_baseline_raises_instead_of_returning_an_empty_set(self) -> None:
        """A failed git call raises instead of classifying every path as not upstream."""
        with pytest.raises(reach.BaselineListingError):
            reach.upstream_paths("no-such-ref-this-repo-will-never-have")

    def test_the_error_carries_gits_own_message(self) -> None:
        with pytest.raises(reach.BaselineListingError) as excinfo:
            reach.upstream_paths("no-such-ref-this-repo-will-never-have")
        assert "git ls-tree exit" in str(excinfo.value)

    def test_main_exits_non_zero_when_the_baseline_cannot_be_listed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(reach, "upstream_baseline_ref", lambda: "no-such-ref-this-repo-will-never-have")
        assert reach.main() == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert "KEEP" not in captured.out and "REVIEW" not in captured.out


class TestMembershipIsExact:
    def test_a_real_upstream_path_is_upstream(self) -> None:
        paths = reach.upstream_paths(_baseline_ref())
        assert "src/holosoma/holosoma/train_agent.py" in paths
        assert reach.is_upstream("src/holosoma/holosoma/train_agent.py", paths)

    def test_a_path_added_by_this_project_is_not(self) -> None:
        paths = reach.upstream_paths(_baseline_ref())
        assert not reach.is_upstream("src/holosoma/holosoma/fada/planner_idm/train.py", paths)

    def test_the_listing_agrees_with_git_for_a_sample_of_paths(self) -> None:
        """`is_upstream` agrees with `git cat-file -e <ref>:<path>` on present and absent paths."""
        ref = _baseline_ref()
        paths = reach.upstream_paths(ref)
        sample = ["LICENSE", "src/holosoma/holosoma/train_agent.py", "no/such/path.py"]
        for path in sample:
            expected = subprocess.run(
                ["git", "cat-file", "-e", f"{ref}:{path}"], cwd=REPO_ROOT, capture_output=True, check=False
            ).returncode == 0
            assert reach.is_upstream(path, paths) is expected, path


class TestTheReportStillComputes:
    def test_both_partitions_are_produced(self, capsys: pytest.CaptureFixture[str]) -> None:
        """`main()` prints both a KEEP and a REVIEW partition and exits 0."""
        assert reach.main() == 0
        captured = capsys.readouterr()
        assert "KEEP" in captured.out, captured.out
        assert "REVIEW" in captured.out, captured.out
