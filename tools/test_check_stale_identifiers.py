"""Tests for `tools/check_stale_identifiers.py`'s handling of files it cannot read.

Contract: `_is_probably_binary` raises `UnscannableFileError` on an unreadable
file rather than classifying it as binary, and `main()` reports such files and
exits 1 instead of printing a clean result.

Run with: pytest tools/test_check_stale_identifiers.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL_PATH = Path(__file__).resolve().parent / "check_stale_identifiers.py"
_spec = importlib.util.spec_from_file_location("check_stale_identifiers", _TOOL_PATH)
csi = importlib.util.module_from_spec(_spec)
sys.modules["check_stale_identifiers"] = csi
_spec.loader.exec_module(csi)


def _unreadable(path: Path) -> None:
    path.chmod(0o000)


@pytest.fixture
def unreadable_py(tmp_path: Path) -> Path:
    path = tmp_path / "sealed.py"
    path.write_text("x = 1\n")
    _unreadable(path)
    try:
        with path.open("rb"):
            pytest.skip("this filesystem/user can read a 000-mode file (running as root?)")
    except OSError:
        pass
    yield path
    path.chmod(0o644)


class TestUnreadableIsNotBinary:
    def test_an_unreadable_file_raises_rather_than_reporting_binary(self, unreadable_py: Path) -> None:
        with pytest.raises(csi.UnscannableFileError):
            csi._is_probably_binary(unreadable_py)

    def test_a_real_binary_is_still_binary(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(b"\x00\x01\x02")
        assert csi._is_probably_binary(path) is True

    def test_ordinary_text_is_not_binary(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.py"
        path.write_text("x = 1\n")
        assert csi._is_probably_binary(path) is False


def _run_over(monkeypatch: pytest.MonkeyPatch, root: Path, tracked: list[str]) -> int:
    monkeypatch.setattr(csi, "REPO_ROOT", root)
    monkeypatch.setattr(csi, "_tracked_files", lambda: tracked)
    monkeypatch.setattr(csi, "SKIPPED_PATHS", frozenset())
    monkeypatch.setattr(csi, "SELF_PATH", "tools/check_stale_identifiers.py")
    return csi.main()


class TestTheGateFailsOnAHoleInItsOwnCoverage:
    def test_an_unreadable_tracked_file_fails_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readable = tmp_path / "fine.py"
        readable.write_text("x = 1\n")
        sealed = tmp_path / "sealed.py"
        sealed.write_text("x = 1\n")
        _unreadable(sealed)
        try:
            with sealed.open("rb"):
                pytest.skip("this filesystem/user can read a 000-mode file (running as root?)")
        except OSError:
            pass
        try:
            rc = _run_over(monkeypatch, tmp_path, ["fine.py", "sealed.py"])
            captured = capsys.readouterr()
        finally:
            sealed.chmod(0o644)

        assert rc == 1, captured.out + captured.err
        assert "could not be scanned" in captured.err
        assert "sealed.py" in captured.err
        assert "OK: 0 stale-identifier mentions" not in captured.out

    def test_a_clean_tree_still_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "fine.py").write_text("x = 1\n")
        rc = _run_over(monkeypatch, tmp_path, ["fine.py"])
        captured = capsys.readouterr()
        assert rc == 0, captured.out + captured.err
        assert "OK: 0 stale-identifier mentions" in captured.out

    def test_a_non_utf8_text_file_is_reported_rather_than_silently_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No NUL in the first 8 KiB, so not detected as binary; not decodable either."""
        path = tmp_path / "latin1.py"
        path.write_bytes("x = 'é'\n".encode("latin-1"))
        rc = _run_over(monkeypatch, tmp_path, ["latin1.py"])
        captured = capsys.readouterr()
        assert rc == 1, captured.out + captured.err
        assert "latin1.py" in captured.err
