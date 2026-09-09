from __future__ import annotations

import json
from pathlib import Path

from tools.bitexact.compare import compare_dirs


def _write(d: Path, name: str, payload: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_identical_dirs_report_no_diff(tmp_path: Path) -> None:
    payload = {"loss_sequence": [{"idm": 1.5}], "state_dict_sha256": "abc"}
    for side in ("golden", "current"):
        _write(tmp_path / side, "train.json", payload)
    diffs = compare_dirs(tmp_path / "golden", tmp_path / "current", names=["train"])
    assert diffs == []


def test_float_difference_is_detected(tmp_path: Path) -> None:
    _write(tmp_path / "golden", "train.json", {"loss_sequence": [{"idm": 1.5}]})
    _write(tmp_path / "current", "train.json", {"loss_sequence": [{"idm": 1.5000001}]})
    diffs = compare_dirs(tmp_path / "golden", tmp_path / "current", names=["train"])
    assert len(diffs) == 1
    assert "loss_sequence" in diffs[0]


def test_missing_current_file_is_reported(tmp_path: Path) -> None:
    _write(tmp_path / "golden", "train.json", {"x": 1})
    (tmp_path / "current").mkdir(parents=True, exist_ok=True)
    diffs = compare_dirs(tmp_path / "golden", tmp_path / "current", names=["train"])
    assert len(diffs) == 1
    assert "missing" in diffs[0].lower()


def test_config_fingerprint_only_difference_is_ignored(tmp_path: Path) -> None:
    """config_fingerprint is diagnostic-only: legitimate config-field deletions across
    later tasks (folding FDM/arch switches, noise/scheduler/normalization flags, inlining
    defaults, ...) change this hash without touching any actual computation. If this key
    gated pass/fail, every such task would be forced to edit golden -- exactly the kind of
    self-certifying change that could hide a real regression. loss_sequence/*_sha256 remain
    the only judged signals.
    """
    _write(
        tmp_path / "golden",
        "train.json",
        {"config_fingerprint": "aaa", "loss_sequence": [{"idm": 1.5}], "state_dict_sha256": "abc"},
    )
    _write(
        tmp_path / "current",
        "train.json",
        {"config_fingerprint": "zzz", "loss_sequence": [{"idm": 1.5}], "state_dict_sha256": "abc"},
    )
    diffs = compare_dirs(tmp_path / "golden", tmp_path / "current", names=["train"])
    assert diffs == []


def test_config_fingerprint_exclusion_does_not_mask_other_diffs(tmp_path: Path) -> None:
    """The exclusion must be exact-key-name only. Prove it does not become a fuzzy match
    that also swallows unrelated differences: change config_fingerprint (ignored) AND
    perturb loss_sequence's last float digit AND every *_sha256-style key in the same
    payload -- all three of the latter must still be caught.
    """
    _write(
        tmp_path / "golden",
        "train.json",
        {
            "config_fingerprint": "aaa",
            "loss_sequence": [{"idm": 1.5}],
            "state_dict_sha256": "abc",
            "onnx_sha256": "def",
            "adapter_sha256": "ghi",
        },
    )
    _write(
        tmp_path / "current",
        "train.json",
        {
            "config_fingerprint": "zzz",  # ignored
            "loss_sequence": [{"idm": 1.5000000000000002}],  # last-digit perturbation
            "state_dict_sha256": "abd",  # sha mismatch
            "onnx_sha256": "deg",  # sha mismatch
            "adapter_sha256": "ghi",  # unchanged, must NOT be flagged
        },
    )
    diffs = compare_dirs(tmp_path / "golden", tmp_path / "current", names=["train"])
    joined = "\n".join(diffs)
    assert "config_fingerprint" not in joined
    assert "loss_sequence" in joined
    assert "state_dict_sha256" in joined
    assert "onnx_sha256" in joined
    assert "adapter_sha256" not in joined
    assert len(diffs) == 3
