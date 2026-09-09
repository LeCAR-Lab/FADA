from __future__ import annotations

import json
from pathlib import Path

from tools.bitexact.harness import run_train_probe, run_train_probe_mixed


def test_probe_is_deterministic(tmp_path: Path) -> None:
    a = run_train_probe(tmp_path / "a.json", steps=3)
    b = run_train_probe(tmp_path / "b.json", steps=3)
    assert a["loss_sequence"] == b["loss_sequence"]
    assert a["state_dict_sha256"] == b["state_dict_sha256"]


def test_probe_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "probe.json"
    result = run_train_probe(out, steps=2)
    assert out.exists()
    assert json.loads(out.read_text()) == result
    assert len(result["loss_sequence"]) == 2


def test_probe_mixed_is_deterministic(tmp_path: Path) -> None:
    a = run_train_probe_mixed(tmp_path / "a_mixed.json", steps=3)
    b = run_train_probe_mixed(tmp_path / "b_mixed.json", steps=3)
    assert a["loss_sequence"] == b["loss_sequence"]
    assert a["state_dict_sha256"] == b["state_dict_sha256"]


def test_probe_mixed_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "probe_mixed.json"
    result = run_train_probe_mixed(out, steps=2)
    assert out.exists()
    assert json.loads(out.read_text()) == result
    assert len(result["loss_sequence"]) == 2
