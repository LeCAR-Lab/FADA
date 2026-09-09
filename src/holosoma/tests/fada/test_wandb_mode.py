"""Tests for the --wandb-mode CLI escape hatch and wandb.init()/log()/finish() failure
handling.

DAgger training (``holosoma.fada.planner_idm.train``) and IDM LoRA finetuning
(``holosoma.fada.planner_idm.finetune_idm_lora``) both call ``wandb.init()``. This
file covers:

1. Both entrypoints' argparsers expose ``--wandb-mode {online,offline,disabled}``,
   defaulting to "online" and overridable.
2. Both ``wandb.init()`` call sites degrade gracefully (print a warning, return
   None, keep training running) instead of propagating the exception when
   ``wandb.init()`` itself fails -- e.g. missing credentials.
3. ``_WandbCommGuard`` (shared by both entrypoints) disables further logging after a
   ``wandb.errors.Error`` from ``.log()``/``.finish()``, while letting programming
   errors (e.g. ``TypeError``) propagate.
4. Both entrypoints raise the same ``RuntimeError`` when wandb is enabled
   (mode != "disabled") but the package cannot be imported.
"""

from __future__ import annotations

import builtins
import dataclasses
import threading
from pathlib import Path
from typing import Any

import pytest
import wandb
from holosoma.fada.common.lora_utils import _init_wandb_run, _WandbCommGuard
from holosoma.fada.planner_idm.config import FADAConfig, build_arg_parser, config_from_args
from holosoma.fada.planner_idm.finetune_idm_lora import _build_arg_parser as build_finetune_arg_parser
from holosoma.fada.planner_idm.trainer import init_wandb_run

# ---------------------------------------------------------------------------
# (1) CLI surface: --wandb-mode on both entrypoints.
# ---------------------------------------------------------------------------


def test_train_cli_wandb_mode_defaults_to_online_and_is_overridable() -> None:
    parser = build_arg_parser()

    default_args = parser.parse_args(
        ["--expert-checkpoint", "ckpt.pt", "--run-name", "t1_loco", "--output-dir", "/tmp/out"]
    )
    assert default_args.wandb_mode == "online"
    default_cfg = config_from_args(default_args)
    assert default_cfg.wandb_mode == "online"

    disabled_args = parser.parse_args(
        [
            "--expert-checkpoint",
            "ckpt.pt",
            "--run-name",
            "t1_loco",
            "--output-dir",
            "/tmp/out",
            "--wandb-mode",
            "disabled",
        ]
    )
    assert disabled_args.wandb_mode == "disabled"
    disabled_cfg = config_from_args(disabled_args)
    assert disabled_cfg.wandb_mode == "disabled"


def test_train_cli_wandb_mode_rejects_unknown_value() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--expert-checkpoint",
                "ckpt.pt",
                "--run-name",
                "t1_loco",
                "--output-dir",
                "/tmp/out",
                "--wandb-mode",
                "bogus",
            ]
        )


def test_finetune_cli_wandb_mode_defaults_to_online_and_is_overridable(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.h5"
    dataset_path.touch()
    parser = build_finetune_arg_parser()
    required = [
        "--checkpoint",
        str(tmp_path / "ckpt.pt"),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        "t1_loco_sft",
        "--output-dir",
        str(tmp_path / "out"),
    ]

    default_args = parser.parse_args(required)
    assert default_args.wandb_mode == "online"

    disabled_args = parser.parse_args([*required, "--wandb-mode", "disabled"])
    assert disabled_args.wandb_mode == "disabled"


def test_finetune_cli_wandb_mode_rejects_unknown_value(tmp_path: Path) -> None:
    parser = build_finetune_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                str(tmp_path / "ckpt.pt"),
                "--target-datasets",
                str(tmp_path / "dataset.h5"),
                "--run-name",
                "t1_loco_sft",
                "--output-dir",
                str(tmp_path / "out"),
                "--wandb-mode",
                "bogus",
            ]
        )


# ---------------------------------------------------------------------------
# (2) Graceful degradation when wandb.init() itself fails.
# ---------------------------------------------------------------------------


def _force_wandb_credentials_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `_wandb_credentials_preflight("online")` report "configured" regardless of
    the environment's real credential state.

    Patches `wandb.sdk.lib.apikey.api_key`, the function
    `_wandb_credentials_preflight` calls, to return a dummy key. Required by any test
    whose `wandb.init` mock must be reached in mode="online": with no
    WANDB_API_KEY/.netrc configured the preflight short-circuits before
    `wandb.init()` is called.
    """
    from wandb.sdk.lib import apikey as wandb_apikey

    monkeypatch.setattr(wandb_apikey, "api_key", lambda *_a, **_k: "dummy-test-key")


def _minimal_fada_config(*, wandb_mode: str) -> FADAConfig:
    return FADAConfig(
        expert_checkpoint="dummy.pt",
        wandb_enable=True,
        wandb_mode=wandb_mode,
        wandb_project="fada_test_project",
    )


def test_train_init_wandb_run_disabled_mode_never_touches_wandb_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _minimal_fada_config(wandb_mode="disabled")

    def _fail_if_called(**_kwargs: Any) -> None:
        raise AssertionError("wandb.init() must not be called when wandb_mode='disabled'")

    monkeypatch.setattr(wandb, "init", _fail_if_called)

    result = init_wandb_run(cfg, run_dir=tmp_path)

    assert result is None


def test_train_init_wandb_run_degrades_gracefully_on_init_comm_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Credentials are configured (preflight passes) but wandb.init() raises a
    CommError (e.g. network/service unavailable): the call degrades gracefully."""
    _force_wandb_credentials_configured(monkeypatch)
    cfg = _minimal_fada_config(wandb_mode="online")

    def _raise_comm_error(**_kwargs: Any) -> None:
        raise wandb.errors.CommError("could not reach wandb backend")

    monkeypatch.setattr(wandb, "init", _raise_comm_error)

    # Does not raise; training continues with W&B disabled.
    result = init_wandb_run(cfg, run_dir=tmp_path)

    assert result is None
    captured = capsys.readouterr()
    assert "wandb.init() failed" in captured.out
    assert "--wandb-mode disabled" in captured.out


def test_train_init_wandb_run_does_not_swallow_usage_errors_when_credentials_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the preflight confirms credentials are present, a UsageError from
    wandb.init() (e.g. an invalid project name) propagates rather than being
    downgraded to "no W&B"."""
    _force_wandb_credentials_configured(monkeypatch)
    cfg = _minimal_fada_config(wandb_mode="online")

    def _raise_usage_error(**_kwargs: Any) -> None:
        raise wandb.errors.UsageError("Invalid project name 'xxx...': exceeded 128 characters")

    monkeypatch.setattr(wandb, "init", _raise_usage_error)

    with pytest.raises(wandb.errors.UsageError):
        init_wandb_run(cfg, run_dir=tmp_path)


def test_train_init_wandb_run_propagates_real_wandb_usage_error_for_invalid_project_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end against the real (unmocked) `wandb.init()`; only the credentials
    precondition is forced, isolating this from the missing-credentials case tested
    elsewhere. `wandb.init(project="x" * 200)` raises `UsageError: Invalid project
    name ... exceeded 128 characters`, and that error propagates."""
    _force_wandb_credentials_configured(monkeypatch)
    cfg = _minimal_fada_config(wandb_mode="online")
    cfg = dataclasses.replace(cfg, wandb_project="x" * 200)

    with pytest.raises(wandb.errors.UsageError, match="exceeded 128 characters"):
        init_wandb_run(cfg, run_dir=tmp_path)


def test_train_init_wandb_run_does_not_swallow_programming_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_wandb_credentials_configured(monkeypatch)
    cfg = _minimal_fada_config(wandb_mode="online")

    def _raise_type_error(**_kwargs: Any) -> None:
        raise TypeError("boom: not a wandb.errors.Error")

    monkeypatch.setattr(wandb, "init", _raise_type_error)

    # A TypeError from the kwargs (not a wandb infra failure) propagates.
    with pytest.raises(TypeError):
        init_wandb_run(cfg, run_dir=tmp_path)


def test_finetune_init_wandb_run_disabled_mode_never_touches_wandb_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_if_called(**_kwargs: Any) -> None:
        raise AssertionError("wandb.init() must not be called when mode='disabled'")

    monkeypatch.setattr(wandb, "init", _fail_if_called)

    result = _init_wandb_run(
        enabled=True,
        mode="disabled",
        project="fada_test_project",
        entity=None,
        group=None,
        name="run",
        tags=[],
        run_dir=tmp_path,
        config={},
    )

    assert result is None


def test_finetune_init_wandb_run_degrades_gracefully_on_init_comm_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Credentials are configured (preflight passes) but wandb.init() raises a
    CommError: the call degrades gracefully."""
    _force_wandb_credentials_configured(monkeypatch)

    def _raise_comm_error(**_kwargs: Any) -> None:
        raise wandb.errors.CommError("could not reach wandb backend")

    monkeypatch.setattr(wandb, "init", _raise_comm_error)

    result = _init_wandb_run(
        enabled=True,
        mode="online",
        project="fada_test_project",
        entity=None,
        group=None,
        name="run",
        tags=[],
        run_dir=tmp_path,
        config={},
    )

    assert result is None
    captured = capsys.readouterr()
    assert "wandb.init() failed" in captured.out
    assert "--wandb-mode disabled" in captured.out


def test_finetune_init_wandb_run_does_not_swallow_usage_errors_when_credentials_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finetune-entrypoint counterpart of the DAgger-trainer test above."""
    _force_wandb_credentials_configured(monkeypatch)

    def _raise_usage_error(**_kwargs: Any) -> None:
        raise wandb.errors.UsageError("Invalid project name 'xxx...': exceeded 128 characters")

    monkeypatch.setattr(wandb, "init", _raise_usage_error)

    with pytest.raises(wandb.errors.UsageError):
        _init_wandb_run(
            enabled=True,
            mode="online",
            project="fada_test_project",
            entity=None,
            group=None,
            name="run",
            tags=[],
            run_dir=tmp_path,
            config={},
        )


# ---------------------------------------------------------------------------
# (3) _WandbCommGuard: log()/finish() communication failures degrade to "stop
#     logging", programming errors still propagate.
# ---------------------------------------------------------------------------


class _FakeWandbRun:
    """Stand-in for a real wandb Run whose .log()/.finish() can be scripted to fail."""

    def __init__(self, *, log_side_effect: Any = None, finish_side_effect: Any = None) -> None:
        self.log_calls: list[tuple[dict, int | None]] = []
        self.finish_calls: int = 0
        self._log_side_effect = log_side_effect
        self._finish_side_effect = finish_side_effect
        self.summary: dict[str, Any] = {}

    def log(self, payload: dict, step: int | None = None) -> None:
        self.log_calls.append((payload, step))
        if self._log_side_effect is not None:
            raise self._log_side_effect

    def finish(self) -> None:
        self.finish_calls += 1
        if self._finish_side_effect is not None:
            raise self._finish_side_effect


def test_wandb_comm_guard_disables_logging_after_comm_failure_in_log() -> None:
    fake_run = _FakeWandbRun(log_side_effect=wandb.errors.CommError("network blip"))
    guard = _WandbCommGuard(fake_run, wandb)

    # First call hits the comm error, which is swallowed; the guard does not raise.
    guard.log({"loss": 1.0}, step=1)
    # Second call is a no-op (circuit broken): the underlying run.log() is not called
    # again.
    guard.log({"loss": 2.0}, step=2)

    assert len(fake_run.log_calls) == 1, "log() must not be retried after a comm failure"


def test_wandb_comm_guard_does_not_swallow_programming_errors_in_log() -> None:
    fake_run = _FakeWandbRun(log_side_effect=TypeError("bad payload -- our bug, not W&B's"))
    guard = _WandbCommGuard(fake_run, wandb)

    with pytest.raises(TypeError):
        guard.log({"loss": 1.0})

    # A programming error propagates and leaves further logging enabled.
    assert not guard._disabled


def test_wandb_comm_guard_disables_after_comm_failure_in_finish() -> None:
    fake_run = _FakeWandbRun(finish_side_effect=wandb.errors.CommError("network blip"))
    guard = _WandbCommGuard(fake_run, wandb)

    guard.finish()  # swallowed
    guard.finish()  # no-op, circuit already broken

    assert fake_run.finish_calls == 1


def test_wandb_comm_guard_does_not_swallow_programming_errors_in_finish() -> None:
    fake_run = _FakeWandbRun(finish_side_effect=RuntimeError("our bug"))
    guard = _WandbCommGuard(fake_run, wandb)

    with pytest.raises(RuntimeError):
        guard.finish()


def test_wandb_comm_guard_finish_does_not_block_indefinitely_on_a_stalled_upload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`Run.finish()` waits for the upload to complete, with no bounded timeout in the
    pinned wandb version (no `Settings.finish_timeout`/`finish_timeout_raises`, no
    `Run.finish(timeout=...)`), so a stalled upload blocks the process.
    `_WandbCommGuard.finish()` runs the real finish() call on a daemon thread with a
    bounded join(), so it returns even if the underlying finish() never does."""
    import time

    never_returns = threading.Event()

    class _HangingRun:
        def finish(self) -> None:
            never_returns.wait()  # blocks "forever" (until the test process exits)

    guard = _WandbCommGuard(_HangingRun(), wandb)
    monkeypatch.setattr(guard, "_FINISH_TIMEOUT_SECONDS", 0.05)

    start = time.monotonic()
    guard.finish()  # returns after the bounded join, without waiting for finish()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"finish() blocked for {elapsed:.2f}s instead of respecting the timeout"
    assert guard._disabled
    assert "did not complete within" in capsys.readouterr().out


def test_wandb_comm_guard_forwards_attribute_access() -> None:
    fake_run = _FakeWandbRun()
    fake_run.summary["run_dir"] = "/tmp/foo"
    guard = _WandbCommGuard(fake_run, wandb)

    assert guard.summary is fake_run.summary


def test_init_wandb_run_returns_a_guarded_run_on_both_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both entrypoints return a `_WandbCommGuard`-wrapped run, so log()/finish()
    communication failures degrade gracefully and not only at init()."""
    _force_wandb_credentials_configured(monkeypatch)
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(wandb, "init", lambda **_kwargs: fake_run)

    trainer_result = init_wandb_run(_minimal_fada_config(wandb_mode="online"), run_dir=tmp_path)
    assert isinstance(trainer_result, _WandbCommGuard)

    finetune_result = _init_wandb_run(
        enabled=True,
        mode="online",
        project="fada_test_project",
        entity=None,
        group=None,
        name="run",
        tags=[],
        run_dir=tmp_path,
        config={},
    )
    assert isinstance(finetune_result, _WandbCommGuard)


# ---------------------------------------------------------------------------
# (4) Behavior when the wandb package itself cannot be imported.
# ---------------------------------------------------------------------------


def _raise_import_error_for_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_train_init_wandb_run_raises_when_wandb_package_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _raise_import_error_for_wandb(monkeypatch)
    cfg = _minimal_fada_config(wandb_mode="online")

    with pytest.raises(RuntimeError, match=r"(?s)not installed.*--wandb-mode disabled"):
        init_wandb_run(cfg, run_dir=tmp_path)


def test_finetune_init_wandb_run_raises_when_wandb_package_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _raise_import_error_for_wandb(monkeypatch)

    with pytest.raises(RuntimeError, match=r"(?s)not installed.*--wandb-mode disabled"):
        _init_wandb_run(
            enabled=True,
            mode="online",
            project="fada_test_project",
            entity=None,
            group=None,
            name="run",
            tags=[],
            run_dir=tmp_path,
            config={},
        )


def test_both_entrypoints_skip_wandb_import_entirely_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mode='disabled' does not attempt the wandb import, so the missing-package
    RuntimeError does not fire."""
    _raise_import_error_for_wandb(monkeypatch)

    assert init_wandb_run(_minimal_fada_config(wandb_mode="disabled"), run_dir=tmp_path) is None
    assert (
        _init_wandb_run(
            enabled=True,
            mode="disabled",
            project="fada_test_project",
            entity=None,
            group=None,
            name="run",
            tags=[],
            run_dir=tmp_path,
            config={},
        )
        is None
    )


# ---------------------------------------------------------------------------
# The init-vs-runtime split.
#
# The `wandb.init()` call site catches CommError unconditionally, and UsageError only
# as a fallback when the credentials preflight itself could not run (see
# `_wandb_credentials_preflight`'s docstring); `_WandbCommGuard` (around
# .log()/.finish()) catches only CommError. The split is by *when* the failure
# happens:
#
#   - Missing credentials are checked locally, before `wandb.init()` is ever called,
#     so `wandb.init()` is not reached in that case.
#   - Once credentials are confirmed present (by the preflight, or in a fallback where
#     the preflight itself is unavailable), a UsageError from `wandb.init()` means a
#     bad argument was passed (e.g. an invalid project name) and it propagates -- as
#     it does around `.log()`/`.finish()`.
#
# Both directions are pinned by the tests below.
# ---------------------------------------------------------------------------


def test_train_init_degrades_when_credentials_missing_without_ever_calling_wandb_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no W&B credentials configured (no WANDB_API_KEY, no .netrc; the credential
    state is not mocked here), the preflight short-circuits *before* wandb.init() is
    called. The real wandb.init() in a no-tty environment raises
    `UsageError: api_key not configured (no-tty)`."""

    def _fail_if_called(**_kwargs: Any) -> None:
        raise AssertionError("wandb.init() must not be called when credentials are missing")

    monkeypatch.setattr(wandb, "init", _fail_if_called)

    assert init_wandb_run(_minimal_fada_config(wandb_mode="online"), run_dir=tmp_path) is None
    assert "--wandb-mode disabled" in capsys.readouterr().out


def test_finetune_init_degrades_when_credentials_missing_without_ever_calling_wandb_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finetune-entrypoint counterpart of the DAgger-trainer test above."""

    def _fail_if_called(**_kwargs: Any) -> None:
        raise AssertionError("wandb.init() must not be called when credentials are missing")

    monkeypatch.setattr(wandb, "init", _fail_if_called)

    assert (
        _init_wandb_run(
            enabled=True,
            mode="online",
            project="fada_test_project",
            entity=None,
            group=None,
            name="fada_test_run",
            tags=[],
            run_dir=tmp_path,
            config={},
        )
        is None
    )
    assert "--wandb-mode disabled" in capsys.readouterr().out


def test_wandb_credentials_preflight_falls_back_to_dual_catch_when_apikey_module_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the preflight cannot determine credential state it reports "unknown"
    rather than raising (the "unknown" branch documented in
    `_wandb_credentials_preflight`, reached if the internal `wandb.sdk.lib.apikey`
    module it relies on is removed or renamed), and the wandb.init() call site then
    also catches UsageError and degrades gracefully.

    Simulated at the preflight function rather than by breaking the
    `wandb.sdk.lib.apikey` import."""
    import holosoma.fada.planner_idm.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "_wandb_credentials_preflight", lambda _mode: "unknown")

    def _raise_usage_error(**_kwargs: Any) -> None:
        raise wandb.errors.UsageError("api_key not configured (no-tty). call wandb.login(...)")

    monkeypatch.setattr(wandb, "init", _raise_usage_error)

    result = init_wandb_run(_minimal_fada_config(wandb_mode="online"), run_dir=tmp_path)

    assert result is None
    assert "--wandb-mode disabled" in capsys.readouterr().out


def test_comm_guard_swallows_comm_error_but_propagates_usage_error() -> None:
    """The runtime half of the split: `.log()` swallows CommError and propagates
    UsageError."""
    import wandb

    from holosoma.fada.common.lora_utils import _WandbCommGuard

    class _Run:
        def __init__(self, exc: BaseException) -> None:
            self._exc = exc

        def log(self, *_a: Any, **_k: Any) -> None:
            raise self._exc

    _WandbCommGuard(_Run(wandb.errors.CommError("network down")), wandb).log({"loss": 1.0})

    with pytest.raises(wandb.errors.UsageError):
        _WandbCommGuard(_Run(wandb.errors.UsageError("bad payload")), wandb).log({"loss": 1.0})


# ---------------------------------------------------------------------------
# (5) --wandb-tags must reach wandb.init().
#
# The training CLI parses `--wandb-tags` into `FADAConfig.wandb_tags`, which
# `init_wandb_run` passes to `wandb.init()` as the `tags` kwarg. The shipped default
# is "fada,planner_idm,dagger"; the value is split on commas with blanks dropped.
# ---------------------------------------------------------------------------


def _capture_wandb_init_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Run:
        def log(self, *_a: Any, **_k: Any) -> None:
            return None

        def finish(self, *_a: Any, **_k: Any) -> None:
            return None

    def _record(**kwargs: Any) -> _Run:
        captured.update(kwargs)
        return _Run()

    monkeypatch.setattr(wandb, "init", _record)
    return captured


def test_train_init_wandb_run_passes_the_configured_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_wandb_credentials_configured(monkeypatch)
    captured = _capture_wandb_init_kwargs(monkeypatch)
    cfg = dataclasses.replace(
        _minimal_fada_config(wandb_mode="online"),
        wandb_tags="alpha, beta ,,gamma",
    )

    init_wandb_run(cfg, run_dir=tmp_path)

    assert captured.get("tags") == ["alpha", "beta", "gamma"], (
        f"--wandb-tags never reached wandb.init(): {captured.get('tags')!r}"
    )


def test_train_init_wandb_run_passes_the_shipped_default_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no tags passed, the shipped default tags reach `wandb.init()`."""
    _force_wandb_credentials_configured(monkeypatch)
    captured = _capture_wandb_init_kwargs(monkeypatch)

    init_wandb_run(_minimal_fada_config(wandb_mode="online"), run_dir=tmp_path)

    assert captured.get("tags") == ["fada", "planner_idm", "dagger"], captured.get("tags")


def test_train_init_wandb_run_omits_tags_entirely_when_there_are_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `wandb_tags` string that yields no tags omits `tags` from the `wandb.init()`
    kwargs entirely; `tags=[]` and no `tags` differ to W&B on resume."""
    _force_wandb_credentials_configured(monkeypatch)
    captured = _capture_wandb_init_kwargs(monkeypatch)
    cfg = dataclasses.replace(_minimal_fada_config(wandb_mode="online"), wandb_tags="  , ,")

    init_wandb_run(cfg, run_dir=tmp_path)

    assert "tags" not in captured, captured
