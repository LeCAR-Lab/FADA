"""The retargeting asset guard must not be silent about the tree it did not check.

`holosoma_retargeting`'s two asset trees -- `demo_data/` (motion data) and `models/`
(URDFs and meshes) -- are deliberately absent from the wheel, and `asset_paths.py` exists
to turn "a default path does not resolve from an installed package" into a sentence that
says so.

The hole this file pins: `require_asset_dir(--data_path)` returns quietly whenever the
supplied path exists, and `--data_path` may legitimately name a directory outside the
package (a user's own capture). On a wheel install that check passes while `models/` is
absent entirely, and the run dies later, deeper, and with a worse message inside a URDF
loader -- once per worker, in the parallel entry point. Checking one argument is not
evidence about the other, so the URDF is now checked on its own.

`asset_paths` is loaded from its file rather than imported as
`holosoma_retargeting.asset_paths` so this runs in any environment: the module itself
needs only `os` and `pathlib`, but the package's entry points pull in tyro, mujoco, viser
and friends, which are installed only in the retargeting environment.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "src" / "holosoma_retargeting" / "holosoma_retargeting"
ENTRY_POINTS = (
    PACKAGE / "examples" / "robot_retarget.py",
    PACKAGE / "examples" / "parallel_robot_retarget.py",
)


@pytest.fixture(scope="module")
def asset_paths():
    spec = importlib.util.spec_from_file_location("_asset_paths_under_test", PACKAGE / "asset_paths.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The two trees are checked separately.
# ---------------------------------------------------------------------------


def test_the_data_and_model_trees_are_distinct_declarations(asset_paths) -> None:
    assert asset_paths.DATA_ASSET_DIRS == ("demo_data",)
    assert asset_paths.MODEL_ASSET_DIRS == ("models",)
    assert set(asset_paths.DATA_ASSET_DIRS) | set(asset_paths.MODEL_ASSET_DIRS) == set(asset_paths.ASSET_DIR_NAMES)


def test_a_missing_urdf_is_reported_against_the_models_tree(asset_paths, tmp_path, monkeypatch) -> None:
    """The message must name `models`, not the tree that happens to be present."""
    installed = tmp_path / "site-packages" / "holosoma_retargeting"
    (installed / "demo_data").mkdir(parents=True)  # data present, models absent
    monkeypatch.setattr(asset_paths, "package_root", lambda: installed)

    with pytest.raises(asset_paths.RetargetingAssetsUnavailable) as excinfo:
        asset_paths.require_asset_file("models/g1/g1_29dof.urdf", what="robot URDF")

    message = str(excinfo.value)
    assert "robot URDF" in message
    assert "['models']" in message, message
    assert "demo_data" not in message.split("missing", 1)[1].splitlines()[0]
    assert "NOT packaged into the wheel" in message


def test_an_existing_external_data_dir_does_not_vouch_for_the_models_tree(asset_paths, tmp_path, monkeypatch) -> None:
    """The demonstrated hole, as a test.

    `--data_path` points at a real directory the user owns; the package install has no
    `models/`. The data check passes -- correctly -- and the URDF check must still fail.
    Against the pre-change code there was no second check to fail, so this test's second
    half had nothing to call.
    """
    installed = tmp_path / "site-packages" / "holosoma_retargeting"
    installed.mkdir(parents=True)
    external_data = tmp_path / "my_own_mocap"
    external_data.mkdir()
    monkeypatch.setattr(asset_paths, "package_root", lambda: installed)

    assert asset_paths.require_asset_dir(external_data, what="--data_path") == external_data

    with pytest.raises(asset_paths.RetargetingAssetsUnavailable):
        asset_paths.require_asset_file("models/g1/g1_29dof.urdf", what="robot URDF")


def test_a_wrong_data_path_is_not_blamed_on_a_missing_install(asset_paths, tmp_path, monkeypatch) -> None:
    """When the tree that argument comes from *is* present, say "wrong path"."""
    installed = tmp_path / "site-packages" / "holosoma_retargeting"
    (installed / "demo_data").mkdir(parents=True)
    monkeypatch.setattr(asset_paths, "package_root", lambda: installed)

    with pytest.raises(asset_paths.RetargetingAssetsUnavailable) as excinfo:
        asset_paths.require_asset_dir(tmp_path / "typo", what="--data_path")

    message = str(excinfo.value)
    assert "wrong path rather than a missing install" in message
    assert "NOT packaged into the wheel" not in message


def test_an_existing_path_is_returned_unchanged(asset_paths, tmp_path) -> None:
    """The guard reports; it must never substitute a package-relative path."""
    target = tmp_path / "demo_data"
    target.mkdir()
    assert asset_paths.require_asset_dir(target) == target
    file_target = tmp_path / "robot.urdf"
    file_target.write_text("<robot/>")
    assert asset_paths.require_asset_file(file_target) == file_target


# ---------------------------------------------------------------------------
# Both entry points actually call it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda p: p.name)
def test_entry_point_validates_the_robot_urdf_separately(entry_point: Path) -> None:
    """Source assertion: importing these modules needs the retargeting environment.

    Fails against the pre-change entry points, which validated only the data directory.
    """
    source = entry_point.read_text()
    assert re.search(r"require_asset_file\(\s*cfg\.robot_config\.ROBOT_URDF_FILE", source), (
        f"{entry_point.name} does not validate the robot URDF. A valid --data_path does not imply "
        "the models/ tree exists, so without this the run fails later and deeper."
    )
    assert "require_asset_dir(" in source, f"{entry_point.name} no longer validates its data directory"


# ---------------------------------------------------------------------------
# The inherited `.pt` / `.pkl` readers in `src/utils.py`.
#
# `holosoma_retargeting/src/utils.py` calls a bare `torch.load(file_path,
# map_location="cpu")` on an InterMimic `.pt` the README tells the reader to download from
# a third party. That is the same threat model as the FADA oracle checkpoint, which
# `holosoma/utils/safe_torch_load.py` exists to close -- and this file is byte-identical to
# upstream, so hardening it in place would be a new upstream deviation in a subsystem this
# release ships no pipeline for.
#
# The decision taken: close the window through the DEPENDENCY FLOOR instead. PyTorch 2.6
# flipped `torch.load`'s `weights_only` default to True, so at `torch>=2.6` that call is
# already restricted and refuses a malicious reducer, with not one inherited line touched.
# The floor is the security control, so it is asserted here rather than left as a comment.
#
# The `pickle.load` on `demo_data/height_dict.pkl` is deliberately left alone: it reads a
# tracked file out of the package's own source tree, not a download, so it is exactly as
# trusted as the checkout it came from.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ["pyproject.toml", "setup.py"])
def test_the_torch_floor_makes_the_inherited_pt_reader_restricted(filename: str) -> None:
    from packaging.requirements import Requirement
    from packaging.version import Version

    text = (REPO_ROOT / "src" / "holosoma_retargeting" / filename).read_text()
    requirements = [r for r in re.findall(r'"(torch(?![-_a-zA-Z])[^"]*)"', text)]
    assert len(requirements) == 1, f"{filename}: expected one torch requirement, got {requirements}"

    specifier = Requirement(requirements[0]).specifier
    assert Version("2.5.1") not in specifier, (
        f"src/holosoma_retargeting/{filename} declares {requirements[0]!r}, which admits torch < 2.6. "
        "There, `torch.load` defaults to weights_only=False, and src/utils.py's bare torch.load on a "
        "downloaded InterMimic .pt executes the file's pickle stream. Either keep the floor at 2.6 or "
        "harden src/utils.py and register the upstream deviation."
    )
    assert Version("2.6.0") in specifier


def test_the_inherited_reader_was_not_edited_in_place() -> None:
    """The other half of the decision: `src/utils.py` stays byte-identical to upstream.

    If a future change does harden it, that is a new deviation and this test is where the
    reminder to register it lives.
    """
    import subprocess

    utils = "src/holosoma_retargeting/holosoma_retargeting/src/utils.py"
    upstream = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "upstream-baseline"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ref = "upstream-baseline" if upstream.returncode == 0 else "main"
    diff = subprocess.run(["git", "diff", "--numstat", ref, "--", utils], cwd=REPO_ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("no upstream ref available in this tree")
    assert diff.stdout.strip() == "", (
        f"{utils} now differs from {ref}:\n{diff.stdout}\n"
        "Hardening it in place is a deliberate option, but it changes a file this repository "
        "otherwise keeps identical to upstream."
    )
