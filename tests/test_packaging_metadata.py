"""What `pip install` actually produces must match what the code actually needs.

A wheel can lose content in three independent ways, none of which any other check in
this repository can see, because every documented environment installs these packages
**editable** -- and an editable install reads the checkout:

1. **Content the package's own defaults resolve at runtime.**
   `holosoma_inference/config/config_values/task.py` defaults the G1 safety policy to
   `<package>/models/loco/g1_29dof/fastsac_g1_29dof.onnx`, so the shipped ONNX files
   have to be in the wheel. `setup.py` discovering packages with `find_packages()`
   reaches only directories that contain `__init__.py`.

2. **Dependency metadata against what is imported at startup.**
   `run_policy.py` imports the WBT and FADA policy modules unconditionally, so `numpy`,
   `pinocchio`, `typing_extensions` and `holosoma` are all reached before the first line
   of `main()` and must be declared.

3. **Data files selected by extension rather than by directory.**
   `simulator/mujoco/scene_manager.py` resolves `asset.xml_file` under the *installed*
   package and hands it to `mujoco.MjSpec.from_file()`, so every robot XML must be in
   the wheel, and the `.gitignore`d IsaacSim conversion caches must not be.

These are static tests on purpose: they need no build, no network and no venv, so they
run in the same gate as everything else and they run inside the built release tree. Where
a question genuinely needs the build machinery -- "which files does this `MANIFEST.in`
select?" -- they drive setuptools' own implementation rather than re-implementing it.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every distribution this repository builds, keyed by its import name. All three are
# checked: module discovery is only one of the ways a wheel loses content, so a
# distribution whose discovery is a non-issue still needs its data files checked.
DISTRIBUTIONS = {
    "holosoma": REPO_ROOT / "src" / "holosoma",
    "holosoma_inference": REPO_ROOT / "src" / "holosoma_inference",
    "holosoma_retargeting": REPO_ROOT / "src" / "holosoma_retargeting",
}

# Directories that are deliberately NOT importable packages, with the reason.
DECLARED_NON_PACKAGES = {
    # The retargeting README instructs the reader to `git clone` a third-party LAFAN
    # repository into this directory. Making it a package would sweep that clone into any
    # wheel built from a working copy that followed those instructions.
    "holosoma_retargeting/data_utils",
}


def _package_dirs_with_modules(root: Path, top: str) -> list[Path]:
    """Directories under `root/top` that contain at least one .py file."""
    found = []
    for path in sorted((root / top).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parent = path.parent
        if parent not in found:
            found.append(parent)
    return found


def _discovered_packages(root: Path) -> set[str]:
    """The package set the **real build** produces for the distribution rooted at `root`.

    Not a re-implementation: this calls the same `setuptools` finders the build calls, with
    the same arguments, resolved from whichever configuration actually governs.

    Getting this wrong is the whole of one shipped defect. The earlier version of this test
    assumed `find_packages()` for both distributions because both have a `setup.py`. That is
    true for `holosoma_inference` (setup.py only) and false for `holosoma_retargeting`, whose
    `pyproject.toml` carries `[project]` and `[tool.setuptools.packages.find]` -- PEP 517
    discovery, where `namespaces` defaults to **true**. Under that default a directory with
    no `__init__.py` is still a (namespace) package, so `data_utils/` matched the
    `holosoma_retargeting*` include glob and shipped, while this test happily reported it as
    excluded.
    """
    # Version-guarded rather than try/except: `mypy.ini` sets `python_version = 3.8`, where
    # `tomllib` does not exist and an unguarded import is an unresolvable-module error. A
    # `sys.version_info` guard is dead code to mypy under that target, so it checks the
    # `tomli` branch and skips the other. Same API either way.
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    from setuptools import find_namespace_packages, find_packages

    pyproject = root / "pyproject.toml"
    config: dict = {}
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text())
        if "project" in data:  # PEP 621 metadata present -> the declarative config governs
            config = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
        else:
            return set(find_packages(where=str(root)))
    else:
        return set(find_packages(where=str(root)))

    finder = find_packages if config.get("namespaces") is False else find_namespace_packages
    kwargs: dict = {"where": str((root / config.get("where", ["."])[0]).resolve())}
    if "include" in config:
        kwargs["include"] = tuple(config["include"])
    if "exclude" in config:
        kwargs["exclude"] = tuple(config["exclude"])
    return set(finder(**kwargs))


@pytest.mark.parametrize("top", sorted(DISTRIBUTIONS))
def test_every_module_directory_is_reachable_by_the_real_package_finder(top: str) -> None:
    root = DISTRIBUTIONS[top]
    discovered = _discovered_packages(root)
    missing = []
    for directory in _package_dirs_with_modules(root, top):
        rel = directory.relative_to(root).as_posix()
        # A declared non-package excuses everything under it too -- `data_utils/lafan1` is
        # unreachable precisely because `data_utils` is deliberately not a package.
        if any(rel == excluded or rel.startswith(f"{excluded}/") for excluded in DECLARED_NON_PACKAGES):
            continue
        if rel.replace("/", ".") not in discovered:
            missing.append(rel)
    assert not missing, (
        f"{top}: the package finder the build actually uses does not reach {sorted(set(missing))}, "
        "so those directories and their modules are silently absent from the built wheel. "
        "Editable installs hide this."
    )


@pytest.mark.parametrize("top", sorted(DISTRIBUTIONS))
def test_no_distribution_packages_anything_outside_its_own_import_name(top: str) -> None:
    """A wheel may only claim top-level names it owns.

    `holosoma`'s `[tool.setuptools.packages.find]` said only `where = ["."]`, and PEP 517
    discovery defaults `namespaces = true`, so `src/holosoma/tests/` was discovered and
    the wheel installed a top-level `tests` package into site-packages -- shadowing any
    other project's `tests` in the same environment. This test fails against the
    pre-change pyproject.toml.
    """
    stray = sorted(name for name in _discovered_packages(DISTRIBUTIONS[top]) if name.split(".")[0] != top)
    assert not stray, (
        f"the package finder for {top} discovers {stray}, which the built wheel installs as top-level "
        f"names {sorted({name.split('.')[0] for name in stray})} in site-packages. Constrain discovery "
        f"with an `include` glob."
    )


@pytest.mark.parametrize("excluded", sorted(DECLARED_NON_PACKAGES))
def test_declared_non_packages_really_are_absent_from_the_build(excluded: str) -> None:
    """"Not a package" has to be enforced by the build, not merely asserted by this file.

    Missing `__init__.py` does not exclude anything under PEP 517 discovery, whose
    `namespaces` option defaults to true. `holosoma_retargeting/data_utils` shipped in the
    wheel on exactly that basis -- two modules that raise `ModuleNotFoundError` on
    undeclared `lafan1` / `human_body_prior`, plus whatever the reader cloned into
    `data_utils/lafan1` per this package's README, which is untracked third-party code.

    This test fails against the pre-change `pyproject.toml`.
    """
    top = excluded.split("/", 1)[0]
    root = DISTRIBUTIONS[top]
    prefix = excluded.replace("/", ".")
    shipped = sorted(
        name for name in _discovered_packages(root) if name == prefix or name.startswith(f"{prefix}.")
    )
    assert not shipped, (
        f"{excluded} is declared a non-package, but the real build discovers {shipped} and packages "
        "them. Exclude it in the build configuration -- omitting __init__.py is not enough."
    )


def _package_data_patterns(setup_py: Path, top: str) -> list[str]:
    """The `package_data` globs `setup.py` declares for `top`, read from its source."""
    tree = ast.parse(setup_py.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "package_data":
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and key.value == top and isinstance(value, ast.List):
                return [e.value for e in value.elts if isinstance(e, ast.Constant)]
    return []


def test_the_defaulted_safety_policy_is_declared_as_package_data() -> None:
    """The default in the code and the globs in setup.py must agree.

    Read out of the *source* rather than by importing the config, so this runs without
    pydantic/tyro and inside the built release tree.
    """
    root = DISTRIBUTIONS["holosoma_inference"]
    task_py = root / "holosoma_inference" / "config" / "config_values" / "task.py"
    source = task_py.read_text()

    # `_MODELS_DIR = Path(__file__).parent.parent.parent / "models"`, then
    # `model_path=str(_MODELS_DIR / "loco" / "g1_29dof" / "fastsac_g1_29dof.onnx")`.
    match = re.search(r"_MODELS_DIR\s*/\s*((?:\"[^\"]+\"\s*/\s*)*\"[^\"]+\")", source)
    assert match, "no _MODELS_DIR-relative default found in task.py -- has the default moved?"
    relative = "models/" + "/".join(re.findall(r'"([^"]+)"', match.group(1)))

    on_disk = root / "holosoma_inference" / relative
    assert on_disk.is_file(), f"the default names {relative}, which is not in the source tree either"

    patterns = _package_data_patterns(root / "setup.py", "holosoma_inference")
    assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns), (
        f"holosoma_inference's default safety policy is {relative}, a package-relative path, but "
        f"setup.py's package_data globs {patterns} do not match it. The built wheel therefore "
        "contains no such file and the default points at nothing in any non-editable install."
    )


# ---------------------------------------------------------------------------
# `holosoma`'s data files: what the presets resolve vs. what the wheel carries.
#
# `holosoma` declares no `package_data`; its non-Python content is selected entirely by
# `src/holosoma/MANIFEST.in` (pyproject-driven builds default `include_package_data` to
# true). So "will the wheel contain this file?" is answered by running that template --
# not by re-implementing it. `FileList.process_template_line` is the same code
# `setuptools` runs during the build, driven here from an explicit file list so the test
# needs no build, no network and no venv, exactly like everything above it.
# ---------------------------------------------------------------------------


def _manifest_selected(dist_root: Path, top: str) -> set[str]:
    """Package-relative paths `dist_root/MANIFEST.in` selects, per setuptools' own engine."""
    from setuptools._distutils.filelist import FileList

    candidates = [
        path.relative_to(dist_root).as_posix()
        for path in sorted((dist_root / top).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    file_list = FileList()
    file_list.allfiles = candidates
    for raw in (dist_root / "MANIFEST.in").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            file_list.process_template_line(line)
    return {Path(name).as_posix() for name in file_list.files}


def _referenced_assets(model: Path) -> set[Path]:
    """Files a MuJoCo XML or a URDF names, resolved the way the loader resolves them.

    Regex rather than an XML parser on purpose: this has to read both formats, both of
    them name files in plain `file=` / `filename=` attributes, and the only structural
    element that matters is MuJoCo's `<compiler meshdir=...>` prefix.
    """
    text = model.read_text()
    if model.suffix.lower() == ".urdf":
        # `package://<name>/rest` and bare relative paths both resolve beside the URDF.
        names = [re.sub(r"^package://[^/]+/", "", n) for n in re.findall(r'filename="([^"]+)"', text)]
        base = model.parent
        return {(base / name).resolve() for name in names}

    meshdir_match = re.search(r'<compiler\b[^>]*\bmeshdir="([^"]*)"', text, re.DOTALL)
    meshdir = meshdir_match.group(1) if meshdir_match else ""
    referenced = set()
    # `<mesh file=...>`, `<hfield file=...>`, `<texture file=...>` -- all meshdir-relative.
    for name in re.findall(r'\bfile="([^"]+)"', text):
        referenced.add((model.parent / meshdir / name).resolve())
    # `<include file=...>` is relative to the including file, not to meshdir.
    for name in re.findall(r'<include\b[^>]*\bfile="([^"]+)"', text, re.DOTALL):
        referenced.add((model.parent / name).resolve())
    return referenced


def _robot_asset_configs() -> list[dict]:
    """Every `RobotAssetConfig(...)` literal in `config_values/robot.py`, as kwarg dicts.

    Read out of the source with `ast` rather than by importing: `config_values` pulls in
    torch, pydantic and tyro, none of which this file has ever needed.
    """
    source = (DISTRIBUTIONS["holosoma"] / "holosoma" / "config_values" / "robot.py").read_text()
    configs = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if name != "RobotAssetConfig":
            continue
        kwargs = {
            kw.arg: kw.value.value
            for kw in node.keywords
            if kw.arg is not None and isinstance(kw.value, ast.Constant)
        }
        configs.append(kwargs)
    return configs


def _resolved_preset_assets() -> dict[str, set[str]]:
    """Package-relative paths the shipped robot presets resolve, keyed by the model naming them.

    This is the whole point of the section: the required set is *derived* from the presets
    and followed transitively into the models they name, so adding a robot, renaming a
    mesh or switching asset format cannot leave the check behind.
    """
    package = DISTRIBUTIONS["holosoma"] / "holosoma"
    resolved: dict[str, set[str]] = {}
    for config in _robot_asset_configs():
        asset_root = config.get("asset_root", "")
        # `@holosoma` -> the *installed* package directory. Four call sites do this same
        # rewrite: `simulator/mujoco/scene_manager.py:340-341` (the one the documented
        # steps 4 and 6 run through), the two other simulator backends, and
        # `envs/locomotion/locomotion_manager.py:121-122`.
        if not asset_root.startswith("@holosoma/"):
            continue
        root = package.parent / asset_root.replace("@holosoma/", "holosoma/", 1)
        for field in ("urdf_file", "xml_file", "usd_file"):
            declared = config.get(field)
            if not declared:
                continue
            model = (root / declared).resolve()
            key = f"{field}={declared}"
            entries = {model} | _referenced_assets(model) if model.is_file() else {model}
            resolved[key] = {
                path.relative_to(package.parent.resolve()).as_posix()
                for path in entries
                if package.resolve() in path.parents
            }
    return resolved


def test_the_preset_asset_walker_has_not_rotted() -> None:
    """Fail-closed companion: an empty resolved set would make the checks below vacuous."""
    resolved = _resolved_preset_assets()
    assert len(_robot_asset_configs()) >= 2, "no RobotAssetConfig literals found in config_values/robot.py"
    every = {path for paths in resolved.values() for path in paths}
    suffixes = {Path(path).suffix.lower() for path in every}
    assert ".xml" in suffixes, "no MuJoCo XML resolved -- the walker or the presets have moved"
    assert ".urdf" in suffixes, "no URDF resolved -- the walker or the presets have moved"
    assert len(every) > 20, f"only {len(every)} asset files resolved; the transitive walk is not running"


def test_every_asset_a_robot_preset_resolves_exists_in_the_source_tree() -> None:
    missing = sorted(
        path
        for paths in _resolved_preset_assets().values()
        for path in paths
        if not (DISTRIBUTIONS["holosoma"] / path).is_file()
    )
    assert not missing, f"robot presets name these files, which are not in the source tree at all: {missing}"


def test_every_asset_a_robot_preset_resolves_is_in_the_built_wheel() -> None:
    """The defect this section exists for, stated as an invariant.

    A preset that names `t1/t1_23dof.xml` and a `MANIFEST.in` that ships `*.STL`,
    `*.urdf`, `*.yaml`, `*.npz` and `*.obj` are individually reasonable and jointly
    broken. Nothing else in this repository can see the gap, because every documented
    environment installs `holosoma` editable and an editable install reads the checkout.

    This test fails against the pre-change `MANIFEST.in`, naming all five XML files.
    """
    root = DISTRIBUTIONS["holosoma"]
    selected = _manifest_selected(root, "holosoma")
    unpackaged: dict[str, list[str]] = {}
    for source_of, paths in _resolved_preset_assets().items():
        absent = sorted(path for path in paths if path not in selected)
        if absent:
            unpackaged[source_of] = absent
    assert not unpackaged, (
        "src/holosoma/MANIFEST.in does not select these files, so a built wheel does not contain "
        f"them, and the robot preset that names each one resolves to nothing once installed: {unpackaged}. "
        "Editable installs hide this entirely."
    )


# ---------------------------------------------------------------------------
# The general case, not just the assets some preset happens to name today.
# ---------------------------------------------------------------------------

#: Tracked non-Python files under `src/holosoma/holosoma` that are deliberately NOT
#: packaged, with the reason. Empty on purpose: everything `holosoma` tracks under its
#: package directory is a runtime asset, and the distribution declares no docs, scripts
#: or service files there (contrast `holosoma_inference`, which tracks READMEs, shell
#: scripts and a Dockerfile beside its code, and `holosoma_retargeting`, whose demo
#: assets are deliberately unpackaged -- neither is reached by the six documented steps).
#: An entry here is a claim that no documented step resolves the file.
HOLOSOMA_UNPACKAGED_DATA: dict[str, str] = {}


def _tracked_files(dist_root: Path, top: str) -> list[str] | None:
    """`git ls-files` under `dist_root/top`, or None when git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(dist_root), "ls-files", "-z", "--", top],
            capture_output=True,
            check=False,
        )
    except (OSError, ValueError):  # pragma: no cover - git absent
        return None
    if completed.returncode != 0:
        return None
    names = [name for name in completed.stdout.decode().split("\0") if name]
    return names or None


def test_the_holosoma_wheel_carries_every_tracked_data_file_and_nothing_untracked() -> None:
    """Generalises the three wheel-content defects this repository has now shipped.

    42 missing modules, then six missing ONNX files, then five missing robot XMLs: each
    was a different mechanism, and each was found by someone installing a wheel rather
    than by a test. The invariant that covers all three without naming any of them is
    *the wheel's non-Python content equals the tracked non-Python content of the package
    directory* -- both directions, because the reverse failure is real too: `MANIFEST.in`
    was sweeping a `.gitignore`d IsaacSim USD-conversion cache (`converted_rank0/`,
    containing absolute paths from the machine that built the wheel) into every build.

    Skipped rather than failed without git: an unpacked sdist has no way to know which
    files were tracked, and the checks above do not need git to catch the real defect.
    """
    root = DISTRIBUTIONS["holosoma"]
    tracked = _tracked_files(root, "holosoma")
    if tracked is None:
        pytest.skip("git is unavailable or this is not a checkout; the preset-driven checks still apply")
    # `pytest.skip` is `NoReturn`, but only in stubs newer than the pinned mypy resolves.
    assert tracked is not None

    tracked_data = {name for name in tracked if not name.endswith(".py")}
    selected = _manifest_selected(root, "holosoma")

    # setuptools packages `py.typed` on its own, without a MANIFEST.in entry.
    absent = sorted(tracked_data - selected - set(HOLOSOMA_UNPACKAGED_DATA) - {"holosoma/py.typed"})
    assert not absent, (
        f"these files are tracked under src/holosoma/holosoma but src/holosoma/MANIFEST.in does not "
        f"select them, so no built wheel contains them: {absent}. Either ship them or add each one to "
        "HOLOSOMA_UNPACKAGED_DATA with the reason no documented step resolves it."
    )

    extra = sorted(name for name in selected - set(tracked) if not name.endswith(".py"))
    assert not extra, (
        f"src/holosoma/MANIFEST.in selects these untracked files, so they ship in any wheel built from "
        f"a working checkout: {extra}. Generated caches and local artifacts must be pruned, not shipped."
    )


# ---------------------------------------------------------------------------
# Declared dependencies vs. imported ones.
# ---------------------------------------------------------------------------

#: import name -> distribution name, where they differ.
_DISTRIBUTION_OF_MODULE = {
    "pinocchio": "pin",  # conda-forge `pinocchio` and PyPI `pin` both provide it
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
}


def _module_level_import_graph(root: Path, entry_module: str, top: str) -> set[str]:
    """Top-level third-party modules reachable from `entry_module` via module-scope imports.

    Module-scope only: an import inside a function is a branch a program may never take,
    but everything walked here runs the moment the entry point is imported.
    """

    def resolve(module: str) -> Path | None:
        candidate = root / (module.replace(".", "/") + ".py")
        if candidate.is_file():
            return candidate
        candidate = root / module.replace(".", "/") / "__init__.py"
        return candidate if candidate.is_file() else None

    seen: set[str] = set()
    external: set[str] = set()
    stack = [entry_module]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        path = resolve(module)
        if path is None:
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == top:
                        stack.append(alias.name)
                    else:
                        external.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.rsplit(".", node.level)[0] if "." in module else top
                    target = f"{base}.{node.module}" if node.module else base
                    stack.append(target)
                    stack.extend(f"{target}.{alias.name}" for alias in node.names)
                elif node.module:
                    if node.module.split(".")[0] == top:
                        stack.append(node.module)
                        stack.extend(f"{node.module}.{alias.name}" for alias in node.names)
                    else:
                        external.add(node.module.split(".")[0])
    return external - set(sys.stdlib_module_names)


def _install_requires(setup_py: Path) -> set[str]:
    tree = ast.parse(setup_py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "install_requires" and isinstance(node.value, ast.List):
            names = set()
            for element in node.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    # "pin>=3.8.0" -> "pin"; normalized the way pip normalizes names.
                    names.add(re.split(r"[<>=!~\[; ]", element.value, 1)[0].strip().lower().replace("_", "-"))
            return names
    return set()


#: Imports that are deliberately NOT declared, with the reason each one cannot be.
#: Every entry here is a promise that the requirement is documented somewhere a reader
#: will hit before running the code; `test_undeclarable_requirements_are_documented`
#: checks that promise rather than taking it on faith.
UNDECLARABLE_IMPORTS = {
    # `holosoma` names a *different*, unrelated project on PyPI (`holosoma==0.0.1`,
    # 558 files, none under `fada`). A bare `holosoma` requirement resolves to it,
    # so declaring it makes `pip install holosoma-inference` succeed, `pip check`
    # pass, and `run_policy.py` die at import with
    # `ModuleNotFoundError: No module named 'holosoma.fada'`. A direct-URL
    # requirement would fix resolution but make the wheel unpublishable to PyPI.
    "holosoma": "must be installed from this repository (`pip install -e src/holosoma`)",
}


def test_run_policy_imports_nothing_undeclared() -> None:
    root = DISTRIBUTIONS["holosoma_inference"]
    imported = _module_level_import_graph(root, "holosoma_inference.run_policy", "holosoma_inference")
    assert imported, "the import walker found nothing -- it has rotted"

    declared = _install_requires(root / "setup.py")
    undeclared = sorted(
        module
        for module in imported
        if module not in UNDECLARABLE_IMPORTS
        and _DISTRIBUTION_OF_MODULE.get(module, module).lower().replace("_", "-") not in declared
    )
    assert not undeclared, (
        "holosoma_inference/run_policy.py imports these at module scope -- i.e. before the "
        f"first line of main(), on every invocation -- and setup.py declares none of them: {undeclared}. "
        "`pip install holosoma-inference` therefore produces an environment that cannot run the "
        "documented step-4 command, and `pip check` cannot see the problem because nothing says "
        "they are required."
    )


def test_no_requirement_resolves_to_a_different_project_of_the_same_name() -> None:
    """A requirement that pip satisfies with the wrong package is worse than a missing one.

    `holosoma` is the demonstrated case: PyPI's `holosoma==0.0.1` has no `fada`
    subpackage, so declaring the bare name buys a green `pip check` and a broken
    import. This test fails against the pre-change setup.py, which declared it.
    """
    declared = _install_requires(DISTRIBUTIONS["holosoma_inference"] / "setup.py")
    collisions = sorted(name for name in UNDECLARABLE_IMPORTS if name in declared)
    assert not collisions, (
        f"holosoma_inference/setup.py declares {collisions}, which PyPI resolves to an unrelated "
        "distribution of the same name. Either make the dependency genuinely resolvable or leave it "
        "undeclared and documented -- do not declare a name that resolves to the wrong project."
    )


@pytest.mark.parametrize("module", sorted(UNDECLARABLE_IMPORTS))
def test_undeclarable_requirements_are_documented(module: str) -> None:
    """Not declaring a real requirement is only acceptable if the reader is told.

    Two places have to say it, because they are the two ways a reader arrives: the
    file that would otherwise have declared it, and the install path the README
    points at.
    """
    setup_py = (DISTRIBUTIONS["holosoma_inference"] / "setup.py").read_text()
    assert re.search(rf"NOT DECLARED.*\n(?:.*\n)*?.*{re.escape(module)}", setup_py, re.IGNORECASE), (
        f"setup.py omits `{module}` without explaining why; the next reader will 'fix' it by adding it back"
    )

    setup_script = (REPO_ROOT / "scripts" / "setup_inference.sh").read_text()
    assert re.search(rf"pip install.*-e.*src/{re.escape(module)}\b", setup_script), (
        f"`{module}` is undeclared, so the documented install path is the only thing that installs it -- "
        f"and scripts/setup_inference.sh does not appear to."
    )


def test_the_numpy_requirement_is_not_a_pin_no_environment_can_satisfy() -> None:
    """`numpy==1.23.5` was false in two of the three shipped environments.

    hsmujoco runs numpy 2.x (it is where the bitexact probes run) and IsaacSim 5.1
    requires `numpy==1.26.0`; an equality pin at 1.23.5 makes both of those environments
    `pip check`-broken, and `scripts/setup_isaacsim.sh` only got away with it by passing
    `--no-deps`. Upstream's own published wheel declares a range, not a pin.
    """
    pyproject = (REPO_ROOT / "src" / "holosoma" / "pyproject.toml").read_text()
    requirements = re.findall(r'^\s*"(numpy[^"]*)"', pyproject, re.MULTILINE)
    numpy_requirements = [r for r in requirements if re.match(r"^numpy(?![-_a-zA-Z])", r)]
    assert len(numpy_requirements) == 1, numpy_requirements
    requirement = numpy_requirements[0]

    assert "==" not in requirement, (
        f"src/holosoma/pyproject.toml declares {requirement!r}. An equality pin cannot be true in "
        "both the mujoco environment (numpy 2.x) and an IsaacSim 5.1 environment "
        "(isaacsim-kernel requires numpy==1.26.0)."
    )
    # The range must actually admit the two versions that are known to work and the one
    # IsaacSim demands -- "not a pin" alone would be satisfied by any nonsense.
    from packaging.requirements import Requirement
    from packaging.version import Version

    specifier = Requirement(requirement).specifier
    for version in ("1.23.5", "1.26.0", "2.2.6"):
        assert Version(version) in specifier, f"{requirement} excludes numpy {version}"


# ---------------------------------------------------------------------------
# A declared `requires-python` must admit the declared dependencies.
# ---------------------------------------------------------------------------

#: NumPy minor series -> the `Requires-Python` floor its PyPI wheels declare.
#:
#: Hardcoded rather than queried, so this test needs no network and runs in the same gate
#: as everything else. Sourced from PyPI (`pip index`/JSON metadata); the failure it exists
#: to catch is a *pin* moving above the declared interpreter floor, and a pin only moves
#: when someone edits it, at which point this table is right next to the assertion.
_NUMPY_SERIES_PYTHON_FLOOR = {
    (1, 23): "3.8",
    (1, 24): "3.8",
    (1, 25): "3.9",
    (1, 26): "3.9",
    (2, 0): "3.9",
    (2, 1): "3.10",
    (2, 2): "3.10",
    (2, 3): "3.11",
    (2, 4): "3.11",
    (2, 5): "3.12",
}

#: Distribution root -> the file that declares its `requires-python`.
_PYTHON_FLOOR_DECLARATIONS = {
    "holosoma_retargeting": ("pyproject.toml", "setup.py"),
}


def _declared_python_floor(text: str) -> str:
    """The lower bound of the single `requires-python` / `python_requires` in `text`."""
    from packaging.specifiers import SpecifierSet

    match = re.search(r'(?:requires-python|python_requires)\s*=\s*"([^"]+)"', text)
    assert match, "no requires-python / python_requires declaration found"
    lower = [s for s in SpecifierSet(match.group(1)) if s.operator in (">=", "==", "~=")]
    assert len(lower) == 1, f"expected exactly one lower bound, got {match.group(1)!r}"
    return lower[0].version


@pytest.mark.parametrize("top", sorted(_PYTHON_FLOOR_DECLARATIONS))
def test_requires_python_admits_the_pinned_numpy(top: str) -> None:
    """`requires-python = ">=3.10"` alongside `numpy==2.3.5` is a promise that cannot be kept.

    NumPy 2.3.x declares `Requires-Python: >=3.11`, so a 3.10 install of this distribution
    fails at resolution with "No matching distribution found for numpy==2.3.5" -- the
    package advertises support for an interpreter on which it cannot be installed at all.
    This test fails against the pre-change declaration.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    root = REPO_ROOT / "src" / top
    for filename in _PYTHON_FLOOR_DECLARATIONS[top]:
        text = (root / filename).read_text()
        floor = Version(_declared_python_floor(text))

        pins = [r for r in re.findall(r'"(numpy[^"]*)"', text) if re.match(r"^numpy(?![-_a-zA-Z])", r)]
        assert len(pins) == 1, f"{top}/{filename}: expected one numpy requirement, got {pins}"
        specifier = Requirement(pins[0]).specifier

        for series, series_floor in sorted(_NUMPY_SERIES_PYTHON_FLOOR.items()):
            # Only series the requirement actually admits constrain the interpreter floor.
            if not any(Version(f"{series[0]}.{series[1]}.{patch}") in specifier for patch in range(0, 8)):
                continue
            if Version(series_floor) <= floor:
                break
        else:
            worst = max(
                Version(v)
                for series, v in _NUMPY_SERIES_PYTHON_FLOOR.items()
                if any(Version(f"{series[0]}.{series[1]}.{p}") in specifier for p in range(0, 8))
            )
            pytest.fail(
                f"src/{top}/{filename} declares python {floor} and {pins[0]!r}, but every NumPy "
                f"release that requirement admits needs python >= {worst}. On {floor} the "
                "dependency set is unsatisfiable: 'No matching distribution found'."
            )


@pytest.mark.parametrize("top", sorted(_PYTHON_FLOOR_DECLARATIONS))
def test_setup_py_and_pyproject_agree_on_the_python_floor(top: str) -> None:
    """Two files stating the interpreter floor is two places for it to be wrong."""
    root = REPO_ROOT / "src" / top
    floors = {name: _declared_python_floor((root / name).read_text()) for name in _PYTHON_FLOOR_DECLARATIONS[top]}
    assert len(set(floors.values())) == 1, f"{top}: disagreeing python floors {floors}"


# ---------------------------------------------------------------------------
# `readme` must name a file that exists.
#
# `src/holosoma_retargeting/pyproject.toml` declared `readme = "README.md"` and there
# is no such file at that level -- the document lives one directory down, beside the
# code. setuptools does not fail on that: it emits `SetuptoolsWarning: File
# '.../README.md' cannot be found` and builds anyway, so the wheel shipped with an
# empty long_description (27-line METADATA against 262 once the path is right) and
# `pip show` / any index page rendered blank. A warning in a build log is exactly the
# kind of defect no other check in this repository can see, which is why it is
# asserted rather than watched for.
# ---------------------------------------------------------------------------


def _pyprojects_declaring_a_readme() -> list[Path]:
    found = []
    for path in sorted((REPO_ROOT / "src").glob("*/pyproject.toml")):
        if re.search(r'^\s*readme\s*=\s*"', path.read_text(), re.M):
            found.append(path)
    return found


def test_at_least_one_distribution_declares_a_readme() -> None:
    """Fail-closed companion: a parametrization over an empty list passes vacuously."""
    assert _pyprojects_declaring_a_readme(), (
        "no src/*/pyproject.toml declares `readme`; the check below would be silently vacuous"
    )


@pytest.mark.parametrize(
    "pyproject", _pyprojects_declaring_a_readme(), ids=lambda p: p.parent.name
)
def test_the_declared_readme_resolves_to_a_real_file(pyproject: Path) -> None:
    match = re.search(r'^\s*readme\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    assert match is not None
    declared = match.group(1)
    target = pyproject.parent / declared

    assert target.is_file(), (
        f"{pyproject.relative_to(REPO_ROOT)} declares readme = {declared!r}, which does not exist. "
        "setuptools only warns about this and then builds a wheel with no long description."
    )
    # A README that exists but is empty produces the same blank metadata by another route.
    assert target.read_text(encoding="utf-8").strip(), f"{target.relative_to(REPO_ROOT)} is empty"
