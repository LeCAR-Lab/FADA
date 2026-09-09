"""Where holosoma-retargeting's data and robot models live, and what to say when they don't.

This package's assets are not in its wheel: `demo_data/` and `models/` (~293 MiB of
URDFs, meshes and jinja templates) ship only in the source checkout.

Every documented retargeting command is checkout-relative, not install-relative:

    cd src/holosoma_retargeting/holosoma_retargeting
    python examples/robot_retarget.py --data_path demo_data/OMOMO_new ...

-- the `examples/` scripts, the `models/g1/g1_29dof.urdf` argument, the `demo_results/`
output directory and the LAFAN `git clone` step in the README all resolve against the
package's *source directory* as the working directory.

The config defaults (`RetargetingConfig.data_path = Path("demo_data/OMOMO_new")`,
`RobotConfig.robot_urdf_file = "models/<robot>/..."`) are therefore not package-relative
paths. From a wheel install they resolve against the working directory;
`require_asset_dir` reports that case by name instead of leaving a bare
`FileNotFoundError` inside a loader or an empty glob.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ASSET_DIR_NAMES",
    "DATA_ASSET_DIRS",
    "MODEL_ASSET_DIRS",
    "RetargetingAssetsUnavailable",
    "assets_present",
    "package_root",
    "require_asset_dir",
    "require_asset_file",
]

# The two asset trees this package ships in the source checkout and omits from the wheel.
ASSET_DIR_NAMES = ("demo_data", "models")

# The two trees back different arguments. `--data_path` may point outside the package (a
# user's own motion capture), in which case `demo_data/` being absent is irrelevant;
# `models/` is still needed for the URDF.
DATA_ASSET_DIRS = ("demo_data",)
MODEL_ASSET_DIRS = ("models",)


class RetargetingAssetsUnavailable(FileNotFoundError):
    """A documented default (or an explicit path) does not resolve from here."""


def package_root() -> Path:
    """Directory of the `holosoma_retargeting` package itself.

    In a source checkout or an editable install this is also the directory every
    documented command expects to be the working directory.
    """
    return Path(__file__).resolve().parent


def assets_present(root: Path | None = None) -> bool:
    """True when `root` (default: the package directory) carries the asset trees.

    False in any non-editable install, which is the case this module exists to explain.
    """
    base = package_root() if root is None else Path(root)
    return all((base / name).is_dir() for name in ASSET_DIR_NAMES)


def _explain(resolved: Path, what: str, asset_dirs: tuple[str, ...]) -> str:
    lines = [f"{what} does not exist: {resolved}"]
    if not resolved.is_absolute():
        lines.append(f"  (resolved against the current working directory, {Path.cwd()})")
    missing = [name for name in asset_dirs if not (package_root() / name).is_dir()]
    if missing:
        lines += [
            "",
            f"holosoma_retargeting is installed at {package_root()}, and that directory is",
            f"missing {missing}. The retargeting assets (~293 MiB of motion data, URDFs and",
            "meshes) are intentionally NOT packaged into the wheel; they exist only in a source",
            "checkout of this repository.",
            "",
            "Every documented retargeting command is run from the package's source directory:",
            "",
            "    cd <repo>/src/holosoma_retargeting/holosoma_retargeting",
            "    python examples/robot_retarget.py --data_path demo_data/OMOMO_new ...",
            "",
            "See holosoma_retargeting/README.md.",
        ]
    else:
        lines += [
            "",
            f"The asset tree(s) {list(asset_dirs)} ARE present at {package_root()}, so this is a",
            "wrong path rather than a missing install. Note the documented commands are run with",
            "that directory as the working directory.",
        ]
    return "\n".join(lines)


def require_asset_dir(
    path: str | os.PathLike[str],
    *,
    what: str = "path",
    asset_dirs: tuple[str, ...] = DATA_ASSET_DIRS,
) -> Path:
    """Return `path` if it exists; otherwise raise naming the likely cause.

    Does not rewrite the path to a package-relative one; it only reports.

    `asset_dirs` names the asset tree(s) whose absence would explain *this* argument
    failing, and defaults to `demo_data/`, the tree a data path comes from. The URDF
    comes from `models/` and is checked separately, by :func:`require_asset_file`; a
    passing call here says nothing about that tree.
    """
    resolved = Path(path)
    if resolved.exists():
        return resolved
    raise RetargetingAssetsUnavailable(_explain(resolved, what, asset_dirs))


def require_asset_file(
    path: str | os.PathLike[str],
    *,
    what: str = "path",
    asset_dirs: tuple[str, ...] = MODEL_ASSET_DIRS,
) -> Path:
    """Same contract as :func:`require_asset_dir`, for a single file (a URDF, a mesh).

    Separate from `require_asset_dir` only so the caller states which of the two asset
    trees it depends on and the message names the right one; `Path.exists()` does not
    distinguish, so a directory passed here still passes.
    """
    resolved = Path(path)
    if resolved.exists():
        return resolved
    raise RetargetingAssetsUnavailable(_explain(resolved, what, asset_dirs))
