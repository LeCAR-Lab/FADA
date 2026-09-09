from setuptools import find_packages, setup  # type: ignore[import-untyped]

# No `package_data` / `include_package_data`: this package's assets
# (`holosoma_retargeting/demo_data/`, `holosoma_retargeting/models/`, ~293 MiB) are not
# packaged into the wheel.
#
# The defaults that name those directories (`RetargetingConfig.data_path =
# Path("demo_data/OMOMO_new")`, `RobotConfig.robot_urdf_file = "models/<robot>/..."`) are
# NOT package-relative: every documented retargeting command runs with the package's
# source directory as the working directory and writes its results back into that
# directory (`demo_results/`, `converted_data/`), and one of them has the reader
# `git clone` a third-party repository into `data_utils/`. holosoma-retargeting is run
# from a checkout, not installed and imported.
#
# `holosoma_retargeting/asset_paths.py` turns "default path does not resolve from a wheel
# install" into a message naming the cause; both documented entry points call it.
# src/holosoma_inference/setup.py does package its ONNX, whose default path is
# package-relative.

setup(
    name="holosoma-retargeting",
    version="0.1.0",
    description="holosoma-retargeting: retargeting components for converting human motions to robot motions",
    author="Amazon FAR Team",
    packages=find_packages(),
    # Must match pyproject.toml's `requires-python`. The `numpy==2.3.5` pin below rules
    # out 3.10.
    python_requires=">=3.11",
    install_requires=[
        # Needs to ping numpy to 2.3.5;
        # reason: later numpy version such as 2.4 will trigger
        # "TypeError: only 0-dimensional arrays can be converted to Python scalars"
        # in yourdf/urdf.py::1078 when converting float(q)
        "numpy==2.3.5",
        # Security floor; must match pyproject.toml. Below torch 2.6, `src/utils.py`'s
        # bare `torch.load` on a downloaded InterMimic `.pt` executes the file's pickle
        # stream.
        "torch>=2.6",
        "tqdm",
        "scipy",
        "matplotlib",
        "trimesh",
        "smplx",
        "jinja2",
        "mujoco",
        "viser",
        "robot_descriptions",
        "yourdfpy",
        "cvxpy",
        "libigl",
        "tyro",
    ],
)
