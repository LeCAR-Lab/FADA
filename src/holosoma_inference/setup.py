import platform

from setuptools import find_packages, setup

# Default pip install uses CPU onnxruntime. For GPU inference (CUDAExecutionProvider),
# use scripts/setup_inference.sh on Linux x86_64, or: pip uninstall -y onnxruntime &&
# pip install onnxruntime-gpu (holosoma also declares onnxruntime; uninstall CPU wheel first).

UNITREE_VERSION = "0.1.1"
UNITREE_REPO = "https://github.com/amazon-far/unitree_sdk2"
BOOSTER_VERSION = "0.1.0"
BOOSTER_REPO = "https://github.com/amazon-far/booster_robotics_sdk"

PLATFORM_MAP = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
}

platform_tag = PLATFORM_MAP.get(platform.machine(), "linux_x86_64")

unitree_extras = []
unitree_url = (
    f"{UNITREE_REPO}/releases/download/{UNITREE_VERSION}/unitree_sdk2-{UNITREE_VERSION}-cp310-cp310-{platform_tag}.whl"
)
unitree_extras.append(f"unitree_sdk2 @ {unitree_url}")

booster_extras = []
booster_url = f"{BOOSTER_REPO}/releases/download/{BOOSTER_VERSION}/booster_robotics_sdk-{BOOSTER_VERSION}-cp310-cp310-{platform_tag}.whl"  # noqa: E501
booster_extras.append(f"booster_robotics_sdk @ {booster_url}")


setup(
    name="holosoma-inference",
    version="0.1.0",
    description="holosoma-inference: inference components for humanoid robot policies",
    long_description="",
    long_description_content_type="text/markdown",
    author="Amazon FAR Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pydantic",
        "loguru",
        "netifaces",
        "onnx",
        "onnxruntime",
        "scipy",
        "sshkeyboard",
        "termcolor",
        "torch>=2.0.0",
        "pyyaml",
        "tyro>=0.10.0a4",
        "wandb",
        "zmq",
        "defusedxml",
        "evdev",
        # --- imported at `run_policy.py` import time ------------------------------
        # None of these is behind an optional branch: run_policy.py imports the WBT
        # *and* FADA policy modules unconditionally at startup.
        "numpy",  # policies/base.py
        "typing_extensions",  # config/config_values/inference.py (Annotated)
        # policies/wbt_utils.py does `import pinocchio as pin`. The distribution that
        # provides that module is named `pin` on PyPI. Both documented install paths
        # satisfy it: setup_inference.sh pip-installs `pin>=3.8.0` on aarch64, and on
        # x86_64 it conda-installs `pinocchio` from conda-forge -- which ships a
        # pip-visible `pin-<version>.dist-info`, so `pip check` sees it as satisfied.
        "pin>=3.8.0",
        # --- `holosoma` is NOT DECLARED HERE -------------------------------------
        # `policies/locomotion_fada.py` imports `holosoma.fada.common.current_command`
        # at module scope, so holosoma is required at run time. It is not listed because
        # the name on PyPI belongs to an unrelated `holosoma==0.0.1` that contains no
        # `fada` package, and a direct-URL requirement (`holosoma @ git+https://...`)
        # cannot appear in metadata PyPI will accept.
        #
        # Install this repository's `src/holosoma` first (`pip install -e src/holosoma`,
        # which is what `scripts/setup_inference.sh` does, before installing this
        # package). `tests/test_packaging_metadata.py` pins both halves.
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "gpu": [
            "onnxruntime-gpu",
        ],
        # `utils/mocap_zmq_publisher.py` needs a Vicon DataStream client. Vicon hardware
        # is not part of the FADA pipeline, so this is an extra rather than a
        # requirement.
        "mocap": ["pyvicon-datastream"],
        "unitree": unitree_extras,
        "booster": booster_extras,
    },
    entry_points={
        "holosoma.sdk": [
            "unitree = holosoma_inference.sdk.unitree.unitree_interface:UnitreeInterface",
            "booster = holosoma_inference.sdk.booster.booster_interface:BoosterInterface",
        ],
        "holosoma.config.robot": [
            "g1-29dof = holosoma_inference.config.config_values.robot:g1_29dof",
            "t1-29dof = holosoma_inference.config.config_values.robot:t1_29dof",
        ],
        "holosoma.config.inference": [
            "g1-29dof-loco = holosoma_inference.config.config_values.inference:g1_29dof_loco",
            "t1-29dof-loco = holosoma_inference.config.config_values.inference:t1_29dof_loco",
            "g1-29dof-wbt = holosoma_inference.config.config_values.inference:g1_29dof_wbt",
        ],
    },
    keywords="humanoid robotics inference policy onnx",
    include_package_data=True,
    # `models/**/*.onnx` is not optional data: config/config_values/task.py builds
    # `_MODELS_DIR = Path(__file__).parent.parent.parent / "models"` and defaults
    # `safety_locomotion_g1.model_path` -- the dual-mode secondary policy -- to a file
    # inside it, so the ONNX files must be in the wheel for that default to resolve in a
    # non-editable install.
    package_data={
        "holosoma_inference": ["configs/**/*.yaml", "py.typed", "models/**/*.onnx"],
    },
)
