#!/usr/bin/env bash
# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

echo "Setting up inference environment"

if ! command -v sudo &> /dev/null; then
  # in docker build sudo isn't avaiable, but its ok
  echo "Warning: sudo could not be found, you may need to run this script with sudo"
  function sudo { "$@"; }
  export -f sudo
fi

OS=$(uname -s)
ARCH=$(uname -m)

case $ARCH in
  "aarch64"|"arm64") ARCH="aarch64" ;;
  "x86_64") ARCH="x86_64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

case $OS in
  "Linux")
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
    PACKAGE_MANAGER="apt-get"
    INSTALL_CMD="sudo apt-get install -y"
    ;;
  "Darwin")
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    PACKAGE_MANAGER="brew"
    INSTALL_CMD="brew install"
    ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Swap CPU onnxruntime for onnxruntime-gpu so task.onnx_provider=cuda|tensorrt gets
# CUDAExecutionProvider / TensorrtExecutionProvider. Opt out: HSINFERENCE_SKIP_ONNX_GPU=1.
# Skipped on macOS and Linux aarch64 (Jetson uses different ORT builds).
install_onnxruntime_gpu_linux_x86_64() {
  local py="$1"
  [[ "${HSINFERENCE_SKIP_ONNX_GPU:-0}" == "1" ]] && return 0
  [[ "$OS" != "Linux" || "$ARCH" != "x86_64" ]] && return 0
  echo "Installing onnxruntime-gpu (CUDA / TensorRT execution providers)..."
  "$py" -m pip uninstall -y onnxruntime 2>/dev/null || true
  "$py" -m pip install 'onnxruntime-gpu>=1.23.0,<1.24'
  echo "Installing TensorRT runtime for onnx_provider=tensorrt..."
  "$py" -m pip install 'tensorrt>=10.0,<11'
}

# Create overall workspace
# Use CONDA_ENV_NAME if provided, otherwise default to "hsinference"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsinference}
echo "conda environment name is set to: $CONDA_ENV_NAME"

source ${SCRIPT_DIR}/source_common.sh
# A global CONDA_ENVS_PATH makes conda/mamba create envs outside $CONDA_ROOT/envs and breaks this script.
unset CONDA_ENVS_PATH
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME

SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_$CONDA_ENV_NAME

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda first so Linux can get swig from conda-forge without sudo.
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl $MINICONDA_URL -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi
  export PATH="$CONDA_ROOT/bin:$PATH"

  # Swig (holosoma_inference build dependency)
  if [[ $OS == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      echo >> $HOME/.zprofile
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> $HOME/.zprofile
      eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
  fi
  if ! command -v swig &> /dev/null; then
    if [[ $OS == "Linux" ]]; then
      $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
      MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba install -y -n base -c conda-forge swig
    fi
  fi
  if ! command -v swig &> /dev/null; then
    $INSTALL_CMD swig
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    # -p forces prefix under $CONDA_ROOT/envs even if ~/.condarc sets envs_dirs elsewhere.
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -p "$ENV_ROOT" python=3.10 -c conda-forge --override-channels
  fi

  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$ENV_ROOT"

  # Install libstdcxx-ng to fix the error: `version `GLIBCXX_3.4.32' not found` on Ubuntu 24.04
  # Only needed on Linux (not macOS)
  if [[ $OS == "Linux" ]]; then
    conda install -p "$ENV_ROOT" -c conda-forge -y libstdcxx-ng
  fi

  # ORDERING NOTE. holosoma_inference is installed LAST in this block, after holosoma
  # and after pinocchio. That is not cosmetic: holosoma_inference/setup.py now declares
  # what it actually imports at run_policy.py import time, which includes `holosoma`
  # (policies/locomotion_fada.py) and `pin` (policies/wbt_utils.py). Installed first, pip
  # would go to PyPI to satisfy both -- fetching the *upstream* holosoma wheel, which has
  # no `fada` subpackage, and a `pin` wheel that the conda-forge pinocchio below then
  # installs over. Installed last, both requirements are already satisfied locally and
  # pip touches no index for them.

  # run_policy / transformer / FADA policies import holosoma (e.g. fada.planner_idm).
  echo "Installing holosoma (policy import dependency)"
  if [[ $OS == "Darwin" ]]; then
    "$ENV_ROOT/bin/python" -m pip install -e "$ROOT_DIR/src/holosoma[unitree]"
  else
    "$ENV_ROOT/bin/python" -m pip install -e "$ROOT_DIR/src/holosoma[unitree,booster]"
  fi
  # Setup a few things for ARM64 Linux (G1 Jetson)
  # Otherwise we get this error:
  # /opt/rh/gcc-toolset-14/root/usr/include/c++/14/bits/stl_vector.h:1130: ...
  if [[ $OS == "Linux" && $ARCH == "aarch64" ]]; then
    sudo nvpmodel -m 0 2>/dev/null || true
    "$ENV_ROOT/bin/python" -m pip install "pin>=3.8.0"
  else
    if [[ ! -d $WORKSPACE_DIR/unitree_sdk2_python ]]; then
      git clone https://github.com/unitreerobotics/unitree_sdk2_python.git $WORKSPACE_DIR/unitree_sdk2_python
    fi
    "$ENV_ROOT/bin/python" -m pip install -e $WORKSPACE_DIR/unitree_sdk2_python/
    # conda-forge's `pinocchio` ships a pip-visible `pin-<version>.dist-info`, which is
    # what makes holosoma_inference's `pin>=3.8.0` requirement resolve here.
    $CONDA_ROOT/bin/conda install -p "$ENV_ROOT" pinocchio -y -c conda-forge --override-channels
  fi

  # Install holosoma_inference
  # Note: On macOS, only Unitree SDK is supported (Booster SDK is Linux-only)
  if [[ $OS == "Darwin" ]]; then
    echo "Note: Installing Unitree SDK only (Booster SDK is not supported on macOS)"
    "$ENV_ROOT/bin/python" -m pip install -e "$ROOT_DIR/src/holosoma_inference[unitree]"
  else
    "$ENV_ROOT/bin/python" -m pip install -e "$ROOT_DIR/src/holosoma_inference[unitree,booster]"
  fi

  install_onnxruntime_gpu_linux_x86_64 "$ENV_ROOT/bin/python"

  cd $ROOT_DIR
  touch $SENTINEL_FILE
fi

# holosoma_inference imports torch; ensure it exists (covers envs created before torch was declared).
if [[ -d "$ENV_ROOT" && -x "$CONDA_ROOT/bin/conda" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$ENV_ROOT"
  if ! "$ENV_ROOT/bin/python" -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch (required by holosoma_inference)..."
    "$ENV_ROOT/bin/python" -m pip install 'torch>=2.0.0'
  fi
  if ! "$ENV_ROOT/bin/python" -c "import holosoma" 2>/dev/null; then
    echo "Installing holosoma (required by holosoma_inference policy stack)..."
    if [[ $OS == "Darwin" ]]; then
      "$ENV_ROOT/bin/python" -m pip install -e "$ROOT_DIR/src/holosoma[unitree]"
    else
      "$ENV_ROOT/bin/python" -m pip install -e "$ROOT_DIR/src/holosoma[unitree,booster]"
    fi
  fi
  if [[ $OS == "Linux" && $ARCH == "x86_64" && "${HSINFERENCE_SKIP_ONNX_GPU:-0}" != "1" ]]; then
    if "$ENV_ROOT/bin/python" -c "import onnxruntime as ort" 2>/dev/null; then
      if ! "$ENV_ROOT/bin/python" -c \
        "import onnxruntime as ort, sys; sys.exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" \
        2>/dev/null; then
        echo "ONNX Runtime lacks CUDAExecutionProvider; installing onnxruntime-gpu..."
        install_onnxruntime_gpu_linux_x86_64 "$ENV_ROOT/bin/python"
      fi
    fi
    # Ensure TensorRT runtime is installed (may be missing in envs created before TRT support)
    if ! "$ENV_ROOT/bin/python" -c "import tensorrt" 2>/dev/null; then
      echo "TensorRT not found; installing..."
      "$ENV_ROOT/bin/python" -m pip install 'tensorrt>=10.0,<11'
    fi
  fi
fi
