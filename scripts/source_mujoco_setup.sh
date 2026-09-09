#!/bin/bash
# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

# Use CONDA_ENV_NAME if provided, otherwise default to "hsmujoco"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsmujoco}
echo "conda environment name is set to: $CONDA_ENV_NAME"

source ${SCRIPT_DIR}/source_common.sh
unset CONDA_ENVS_PATH
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"

# Set MuJoCo-specific environment variables
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT}/envs/$CONDA_ENV_NAME/lib

# MuJoCo-specific environment variables (if needed)
export MUJOCO_GL=egl  # For headless rendering
# export MUJOCO_GL=osmesa  # Alternative headless option

# Validate environment is properly activated
if python -c "import mujoco" 2>/dev/null; then
    echo "MuJoCo environment activated successfully"
    echo "MuJoCo version: $(python -c 'import mujoco; print(mujoco.__version__)')"
    if python -c "import torch" 2>/dev/null; then
        echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    else
        echo "Warning: PyTorch not installed. Run: bash scripts/setup_mujoco.sh"
    fi

    # Print mujoco-warp commit if installed
    if python -c "import mujoco_warp" 2>/dev/null; then
        MUJOCO_WARP_COMMIT=$(git -C ${WORKSPACE_DIR}/mujoco_warp rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "MuJoCo Warp commit: ${MUJOCO_WARP_COMMIT}"
    fi
else
    echo "Warning: MuJoCo environment activation may have issues"
fi
