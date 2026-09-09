# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi
# Use CONDA_ENV_NAME if provided, otherwise default to "hsinference"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsinference}
echo "conda environment name is set to: $CONDA_ENV_NAME"

source ${SCRIPT_DIR}/source_common.sh
unset CONDA_ENVS_PATH
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
# Safe when LD_LIBRARY_PATH is unset (e.g. caller has set -u)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_ROOT}/envs/$CONDA_ENV_NAME/lib/python3.10/site-packages/lib"

# Make pip-installed CUDA/cuDNN runtime libraries visible to onnxruntime-gpu.
# Newer ORT GPU wheels may advertise CUDAExecutionProvider but still fail to create
# it unless these site-packages/nvidia/*/lib directories are on LD_LIBRARY_PATH.
_hs_sitepkg="${CONDA_ROOT}/envs/$CONDA_ENV_NAME/lib/python3.10/site-packages"
if [[ -d "$_hs_sitepkg/nvidia" ]]; then
    while IFS= read -r -d '' _libdir; do
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${_libdir}"
    done < <(find "$_hs_sitepkg/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print0 2>/dev/null)
fi
# TensorRT pip package places libnvinfer*.so under site-packages/tensorrt_libs/
if [[ -d "$_hs_sitepkg/tensorrt_libs" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${_hs_sitepkg}/tensorrt_libs"
fi
unset _hs_sitepkg _libdir

if ! python -c "import torch" 2>/dev/null; then
    echo "Warning: PyTorch not installed in this env. Run: bash scripts/setup_inference.sh"
fi

# Check UFW status if ufw command exists (non-interactive, no sudo)
if command -v ufw >/dev/null 2>&1; then
    _ufw_status=$(sudo -n ufw status 2>/dev/null) || _ufw_status=""
    if [[ -z "$_ufw_status" ]]; then
        :  # sudo needs password, skip silently
    elif echo "$_ufw_status" | grep -q "Status: inactive"; then
        echo "✓ UFW disabled"
    else
        echo "Warning: UFW is currently enabled."
    fi
    unset _ufw_status
fi
