# Workspace for conda envs, Isaac Gym/Sim, pip/conda caches, temp files, etc.
#
# Choose root (first match):
#   1. WORKSPACE_DIR if already set in the environment
#   2. HOLOSOMA_WORKSPACE_DIR if set (useful in shell rc to force /usr0/...)
#   3. /usr0/$USER/.holosoma_deps if that parent dir exists and is writable
#   4. $HOME/.holosoma_deps
#
# Keep large downloads and caches under WORKSPACE_DIR so installs do not fill $HOME or /tmp
# (pip/conda temp files and default ~/.cache/pip often cause "No space left on device" on /home).
if [[ -z "${WORKSPACE_DIR:-}" ]]; then
  if [[ -n "${HOLOSOMA_WORKSPACE_DIR:-}" ]]; then
    WORKSPACE_DIR="$HOLOSOMA_WORKSPACE_DIR"
  else
    _usr0_base="/usr0/$USER"
    if [[ -d "$_usr0_base" && -w "$_usr0_base" ]]; then
      WORKSPACE_DIR="$_usr0_base/.holosoma_deps"
    else
      WORKSPACE_DIR="$HOME/.holosoma_deps"
    fi
  fi
  unset _usr0_base
fi
export WORKSPACE_DIR
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
# Override with PYTHONNOUSERSITE=0 if you intentionally rely on pip --user packages.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORKSPACE_DIR/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_DIR/pip-cache}"
export TMPDIR="${TMPDIR:-$WORKSPACE_DIR/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE_DIR/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$WORKSPACE_DIR/.config}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$WORKSPACE_DIR/.local/share}"
export TORCH_HOME="${TORCH_HOME:-$WORKSPACE_DIR/torch}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$WORKSPACE_DIR/pycache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE_DIR/uv-cache}"

mkdir -p "$WORKSPACE_DIR" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TORCH_HOME" \
  "$PYTHONPYCACHEPREFIX" "$UV_CACHE_DIR"
