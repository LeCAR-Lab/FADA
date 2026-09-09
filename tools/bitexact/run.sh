#!/bin/bash
set -euo pipefail
# Ensure a torch-capable interpreter is used regardless of the caller's shell/PATH.
# Set BITEXACT_PYTHON to point at a specific (e.g. conda) interpreter if `python`
# on PATH isn't torch-capable.
PYTHON_BIN="${BITEXACT_PYTHON:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: interpreter '$PYTHON_BIN' not found on PATH." >&2
  echo "Set BITEXACT_PYTHON to a torch-capable interpreter, e.g.:" >&2
  echo "  BITEXACT_PYTHON=/path/to/env/bin/python bash tools/bitexact/run.sh" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' cannot import torch." >&2
  echo "This script needs a torch-capable interpreter. Set BITEXACT_PYTHON to point" >&2
  echo "at one, e.g. a conda/venv environment with torch installed:" >&2
  echo "  BITEXACT_PYTHON=/path/to/env/bin/python bash tools/bitexact/run.sh" >&2
  exit 1
fi

export PYTHONPATH=".:src/holosoma:src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -c "
from pathlib import Path
from tools.bitexact.harness import run_train_probe, run_train_probe_mixed
from tools.bitexact.export_probe import run_export_probe
from tools.bitexact.sft_probe import run_sft_probe
c = Path('tools/bitexact/current')
run_train_probe(c / 'train.json', steps=20)
run_train_probe_mixed(c / 'train_mixed.json', steps=20)
run_export_probe(c / 'export.json')
run_sft_probe(c / 'sft.json', steps=50)
"
"$PYTHON_BIN" tools/bitexact/compare.py
