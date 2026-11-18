#!/usr/bin/env bash
# Create the inference environment without installing training/cloud dependencies.
set -euo pipefail
SENNA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${SENNA_CONDA:-/data/miniconda3/bin/conda}"
SENNA_ENV_PREFIX="$("${CONDA_BIN}" info --base)/envs/Senna"
if [[ ! -x "${SENNA_ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_BIN}" create -n Senna python=3.10 pip=24.0 -y \
        --override-channels -c https://repo.anaconda.com/pkgs/main
fi
SENNA_REQUIREMENTS="${SENNA_ROOT}/requirements-inference.lock.txt"
if [[ ! -f "${SENNA_REQUIREMENTS}" ]]; then
    SENNA_REQUIREMENTS="${SENNA_ROOT}/requirements-inference.txt"
fi
"${SENNA_ENV_PREFIX}/bin/python" -m pip install --index-url https://pypi.org/simple \
    -r "${SENNA_REQUIREMENTS}"
"${SENNA_ENV_PREFIX}/bin/python" -m pip check
