#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SENNA_ROOT="${SENNA_ROOT:-/data/qh/phdCode/work4/Senna}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/data/qh/nuscenes/v1.0-trainval}"
OUTPUT_DIR="${SENNA_NUSC_OUTPUT_DIR:-${NUSCENES_ROOT}/vqa}"
PYTHON_BIN="${SENNA_PYTHON:-/home/qh/miniconda3/envs/Senna/bin/python}"

if [[ -z "${LLAVA_MODEL_PATH:-}" ]]; then
    echo "LLAVA_MODEL_PATH must point to a LLaVA-v1.6-34b checkpoint." >&2
    exit 2
fi

export PYTHONPATH="${SENNA_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

cd "${SENNA_ROOT}"

echo "Start generating dataset..."

"${PYTHON_BIN}" \
    data_tools/senna_nusc_data_converter.py \
    nuscenes \
    --root-path "${NUSCENES_ROOT}" \
    --out-dir "${OUTPUT_DIR}" \
    --extra-tag senna_nusc \
    --version v1.0 \
    --skip-test \
    --canbus "${NUSCENES_ROOT}"
