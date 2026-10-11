#!/usr/bin/env bash
set -euo pipefail

SENNA_ROOT="${SENNA_ROOT:-/data/qh/phdCode/work4/Senna}"
SENNA_MODEL_PATH="${SENNA_MODEL_PATH:-/data/qh/phdCode/work4/models/Senna}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/data/qh/nuscenes/v1.0-trainval}"
SENNA_GPU="${SENNA_GPU:-0}"
PYTHON_BIN="${SENNA_PYTHON:-/home/qh/miniconda3/envs/Senna/bin/python}"

export PYTHONPATH="${SENNA_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/qh/phdCode/work4/.cache/Senna/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/qh/phdCode/work4/.cache/Senna/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/qh/phdCode/work4/.cache/Senna/triton}"
export CUDA_VISIBLE_DEVICES="${SENNA_GPU}"

mkdir -p "${MPLCONFIGDIR}" "${TRITON_CACHE_DIR}"
cd "${SENNA_ROOT}"
exec "${PYTHON_BIN}" -m eval_tools.senna_inference \
    --model-path "${SENNA_MODEL_PATH}" \
    --dataroot "${NUSCENES_ROOT}" \
    --gpu 0 \
    "$@"
