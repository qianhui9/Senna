#!/usr/bin/env bash
set -euo pipefail

SENNA_ROOT="${SENNA_ROOT:-/data/qh/phdCode/work4/Senna}"
SENNA_MODEL_PATH="${SENNA_MODEL_PATH:-/data/qh/phdCode/work4/models/Senna}"
SENNA_EVAL_DATA="${SENNA_EVAL_DATA:-${SENNA_ROOT}/data/senna_nusc_val_plan_corrected.json}"
SENNA_EVAL_OUTPUT="${SENNA_EVAL_OUTPUT:-${SENNA_ROOT}/outputs/senna_val_eval_corrected.json}"
SENNA_GPU="${SENNA_GPU:-0}"
PYTHON_BIN="${SENNA_PYTHON:-/home/qh/miniconda3/envs/Senna/bin/python}"

export PYTHONPATH="${SENNA_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/qh/phdCode/work4/.cache/Senna/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/qh/phdCode/work4/.cache/Senna/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/qh/phdCode/work4/.cache/Senna/triton}"
export CUDA_VISIBLE_DEVICES="${SENNA_GPU}"
mkdir -p "${MPLCONFIGDIR}" "${TRITON_CACHE_DIR}"

cd "${SENNA_ROOT}"
exec "${PYTHON_BIN}" eval_tools/senna_plan_cmd_eval_multi_img.py \
    --eval-data-path "${SENNA_EVAL_DATA}" \
    --model-path "${SENNA_MODEL_PATH}" \
    --save-path "${SENNA_EVAL_OUTPUT}" \
    --gpu 0 \
    "$@"
