#!/usr/bin/env bash
set -euo pipefail

SENNA_ROOT="${SENNA_ROOT:-/data/qh/phdCode/work4/Senna}"
SENNA_MODEL_PATH="${SENNA_MODEL_PATH:-/data/qh/phdCode/work4/models/Senna}"
SENNA_TRAIN_DATA="${SENNA_TRAIN_DATA:-${SENNA_ROOT}/data/senna_nusc_train_plan_corrected.json}"
SENNA_EVAL_DATA="${SENNA_EVAL_DATA:-${SENNA_ROOT}/data/senna_nusc_val_plan_corrected.json}"
SENNA_TRAIN_OUTPUT="${SENNA_TRAIN_OUTPUT:-${SENNA_ROOT}/outputs/senna_lora_nusc_3090}"
SENNA_TRAIN_LOG="${SENNA_TRAIN_LOG:-${SENNA_TRAIN_OUTPUT}.log}"
PYTHON_BIN="${SENNA_PYTHON:-/home/qh/miniconda3/envs/Senna/bin/python}"

export PYTHONPATH="${SENNA_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${SENNA_GPU:-0}"
export HF_HOME="${HF_HOME:-/data/qh/phdCode/work4/.cache/Senna/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/qh/phdCode/work4/.cache/Senna/matplotlib}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p "${MPLCONFIGDIR}" "$(dirname "${SENNA_TRAIN_LOG}")"
cd "${SENNA_ROOT}"
"${PYTHON_BIN}" -m train_tools.senna_lora \
  --model-path "${SENNA_MODEL_PATH}" --train-data "${SENNA_TRAIN_DATA}" \
  --eval-data "${SENNA_EVAL_DATA}" --output-dir "${SENNA_TRAIN_OUTPUT}" "$@" \
  2>&1 | tee -a "${SENNA_TRAIN_LOG}"
