#!/usr/bin/env bash
set -euo pipefail
SENNA_ROOT="${SENNA_ROOT:-/data/qh/phdCode/work4/Senna}"
SENNA_TRAIN_DATA="${SENNA_TRAIN_DATA:-${SENNA_ROOT}/data/senna_nusc_train_plan_corrected.json}"
PYTHON_BIN="${SENNA_PYTHON:-/home/qh/miniconda3/envs/Senna/bin/python}"
export PYTHONPATH="${SENNA_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/qh/phdCode/work4/.cache/Senna/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p "${MPLCONFIGDIR}"
cd "${SENNA_ROOT}"
if [[ -e "${SENNA_TRAIN_DATA}" ]]; then
  printf 'Training JSON already exists (not overwritten): %s\n' "${SENNA_TRAIN_DATA}"
else
  "${PYTHON_BIN}" -m eval_tools.prepare_senna_plan_eval --split train \
    --dataroot "${NUSCENES_ROOT:-/data/qh/nuscenes/v1.0-trainval}" \
    --output "${SENNA_TRAIN_DATA}"
fi
bash train_tools/train_senna_lora_3090.sh --validate-only
