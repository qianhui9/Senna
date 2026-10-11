#!/usr/bin/env bash
set -euo pipefail

SENNA_ROOT="${SENNA_ROOT:-/data/qh/phdCode/work4/Senna}"
SENNA_EVAL_LOG="${SENNA_EVAL_LOG:-${SENNA_ROOT}/outputs/senna_val_eval_corrected.log}"

mkdir -p "$(dirname "${SENNA_EVAL_LOG}")"
cd "${SENNA_ROOT}"
./eval_tools/senna_plan_cmd_eval_multi_img.sh "$@" 2>&1 | tee -a "${SENNA_EVAL_LOG}"
