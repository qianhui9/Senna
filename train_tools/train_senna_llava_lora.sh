#!/usr/bin/env bash
# Local single-RTX3090 planning LoRA entry; see 运行文件.md for data and flags.
set -euo pipefail
SENNA_ROOT="${SENNA_ROOT:-/data/qh/phdCode/work4/Senna}"
exec bash "${SENNA_ROOT}/train_tools/train_senna_lora_3090.sh" "$@"
