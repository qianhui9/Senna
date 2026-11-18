#!/usr/bin/env bash
set -euo pipefail
cd /data/qh/phdCode/work4/Senna
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec /data/miniconda3/envs/Senna/bin/python -m experiments.motivation_exp3.run_queries \
  --config experiments/motivation_exp3/config.yaml --mode all --existing resume \
  --wait-for-idle >> outputs/motivation_exp3/semantic_query_idle_run.log 2>&1
