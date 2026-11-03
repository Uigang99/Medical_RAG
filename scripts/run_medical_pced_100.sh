#!/usr/bin/env bash
set -euo pipefail
phase="${1:?use preflight|prepare|evaluate|all}"
shift
case "$phase" in preflight|prepare|evaluate|all) ;; *) echo "Invalid phase: $phase" >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
python=/home/user/Uiheon/Medical_RAG/.venv_pced/bin/python
runner=/home/user/Uiheon/Medical_RAG/scripts/run_medical_pced_100.py
for dataset in bioasq11_factoid papersearchqa covid_qa; do
  if [[ "$phase" == all ]]; then
    "$python" -u "$runner" --phase prepare --dataset "$dataset" "$@"
    "$python" -u "$runner" --phase evaluate --dataset "$dataset" "$@"
  else
    "$python" -u "$runner" --phase "$phase" --dataset "$dataset" "$@"
  fi
done
