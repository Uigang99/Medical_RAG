#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
python=/home/user/Uiheon/Medical_RAG/.venv_pced/bin/python
runner=/home/user/Uiheon/Medical_RAG/scripts/evaluate_medical_pced_100_baselines.py
for dataset in bioasq11_factoid papersearchqa covid_qa; do
  for arm in no-rag single all; do
    "$python" -u "$runner" --dataset "$dataset" --arm "$arm" "$@"
  done
done
