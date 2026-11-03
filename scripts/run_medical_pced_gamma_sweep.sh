#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
python=/home/user/Uiheon/Medical_RAG/.venv_pced/bin/python
runner=/home/user/Uiheon/Medical_RAG/scripts/sweep_medical_pced_gamma.py

for dataset in bioasq11_factoid papersearchqa covid_qa; do
  "$python" -u "$runner" --phase preflight --dataset "$dataset" "$@"
done
for dataset in bioasq11_factoid papersearchqa covid_qa; do
  "$python" -u "$runner" --phase evaluate --dataset "$dataset" "$@"
done
"$python" -u "$runner" --phase summary --dataset all "$@"
