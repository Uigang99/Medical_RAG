#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/user/Uiheon/.venv_vllm/bin/python -u \
  /home/user/Uiheon/Medical_RAG/scripts/evaluate_question_only_cot_baselines.py "$@"
