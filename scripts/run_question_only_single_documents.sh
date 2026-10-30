#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PATH="/home/user/Uiheon/.venv_vllm/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

exec /home/user/Uiheon/.venv_vllm/bin/python -u \
  /home/user/Uiheon/Medical_RAG/scripts/evaluate_question_only_single_documents.py "$@"
