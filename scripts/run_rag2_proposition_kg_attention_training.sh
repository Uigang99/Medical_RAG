#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/home/user/Uiheon/Medical_RAG"
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec /home/user/Uiheon/.venv_vllm/bin/python -u \
  "$TASK_ROOT/scripts/train_rag2_proposition_kg_attention.py" "$@"
