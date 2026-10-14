#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Downloads the public sentence encoder only if it is not already cached.
exec "${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}" \
  /home/user/Uiheon/Medical_RAG/scripts/train_rag2_text_kg_input_pilot.py --arm graph --download-encoder "$@"
