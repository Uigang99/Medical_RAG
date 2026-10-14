#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/user/Uiheon/Medical_RAG
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
exec /home/user/Uiheon/.venv_vllm/bin/python "$ROOT/scripts/run_rag2_kg_attention_pilot.py" "$@"
