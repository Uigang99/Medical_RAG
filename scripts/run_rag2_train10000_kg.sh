#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/home/user/Uiheon/Medical_RAG"
export CUDA_VISIBLE_DEVICES=1
exec /home/user/Uiheon/.venv_vllm/bin/python -u \
  "$TASK_ROOT/scripts/build_rag2_train10000_kg.py" "$@"
