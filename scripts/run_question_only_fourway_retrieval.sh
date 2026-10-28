#!/usr/bin/env bash
set -euo pipefail

# The host's physical H200 GPU 1 becomes cuda:0 inside the process.
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec /home/user/Uiheon/.venv_vllm/bin/python -u \
  /home/user/Uiheon/Medical_RAG/scripts/build_question_only_fourway_retrieval.py "$@"
