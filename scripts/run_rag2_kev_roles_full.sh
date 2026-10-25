#!/usr/bin/env bash
set -euo pipefail
project_root=/home/user/Uiheon/Medical_RAG
cd "$project_root"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /home/user/Uiheon/.venv_vllm/bin/python \
  "$project_root/scripts/label_rag2_kev_roles_full.py" "$@"
