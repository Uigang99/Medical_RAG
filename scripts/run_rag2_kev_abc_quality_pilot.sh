#!/usr/bin/env bash
set -euo pipefail
project_root=/home/user/Uiheon/Medical_RAG
python_bin=/home/user/Uiheon/.venv_vllm/bin/python
cd "$project_root"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_OFFLINE=1
exec "$python_bin" "$project_root/scripts/pilot_rag2_kev_abc_quality.py" "$@"
