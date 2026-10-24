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
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mode=--run
if [[ "${1:-}" == "--preflight" || "${1:-}" == "--report" ]]; then
  mode="$1"
  shift
fi
exec "$python_bin" "$project_root/scripts/pilot_rag2_kev_evidence_graphs.py" "$mode" "$@"
