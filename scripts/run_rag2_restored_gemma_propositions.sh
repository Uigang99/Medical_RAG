#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/home/user/Uiheon/Medical_RAG"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=ERROR
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
echo '[plan] Fixed restored union 88,294 documents; Gemma-4-31B-it; frozen standalone_faithful_v3; batch=8; GPU 1 only'
echo '[context] Capacity 16384 instead of pilot8192 for 9 long inputs; source text/prompt/output budget unchanged; all 58 boundary-review documents retained and flagged.'
echo '[estimate] Pilot throughput extrapolation: roughly 70-80 hours cold; not a guaranteed duration. ETA recalibrates after 32 newly generated documents.'
exec /home/user/Uiheon/.venv_vllm/bin/python -u "${TASK_ROOT}/scripts/extract_rag2_restored_gemma_propositions.py" "$@"
