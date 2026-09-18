#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
MODEL="${MODEL:-/home/user/Uiheon/models/Qwen3.5-9B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_context_preserving_proposition_graph_pilot_768_v3}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"
echo "[command plan] 768 grounded propositions -> participant extraction without hard scope filtering -> exact-span UMLS normalization -> proposition factor graph quality report"
echo "[command estimate] approximately 10-30 seconds when the validated participant cache is reusable; 3-8 minutes from an empty cache on H200 GPU 1"
exec "${PYTHON_BIN}" "${ROOT}/scripts/build_rag2_context_preserving_proposition_graph_pilot.py" \
  --output-root "${OUTPUT_ROOT}" --model "${MODEL}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" --generation-shard-size 128 \
  --gpu-memory-utilization 0.82 \
  "$@"
