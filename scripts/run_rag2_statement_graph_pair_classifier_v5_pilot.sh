#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
V2_ROOT="${V2_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_v2_pilot_1k_v2}"
V4_ROOT="${V4_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_semantic_frames_pilot_1k_v4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_pair_classifier_pilot_1k_v5}"
QWEN_MODEL="${QWEN_MODEL:-/home/user/Uiheon/models/Qwen3.5-9B}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"

echo "[command plan] 1,000 propositions -> fixed type-compatible pairs -> Qwen relation/NO_RELATION classification -> same 200-case v4/v5 review"
echo "[command estimate] H200 GPU 1, batch=${GENERATION_BATCH_SIZE}: approximately 5-12 minutes from empty v5 cache; identical reruns resume durable shards"

exec "${PYTHON_BIN}" "${ROOT}/scripts/extract_rag2_statement_graph_pair_classifier_v5.py" \
  --v2-root "${V2_ROOT}" \
  --v4-root "${V4_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --qwen-model "${QWEN_MODEL}" \
  --llm-python "${PYTHON_BIN}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  "$@"
