#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
V2_ROOT="${V2_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_v2_pilot_1k_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_canonical_direction_pilot_1k_v3}"
QWEN_MODEL="${QWEN_MODEL:-/home/user/Uiheon/models/Qwen3.5-9B}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"

echo "[command plan] reuse 1,000 v2 rows / 972 structural relations -> Qwen direct canonical subject-relation-object -> deterministic clear-direction audit"
echo "[command estimate] H200 GPU 1, batch=${GENERATION_BATCH_SIZE}: approximately 2-5 minutes from empty v3 cache; identical reruns resume durable shards"

exec "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_rag2_statement_graph_canonical_direction_v3.py" \
  --v2-root "${V2_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --qwen-model "${QWEN_MODEL}" \
  --llm-python "${PYTHON_BIN}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  "$@"
