#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_reference_1k_codex_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_relation_oracle_1k_v1}"
QWEN_MODEL="${QWEN_MODEL:-/home/user/Uiheon/models/Qwen3.5-9B}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"

echo "[command plan] 768 high-confidence propositions -> gold endpoints + within-proposition hard negatives -> unchanged v5 Qwen relation classifier -> oracle/full-pipeline comparison"
echo "[command estimate] H200 GPU 1, Qwen3.5-9B, batch=${GENERATION_BATCH_SIZE}: approximately 5-12 minutes from empty cache; identical reruns resume durable shards"

exec "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_rag2_statement_graph_relation_oracle.py" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --qwen-model "${QWEN_MODEL}" \
  --llm-python "${PYTHON_BIN}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  "$@"
