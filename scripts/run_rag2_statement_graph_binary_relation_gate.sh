#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
ORACLE_ROOT="${ORACLE_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_relation_oracle_1k_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_binary_relation_gate_bounded256_v3}"
QWEN_MODEL="${QWEN_MODEL:-/home/user/Uiheon/models/Qwen3.5-9B}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"

echo "[command plan] attempt 3/3: corpus-stratified development=128 + validation=128 gold endpoints -> pair-isolated Qwen R/N/A gate -> fixed stop rule; test remains untouched"
echo "[command estimate] H200 GPU 1, Qwen3.5-9B, batch=${GENERATION_BATCH_SIZE}: approximately 2-5 minutes from empty cache; identical reruns resume durable shards"

exec "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_rag2_statement_graph_binary_relation_gate.py" \
  --oracle-root "${ORACLE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --qwen-model "${QWEN_MODEL}" \
  --llm-python "${PYTHON_BIN}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --prompt-version minimal \
  --pair-isolated \
  "$@"
