#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
V2_ROOT="${V2_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_v2_pilot_1k_v2}"
V3_ROOT="${V3_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_canonical_direction_pilot_1k_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_semantic_frames_pilot_1k_v4}"
QWEN_MODEL="${QWEN_MODEL:-/home/user/Uiheon/models/Qwen3.5-9B}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"

echo "[command plan] 1,000 standalone propositions + grounded mentions -> Qwen semantic frames -> deterministic grounding audit -> fixed 200-case v3/v4 review"
echo "[command estimate] H200 GPU 1, batch=${GENERATION_BATCH_SIZE}: approximately 3-7 minutes from empty v4 cache; identical reruns resume durable shards"

exec "${PYTHON_BIN}" "${ROOT}/scripts/extract_rag2_statement_graph_semantic_frames_v4.py" \
  --v2-root "${V2_ROOT}" \
  --v3-root "${V3_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --qwen-model "${QWEN_MODEL}" \
  --llm-python "${PYTHON_BIN}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  "$@"
