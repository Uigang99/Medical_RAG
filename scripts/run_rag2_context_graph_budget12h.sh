#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
TARGET_PROPOSITIONS="${TARGET_PROPOSITIONS:-600000}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-256}"
NORMALIZATION_BATCH_SIZE="${NORMALIZATION_BATCH_SIZE:-2048}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_context_graph_existing_propositions_600k_v1}"

echo "[command plan] reuse existing propositions -> validate no-hint extraction -> select ${TARGET_PROPOSITIONS} -> participant extraction -> cached UMLS normalization -> graph"
echo "[command estimate] H200 GPU 1, target=${TARGET_PROPOSITIONS}: approximately 6-12 hours; measured ETA appears after the first 10k extractions"

exec "${PYTHON_BIN}" "${ROOT}/scripts/build_rag2_context_graph_budgeted.py" \
  --target-propositions "${TARGET_PROPOSITIONS}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --generation-shard-size "${GENERATION_BATCH_SIZE}" \
  --normalization-batch-size "${NORMALIZATION_BATCH_SIZE}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
