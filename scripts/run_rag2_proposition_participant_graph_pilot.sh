#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
MODEL="${MODEL:-/home/user/Uiheon/models/Qwen3.5-9B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_proposition_participant_graph_pilot_768_v1}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"

echo "[command plan] fixed high-confidence 768 propositions -> Qwen graph-use + extractive participants -> held-out validation/test gates -> proposition factor graph"
echo "[command estimate] H200 GPU 1, Qwen3.5-9B, batch=${GENERATION_BATCH_SIZE}: approximately 3-8 minutes from empty cache; identical reruns resume durable shards"

exec "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_rag2_proposition_participant_graph_pilot.py" \
  --output-root "${OUTPUT_ROOT}" \
  --model "${MODEL}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --generation-shard-size 128 \
  --gpu-memory-utilization 0.82
