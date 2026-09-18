#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
V1_ROOT="${V1_ROOT:-${ROOT}/datasets/kg/rag2_open_statement_kg_pilot_1k_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_v2_pilot_1k_v2}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"
GENERATION_SHARD_SIZE="${GENERATION_SHARD_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"

export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export TRANSFORMERS_VERBOSITY=error
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[command plan] 1,000 cached propositions -> mention cleanup -> Qwen explicit relations -> Qwen direction normalization -> Llama source-fidelity verification -> Mention/Concept/Statement graphs"
echo "[command estimate] H200 GPU 1, batch=${GENERATION_BATCH_SIZE}: approximately 30-75 minutes from empty v2 caches; rerunning the identical command resumes durable shards"

exec "${PYTHON_BIN}" "${ROOT}/scripts/build_rag2_statement_graph_v2_pilot.py" \
  --llm-python "${PYTHON_BIN}" \
  --v1-root "${V1_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --generation-shard-size "${GENERATION_SHARD_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "$@"
