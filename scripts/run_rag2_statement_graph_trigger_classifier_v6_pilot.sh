#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
ENTITY_PYTHON="${ENTITY_PYTHON:-/home/user/Uiheon/.venv_rag2_kg_mvp_py312/bin/python}"
LLM_PYTHON="${LLM_PYTHON:-/home/user/Uiheon/.venv_vllm/bin/python}"
V2_ROOT="${V2_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_v2_pilot_1k_v2}"
V4_ROOT="${V4_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_semantic_frames_pilot_1k_v4}"
V5_ROOT="${V5_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_pair_classifier_pilot_1k_v5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_statement_graph_trigger_classifier_pilot_1k_v6_2}"
QWEN_MODEL="${QWEN_MODEL:-/home/user/Uiheon/models/Qwen3.5-9B}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"

echo "[command plan] same 1,000 propositions -> noun-headed core nodes -> explicit trigger/type assertions -> Qwen binary source verification -> fixed 200-edge precision audit"
echo "[command estimate] H200 GPU 1, batch=${GENERATION_BATCH_SIZE}: approximately 4-10 minutes from empty v6 cache; identical reruns resume durable Qwen shards"
echo "[decision rule] do not expand or scale unless exact precision >=75%, direction >=90%, wrong edges <=10%, and correct-edge coverage >=60% of eligible / >=12% of all propositions"

exec "${LLM_PYTHON}" "${ROOT}/scripts/extract_rag2_statement_graph_trigger_classifier_v6.py" \
  --v2-root "${V2_ROOT}" \
  --v4-root "${V4_ROOT}" \
  --v5-root "${V5_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --qwen-model "${QWEN_MODEL}" \
  --entity-python "${ENTITY_PYTHON}" \
  --llm-python "${LLM_PYTHON}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  "$@"
