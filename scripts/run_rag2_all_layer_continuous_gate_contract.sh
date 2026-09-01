#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON=${PYTHON:-/home/user/Uiheon/.venv_vllm/bin/python}
BASE=${BASE:-${PROJECT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1}
FEATURE_DIR=${FEATURE_DIR:-${BASE}/semantic_attention_controller_v1/medqa_pilot_top8_rationale_wide_v1/prepared_features}
LLM_MODEL=${LLM_MODEL:-/home/user/Uiheon/models/Llama-3-8B-Instruct}
REFERENCE_DIR=${REFERENCE_DIR:-${PROJECT}/results/rag2_all_layer_document_mask_contract_v1/medqa_test}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/results/rag2_all_layer_continuous_gate_contract_v1/medqa_test}
GATE_BATCH_SIZE=${GATE_BATCH_SIZE:-8}

START_SECONDS=${SECONDS}
format_duration() {
  local total=${1}
  printf '%02dh%02dm%02ds' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

printf '[overall 0/1 | elapsed %s | expected ETA 00h06m-00h09m] stage=all-layer continuous-gate audit\n' \
  "$(format_duration "$((SECONDS - START_SECONDS))")"
printf 'Plan: reuse 64-question physical-deletion cache; evaluate 8 documents x gates 1,.75,.5,.25,0 with all-layer/all-query control.\n'
printf 'Assumptions: one H200, BF16, gate batch=%s, warm local model/cache; sequence length is the main runtime variable.\n' \
  "${GATE_BATCH_SIZE}"
printf 'Resume: durable cache=%s/continuous_gate_details.jsonl; rerunning this command safely resumes.\n' \
  "${OUTPUT_DIR}"

"${PYTHON}" "${PROJECT}/scripts/evaluate_rag2_all_layer_continuous_gate_contract.py" \
  --feature-dir "${FEATURE_DIR}" \
  --deletion-reference-dir "${REFERENCE_DIR}" \
  --dataset medqa \
  --split test \
  --llm-model "${LLM_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --gate-factors 1 0.75 0.5 0.25 0 \
  --gate-batch-size "${GATE_BATCH_SIZE}" \
  --document-count 8 \
  --meaningful-probability-delta 0.01 \
  --probability-tolerance 0.001 \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume

printf '[overall 1/1 complete | elapsed %s] report=%s/continuous_gate_report.md\n' \
  "$(format_duration "$((SECONDS - START_SECONDS))")" "${OUTPUT_DIR}"
