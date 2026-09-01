#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON=${PYTHON:-/home/user/Uiheon/.venv_vllm/bin/python}
BASE=${BASE:-${PROJECT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1}
FEATURE_DIR=${FEATURE_DIR:-${BASE}/semantic_attention_controller_v1/medqa_pilot_top8_rationale_wide_v1/prepared_features}
LLM_MODEL=${LLM_MODEL:-/home/user/Uiheon/models/Llama-3-8B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/results/rag2_all_layer_document_mask_contract_v1/medqa_test}
MAX_SAMPLES=${MAX_SAMPLES:-64}

START_SECONDS=${SECONDS}
format_duration() {
  local total=${1}
  printf '%02dh%02dm%02ds' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

printf '[overall 0/1 | elapsed %s | overall ETA calibrating] stage=all-layer document-mask contract\n' \
  "$(format_duration "$((SECONDS - START_SECONDS))")"
printf 'Plan: dataset=medqa split=test questions<=%s documents/question=8 conditions=physical-delete,all-layer-mask,all-layer-mask+compact-position,legacy-layer16\n' \
  "${MAX_SAMPLES}"
printf 'Resume: durable cache=%s/mask_contract_details.jsonl; rerunning this command safely resumes.\n' \
  "${OUTPUT_DIR}"

"${PYTHON}" "${PROJECT}/scripts/evaluate_rag2_all_layer_document_mask_contract.py" \
  --feature-dir "${FEATURE_DIR}" \
  --dataset medqa \
  --split test \
  --llm-model "${LLM_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --sample-seed 42 \
  --document-count 8 \
  --legacy-layer-start 16 \
  --legacy-zero-log-bias -20 \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume

printf '[overall 1/1 complete | elapsed %s] report=%s/mask_contract_report.md\n' \
  "$(format_duration "$((SECONDS - START_SECONDS))")" "${OUTPUT_DIR}"
