#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
EXPECTED_CANDIDATE_SECONDS=${EXPECTED_CANDIDATE_SECONDS:-20}
EXPECTED_PREFLIGHT_SECONDS=${EXPECTED_PREFLIGHT_SECONDS:-10}
EXPECTED_SCORING_SECONDS=${EXPECTED_SCORING_SECONDS:-240}
EXPECTED_TOTAL_SECONDS=$((EXPECTED_CANDIDATE_SECONDS + EXPECTED_PREFLIGHT_SECONDS + EXPECTED_SCORING_SECONDS + 15))
STARTED=$(date +%s)

format_duration() {
  local seconds=$1
  printf '%02dh%02dm%02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

on_error() {
  local status=$?
  local elapsed=$(( $(date +%s) - STARTED ))
  printf '[command FAILED | elapsed %s] Completed score shards remain valid; rerun the identical command to resume.\n' \
    "$(format_duration "$elapsed")" >&2
  exit "$status"
}
trap on_error ERR

printf '[command plan | expected wall time %s] 1) cohort 2) Top-8 join/preflight 3) frozen scoring 4) paired report\n' \
  "$(format_duration "$EXPECTED_TOTAL_SECONDS")"

"$PYTHON_BIN" "$PROJECT/scripts/evaluate_rag2_document_first_prompt_order.py" \
  --max-questions "${MAX_QUESTIONS:-512}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --questions-per-shard "${QUESTIONS_PER_SHARD:-64}" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-2048}" \
  --top8-document-token-budget "${TOP8_DOCUMENT_TOKEN_BUDGET:-1500}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-3000}" \
  --expected-candidate-seconds "$EXPECTED_CANDIDATE_SECONDS" \
  --expected-preflight-seconds "$EXPECTED_PREFLIGHT_SECONDS" \
  --expected-scoring-seconds "$EXPECTED_SCORING_SECONDS" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device cuda:0

ELAPSED=$(( $(date +%s) - STARTED ))
printf '[command complete | elapsed %s | ETA 00h00m00s] report=%s\n' \
  "$(format_duration "$ELAPSED")" \
  "$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/document_first_prompt_order_validity_v1/medmcqa_val512/summary.md"
