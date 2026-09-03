#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
PAIR_FILE=${PAIR_FILE:-$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/direct_semantic_mismatch_pilot_pairs_v1/medmcqa/test.jsonl}
BASE_MODEL=${BASE_MODEL:-/home/user/Uiheon/models/Llama-3-8B-Instruct}
ADAPTER=${ADAPTER:-/home/user/Uiheon/models/RAG2-Direct-Semantic-Contrastive-LoRA/medmcqa/medmcqa_pilot_direct_semantic_contrastive_v3/final_model}
OUTPUT_DIR=${OUTPUT_DIR:-/home/user/Uiheon/models/RAG2-Direct-Semantic-Contrastive-LoRA/medmcqa/medmcqa_pilot_direct_semantic_contrastive_v3/document_dependence_eval_v1}

EXPECTED_MODEL_PHASE_SECONDS=${EXPECTED_MODEL_PHASE_SECONDS:-480}
EXPECTED_TOTAL_SECONDS=$((2 * EXPECTED_MODEL_PHASE_SECONDS + 90))
STARTED=$(date +%s)

format_duration() {
  local seconds=$1
  printf '%02dh%02dm%02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

on_error() {
  local status=$?
  local elapsed=$(( $(date +%s) - STARTED ))
  printf '[command FAILED | elapsed %s] Durable scoring shards were preserved. Re-run this exact command to resume.\n' \
    "$(format_duration "$elapsed")" >&2
  exit "$status"
}
trap on_error ERR

printf '[command plan | expected wall time %s] 1) preflight 2) frozen model 3) adapter 4) paired report\n' \
  "$(format_duration "$EXPECTED_TOTAL_SECONDS")"

"$PYTHON_BIN" "$PROJECT/scripts/evaluate_rag2_semantic_contrastive_document_dependence.py" \
  --pair-file "$PAIR_FILE" \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --expected-questions "${EXPECTED_QUESTIONS:-2054}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --questions-per-shard "${QUESTIONS_PER_SHARD:-256}" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-2048}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-3000}" \
  --expected-model-phase-seconds "$EXPECTED_MODEL_PHASE_SECONDS" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device cuda:0

ELAPSED=$(( $(date +%s) - STARTED ))
printf '[command complete | elapsed %s | ETA 00h00m00s] report=%s/summary.md\n' \
  "$(format_duration "$ELAPSED")" "$OUTPUT_DIR"
