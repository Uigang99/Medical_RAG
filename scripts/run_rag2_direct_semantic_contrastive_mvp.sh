#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medmcqa}
MODE=${2:-pilot}

if [[ "${DATASET}" != "medmcqa" && "${DATASET}" != "medqa" ]]; then
  echo "ERROR: dataset must be medmcqa or medqa" >&2
  exit 2
fi
if [[ "${MODE}" != "pilot" ]]; then
  echo "ERROR: only the bounded pilot is enabled; scale only after it passes" >&2
  exit 2
fi

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
PAIR_ROOT=${PAIR_ROOT:-"$BASE/direct_semantic_mismatch_${MODE}_pairs_v1"}
MODEL_ROOT=${MODEL_ROOT:-/home/user/Uiheon/models/RAG2-Direct-Semantic-Contrastive-LoRA}
LLAMA_MODEL=${LLAMA_MODEL:-/home/user/Uiheon/models/Llama-3-8B-Instruct}
MAX_TRAIN_QUESTIONS=${MAX_TRAIN_QUESTIONS:-4000}
if [[ "${DATASET}" == "medqa" ]]; then
  MAX_EVAL_QUESTIONS=${MAX_EVAL_QUESTIONS:-0}
else
  MAX_EVAL_QUESTIONS=${MAX_EVAL_QUESTIONS:-4000}
fi

EPOCHS=${EPOCHS:-3}
EXPECTED_PREP_SECONDS=${EXPECTED_PREP_SECONDS:-120}
EXPECTED_TRAIN_SECONDS=${EXPECTED_TRAIN_SECONDS:-900}
EXPECTED_EVAL_SECONDS=${EXPECTED_EVAL_SECONDS:-900}
EXPECTED_SAVE_SECONDS=${EXPECTED_SAVE_SECONDS:-15}
expected_training_seconds=$((
  EPOCHS * (EXPECTED_TRAIN_SECONDS + EXPECTED_EVAL_SECONDS)
  + EXPECTED_EVAL_SECONDS
  + EXPECTED_SAVE_SECONDS
))
expected_total_seconds=$((EXPECTED_PREP_SECONDS + expected_training_seconds))
workflow_start=$(date +%s)
active_stage="startup"

on_error() {
  local status=$?
  local elapsed=$(( $(date +%s) - workflow_start ))
  printf '[%s] [command FAILED | stage=%s | elapsed %s] Re-run the identical command to resume from the last durable checkpoint.\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$active_stage" "$(format_duration "$elapsed")" >&2
  exit "$status"
}
trap on_error ERR

format_duration() {
  local seconds=$1
  printf '%02dh%02dm%02ds' \
    "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

announce() {
  local stage=$1
  local name=$2
  local remaining=$3
  local elapsed=$(( $(date +%s) - workflow_start ))
  printf '[%s] [command %s/2 | elapsed %s | overall ETA %s] %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$stage" \
    "$(format_duration "$elapsed")" "$(format_duration "$remaining")" "$name"
}

announce 1 "reuse/validate direct-choice pair data" "$expected_total_seconds"
active_stage="1/2 pair-data validation"
"$PYTHON_BIN" "$PROJECT/scripts/prepare_rag2_direct_semantic_mismatch_mvp.py" \
  --dataset "$DATASET" \
  --output-root "$PAIR_ROOT" \
  --max-train-questions "$MAX_TRAIN_QUESTIONS" \
  --max-eval-questions "$MAX_EVAL_QUESTIONS" \
  --train-failure-fraction "${TRAIN_FAILURE_FRACTION:-0.80}" \
  --seed "${SEED:-42}" \
  --resume

announce 2 "four-group semantic-contrastive LoRA pilot; inner bars show phase and command-level ETA" "$expected_training_seconds"
active_stage="2/2 semantic-contrastive LoRA training"
prep_elapsed=$(( $(date +%s) - workflow_start ))
run_name="${DATASET}_${MODE}_direct_semantic_contrastive_v3"
"$PYTHON_BIN" "$PROJECT/scripts/train_rag2_direct_semantic_mismatch_lora.py" \
  --dataset "$DATASET" \
  --pair-root "$PAIR_ROOT" \
  --model-name-or-path "$LLAMA_MODEL" \
  --output-root "$MODEL_ROOT" \
  --run-name "$run_name" \
  --objective semantic_contrastive \
  --epochs "$EPOCHS" \
  --patience "${PATIENCE:-2}" \
  --examples-per-group-batch "${EXAMPLES_PER_GROUP_BATCH:-4}" \
  --eval-examples-per-batch "${EVAL_EXAMPLES_PER_BATCH:-32}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --warmup-ratio "${WARMUP_RATIO:-0.03}" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-2048}" \
  --boundary-margin "${BOUNDARY_MARGIN:-0.5}" \
  --min-pair-teacher-gap "${MIN_PAIR_TEACHER_GAP:-0.5}" \
  --pair-margin "${PAIR_MARGIN:-0.5}" \
  --expected-train-seconds "$EXPECTED_TRAIN_SECONDS" \
  --expected-eval-seconds "$EXPECTED_EVAL_SECONDS" \
  --prior-workflow-seconds "$prep_elapsed" \
  --lora-rank "${LORA_RANK:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --dtype "${DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --device cuda:0 \
  --seed "${SEED:-42}" \
  --resume

elapsed=$(( $(date +%s) - workflow_start ))
printf '[%s] [command 2/2 complete | elapsed %s | overall ETA 00h00m00s] model=%s/%s/%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "$(format_duration "$elapsed")" \
  "$MODEL_ROOT" "$DATASET" "$run_name"
