#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medqa}

if [[ "$DATASET" != "medqa" ]]; then
  echo "Teacher validity pilot currently supports medqa only" >&2
  exit 2
fi

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
FIXED_TEACHER_DIR=${FIXED_TEACHER_DIR:-"$BASE/target_llm_attribution_v1/medqa_pilot_k8_layers20_28_relative_share_rank_v1"}
INDEX_PATH=${INDEX_PATH:-"$BASE/semantic_attention_controller_v1/shared_indices/medqa.sqlite"}
RUN_ROOT=${RUN_ROOT:-"$BASE/teacher_validity_v1/medqa_internal_test128_v1"}
END_TO_END_DIR="$RUN_ROOT/end_to_end_regenerated_rationale"
AUDIT_DIR="$RUN_ROOT/construct_validity_audit"
# Use only half of the completed 256-question internal test split for this
# exploratory audit.  The untouched half remains available for a fresh
# confirmation run if the pre-declared criteria pass.
MAX_QUESTIONS=${MAX_QUESTIONS:-128}
SAMPLE_SEED=${SAMPLE_SEED:-42}

if ! [[ "$MAX_QUESTIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_QUESTIONS must be a positive integer" >&2
  exit 2
fi

workflow_start=$(date +%s)
announce_stage() {
  local index=$1
  local name=$2
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - workflow_start))
  printf '[%s] Teacher-validity workflow %d/2: %s | elapsed=%02dh%02dm%02ds | overall ETA=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$index" "$name" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
    "unknown; active Python stage reports measured ETA"
}

announce_stage 1 "regenerate rationale and choice for full/repeat/eight physical removals"
"$PYTHON_BIN" "$PROJECT/scripts/generate_rag2_teacher_validity_end_to_end.py" \
  --dataset medqa \
  --split test \
  --fixed-teacher-dir "$FIXED_TEACHER_DIR" \
  --index-path "$INDEX_PATH" \
  --model-name-or-path /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --output-dir "$END_TO_END_DIR" \
  --max-questions "$MAX_QUESTIONS" \
  --sample-seed "$SAMPLE_SEED" \
  --questions-per-batch "${QUESTIONS_PER_BATCH:-4}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE:-64}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --retry-max-new-tokens "${RETRY_MAX_NEW_TOKENS:-768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
  --llm-max-model-len "${LLM_MAX_MODEL_LEN:-8192}" \
  --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS:-80}" \
  --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}" \
  --resume

announce_stage 2 "compute direct-choice LOO/one-forward proxies and compare all teachers"
"$PYTHON_BIN" "$PROJECT/scripts/evaluate_rag2_teacher_validity.py" \
  --dataset medqa \
  --split test \
  --fixed-teacher-dir "$FIXED_TEACHER_DIR" \
  --end-to-end-dir "$END_TO_END_DIR" \
  --index-path "$INDEX_PATH" \
  --llm-model /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --output-dir "$AUDIT_DIR" \
  --minimum-reference-jsd "${MINIMUM_REFERENCE_JSD:-1e-4}" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-8192}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume

workflow_end=$(date +%s)
elapsed=$((workflow_end - workflow_start))
printf '[%s] Teacher-validity workflow complete | elapsed=%02dh%02dm%02ds | report=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
  "$AUDIT_DIR/summary.md"
sed -n '1,160p' "$AUDIT_DIR/summary.md"
