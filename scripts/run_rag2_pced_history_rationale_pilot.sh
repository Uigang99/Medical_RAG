#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="true"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
QUESTIONS_PER_DATASET="${QUESTIONS_PER_DATASET:-64}"
FIDELITY_PER_DATASET="${FIDELITY_PER_DATASET:-8}"
PCED_QUESTION_BATCH_SIZE="${PCED_QUESTION_BATCH_SIZE:-32}"
PLAIN_QUESTION_BATCH_SIZE="${PLAIN_QUESTION_BATCH_SIZE:-64}"
CHOICE_PROMPT_BATCH_SIZE="${CHOICE_PROMPT_BATCH_SIZE:-64}"
SHARD_SIZE="${SHARD_SIZE:-32}"
GAMMA="${GAMMA:-2.5}"
HISTORY_STRENGTH="${HISTORY_STRENGTH:-0.10}"
HISTORY_DECAY="${HISTORY_DECAY:-0.95}"
HISTORY_TEMPERATURE="${HISTORY_TEMPERATURE:-1.0}"
HISTORY_CAP="${HISTORY_CAP:-2.0}"
MAX_RATIONALE_TOKENS="${MAX_RATIONALE_TOKENS:-512}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/results/rag2_pced_history_rationale_answer_top8_pilot_v2}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/workflow.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

TOTAL_QUESTIONS=$((QUESTIONS_PER_DATASET * 8))
STARTED=$(date +%s)

duration() {
  local seconds="$1"
  printf "%02dh%02dm%02ds" "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

trap 'code=$?; now=$(date +%s); echo "[workflow FAILED | elapsed $(duration $((now-STARTED))) | exit=$code] rerun this identical command to resume durable shards"; exit $code' ERR

echo "[workflow plan] 1 preflight | 2 load one Llama | 3 fixed rerank PCED + fidelity | 4 No-RAG | 5 Base-RAG | 6 semantic PCED | 7 accumulated-history PCED | 8 paired report"
echo "[experiment] Rationale+Answer Top-8, balanced pilot=${TOTAL_QUESTIONS} (${QUESTIONS_PER_DATASET} per dataset)"
echo "[controlled variable] fixed rerank PCED vs identical PCED plus bounded past-token expert history; semantic labels are not used by the proposed condition"
echo "[fallback] if legacy output differs, only this ${TOTAL_QUESTIONS}-question pilot regenerates all five conditions with the new engine; this command never starts the 6,545-question full run"
echo "[performance] GPU=${CUDA_VISIBLE_DEVICES} PCED_question_batch=${PCED_QUESTION_BATCH_SIZE} plain_batch=${PLAIN_QUESTION_BATCH_SIZE} choice_prompt_batch=${CHOICE_PROMPT_BATCH_SIZE}"
echo "[cold-run estimate] exact legacy match: about 25-55 minutes; mismatch and five pilot conditions regenerated: about 50-110 minutes. Active stages report measured ETA."

ARGS=(
  --candidate-cache "$PROJECT/databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2/top8/candidates.jsonl"
  --semantic-score-cache "$PROJECT/results/rag2_pced_topk_answer_mode_sweep_three_anchor_v2/direct_choice/top8/semantic_support_probabilities.jsonl"
  --legacy-output-dir "$PROJECT/results/rag2_pced_topk_answer_mode_sweep_three_anchor_v2/rationale_answer/top8"
  --output-dir "$OUTPUT_DIR"
  --top-k 8
  --questions-per-dataset "$QUESTIONS_PER_DATASET"
  --fidelity-per-dataset "$FIDELITY_PER_DATASET"
  --pced-question-batch-size "$PCED_QUESTION_BATCH_SIZE"
  --plain-question-batch-size "$PLAIN_QUESTION_BATCH_SIZE"
  --choice-prompt-batch-size "$CHOICE_PROMPT_BATCH_SIZE"
  --shard-size "$SHARD_SIZE"
  --gamma "$GAMMA"
  --history-strength "$HISTORY_STRENGTH"
  --history-decay "$HISTORY_DECAY"
  --history-temperature "$HISTORY_TEMPERATURE"
  --history-cap "$HISTORY_CAP"
  --max-rationale-tokens "$MAX_RATIONALE_TOKENS"
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
)

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  ARGS+=(--preflight-only)
fi

"$PYTHON" "$PROJECT/scripts/evaluate_rag2_pced_history_pilot.py" "${ARGS[@]}"

ENDED=$(date +%s)
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "[workflow complete | elapsed $(duration $((ENDED-STARTED))) | ETA 00h00m00s] preflight_manifest=$OUTPUT_DIR/experiment_manifest.json log=$LOG_FILE"
else
  echo "[workflow complete | elapsed $(duration $((ENDED-STARTED))) | ETA 00h00m00s] summary=$OUTPUT_DIR/summary.json log=$LOG_FILE"
fi
