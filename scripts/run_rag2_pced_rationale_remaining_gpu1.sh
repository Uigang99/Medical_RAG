#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="true"

PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT/results/rag2_pced_topk_answer_mode_sweep_three_anchor_v2}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$PROJECT/databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
GAMMA="${GAMMA:-2.5}"
MAX_RATIONALE_TOKENS="${MAX_RATIONALE_TOKENS:-512}"
ANSWER_RESERVE_TOKENS="${ANSWER_RESERVE_TOKENS:-128}"
SHARD_SIZE="${RATIONAL_SHARD_SIZE:-16}"
LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/gpu1_remaining_workflow.log}"

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

START_EPOCH=$(date +%s)
CURRENT_STAGE=0
TOTAL_STAGES=3

duration() {
  local seconds="$1"
  printf "%02dh%02dm%02ds" "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

begin_stage() {
  CURRENT_STAGE="$1"
  local label="$2"
  local estimate="$3"
  local now elapsed percent
  now=$(date +%s)
  elapsed=$((now - START_EPOCH))
  percent=$((100 * (CURRENT_STAGE - 1) / TOTAL_STAGES))
  echo "Overall: ${percent}% [stage ${CURRENT_STAGE}/${TOTAL_STAGES}, elapsed $(duration "$elapsed"), remaining estimate ${estimate}]"
  echo "Stage ${CURRENT_STAGE}/${TOTAL_STAGES} - ${label}"
}

trap 'code=$?; now=$(date +%s); echo "[workflow stopped | stage ${CURRENT_STAGE}/${TOTAL_STAGES} | elapsed $(duration $((now-START_EPOCH))) | exit=${code}]"; echo "[resume] rerun the identical command; completed 16-question shards are preserved"; exit $code' ERR

for top_k in 16 32; do
  candidate="$CANDIDATE_ROOT/top${top_k}/candidates.jsonl"
  semantic="$OUTPUT_ROOT/direct_choice/top${top_k}/semantic_support_probabilities.jsonl"
  [[ -s "$candidate" ]] || { echo "ERROR: missing $candidate" >&2; exit 2; }
  [[ -s "$semantic" ]] || { echo "ERROR: missing $semantic" >&2; exit 2; }
done
[[ -s "$OUTPUT_ROOT/rationale_answer/top8/summary.json" ]] || {
  echo "ERROR: completed Top-8 summary is missing" >&2
  exit 2
}

echo "[workflow plan] stage 1 Top-16 from its current durable state | stage 2 Top-32 | stage 3 combined report"
echo "[device] physical GPU 1 only"
echo "[method contract] original one-question greedy PCED decoding; no batching that changes token/expert choices"
echo "[data contract] exact stored 3-anchor candidates; no retrieval or reranking is repeated"
echo "[runtime estimate] Top-16 about 14-17h; Top-32 about 20-26h; total about 34-43h from empty caches, based on completed Top-1/2/4/8 throughput"
echo "[resume granularity] 16 questions; interruption recomputes at most the active incomplete shard"
echo "[log] $LOG_FILE"

for top_k in 16 32; do
  if [[ "$top_k" == "16" ]]; then
    stage=1
    estimate="34-43h"
  else
    stage=2
    estimate="20-26h"
  fi
  begin_stage "$stage" "Rationale+Answer Top-${top_k}" "$estimate"
  "$PYTHON" "$PROJECT/scripts/evaluate_rag2_pced_rationale_answer.py" \
    --candidate-cache "$CANDIDATE_ROOT/top${top_k}/candidates.jsonl" \
    --semantic-score-cache "$OUTPUT_ROOT/direct_choice/top${top_k}/semantic_support_probabilities.jsonl" \
    --output-dir "$OUTPUT_ROOT/rationale_answer/top${top_k}" \
    --top-k "$top_k" \
    --gamma "$GAMMA" \
    --max-rationale-tokens "$MAX_RATIONALE_TOKENS" \
    --answer-reserve-tokens "$ANSWER_RESERVE_TOKENS" \
    --shard-size "$SHARD_SIZE" \
    --max-questions "$MAX_QUESTIONS"
  now=$(date +%s)
  echo "[stage ${stage}/${TOTAL_STAGES} complete | elapsed $(duration $((now-START_EPOCH)))] summary=$OUTPUT_ROOT/rationale_answer/top${top_k}/summary.json"
done

begin_stage 3 "combine Direct Choice and Rationale+Answer results" "under 2m"
"$PYTHON" "$PROJECT/scripts/summarize_rag2_pced_topk_answer_modes.py" --root "$OUTPUT_ROOT"

END_EPOCH=$(date +%s)
echo "Overall: 100% [stage 3/3 complete, elapsed $(duration $((END_EPOCH-START_EPOCH))), ETA 00h00m00s]"
echo "[workflow complete] report=$OUTPUT_ROOT/combined_summary_table.txt"
