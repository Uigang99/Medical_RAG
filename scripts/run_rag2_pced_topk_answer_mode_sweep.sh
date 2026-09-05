#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="true"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
TOP_K_VALUES=(1 2 4 8 16 32)
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
GAMMA="${GAMMA:-2.5}"
MAX_RATIONALE_TOKENS="${MAX_RATIONALE_TOKENS:-512}"
ANSWER_RESERVE_TOKENS="${ANSWER_RESERVE_TOKENS:-128}"
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-128}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-64}"
RATIONAL_SHARD_SIZE="${RATIONAL_SHARD_SIZE:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT/results/rag2_pced_topk_answer_mode_sweep_three_anchor_v2}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$PROJECT/databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2}"
CANDIDATE_UNION_ROOT="${CANDIDATE_UNION_ROOT:-$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/external_test_dynamic_topk_rag2_oracle_v1/candidates_topk_union}"
PSEUDO_SEMANTIC_LABELS="${PSEUDO_SEMANTIC_LABELS:-$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/external_test_dynamic_topk_semantic_oracle_v1/dynamic_semantic_oracle_labels.jsonl}"
PSEUDO_SEMANTIC_MANIFEST="${PSEUDO_SEMANTIC_MANIFEST:-$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/external_test_dynamic_topk_semantic_oracle_v1/dynamic_semantic_oracle_labels_manifest.json}"
mkdir -p "$OUTPUT_ROOT"

LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/workflow.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

START_EPOCH=$(date +%s)
TOTAL_STAGES=14
CURRENT_STAGE=0

duration() {
  local seconds="$1"
  printf "%02dh%02dm%02ds" "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

begin_stage() {
  CURRENT_STAGE="$1"
  local label="$2"
  local now elapsed percent
  now=$(date +%s)
  elapsed=$((now - START_EPOCH))
  percent=$((100 * (CURRENT_STAGE - 1) / TOTAL_STAGES))
  echo "Overall: ${percent}% [stage ${CURRENT_STAGE}/${TOTAL_STAGES}, elapsed $(duration "$elapsed"), ETA calibrated by active Python stage]"
  echo "Stage ${CURRENT_STAGE}/${TOTAL_STAGES} - ${label}"
}

trap 'code=$?; now=$(date +%s); echo "[workflow FAILED | stage ${CURRENT_STAGE}/${TOTAL_STAGES} | elapsed $(duration $((now-START_EPOCH))) | exit=${code}] durable shards are resumable with the identical command"; exit $code' ERR

echo "[workflow plan] 1 verify/copy frozen pseudo-semantic Top-k pairs | 2-7 Direct Choice | 8-13 Rationale+Answer | 14 combined report"
echo "[comparison per k] No-RAG | concatenated Base-RAG | PCED rerank prior | PCED semantic-support prior"
echo "[candidate contract] exact stored 3-anchor pseudo-semantic-label pairs for k=1,2,4,8,16,32; no retrieval or reranking in this workflow"
echo "[Rationale+Answer contract] PCED full-vocabulary fusion at every rationale token; same fixed beta at constrained final answer"
echo "[settings] GPU=${CUDA_VISIBLE_DEVICES} questions=${MAX_QUESTIONS:-0} gamma=${GAMMA} max_rationale_tokens=${MAX_RATIONALE_TOKENS} output=${OUTPUT_ROOT}"
echo "[cold-run estimate] candidate identity audit about 1-4 minutes; Direct sweep about 2-5 hours from measured Top-8 throughput. Rationale+Answer has no compatible prior throughput; expect roughly 12-30 hours, with a measured ETA after the first 16-question shard. Completed shards resume safely."
echo "[memory policy] Direct batches shrink automatically on CUDA OOM; Rationale+Answer processes one question and at most 33 streams at a time; concatenated prompts use equal per-document token caps below 8192 tokens."

DIRECT_MAX_INPUT_TOKENS=$((8192 - MAX_RATIONALE_TOKENS - ANSWER_RESERVE_TOKENS))
if (( DIRECT_MAX_INPUT_TOKENS <= 0 )); then
  echo "ERROR: rationale/answer token reserves leave no prompt budget" >&2
  exit 2
fi

begin_stage 1 "verify and materialize exact 3-anchor pseudo-semantic Top-k candidate pairs"
"$PYTHON" "$PROJECT/scripts/prepare_rag2_pced_topk_candidates.py" \
  --candidate-union-root "$CANDIDATE_UNION_ROOT" \
  --semantic-labels "$PSEUDO_SEMANTIC_LABELS" \
  --semantic-label-manifest "$PSEUDO_SEMANTIC_MANIFEST" \
  --output-root "$CANDIDATE_ROOT" \
  --top-k-values "${TOP_K_VALUES[@]}"

for index in "${!TOP_K_VALUES[@]}"; do
  k="${TOP_K_VALUES[$index]}"
  stage=$((2 + index))
  # Keep this identical across k so the No-RAG numerical baseline is not
  # changed by a different batch shape. Prompt micro-batches still auto-shrink.
  question_batch=8
  begin_stage "$stage" "Direct Choice Top-${k}"
  "$PYTHON" "$PROJECT/scripts/evaluate_rag2_pced_direct_choice_topk.py" \
    --candidate-cache "$CANDIDATE_ROOT/top${k}/candidates.jsonl" \
    --output-dir "$OUTPUT_ROOT/direct_choice/top${k}" \
    --top-k "$k" \
    --gamma "$GAMMA" \
    --question-batch-size "$question_batch" \
    --prompt-batch-size "$PROMPT_BATCH_SIZE" \
    --max-input-tokens "$DIRECT_MAX_INPUT_TOKENS" \
    --semantic-batch-size "$SEMANTIC_BATCH_SIZE" \
    --max-questions "$MAX_QUESTIONS"
done

for index in "${!TOP_K_VALUES[@]}"; do
  k="${TOP_K_VALUES[$index]}"
  stage=$((8 + index))
  begin_stage "$stage" "Rationale+Answer Top-${k}"
  "$PYTHON" "$PROJECT/scripts/evaluate_rag2_pced_rationale_answer.py" \
    --candidate-cache "$CANDIDATE_ROOT/top${k}/candidates.jsonl" \
    --semantic-score-cache "$OUTPUT_ROOT/direct_choice/top${k}/semantic_support_probabilities.jsonl" \
    --output-dir "$OUTPUT_ROOT/rationale_answer/top${k}" \
    --top-k "$k" \
    --gamma "$GAMMA" \
    --max-rationale-tokens "$MAX_RATIONALE_TOKENS" \
    --answer-reserve-tokens "$ANSWER_RESERVE_TOKENS" \
    --shard-size "$RATIONAL_SHARD_SIZE" \
    --max-questions "$MAX_QUESTIONS"
done

begin_stage 14 "combine both answer modes and all Top-k results"
"$PYTHON" "$PROJECT/scripts/summarize_rag2_pced_topk_answer_modes.py" --root "$OUTPUT_ROOT"

END_EPOCH=$(date +%s)
echo "Overall: 100% [stage 14/14 complete, elapsed $(duration $((END_EPOCH-START_EPOCH))), ETA 00h00m00s]"
echo "[workflow complete] report=$OUTPUT_ROOT/combined_summary_table.txt log=$LOG_FILE"
