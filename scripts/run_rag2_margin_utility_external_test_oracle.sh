#!/usr/bin/env bash
# Margin-utility gold Oracle on the exact 6,545-question external MCQ test cohort.
#
# Reuses cached no-RAG and one-document exact A/B/C/D logits, as well as the
# existing retrieval/rerank master cache.  Only final Helpful-only answer
# generation is run for k=1,2,4,8,16,32.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
LLAMA="/home/user/Uiheon/models/Llama-3-8B-Instruct"
DATA_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RUN_CACHE_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1"
EXTERNAL_ROOT="$DATA_ROOT/external_test_dynamic_topk_rag2_oracle_v1"

MASTER_CANDIDATES="${MASTER_CANDIDATES:-$RUN_CACHE_ROOT/all_mcq_paper_balanced_max32_rationale_answer_rerank128/candidates/521e23c599352822/candidates.jsonl}"
NO_RAG_ROOT="${NO_RAG_ROOT:-$RUN_CACHE_ROOT/no_rag_rationales_all_mcq_test}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$EXTERNAL_ROOT/candidates_topk_union}"
FEATURE_ROOT="${FEATURE_ROOT:-$EXTERNAL_ROOT/hidden_gold_oracle_tau0p4_v1}"
NO_RAG_FEATURE_ROOT="$FEATURE_ROOT/no_rag_features"
DOCUMENT_FEATURE_ROOT="$FEATURE_ROOT/document_features"
SCORE_ROOT="${SCORE_ROOT:-$EXTERNAL_ROOT/gold_margin_utility_v1}"
UTILITY_THRESHOLD="${UTILITY_THRESHOLD:-0.1}"
THRESHOLD_TAG="${THRESHOLD_TAG:-tau0p1}"
LABEL_ROOT="${LABEL_ROOT:-$EXTERNAL_ROOT/margin_utility_${THRESHOLD_TAG}_v1}"

ORACLE_CACHE_ROOT="${ORACLE_CACHE_ROOT:-$RUN_CACHE_ROOT/all_mcq_paper_balanced_dynamic_topk_margin_utility_${THRESHOLD_TAG}_oracle_v1}"
ORACLE_RESULTS_ROOT="${ORACLE_RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/margin_utility_${THRESHOLD_TAG}_oracle}"
REFERENCE_RESULTS_ROOT="${REFERENCE_RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/paper_reproduction_filter}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/margin_utility_${THRESHOLD_TAG}_comparison}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-80}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
TOP_K_VALUES=(1 2 4 8 16 32)
DATASETS=(medmcqa medqa mmlu_anatomy mmlu_clinical_knowledge mmlu_college_biology mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine)

test -f "$MASTER_CANDIDATES"
test -f "$NO_RAG_ROOT/generation_manifest.json"
test -f "$NO_RAG_FEATURE_ROOT/feature_manifest.json"
test -f "$DOCUMENT_FEATURE_ROOT/document_feature_manifest.json"
test -f "$LLAMA/config.json"

TOTAL_STAGES=9
PIPELINE_START=$(date +%s)
COMPLETED_STAGES=0

format_seconds() {
  local seconds="$1"
  printf '%02dh%02dm%02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

announce_stage() {
  local index="$1"
  local name="$2"
  local now elapsed eta_text
  now=$(date +%s)
  elapsed=$((now - PIPELINE_START))
  if (( COMPLETED_STAGES > 0 )); then
    eta_text=$(format_seconds "$((elapsed * (TOTAL_STAGES - COMPLETED_STAGES) / COMPLETED_STAGES))")
  else
    eta_text="unknown"
  fi
  printf '[overall %d/%d] stage=%s elapsed=%s measured_overall_eta=%s\n' \
    "$index" "$TOTAL_STAGES" "$name" "$(format_seconds "$elapsed")" "$eta_text"
}

finish_stage() {
  COMPLETED_STAGES=$((COMPLETED_STAGES + 1))
}

complete_result_exists() {
  local case_root="$1"
  [[ -d "$case_root" ]] || return 1
  find "$case_root" -mindepth 2 -maxdepth 2 -type f -name results.jsonl \
    -exec sh -c '[ "$(wc -l < "$1")" -eq 6545 ]' _ {} \; -print -quit | grep -q .
}

announce_stage 1 "compute exact cached margin utility (CPU; no model forward)"
"$PYTHON" "$PROJECT/scripts/build_rag2_anchored_gold_margin_scores.py" \
  --no-rag-feature-root "$NO_RAG_FEATURE_ROOT" \
  --document-feature-root "$DOCUMENT_FEATURE_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --output-root "$SCORE_ROOT" \
  --datasets "${DATASETS[@]}" \
  --source-split test \
  --candidate-contract dynamic_topk_union \
  --temperature 1.0 \
  --resume \
  --log-level INFO
finish_stage

announce_stage 2 "materialize Helpful/Neutral/Harmful at utility tau=${UTILITY_THRESHOLD}"
"$PYTHON" "$PROJECT/scripts/materialize_rag2_margin_utility_oracle_labels.py" \
  --candidate-root "$CANDIDATE_ROOT" \
  --score-root "$SCORE_ROOT" \
  --output-root "$LABEL_ROOT" \
  --datasets "${DATASETS[@]}" \
  --split test \
  --candidate-file candidates_topk_union.jsonl \
  --utility-threshold "$UTILITY_THRESHOLD" \
  --resume \
  --log-level INFO
finish_stage

COMMON=(
  --datasets "${DATASETS[@]}"
  --collection unified
  --split test
  --prompt-profile paper_compatible_three_anchor
  --answer-decision-mode free_generation
  --rationale-artifact-root "$NO_RAG_ROOT"
  --rationale-artifact-policy reuse_only
  --dense-query-mode rationale
  --candidate-cache-source-path "$MASTER_CANDIDATES"
  --cache-root "$ORACLE_CACHE_ROOT"
  --vector-db-root "$PROJECT/databases/vector_db/RAG_Square"
  --sources pubmed pmc cpg textbooks
  --candidate-layout source_balanced
  --per-source-top-k 32
  --candidate-pool-top-k 128
  --rerank-top-k 128
  --query-encoder-path /home/user/Uiheon/models/MedCPT-Query-Encoder
  --cross-encoder-path /home/user/Uiheon/models/MedCPT-Cross-Encoder
  --query-max-length 512
  --cross-encoder-max-length 512
  --max-doc-chars 0
  --document-packing dynamic_token_budget
  --document-token-safety-margin 128
  --llm-model-path "$LLAMA"
  --generation-batch-size "$GENERATION_BATCH_SIZE"
  --max-new-tokens 768
  --rationale-max-new-tokens 512
  --rationale-length-retry-attempts 1
  --rationale-length-retry-max-new-tokens 768
  --rationale-invalid-retry-attempts 0
  --no-rationale-retry-quality
  --no-rationale-retry-invalid
  --no-rationale-choice-anchored-retry
  --temperature 0.0
  --top-p 1.0
  --format-retry-attempts 0
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --llm-max-model-len 8192
  --gdn-prefill-backend triton
  --vllm-performance-mode throughput
  --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --vllm-max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
  --case oracle_rag
  --oracle-policy margin_utility
  --oracle-labels-path "$LABEL_ROOT/margin_utility_oracle_labels.jsonl"
  --results-root "$ORACLE_RESULTS_ROOT"
)

STAGE=2
for TOP_K in "${TOP_K_VALUES[@]}"; do
  STAGE=$((STAGE + 1))
  announce_stage "$STAGE" "margin Helpful-only Oracle: paper-balanced 4x${TOP_K} -> Top-${TOP_K}"
  CASE_ROOT="$ORACLE_RESULTS_ROOT/oracle_rag_margin_utility_top${TOP_K}"
  if complete_result_exists "$CASE_ROOT"; then
    echo "Complete 6,545-row margin Oracle result exists; reusing: $CASE_ROOT"
  else
    "$PYTHON" "$PROJECT/scripts/run_rag2_mcq_eval.py" "${COMMON[@]}" \
      --paper-balanced-top-k "$TOP_K"
  fi
  finish_stage
done

announce_stage 9 "summarize reference conditions versus margin-utility gold Oracle"
"$PYTHON" "$PROJECT/scripts/summarize_rag2_anchored_paper_reproduction_sweep.py" \
  --results-root "$REFERENCE_RESULTS_ROOT" \
  --oracle-results-root "$ORACLE_RESULTS_ROOT" \
  --expected-oracle-policy margin_utility \
  --oracle-case-prefix oracle_rag_margin_utility \
  --oracle-display-label "Margin utility gold Oracle (tau=${UTILITY_THRESHOLD})" \
  --output-dir "$SUMMARY_ROOT" \
  --expected-prompt-profile paper_compatible_three_anchor \
  --expected-answer-decision-mode free_generation \
  --expected-per-source-top-k 32 \
  --expected-candidate-pool-top-k 128 \
  --expected-rerank-top-k 128 \
  --expected-paper-balanced-projection
finish_stage

FINISHED=$(date +%s)
printf '[overall %d/%d] complete elapsed=%s summary=%s\n' \
  "$TOTAL_STAGES" "$TOTAL_STAGES" "$(format_seconds "$((FINISHED - PIPELINE_START))")" \
  "$SUMMARY_ROOT/summary_table_pretty.txt"
