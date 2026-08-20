#!/usr/bin/env bash
# MedQA-only downstream sweep for the tau=0.4 no-RAG-state-balanced filter.
#
# This keeps the latest final-evaluation contract fixed:
#   paper-exact terminal rationale query
#   -> 8 candidates from each of PubMed/PMC/CPG/Textbooks (32 total)
#   -> MedCPT rerank Top-32
#   -> filter each reranked Top-k prefix
#   -> augment the surviving original documents.
#
# The completed all-MCQ candidate cache is materialized as an exact MedQA
# subset. h0/hD features and filter decisions are computed once and shared by
# rationale-terminal and constrained direct-choice answer generation.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
PROJECT=/home/user/Uiheon/Medical_RAG
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"

LLAMA_MODEL=/home/user/Uiheon/models/Llama-3-8B-Instruct
FLAN_BACKBONE=/home/user/Uiheon/models/Flan-T5-large
MEDQA_FILTER="${MEDQA_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0p4-NoRAGStateBalanced/medqa/medqa_tau0p4_text_hidden_no_rag_state_balanced_epoch5/20260819_205154/final_model}"

EXPERIMENT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$EXPERIMENT_ROOT/no_rag_rationales}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-$EXPERIMENT_ROOT/all_mcq_source_balanced32_rationale_full_rerank32/candidates/07083d5bac341d9b/candidates.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/medqa_source_balanced32_rationale_full_rerank32}"
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-$PROJECT/results/rag2_llama3_hidden_tau0p4_no_rag_state_balanced_v1/medqa_source_balanced32_rerank32}"
RATIONALE_RESULTS_ROOT="${RATIONALE_RESULTS_ROOT:-$BASE_RESULTS_ROOT/rationale_answer/hidden_state_filter}"
DIRECT_RESULTS_ROOT="${DIRECT_RESULTS_ROOT:-$BASE_RESULTS_ROOT/direct_choice/hidden_state_filter}"
RATIONALE_NO_RAG_ROOT="${RATIONALE_NO_RAG_ROOT:-$BASE_RESULTS_ROOT/rationale_answer/no_rag_reference}"
DIRECT_NO_RAG_ROOT="${DIRECT_NO_RAG_ROOT:-$BASE_RESULTS_ROOT/direct_choice/no_rag_reference}"

test -x "$PYTHON"
test -f "$EVALUATOR"
test -f "$LLAMA_MODEL/config.json"
test -f "$FLAN_BACKBONE/model.safetensors"
test -f "$MEDQA_FILTER/pytorch_model.bin"
test -f "$MEDQA_FILTER/rag2_hidden_filter_architecture.json"
test -f "$CANDIDATE_SOURCE"
test -f "$(dirname "$CANDIDATE_SOURCE")/manifest.json"

COMMON_ARGS=(
  --datasets medqa
  --collection unified
  --split test
  --prompt-profile paper_exact_terminal
  --rationale-artifact-root "$ARTIFACT_ROOT"
  --rationale-artifact-policy repair_invalid
  --dense-query-mode rationale
  --vector-db-root "$PROJECT/databases/vector_db/RAG_Square"
  --sources pubmed pmc cpg textbooks
  --candidate-layout source_balanced
  --per-source-top-k 8
  --candidate-pool-top-k 32
  --rerank-top-k 32
  --candidate-cache-source-path "$CANDIDATE_SOURCE"
  --query-encoder-path /home/user/Uiheon/models/MedCPT-Query-Encoder
  --cross-encoder-path /home/user/Uiheon/models/MedCPT-Cross-Encoder
  --query-max-length 512
  --embedding-batch-size 1024
  --retrieval-batch-size 2048
  --rerank-batch-size 1024
  --cross-encoder-max-length 512
  --cross-encoder-attn-implementation eager
  --faiss-gpu-device 0
  --faiss-gpu-use-float16
  --faiss-gpu-temp-memory-mb 2048
  --medqa-filter-model-path "$MEDQA_FILTER"
  --hidden-filter-backbone-path "$FLAN_BACKBONE"
  --filter-evidence-unit preanswer_text_hidden
  --filter-generation-context-unit document
  --filter-batch-size 64
  --filter-max-input-length 768
  --filter-max-doc-chars 0
  --filter-device cuda:0
  --filter-bf16
  --hidden-feature-layer 28
  --hidden-feature-batch-size 64
  --hidden-filter-question-batch-size 32
  --hidden-feature-max-input-tokens 2048
  --hidden-feature-dtype bfloat16
  --hidden-feature-attn-implementation eager
  --hidden-filter-helpful-threshold 0.5
  --max-doc-chars 0
  --document-packing dynamic_token_budget
  --document-token-safety-margin 128
  --llm-model-path "$LLAMA_MODEL"
  --rationale-max-new-tokens 768
  --rationale-length-retry-attempts 0
  --rationale-length-retry-max-new-tokens 768
  --rationale-invalid-retry-attempts 0
  --rationale-invalid-retry-max-new-tokens 768
  --no-rationale-retry-quality
  --no-rationale-retry-invalid
  --no-rationale-choice-anchored-retry
  --temperature 0.0
  --top-p 1.0
  --format-retry-attempts 0
  --gpu-memory-utilization 0.92
  --llm-max-model-len 8192
  --gdn-prefill-backend triton
  --vllm-performance-mode throughput
  --vllm-max-num-batched-tokens 65536
  --cache-root "$CACHE_ROOT"
  --log-level INFO
)

HIDDEN_FILTER_ARGS=(
  --case filter_rag
)

RATIONALE_ARGS=(
  --answer-decision-mode free_generation
  --generation-batch-size 32
  --max-new-tokens 768
  --vllm-max-num-seqs 80
)

DIRECT_ARGS=(
  --answer-decision-mode constrained_choice
  --generation-batch-size 128
  --max-new-tokens 1
  --vllm-max-num-seqs 160
)

RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RUN_ARGS+=(--dry-run)
fi

is_complete() {
  local case_root="$1"
  local expected_mode="$2"
  local require_filter="${3:-0}"
  local run_dir=""
  [[ -d "$case_root" ]] || return 1
  while IFS= read -r run_dir; do
    [[ -s "$run_dir/results.jsonl" ]] || continue
    [[ "$(wc -l < "$run_dir/results.jsonl")" -eq 1273 ]] || continue
    [[ -s "$run_dir/run_config.json" ]] || continue
    grep -Fq "\"answer_decision_mode\": \"$expected_mode\"" "$run_dir/run_config.json" || continue
    if [[ "$require_filter" == "1" ]]; then
      grep -Fq "\"medqa_filter_model_path\": \"$MEDQA_FILTER\"" "$run_dir/run_config.json" || continue
      grep -Fq '"filter_evidence_unit": "preanswer_text_hidden"' "$run_dir/run_config.json" || continue
    fi
    return 0
  done < <(find "$case_root" -mindepth 1 -maxdepth 1 -type d | sort -r)
  return 1
}

run_no_rag() {
  local mode_name="$1"
  local results_root="$2"
  local expected_mode="$3"
  shift 3
  local mode_args=("$@")
  if [[ "${DRY_RUN:-0}" != "1" ]] && is_complete "$results_root/no_rag" "$expected_mode"; then
    echo "[$mode_name] MedQA No-RAG already complete; reusing."
    return
  fi
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${mode_args[@]}" \
    --case no_rag \
    --results-root "$results_root" \
    "${RUN_ARGS[@]}"
}

run_filter_sweep() {
  local mode_name="$1"
  local results_root="$2"
  local expected_mode="$3"
  shift 3
  local mode_args=("$@")
  local top_k=""
  for top_k in 1 2 4 8 16 32; do
    if [[ "${DRY_RUN:-0}" != "1" ]] \
      && is_complete "$results_root/filter_rag_top${top_k}" "$expected_mode" 1; then
      echo "[$mode_name] MedQA Top-k $top_k already complete; reusing."
      continue
    fi
    "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${HIDDEN_FILTER_ARGS[@]}" "${mode_args[@]}" \
      --filter-rerank-top-k "$top_k" \
      --results-root "$results_root" \
      "${RUN_ARGS[@]}"
  done
}

ANSWER_MODE="${1:-both}"
case "$ANSWER_MODE" in
  both|rationale|direct_choice) ;;
  *)
    echo "Usage: $0 [both|rationale|direct_choice]" >&2
    exit 2
    ;;
esac

# Validate the exact MedQA subset and all artifact/model contracts first.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${HIDDEN_FILTER_ARGS[@]}" "${RATIONALE_ARGS[@]}" \
  --filter-cache-only \
  --results-root "$RATIONALE_RESULTS_ROOT" \
  --dry-run

# Materialize the exact 1,273-row MedQA subset of the completed 6,545-row
# candidate cache. No dense retrieval or MedCPT reranking is repeated.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${RATIONALE_ARGS[@]}" \
  --case rerank_rag \
  --candidate-cache-only \
  --results-root "$BASE_RESULTS_ROOT/cache_build" \
  "${RUN_ARGS[@]}"

# Compute h0/hD and the new filter's decisions over all 1,273 x 32 pairs once.
# This cache is independent of the final-answer mode and is reused below.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${HIDDEN_FILTER_ARGS[@]}" "${RATIONALE_ARGS[@]}" \
  --filter-cache-only \
  --results-root "$RATIONALE_RESULTS_ROOT" \
  "${RUN_ARGS[@]}"

if [[ "$ANSWER_MODE" == "both" || "$ANSWER_MODE" == "rationale" ]]; then
  run_no_rag rationale "$RATIONALE_NO_RAG_ROOT" free_generation "${RATIONALE_ARGS[@]}"
  run_filter_sweep rationale "$RATIONALE_RESULTS_ROOT" free_generation "${RATIONALE_ARGS[@]}"
fi

if [[ "$ANSWER_MODE" == "both" || "$ANSWER_MODE" == "direct_choice" ]]; then
  run_no_rag direct_choice "$DIRECT_NO_RAG_ROOT" constrained_choice "${DIRECT_ARGS[@]}"
  run_filter_sweep direct_choice "$DIRECT_RESULTS_ROOT" constrained_choice "${DIRECT_ARGS[@]}"
fi

echo "MedQA two-group Hidden-State evaluation complete."
echo "Rationale results: $RATIONALE_RESULTS_ROOT"
echo "Direct-choice results: $DIRECT_RESULTS_ROOT"
