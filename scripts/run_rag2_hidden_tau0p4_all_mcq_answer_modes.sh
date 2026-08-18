#!/usr/bin/env bash
# Evaluate the trained Hidden-State utility filters labelled at tau=0.4 on
# the same 6,545-question all-MCQ cohort with both final-answer protocols.
#
# Expensive work is intentionally shared:
#   * the paper-terminal no-RAG rationale is the dense-retrieval query;
#   * source-balanced retrieval (8 x 4) and MedCPT reranking are reused;
#   * h0/hD extraction and tau=0.4 filter decisions for all reranked-32
#     documents are computed once and reused by both answer protocols and all k.
# Only the final Llama answer generation differs between the two modes.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"
SUMMARIZER="$PROJECT/scripts/summarize_rag2_hidden_tau0p4_answer_modes.py"

LLAMA_MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
FLAN_BACKBONE="/home/user/Uiheon/models/Flan-T5-large"
MEDMCQA_HIDDEN_FILTER="${MEDMCQA_HIDDEN_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0p4/medmcqa/medmcqa_tau0p4_text_hidden_epoch5/20260817_144906/final_model}"
MEDQA_HIDDEN_FILTER="${MEDQA_HIDDEN_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0p4/medqa/medqa_tau0p4_text_hidden_epoch5/20260818_112610/final_model}"

EXPERIMENT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$EXPERIMENT_ROOT/no_rag_rationales}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/all_mcq_source_balanced32_rationale_full_rerank32}"
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-$PROJECT/results/rag2_llama3_hidden_tau0p4_v1/all_mcq_source_balanced32_rerank32}"
RATIONALE_RESULTS_ROOT="${RATIONALE_RESULTS_ROOT:-$BASE_RESULTS_ROOT/rationale_answer/hidden_state_filter}"
DIRECT_RESULTS_ROOT="${DIRECT_RESULTS_ROOT:-$BASE_RESULTS_ROOT/direct_choice/hidden_state_filter}"

# These are the already completed, protocol-matched no-RAG baselines.  They are
# read only by the final summarizer, so no baseline generation is duplicated.
RATIONALE_NO_RAG_ROOT="${RATIONALE_NO_RAG_ROOT:-$PROJECT/results/rag2_llama3_paper_exact_terminal_v1/all_mcq_source_balanced32_rerank32/rag2_filter/no_rag}"
DIRECT_NO_RAG_ROOT="${DIRECT_NO_RAG_ROOT:-$PROJECT/results/rag2_llama3_direct_choice_v1/all_mcq_source_balanced32_rerank32/no_rag_reference/no_rag}"

test -x "$PYTHON"
test -f "$EVALUATOR"
test -f "$SUMMARIZER"
test -f "$LLAMA_MODEL/config.json"
test -f "$FLAN_BACKBONE/model.safetensors"
test -f "$MEDMCQA_HIDDEN_FILTER/pytorch_model.bin"
test -f "$MEDMCQA_HIDDEN_FILTER/rag2_hidden_filter_architecture.json"
test -f "$MEDQA_HIDDEN_FILTER/pytorch_model.bin"
test -f "$MEDQA_HIDDEN_FILTER/rag2_hidden_filter_architecture.json"

COMMON_ARGS=(
  --datasets medmcqa medqa
             mmlu_anatomy mmlu_clinical_knowledge
             mmlu_college_biology mmlu_college_medicine
             mmlu_medical_genetics mmlu_professional_medicine
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
  --medmcqa-filter-model-path "$MEDMCQA_HIDDEN_FILTER"
  --medqa-filter-model-path "$MEDQA_HIDDEN_FILTER"
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
  local run_dir=""
  [[ -d "$case_root" ]] || return 1
  while IFS= read -r run_dir; do
    if [[ -s "$run_dir/results.jsonl" ]] \
      && [[ "$(wc -l < "$run_dir/results.jsonl")" -eq 6545 ]] \
      && grep -q "\"answer_decision_mode\": \"$expected_mode\"" "$run_dir/run_config.json" \
      && grep -Fq "\"medmcqa_filter_model_path\": \"$MEDMCQA_HIDDEN_FILTER\"" "$run_dir/run_config.json" \
      && grep -Fq "\"medqa_filter_model_path\": \"$MEDQA_HIDDEN_FILTER\"" "$run_dir/run_config.json" \
      && grep -Eq '^\| overall[[:space:]]+\|[[:space:]]+6545[[:space:]]+\|[[:space:]]+6545[[:space:]]+\|' \
           "$run_dir/summary_table_pretty.txt"; then
      return 0
    fi
  done < <(find "$case_root" -mindepth 1 -maxdepth 1 -type d | sort -r)
  return 1
}

run_mode() {
  local mode_name="$1"
  local results_root="$2"
  local expected_mode="$3"
  shift 3
  local mode_args=("$@")
  local top_k=""

  for top_k in 1 2 4 8 16 32; do
    if [[ "${DRY_RUN:-0}" != "1" ]] \
      && is_complete "$results_root/filter_rag_top${top_k}" "$expected_mode"; then
      echo "[$mode_name] Top-k $top_k already complete; skipping."
      continue
    fi
    "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${mode_args[@]}" \
      --case filter_rag \
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

# One resumable filter-score pass over 6,545 x 32 pairs.  The filter-cache
# fingerprint deliberately excludes the final-answer mode, so both generation
# protocols below consume these exact same decisions.
if [[ "${SUMMARY_ONLY:-0}" != "1" ]]; then
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${RATIONALE_ARGS[@]}" \
    --case filter_rag \
    --filter-cache-only \
    --results-root "$RATIONALE_RESULTS_ROOT" \
    "${RUN_ARGS[@]}"

  if [[ "$ANSWER_MODE" == "both" || "$ANSWER_MODE" == "rationale" ]]; then
    run_mode rationale "$RATIONALE_RESULTS_ROOT" free_generation "${RATIONALE_ARGS[@]}"
  fi
  if [[ "$ANSWER_MODE" == "both" || "$ANSWER_MODE" == "direct_choice" ]]; then
    run_mode direct_choice "$DIRECT_RESULTS_ROOT" constrained_choice "${DIRECT_ARGS[@]}"
  fi
fi

if [[ "${DRY_RUN:-0}" != "1" && "$ANSWER_MODE" == "both" ]]; then
  "$PYTHON" "$SUMMARIZER" \
    --rationale-no-rag-root "$RATIONALE_NO_RAG_ROOT" \
    --rationale-results-root "$RATIONALE_RESULTS_ROOT" \
    --direct-no-rag-root "$DIRECT_NO_RAG_ROOT" \
    --direct-results-root "$DIRECT_RESULTS_ROOT" \
    --medmcqa-filter-model-path "$MEDMCQA_HIDDEN_FILTER" \
    --medqa-filter-model-path "$MEDQA_HIDDEN_FILTER" \
    --output-dir "$BASE_RESULTS_ROOT" \
    --expected-questions 6545
fi
