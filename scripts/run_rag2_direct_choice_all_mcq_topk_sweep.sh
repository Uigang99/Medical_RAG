#!/usr/bin/env bash
# Repeat the completed all-MCQ rationale-answer table with only the final
# answer protocol changed to direct choice.  Rationale retrieval queries,
# source-balanced candidates, MedCPT reranking, and filter decisions reuse the
# exact existing caches.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"
SUMMARIZER="$PROJECT/scripts/summarize_rag2_direct_choice_all_mcq.py"

LLAMA_MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
FLAN_BACKBONE="/home/user/Uiheon/models/Flan-T5-large"
MEDMCQA_RAG2_FILTER="${MEDMCQA_RAG2_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperExactFreeResponse-CorrectedNoDocLabels/medmcqa/medmcqa_top10_paper_exact_corrected_nodoc_epoch5_len768/20260803_101200/final_model}"
MEDQA_RAG2_FILTER="${MEDQA_RAG2_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperExactFreeResponse-CorrectedNoDocLabels/medqa/medqa_top10_paper_exact_corrected_nodoc_epoch5_len768/20260803_101728/final_model}"
MEDMCQA_HIDDEN_FILTER="${MEDMCQA_HIDDEN_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0/medmcqa/medmcqa_tau0_text_hidden_epoch5/20260813_212343/final_model}"
MEDQA_HIDDEN_FILTER="${MEDQA_HIDDEN_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0/medqa/medqa_tau0_text_hidden_epoch5/20260813_132853/final_model}"

EXPERIMENT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$EXPERIMENT_ROOT/no_rag_rationales}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/all_mcq_source_balanced32_rationale_full_rerank32}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT/results/rag2_llama3_direct_choice_v1/all_mcq_source_balanced32_rerank32}"

test -x "$PYTHON"
test -f "$EVALUATOR"
test -f "$SUMMARIZER"
test -f "$LLAMA_MODEL/config.json"
test -f "$FLAN_BACKBONE/model.safetensors"
test -f "$MEDMCQA_RAG2_FILTER/model.safetensors"
test -f "$MEDQA_RAG2_FILTER/model.safetensors"
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
  --answer-decision-mode constrained_choice
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
  --max-doc-chars 0
  --document-packing dynamic_token_budget
  --document-token-safety-margin 128
  --llm-model-path "$LLAMA_MODEL"
  --generation-batch-size 128
  --max-new-tokens 1
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
  --vllm-max-num-seqs 160
  --vllm-max-num-batched-tokens 65536
  --cache-root "$CACHE_ROOT"
)

RAG2_FILTER_ARGS=(
  --medmcqa-filter-model-path "$MEDMCQA_RAG2_FILTER"
  --medqa-filter-model-path "$MEDQA_RAG2_FILTER"
  --filter-evidence-unit document
  --filter-generation-context-unit document
  --filter-batch-size 128
  --filter-max-input-length 768
  --filter-max-new-tokens 1
  --filter-max-doc-chars 0
  --filter-device cuda:0
  --filter-bf16
  --filter-scoring-method special_token
  --filter-input-format official
  --filter-score-normalization mean
)

HIDDEN_FILTER_ARGS=(
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
)

is_complete() {
  local case_root="$1"
  local run_dir=""
  [[ -d "$case_root" ]] || return 1
  while IFS= read -r run_dir; do
    if [[ -s "$run_dir/results.jsonl" ]] \
      && [[ "$(wc -l < "$run_dir/results.jsonl")" -eq 6545 ]] \
      && grep -q '"answer_decision_mode": "constrained_choice"' "$run_dir/run_config.json" \
      && grep -Eq '^\| overall[[:space:]]+\|[[:space:]]+6545[[:space:]]+\|[[:space:]]+6545[[:space:]]+\|' \
           "$run_dir/summary_table_pretty.txt"; then
      return 0
    fi
  done < <(find "$case_root" -mindepth 1 -maxdepth 1 -type d | sort -r)
  return 1
}

run_if_incomplete() {
  local case_root="$1"
  shift
  if is_complete "$case_root"; then
    echo "Complete result exists; skipping: $case_root"
    return
  fi
  "$@"
}

RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RUN_ARGS+=(--dry-run)
fi

# Direct-choice No-RAG is a fresh one-token baseline.  Stored rationales are
# still reused solely as dense-retrieval queries in every RAG condition.
run_if_incomplete "$RESULTS_ROOT/no_rag_reference/no_rag" \
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case no_rag \
  --results-root "$RESULTS_ROOT/no_rag_reference" \
  "${RUN_ARGS[@]}"

# Validate/reuse the exact existing embeddings, source-balanced candidates,
# and MedCPT reranking.  No answer generation is performed by this command.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case rerank_rag \
  --candidate-cache-only \
  --results-root "$RESULTS_ROOT/cache_validation" \
  "${RUN_ARGS[@]}"

for TOP_K in 1 2 4 8 16 32; do
  run_if_incomplete "$RESULTS_ROOT/unfiltered_rag/rerank_rag_top${TOP_K}" \
    "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
    --case rerank_rag \
    --generation-top-k "$TOP_K" \
    --results-root "$RESULTS_ROOT/unfiltered_rag" \
    "${RUN_ARGS[@]}"
done

# Both filter-score passes should hit the completed rationale-answer caches;
# they remain resumable if any cache row was previously missing.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${RAG2_FILTER_ARGS[@]}" \
  --case filter_rag \
  --filter-cache-only \
  --results-root "$RESULTS_ROOT/rag2_filter" \
  "${RUN_ARGS[@]}"

for TOP_K in 1 2 4 8 16 32; do
  run_if_incomplete "$RESULTS_ROOT/rag2_filter/filter_rag_top${TOP_K}" \
    "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${RAG2_FILTER_ARGS[@]}" \
    --case filter_rag \
    --filter-rerank-top-k "$TOP_K" \
    --results-root "$RESULTS_ROOT/rag2_filter" \
    "${RUN_ARGS[@]}"
done

"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${HIDDEN_FILTER_ARGS[@]}" \
  --case filter_rag \
  --filter-cache-only \
  --results-root "$RESULTS_ROOT/hidden_state_filter" \
  "${RUN_ARGS[@]}"

for TOP_K in 1 2 4 8 16 32; do
  run_if_incomplete "$RESULTS_ROOT/hidden_state_filter/filter_rag_top${TOP_K}" \
    "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${HIDDEN_FILTER_ARGS[@]}" \
    --case filter_rag \
    --filter-rerank-top-k "$TOP_K" \
    --results-root "$RESULTS_ROOT/hidden_state_filter" \
    "${RUN_ARGS[@]}"
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "$PYTHON" "$SUMMARIZER" \
    --results-root "$RESULTS_ROOT" \
    --expected-questions 6545
fi
