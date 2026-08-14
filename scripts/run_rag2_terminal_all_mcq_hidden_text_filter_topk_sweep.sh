#!/usr/bin/env bash
# Fixed-terminal all-MCQ Top-k sweep using the text+hidden-state utility
# filters.  MedQA uses its MedQA checkpoint; MedMCQA and all medical MMLU
# subsets use the MedMCQA checkpoint.  h0/hD and filter decisions are computed
# for all reranked-32 documents once, committed in resumable question batches,
# and reused for every Top-k generation.

set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"

LLAMA_MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
FLAN_BACKBONE="/home/user/Uiheon/models/Flan-T5-large"
MEDMCQA_HIDDEN_FILTER="${MEDMCQA_HIDDEN_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0/medmcqa/medmcqa_tau0_text_hidden_epoch5/20260813_212343/final_model}"
MEDQA_HIDDEN_FILTER="${MEDQA_HIDDEN_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0/medqa/medqa_tau0_text_hidden_epoch5/20260813_132853/final_model}"

EXPERIMENT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$EXPERIMENT_ROOT/no_rag_rationales}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/all_mcq_source_balanced32_rationale_full_rerank32}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_exact_terminal_v1/all_mcq_source_balanced32_rerank32/hidden_state_text_filter}"

test -x "$PYTHON"
test -f "$EVALUATOR"
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
  --answer-decision-mode free_generation
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
  --generation-batch-size 32
  --max-new-tokens 768
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
  --vllm-max-num-seqs 80
  --vllm-max-num-batched-tokens 65536
  --cache-root "$CACHE_ROOT"
  --results-root "$RESULTS_ROOT"
)

RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RUN_ARGS+=(--dry-run)
fi

# Phase 1: one resumable pass over all 6,545 x 32 question-document pairs.
# Re-running this command validates and retains every completed question row.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case filter_rag \
  --filter-cache-only \
  "${RUN_ARGS[@]}"

is_complete() {
  local top_k="$1"
  local case_root="$RESULTS_ROOT/filter_rag_top${top_k}"
  local latest=""
  if [[ -d "$case_root" ]]; then
    latest="$(find "$case_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  fi
  [[ -n "$latest" ]] \
    && [[ -s "$latest/results.jsonl" ]] \
    && [[ "$(wc -l < "$latest/results.jsonl")" -eq 6545 ]] \
    && grep -Eq '^\| overall[[:space:]]+\|[[:space:]]+6545[[:space:]]+\|[[:space:]]+6545[[:space:]]+\|' \
         "$latest/summary_table_pretty.txt"
}

# Phase 2: all Top-k runs reuse the completed hidden-filter decision cache.
for TOP_K in 1 2 4 8 16 32; do
  if [[ "${DRY_RUN:-0}" != "1" ]] && is_complete "$TOP_K"; then
    echo "Top-k $TOP_K already complete; skipping."
    continue
  fi
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
    --case filter_rag \
    --filter-rerank-top-k "$TOP_K" \
    "${RUN_ARGS[@]}"
done
