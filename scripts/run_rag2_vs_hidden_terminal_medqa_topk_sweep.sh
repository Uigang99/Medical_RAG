#!/usr/bin/env bash
# Fair MedQA comparison between the original RAG2 document filter and the
# text+hidden-state filter.  Both methods share one fixed-terminal no-RAG
# rationale artifact and one source-balanced-32 / MedCPT-rerank-32 cache.

set -euo pipefail
# Default to physical GPU 1, while allowing an explicit caller override.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"

LLAMA_MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
FLAN_BACKBONE="/home/user/Uiheon/models/Flan-T5-large"
RAG2_FILTER="${RAG2_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperExactFreeResponse-CorrectedNoDocLabels/medqa/medqa_top10_paper_exact_corrected_nodoc_epoch5_len768/20260803_101728/final_model}"
HIDDEN_FILTER="${HIDDEN_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0/medqa/medqa_tau0_text_hidden_epoch5/20260813_132853/final_model}"

EXPERIMENT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$EXPERIMENT_ROOT/no_rag_rationales}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/medqa_source_balanced32_rationale_full_rerank32}"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT/results/rag2_llama3_paper_exact_terminal_v1/medqa_source_balanced32_rerank32}"

test -f "$LLAMA_MODEL/config.json"
test -f "$FLAN_BACKBONE/model.safetensors"
test -f "$RAG2_FILTER/model.safetensors"
test -f "$HIDDEN_FILTER/pytorch_model.bin"
test -f "$HIDDEN_FILTER/rag2_hidden_filter_architecture.json"

COMMON_ARGS=(
  --datasets medqa
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
)

# 1) Generate once and store the complete no-RAG rationale+terminal answer.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case no_rag \
  --results-root "$RESULTS_BASE/no_rag_reference"

# 2) Embed the complete response, retrieve 8 documents from each source
#    (32 total), and MedCPT-rerank all 32 exactly once.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case rerank_rag \
  --candidate-cache-only \
  --results-root "$RESULTS_BASE/cache_build"

RAG2_FILTER_ARGS=(
  --medqa-filter-model-path "$RAG2_FILTER"
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

# 3) Score all reranked documents once with the original RAG2 filter.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${RAG2_FILTER_ARGS[@]}" \
  --case filter_rag \
  --filter-cache-only \
  --results-root "$RESULTS_BASE/rag2_filter"

# k is the reranked prefix eligible for filtering.  It is not a post-filter cap.
for TOP_K in 1 2 4 8 16 32; do
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${RAG2_FILTER_ARGS[@]}" \
    --case filter_rag \
    --filter-rerank-top-k "$TOP_K" \
    --results-root "$RESULTS_BASE/rag2_filter"
done

HIDDEN_FILTER_ARGS=(
  --medqa-filter-model-path "$HIDDEN_FILTER"
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
  --hidden-feature-max-input-tokens 2048
  --hidden-feature-dtype bfloat16
  --hidden-feature-attn-implementation eager
  --hidden-filter-helpful-threshold 0.5
)

# 4) Extract h0/hD and score all 32 documents once.  The resulting cache is
#    shared by every Hidden State Top-k generation below.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${HIDDEN_FILTER_ARGS[@]}" \
  --case filter_rag \
  --filter-cache-only \
  --results-root "$RESULTS_BASE/hidden_state_filter"

for TOP_K in 1 2 4 8 16 32; do
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" "${HIDDEN_FILTER_ARGS[@]}" \
    --case filter_rag \
    --filter-rerank-top-k "$TOP_K" \
    --results-root "$RESULTS_BASE/hidden_state_filter"
done
