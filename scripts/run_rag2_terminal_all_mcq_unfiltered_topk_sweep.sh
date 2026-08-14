#!/usr/bin/env bash
# Unfiltered RAG Top-k sweep on MedMCQA, MedQA, and six medical MMLU
# subsets.  This is the direct comparison arm for
# run_rag2_terminal_all_mcq_topk_sweep.sh: it reuses the exact same no-RAG
# rationale queries and source-balanced-32 / MedCPT-rerank-32 candidate cache,
# but sends the reranked Top-k prefix to the answer LLM without any filter.

set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"

LLAMA_MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
EXPERIMENT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$EXPERIMENT_ROOT/no_rag_rationales}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/all_mcq_source_balanced32_rationale_full_rerank32}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_exact_terminal_v1/all_mcq_source_balanced32_rerank32/unfiltered_rag}"

test -x "$PYTHON"
test -f "$EVALUATOR"
test -f "$LLAMA_MODEL/config.json"

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

# --generation-top-k selects a prefix of the already reranked 32 documents.
# No filtering model is loaded or consulted in any of these runs.
for TOP_K in 1 2 4 8 16 32; do
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
    --case rerank_rag \
    --generation-top-k "$TOP_K" \
    "${RUN_ARGS[@]}"
done
