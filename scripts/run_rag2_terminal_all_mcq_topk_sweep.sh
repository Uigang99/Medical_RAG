#!/usr/bin/env bash
# RAG2 document-filter evaluation on MedMCQA, MedQA, and six medical MMLU
# subsets.  All expensive artifacts are persisted so the later Hidden State
# filter run can reuse the exact same no-RAG queries and reranked candidates.

set -euo pipefail
# Default to physical GPU 1, but respect an explicit caller override such as
# ``CUDA_VISIBLE_DEVICES=0 bash ...`` when GPU 0 has been authorised.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"

LLAMA_MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
MEDMCQA_FILTER="${MEDMCQA_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperExactFreeResponse-CorrectedNoDocLabels/medmcqa/medmcqa_top10_paper_exact_corrected_nodoc_epoch5_len768/20260803_101200/final_model}"
MEDQA_FILTER="${MEDQA_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperExactFreeResponse-CorrectedNoDocLabels/medqa/medqa_top10_paper_exact_corrected_nodoc_epoch5_len768/20260803_101728/final_model}"

EXPERIMENT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$EXPERIMENT_ROOT/no_rag_rationales}"
# Keep this root unchanged in the later all-dataset Hidden State run.  The
# candidate cache fingerprint is independent of the selected filter model.
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/all_mcq_source_balanced32_rationale_full_rerank32}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_exact_terminal_v1/all_mcq_source_balanced32_rerank32/rag2_filter}"

test -f "$LLAMA_MODEL/config.json"
test -f "$MEDMCQA_FILTER/model.safetensors"
test -f "$MEDQA_FILTER/model.safetensors"

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
  --medmcqa-filter-model-path "$MEDMCQA_FILTER"
  --medqa-filter-model-path "$MEDQA_FILTER"
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

# 1) One shared no-RAG baseline and full rationale+terminal-answer queries.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" --case no_rag

# 2) One shared embedding/retrieval/reranking pass:
#    8 candidates x 4 sources = 32, then MedCPT reranks all 32.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case rerank_rag \
  --candidate-cache-only

# 3) Score every reranked document once.  Routing is MedQA -> MedQA filter;
#    MedMCQA and all MMLU subsets -> MedMCQA filter.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case filter_rag \
  --filter-cache-only

# 4) k is the reranked prefix eligible for filtering, not a post-filter cap.
#    Embeddings, retrieval, reranking, and filter decisions are all reused.
for TOP_K in 1 2 4 8 16 32; do
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
    --case filter_rag \
    --filter-rerank-top-k "$TOP_K"
done
