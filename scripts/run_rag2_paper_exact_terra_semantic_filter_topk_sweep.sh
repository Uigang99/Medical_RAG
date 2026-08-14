#!/usr/bin/env bash
# Evaluate the document-level Flan-T5 filter trained with GPT-5.6-Terra
# semantic labels under the *identical* external-MCQ protocol used by the
# source_balanced40_corrected_filter_rerank_topk_sweep baseline.
#
# Fixed protocol:
#   cached paper-exact no-RAG rationale/answer -> rationale dense query
#   10 candidates from each of PubMed / PMC / CPG / Textbooks (40 total)
#   MedCPT cross-encoder rerank Top-32 (reused verbatim from the prior sweep)
#   score all 32 once with the selected filter
#   generate answers for rerank prefixes k = 1,2,4,8,16,32, retaining only
#   filter-predicted Helpful chunks in each prefix.  There is no backfill.
#
# Default scope is MedQA only because the completed Terra-label model currently
# exists only for MedQA.  Once the MedMCQA Terra-label filter is trained, run
# EVAL_SCOPE=all MEDMCQA_FILTER=/absolute/path/to/best/checkpoint "$0".

set -euo pipefail

# Physical GPU 1 is deliberately exposed as logical cuda:0 to PyTorch/FAISS/vLLM.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"

EVAL_SCOPE="${EVAL_SCOPE:-medqa}"
MEDQA_FILTER="${MEDQA_FILTER:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-CodexSemanticTop8/medqa/medqa_codex_semantic_top8_countmatched_epoch5_len768/20260812_120009/checkpoint-8352}"

# These exact artifacts/candidates are the reference used by the prior
# document-level PPL-filter sweep.  The evaluator materialises only the
# requested dataset subset after verifying question keys and dense queries.
ARTIFACT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_free_response_v1/no_rag_rationales_reparsed_v2"
CANDIDATE_SOURCE="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_free_response_v1/source_balanced40_rationale_answer_top32/candidates/937e3924fc474732/candidates.jsonl"

# New roots prevent accidental reuse of a PPL-filter decision cache.  They do
# not duplicate retrieval/reranking; candidate rows are materialised exactly
# from CANDIDATE_SOURCE and filter decisions are model-fingerprinted.
CACHE_ROOT="${CACHE_ROOT:-$PROJECT/databases/run_cache/rag2_llama3_paper_exact_free_response_v1/source_balanced40_terra_semantic_filter_top32}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_exact_free_response_v2/source_balanced40_terra_semantic_filter_rerank_topk_sweep}"

case "$EVAL_SCOPE" in
  medqa)
    DATASETS=(medqa)
    ROUTE_ARGS=(--medqa-filter-model-path "$MEDQA_FILTER")
    ;;
  all)
    : "${MEDMCQA_FILTER:?EVAL_SCOPE=all requires MEDMCQA_FILTER=/absolute/path/to/the completed Terra-label checkpoint}"
    DATASETS=(
      medmcqa medqa mmlu_anatomy mmlu_clinical_knowledge
      mmlu_college_biology mmlu_college_medicine
      mmlu_medical_genetics mmlu_professional_medicine
    )
    ROUTE_ARGS=(
      --medmcqa-filter-model-path "$MEDMCQA_FILTER"
      --medqa-filter-model-path "$MEDQA_FILTER"
    )
    ;;
  *)
    echo "Unsupported EVAL_SCOPE=$EVAL_SCOPE (use medqa or all)." >&2
    exit 2
    ;;
esac

test -f "$MEDQA_FILTER/model.safetensors"
test -f "$CANDIDATE_SOURCE"
test -f "$(dirname "$CANDIDATE_SOURCE")/manifest.json"
if [[ "$EVAL_SCOPE" == "all" ]]; then
  test -f "$MEDMCQA_FILTER/model.safetensors"
fi

COMMON_ARGS=(
  --datasets "${DATASETS[@]}"
  --collection unified
  --split test
  --prompt-profile paper_exact
  --rationale-artifact-root "$ARTIFACT_ROOT"
  --rationale-artifact-policy reuse_only
  --dense-query-mode rationale
  --vector-db-root "$PROJECT/databases/vector_db/RAG_Square"
  --sources pubmed pmc cpg textbooks
  --candidate-layout source_balanced
  --per-source-top-k 10
  --candidate-pool-top-k 40
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
  "${ROUTE_ARGS[@]}"
  --filter-batch-size 128
  --filter-max-input-length 768
  --filter-max-new-tokens 1
  --filter-max-doc-chars 0
  --filter-device cuda:0
  --filter-bf16
  --filter-scoring-method special_token
  --filter-input-format official
  --filter-score-normalization mean
  --filter-evidence-unit document
  --filter-generation-context-unit document
  --max-doc-chars 0
  --document-packing dynamic_token_budget
  --document-token-safety-margin 128
  --llm-model-path /home/user/Uiheon/models/Llama-3-8B-Instruct
  --generation-batch-size 32
  --max-new-tokens 768
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

# 0) Fail fast if the requested sample subset and cache contract do not match.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" --case filter_rag --filter-cache-only --dry-run

# 1) Record the unchanged no-RAG reference on the same parse-valid examples.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" --case no_rag

# 2) Materialise the exact subset of the old dense-40/rerank-32 cache once.
#    No dense retrieval or cross-encoder reranking is recomputed.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" --case rerank_rag --candidate-cache-only

# 3) Score all reranked Top-32 chunks once and write a checkpoint-fingerprinted
#    filter cache.  Subsequent prefix runs reuse this cache unchanged.
"$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" --case filter_rag --filter-cache-only

# 4) k is the MedCPT rerank prefix, not a post-filter cap.  Every passing
#    document within each prefix is given to the answer LLM in rerank order.
for TOP_K in 1 2 4 8 16 32; do
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
    --case filter_rag \
    --filter-rerank-top-k "$TOP_K"
done
