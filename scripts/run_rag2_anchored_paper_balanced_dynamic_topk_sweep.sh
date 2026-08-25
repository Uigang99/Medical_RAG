#!/usr/bin/env bash
# Paper-described RAG2 balanced-retrieval sweep:
#   for every k, dense Top-k per logical corpus -> 4k candidates -> MedCPT Top-k.
#
# Expensive work is performed once at the maximum k=32:
#   1) no-RAG rationale+answer artifacts,
#   2) dense Top-32 x four corpora and cross-encoder scores for all 128 docs,
#   3) dataset-routed filter scores for all 128 docs.
# Every generation stage reconstructs its exact 4k pool from the cached
# source-local dense ranks, then selects the MedCPT Top-k. This is equivalent
# to rerunning retrieval/reranking per k without repeating either operation.

set -euo pipefail

# The research workflow is pinned to physical GPU 1. Inside the process it is
# exposed as logical cuda:0, so FAISS/filter device arguments remain zero.
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"
SUMMARIZER="$PROJECT/scripts/summarize_rag2_anchored_paper_reproduction_sweep.py"

LLAMA_MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
MEDMCQA_FILTER="/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperReproduction-Anchored/medmcqa/medmcqa_rag2_paper_reproduction_epoch6_len512_stride128/20260824_213214/final_model"
MEDQA_FILTER="/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperReproduction-Anchored/medqa/medqa_rag2_paper_reproduction_epoch15_len512_stride128/20260825_075321/final_model"

EXPERIMENT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$EXPERIMENT_ROOT/no_rag_rationales_all_mcq_test}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/all_mcq_paper_balanced_max32_rationale_answer_rerank128}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/paper_reproduction_filter}"

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
  --prompt-profile paper_compatible_three_anchor
  --answer-decision-mode free_generation
  --rationale-artifact-root "$ARTIFACT_ROOT"
  --rationale-artifact-policy repair_invalid
  --dense-query-mode rationale
  --vector-db-root "$PROJECT/databases/vector_db/RAG_Square"
  --sources pubmed pmc cpg textbooks
  --candidate-layout source_balanced
  --per-source-top-k 32
  --candidate-pool-top-k 128
  # Retain cross-encoder scores and text for every master candidate. Individual
  # k runs reconstruct 4k candidates and take their Top-k with no rescoring.
  --rerank-top-k 128
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
  --filter-max-input-length 512
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
  --rationale-max-new-tokens 512
  --rationale-length-retry-attempts 1
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

TOTAL_STAGES=16
COMPLETED_STAGES=0
PIPELINE_STARTED_AT=$(date +%s)

format_seconds() {
  local value="$1"
  printf '%02dh%02dm%02ds' "$((value / 3600))" "$(((value % 3600) / 60))" "$((value % 60))"
}

run_stage() {
  local stage_index="$1"
  local stage_name="$2"
  shift 2
  local elapsed now eta_text
  now=$(date +%s)
  elapsed=$((now - PIPELINE_STARTED_AT))
  eta_text="unknown"
  if (( COMPLETED_STAGES > 0 )); then
    eta_text=$(format_seconds "$((elapsed * (TOTAL_STAGES - COMPLETED_STAGES) / COMPLETED_STAGES))")
  fi
  printf '[overall %d/%d] stage=%s elapsed=%s overall_eta=%s\n' \
    "$stage_index" "$TOTAL_STAGES" "$stage_name" "$(format_seconds "$elapsed")" "$eta_text"
  "$@"
  COMPLETED_STAGES="$stage_index"
}

# 1/16: this reuses the completed anchored no-RAG artifact when available.
run_stage 1 "no-RAG rationale+answer" \
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" --case no_rag

# 2/16: one exact dense retrieval and one cross-encoder pass at maximum k.
run_stage 2 "master balanced retrieval 4x32 and rerank-score cache (128 docs)" \
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case rerank_rag \
  --candidate-cache-only

# 3/16: score every master candidate once; all k conditions reuse this cache.
run_stage 3 "master dataset-routed filter-score cache (128 docs/question)" \
  "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
  --case filter_rag \
  --filter-cache-only

STAGE_INDEX=3
for TOP_K in 1 2 4 8 16 32; do
  STAGE_INDEX=$((STAGE_INDEX + 1))
  run_stage "$STAGE_INDEX" "unfiltered paper-balanced 4x${TOP_K} -> top-${TOP_K}" \
    "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
    --case rerank_rag \
    --paper-balanced-top-k "$TOP_K"
done

for TOP_K in 1 2 4 8 16 32; do
  STAGE_INDEX=$((STAGE_INDEX + 1))
  run_stage "$STAGE_INDEX" "filtered paper-balanced 4x${TOP_K} -> top-${TOP_K}" \
    "$PYTHON" "$EVALUATOR" "${COMMON_ARGS[@]}" \
    --case filter_rag \
    --paper-balanced-top-k "$TOP_K"
done

STAGE_INDEX=$((STAGE_INDEX + 1))
run_stage "$STAGE_INDEX" "combined result validation and summary" \
  "$PYTHON" "$SUMMARIZER" \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$RESULTS_ROOT/combined_summary" \
  --expected-prompt-profile paper_compatible_three_anchor \
  --expected-answer-decision-mode free_generation \
  --expected-per-source-top-k 32 \
  --expected-candidate-pool-top-k 128 \
  --expected-rerank-top-k 128 \
  --expected-paper-balanced-projection

PIPELINE_FINISHED_AT=$(date +%s)
printf '[overall %d/%d] complete elapsed=%s overall_eta=00h00m00s\n' \
  "$TOTAL_STAGES" "$TOTAL_STAGES" "$(format_seconds "$((PIPELINE_FINISHED_AT - PIPELINE_STARTED_AT))")"
