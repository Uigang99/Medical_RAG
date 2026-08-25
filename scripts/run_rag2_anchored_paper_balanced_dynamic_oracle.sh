#!/usr/bin/env bash
# Gold-label Oracle for the exact paper-balanced dynamic Top-k MCQ sweep.
#
# Stages:
#   1) reconstruct each 4k -> Top-k condition and materialize its document union,
#   2) independently generate one-document anchored rationale/answer traces,
#   3) apply the frozen train-derived RAG2 tau to produce gold labels,
#   4) evaluate Helpful-only Oracle context at k=1,2,4,8,16,32,
#   5) merge No-RAG, unfiltered, learned-filter, and Oracle results.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
LLAMA="/home/user/Uiheon/models/Llama-3-8B-Instruct"
DATA_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RUN_CACHE_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1"

MASTER_CANDIDATES="${MASTER_CANDIDATES:-$RUN_CACHE_ROOT/all_mcq_paper_balanced_max32_rationale_answer_rerank128/candidates/521e23c599352822/candidates.jsonl}"
NO_RAG_ROOT="${NO_RAG_ROOT:-$RUN_CACHE_ROOT/no_rag_rationales_all_mcq_test}"
TRAIN_LABEL_ROOT="${TRAIN_LABEL_ROOT:-$DATA_ROOT/filter_training_inputs_rag2_paper_reproduction_v1}"
ORACLE_ROOT="${ORACLE_ROOT:-$DATA_ROOT/external_test_dynamic_topk_rag2_oracle_v1}"
CANDIDATE_ROOT="$ORACLE_ROOT/candidates_topk_union"
TRACE_ROOT="$ORACLE_ROOT/document_traces_topk_union"
LABEL_ROOT="$ORACLE_ROOT/rag2_labels"
ORACLE_CACHE_ROOT="${ORACLE_CACHE_ROOT:-$RUN_CACHE_ROOT/all_mcq_paper_balanced_dynamic_topk_rag2_oracle_v1}"
ORACLE_RESULTS_ROOT="${ORACLE_RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/rag2_gold_oracle}"
REFERENCE_RESULTS_ROOT="${REFERENCE_RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/paper_reproduction_filter}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/oracle_comparison_summary}"

# 0.50 reserves about 72 GiB on an H200. Together with the currently observed
# 43-48 GiB training process this leaves roughly 24 GiB headroom. When GPU 1 is
# otherwise idle, override with GPU_MEMORY_UTILIZATION=0.92 for best throughput.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-48}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
TOP_K_VALUES=(1 2 4 8 16 32)
DATASETS=(medmcqa medqa mmlu_anatomy mmlu_clinical_knowledge mmlu_college_biology mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine)

test -f "$MASTER_CANDIDATES"
test -f "$NO_RAG_ROOT/generation_manifest.json"
test -f "$TRAIN_LABEL_ROOT/medmcqa/manifest.json"
test -f "$TRAIN_LABEL_ROOT/medqa/manifest.json"
test -f "$LLAMA/config.json"

TOTAL_STAGES=10
PIPELINE_START=$(date +%s)

format_seconds() {
  local seconds="$1"
  printf '%02dh%02dm%02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

announce_stage() {
  local index="$1"
  local name="$2"
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - PIPELINE_START))
  printf '[overall %d/%d] stage=%s elapsed=%s\n' "$index" "$TOTAL_STAGES" "$name" "$(format_seconds "$elapsed")"
}

complete_result_exists() {
  local case_root="$1"
  [[ -d "$case_root" ]] || return 1
  find "$case_root" -mindepth 2 -maxdepth 2 -type f -name results.jsonl \
    -exec sh -c '[ "$(wc -l < "$1")" -eq 6545 ]' _ {} \; -print -quit | grep -q .
}

announce_stage 1 "materialize exact dynamic Top-k document union"
"$PYTHON" "$PROJECT/scripts/prepare_rag2_anchored_dynamic_oracle_candidates.py" \
  --master-candidates-path "$MASTER_CANDIDATES" \
  --no-rag-root "$NO_RAG_ROOT" \
  --output-root "$CANDIDATE_ROOT" \
  --datasets "${DATASETS[@]}" \
  --split test \
  --top-k-values "${TOP_K_VALUES[@]}" \
  --sources pubmed pmc cpg textbooks \
  --master-per-source-top-k 32 \
  --log-level INFO

announce_stage 2 "independent one-document anchored rationale+answer generation"
"$PYTHON" "$PROJECT/scripts/generate_rag2_anchored_document_traces.py" \
  --datasets "${DATASETS[@]}" \
  --split test \
  --candidate-root "$CANDIDATE_ROOT" \
  --candidate-file candidates_topk_union.jsonl \
  --docs-per-question 1 \
  --allow-variable-docs-per-question \
  --model-name-or-path "$LLAMA" \
  --output-root "$TRACE_ROOT" \
  --questions-per-shard 128 \
  --generation-batch-size "$GENERATION_BATCH_SIZE" \
  --max-new-tokens 512 \
  --retry-max-new-tokens 768 \
  --temperature 0.0 \
  --top-p 1.0 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --llm-max-model-len 8192 \
  --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --vllm-max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --vllm-performance-mode throughput \
  --max-doc-chars 0 \
  --resume \
  --log-level INFO

announce_stage 3 "apply frozen train RAG2 tau and materialize gold labels"
"$PYTHON" "$PROJECT/scripts/materialize_rag2_anchored_external_oracle_labels.py" \
  --candidate-root "$CANDIDATE_ROOT" \
  --document-trace-root "$TRACE_ROOT" \
  --no-rag-root "$NO_RAG_ROOT" \
  --training-label-root "$TRAIN_LABEL_ROOT" \
  --output-root "$LABEL_ROOT" \
  --datasets "${DATASETS[@]}" \
  --split test \
  --candidate-file candidates_topk_union.jsonl \
  --log-level INFO

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
  --oracle-policy rag2
  --oracle-labels-path "$LABEL_ROOT/rag2_oracle_labels.jsonl"
  --results-root "$ORACLE_RESULTS_ROOT"
)

STAGE=3
for TOP_K in "${TOP_K_VALUES[@]}"; do
  STAGE=$((STAGE + 1))
  announce_stage "$STAGE" "RAG2 gold Oracle: paper-balanced 4x${TOP_K} -> Top-${TOP_K}"
  CASE_ROOT="$ORACLE_RESULTS_ROOT/oracle_rag_rag2_top${TOP_K}"
  if complete_result_exists "$CASE_ROOT"; then
    echo "Complete 6,545-row Oracle result exists; reusing: $CASE_ROOT"
    continue
  fi
  "$PYTHON" "$PROJECT/scripts/run_rag2_mcq_eval.py" "${COMMON[@]}" \
    --paper-balanced-top-k "$TOP_K"
done

announce_stage 10 "validate and summarize learned-filter versus gold-label Oracle"
"$PYTHON" "$PROJECT/scripts/summarize_rag2_anchored_paper_reproduction_sweep.py" \
  --results-root "$REFERENCE_RESULTS_ROOT" \
  --oracle-results-root "$ORACLE_RESULTS_ROOT" \
  --output-dir "$SUMMARY_ROOT" \
  --expected-prompt-profile paper_compatible_three_anchor \
  --expected-answer-decision-mode free_generation \
  --expected-per-source-top-k 32 \
  --expected-candidate-pool-top-k 128 \
  --expected-rerank-top-k 128 \
  --expected-paper-balanced-projection

FINISHED=$(date +%s)
printf '[overall %d/%d] complete elapsed=%s\n' "$TOTAL_STAGES" "$TOTAL_STAGES" "$(format_seconds "$((FINISHED - PIPELINE_START))")"
