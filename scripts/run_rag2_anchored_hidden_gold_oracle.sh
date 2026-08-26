#!/usr/bin/env bash
# Hidden-state three-class gold oracle on the exact existing RAG2 MCQ cohort.
#
# No no-RAG or unfiltered answer experiment is rerun.  Existing no-RAG traces,
# retrieval/rerank candidates, and one-document traces are replayed only to
# obtain h0/hD/c.  At evaluation, only Hidden Helpful documents are passed;
# Neutral and Harmful are blocked, and zero Helpful falls back to cached no-RAG.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
LLAMA="/home/user/Uiheon/models/Llama-3-8B-Instruct"
DATA_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RUN_CACHE_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1"

MASTER_CANDIDATES="${MASTER_CANDIDATES:-$RUN_CACHE_ROOT/all_mcq_paper_balanced_max32_rationale_answer_rerank128/candidates/521e23c599352822/candidates.jsonl}"
NO_RAG_TRACE_ROOT="${NO_RAG_TRACE_ROOT:-$RUN_CACHE_ROOT/no_rag_rationales_all_mcq_test}"
ORACLE_ROOT="${ORACLE_ROOT:-$DATA_ROOT/external_test_dynamic_topk_rag2_oracle_v1}"
DOCUMENT_TRACE_ROOT="${DOCUMENT_TRACE_ROOT:-$ORACLE_ROOT/document_traces_topk_union}"
RAG2_LABELS_PATH="${RAG2_LABELS_PATH:-$ORACLE_ROOT/rag2_labels/rag2_oracle_labels.jsonl}"

HIDDEN_ROOT="${HIDDEN_ROOT:-$ORACLE_ROOT/hidden_gold_oracle_tau0p4_v1}"
NO_RAG_FEATURE_ROOT="$HIDDEN_ROOT/no_rag_features"
DOCUMENT_FEATURE_ROOT="$HIDDEN_ROOT/document_features"
DIRECTION_ROOT="$HIDDEN_ROOT/gold_directions"
LABEL_ROOT="$HIDDEN_ROOT/hidden_labels"

ORACLE_CACHE_ROOT="${ORACLE_CACHE_ROOT:-$RUN_CACHE_ROOT/all_mcq_paper_balanced_dynamic_topk_hidden_gold_oracle_tau0p4_v1}"
HIDDEN_RESULTS_ROOT="${HIDDEN_RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/hidden_gold_oracle_tau0p4}"
RAG2_RESULTS_ROOT="${RAG2_RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/rag2_gold_oracle}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/rag2_vs_hidden_gold_oracle_tau0p4}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-80}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
HIDDEN_THRESHOLD="${HIDDEN_THRESHOLD:-0.4}"
LAYER="${LAYER:-28}"
ANCHOR="${ANCHOR:-pre_choice}"

TOP_K_VALUES=(1 2 4 8 16 32)
DATASETS=(medmcqa medqa mmlu_anatomy mmlu_clinical_knowledge mmlu_college_biology mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine)

test -f "$MASTER_CANDIDATES"
test -f "$NO_RAG_TRACE_ROOT/generation_manifest.json"
test -f "$DOCUMENT_TRACE_ROOT/generation_manifest.json"
test -f "$RAG2_LABELS_PATH"
test -f "$LLAMA/config.json"

TOTAL_STAGES=11
PIPELINE_START=$(date +%s)
COMPLETED_STAGES=0

format_seconds() {
  local seconds="$1"
  printf '%02dh%02dm%02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

announce_stage() {
  local index="$1"
  local name="$2"
  local now elapsed eta_text
  now=$(date +%s)
  elapsed=$((now - PIPELINE_START))
  if (( COMPLETED_STAGES > 0 )); then
    eta_text=$(format_seconds "$((elapsed * (TOTAL_STAGES - COMPLETED_STAGES) / COMPLETED_STAGES))")
  else
    eta_text="unknown"
  fi
  printf '[overall %d/%d] stage=%s elapsed=%s measured_stage_eta=%s\n' \
    "$index" "$TOTAL_STAGES" "$name" "$(format_seconds "$elapsed")" "$eta_text"
}

finish_stage() {
  COMPLETED_STAGES=$((COMPLETED_STAGES + 1))
}

complete_result_exists() {
  local case_root="$1"
  [[ -d "$case_root" ]] || return 1
  find "$case_root" -mindepth 2 -maxdepth 2 -type f -name results.jsonl \
    -exec sh -c '[ "$(wc -l < "$1")" -eq 6545 ]' _ {} \; -print -quit | grep -q .
}

announce_stage 1 "extract cached no-RAG Block-${LAYER}/${ANCHOR} states"
"$PYTHON" "$PROJECT/scripts/extract_rag2_anchored_no_rag_features.py" \
  --trace-root "$NO_RAG_TRACE_ROOT" \
  --output-root "$NO_RAG_FEATURE_ROOT" \
  --model-name-or-path "$LLAMA" \
  --datasets "${DATASETS[@]}" \
  --split test \
  --layers "$LAYER" \
  --batch-size "$FEATURE_BATCH_SIZE" \
  --max-input-tokens 8192 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --resume \
  --log-level INFO
finish_stage

announce_stage 2 "extract cached one-document Block-${LAYER}/${ANCHOR} states"
"$PYTHON" "$PROJECT/scripts/extract_rag2_anchored_document_features.py" \
  --trace-root "$DOCUMENT_TRACE_ROOT" \
  --output-root "$DOCUMENT_FEATURE_ROOT" \
  --no-rag-feature-root "$NO_RAG_FEATURE_ROOT" \
  --model-name-or-path "$LLAMA" \
  --datasets "${DATASETS[@]}" \
  --split test \
  --layers "$LAYER" \
  --batch-size "$FEATURE_BATCH_SIZE" \
  --max-input-tokens 8192 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --minimum-free-space-gib 20 \
  --resume \
  --log-level INFO
finish_stage

announce_stage 3 "extract no-RAG gold-answer directions"
"$PYTHON" "$PROJECT/scripts/extract_rag2_anchored_gold_directions.py" \
  --no-rag-root "$NO_RAG_FEATURE_ROOT" \
  --model-name-or-path "$LLAMA" \
  --output-root "$DIRECTION_ROOT" \
  --datasets "${DATASETS[@]}" \
  --split test \
  --layer "$LAYER" \
  --anchor "$ANCHOR" \
  --question-batch-size "$FEATURE_BATCH_SIZE" \
  --max-input-tokens 8192 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --resume \
  --log-level INFO
finish_stage

announce_stage 4 "materialize Helpful/Neutral/Harmful at tau=${HIDDEN_THRESHOLD}"
"$PYTHON" "$PROJECT/scripts/materialize_rag2_anchored_external_hidden_labels.py" \
  --no-rag-feature-root "$NO_RAG_FEATURE_ROOT" \
  --document-feature-root "$DOCUMENT_FEATURE_ROOT" \
  --direction-root "$DIRECTION_ROOT" \
  --reference-rag2-labels-path "$RAG2_LABELS_PATH" \
  --output-root "$LABEL_ROOT" \
  --datasets "${DATASETS[@]}" \
  --split test \
  --layer "$LAYER" \
  --anchor "$ANCHOR" \
  --threshold "$HIDDEN_THRESHOLD" \
  --tensor-cache-shards 24 \
  --resume \
  --log-level INFO
finish_stage

COMMON=(
  --datasets "${DATASETS[@]}"
  --collection unified
  --split test
  --prompt-profile paper_compatible_three_anchor
  --answer-decision-mode free_generation
  --rationale-artifact-root "$NO_RAG_TRACE_ROOT"
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
  --oracle-policy hidden_three_class
  --oracle-labels-path "$LABEL_ROOT/hidden_oracle_labels.jsonl"
  --results-root "$HIDDEN_RESULTS_ROOT"
)

STAGE=4
for TOP_K in "${TOP_K_VALUES[@]}"; do
  STAGE=$((STAGE + 1))
  announce_stage "$STAGE" "Hidden Helpful-only oracle: paper-balanced 4x${TOP_K} -> Top-${TOP_K}"
  CASE_ROOT="$HIDDEN_RESULTS_ROOT/oracle_rag_hidden_three_class_top${TOP_K}"
  if complete_result_exists "$CASE_ROOT"; then
    echo "Complete 6,545-row Hidden Oracle result exists; reusing: $CASE_ROOT"
  else
    "$PYTHON" "$PROJECT/scripts/run_rag2_mcq_eval.py" "${COMMON[@]}" \
      --paper-balanced-top-k "$TOP_K"
  fi
  finish_stage
done

announce_stage 11 "summarize RAG2 versus hidden-state gold oracle"
"$PYTHON" "$PROJECT/scripts/summarize_rag2_anchored_hidden_oracle.py" \
  --rag2-oracle-results-root "$RAG2_RESULTS_ROOT" \
  --hidden-oracle-results-root "$HIDDEN_RESULTS_ROOT" \
  --hidden-label-manifest "$LABEL_ROOT/manifest.json" \
  --output-dir "$SUMMARY_ROOT" \
  --expected-prompt-profile paper_compatible_three_anchor \
  --expected-answer-decision-mode free_generation \
  --expected-per-source-top-k 32 \
  --expected-candidate-pool-top-k 128 \
  --expected-rerank-top-k 128
finish_stage

FINISHED=$(date +%s)
printf '[overall %d/%d] complete elapsed=%s summary=%s\n' \
  "$TOTAL_STAGES" "$TOTAL_STAGES" "$(format_seconds "$((FINISHED - PIPELINE_START))")" \
  "$SUMMARY_ROOT/summary_table_pretty.txt"
