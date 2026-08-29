#!/usr/bin/env bash
# Exact paper-balanced semantic gold-oracle sweep:
#   each corpus dense Top-k (4k total) -> MedCPT rerank Top-k -> semantic Oracle
# Policies:
#   1) direct_support only
#   2) direct_support + supporting_evidence

set -euo pipefail

# Physical GPU 1 is the default and appears as logical cuda:0 inside the job.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
LLAMA="/home/user/Uiheon/models/Llama-3-8B-Instruct"
DATA_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RUN_CACHE_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1"
MASTER_CACHE_ROOT="$RUN_CACHE_ROOT/all_mcq_paper_balanced_max32_rationale_answer_rerank128"
MASTER_CANDIDATES="${MASTER_CANDIDATES:-$MASTER_CACHE_ROOT/candidates/521e23c599352822/candidates.jsonl}"
NO_RAG_ROOT="${NO_RAG_ROOT:-$RUN_CACHE_ROOT/no_rag_rationales_all_mcq_test}"

SEMANTIC_RUN_ROOT="${SEMANTIC_RUN_ROOT:-/home/user/codex_rag2_outputs/codex_evidence_utility_labels_external_oracle_dynamic_topk_union_terra_medium_v1}"
SEMANTIC_LABEL_ROOT="${SEMANTIC_LABEL_ROOT:-$SEMANTIC_RUN_ROOT/terra_medium}"
CANDIDATE_UNION_ROOT="${CANDIDATE_UNION_ROOT:-$DATA_ROOT/external_test_dynamic_topk_rag2_oracle_v1/candidates_topk_union}"
ORACLE_EXPORT_ROOT="${ORACLE_EXPORT_ROOT:-$DATA_ROOT/external_test_dynamic_topk_semantic_oracle_v1}"
ORACLE_LABELS="${ORACLE_LABELS:-$ORACLE_EXPORT_ROOT/dynamic_semantic_oracle_labels.jsonl}"
ORACLE_LABEL_MANIFEST="${ORACLE_LABEL_MANIFEST:-$ORACLE_EXPORT_ROOT/dynamic_semantic_oracle_labels_manifest.json}"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT/results/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_dynamic_topk/semantic_gold_oracle}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RESULTS_ROOT/combined_summary}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-80}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"

DATASETS=(
  medmcqa medqa mmlu_anatomy mmlu_clinical_knowledge
  mmlu_college_biology mmlu_college_medicine
  mmlu_medical_genetics mmlu_professional_medicine
)
POLICIES=(semantic_direct semantic_direct_supporting)
TOP_K_VALUES=(1 2 4 8 16 32)

test -f "$MASTER_CANDIDATES"
test -f "$(dirname "$MASTER_CANDIDATES")/manifest.json"
test -f "$NO_RAG_ROOT/generation_manifest.json"
test -f "$CANDIDATE_UNION_ROOT/manifest.json"
test -f "$SEMANTIC_LABEL_ROOT/manifest.json"
test -f "$LLAMA/config.json"

TOTAL_STAGES=14
PIPELINE_START=$(date +%s)
COMPLETED_STAGES=0
MEASURED_STAGE_COUNT=0
MEASURED_STAGE_SECONDS=0
STAGE_START=$PIPELINE_START

format_seconds() {
  local seconds="$1"
  printf '%02dh%02dm%02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

eta_text() {
  if (( MEASURED_STAGE_COUNT == 0 )); then
    printf 'estimating'
  else
    local mean remaining
    mean=$((MEASURED_STAGE_SECONDS / MEASURED_STAGE_COUNT))
    remaining=$((TOTAL_STAGES - COMPLETED_STAGES))
    format_seconds "$((mean * remaining))"
  fi
}

announce_stage() {
  local name="$1"
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - PIPELINE_START))
  STAGE_START=$now
  printf '[overall %d/%d] stage=%s elapsed=%s overall_eta=%s\n' \
    "$COMPLETED_STAGES" "$TOTAL_STAGES" "$name" "$(format_seconds "$elapsed")" "$(eta_text)"
}

finish_stage() {
  local measured="${1:-1}"
  local now duration
  now=$(date +%s)
  duration=$((now - STAGE_START))
  COMPLETED_STAGES=$((COMPLETED_STAGES + 1))
  if [[ "$measured" == "1" ]]; then
    MEASURED_STAGE_COUNT=$((MEASURED_STAGE_COUNT + 1))
    MEASURED_STAGE_SECONDS=$((MEASURED_STAGE_SECONDS + duration))
  fi
  printf '[overall %d/%d] stage_complete elapsed=%s overall_eta=%s\n' \
    "$COMPLETED_STAGES" "$TOTAL_STAGES" \
    "$(format_seconds "$((now - PIPELINE_START))")" "$(eta_text)"
}

complete_result_exists() {
  local case_root="$1"
  [[ -d "$case_root" ]] || return 1
  find "$case_root" -mindepth 2 -maxdepth 2 -type f -name results.jsonl \
    -exec sh -c '[ "$(wc -l < "$1")" -eq 6545 ]' _ {} \; -print -quit | grep -q .
}

announce_stage "export and verify all 211,875 dynamic-k semantic decisions"
"$PYTHON" "$PROJECT/scripts/materialize_rag2_external_dynamic_semantic_delta.py" \
  --log-level INFO export-oracle \
  --candidate-union-root "$CANDIDATE_UNION_ROOT" \
  --semantic-label-root "$SEMANTIC_LABEL_ROOT" \
  --output-path "$ORACLE_LABELS" \
  --datasets "${DATASETS[@]}" \
  --resume
finish_stage

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
  --cache-root "$MASTER_CACHE_ROOT"
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
  --oracle-labels-path "$ORACLE_LABELS"
  --results-root "$RESULTS_ROOT"
)

for TOP_K in "${TOP_K_VALUES[@]}"; do
  for POLICY in "${POLICIES[@]}"; do
    announce_stage "$POLICY: 4 corpora x $TOP_K -> MedCPT Top-$TOP_K"
    CASE_ROOT="$RESULTS_ROOT/oracle_rag_${POLICY}_top${TOP_K}"
    if complete_result_exists "$CASE_ROOT"; then
      echo "Complete 6,545-row result exists; reusing: $CASE_ROOT"
      finish_stage 0
      continue
    fi
    "$PYTHON" "$PROJECT/scripts/run_rag2_mcq_eval.py" "${COMMON[@]}" \
      --oracle-policy "$POLICY" \
      --paper-balanced-top-k "$TOP_K"
    finish_stage
  done
done

announce_stage "validate all conditions and write one comparison table"
"$PYTHON" "$PROJECT/scripts/summarize_rag2_semantic_oracle_topk_sweep.py" \
  --results-root "$RESULTS_ROOT" \
  --semantic-label-manifest "$ORACLE_LABEL_MANIFEST" \
  --output-dir "$SUMMARY_ROOT" \
  --expected-prompt-profile paper_compatible_three_anchor \
  --expected-answer-decision-mode free_generation \
  --expected-per-source-top-k 32 \
  --expected-candidate-pool-top-k 128 \
  --expected-rerank-top-k 128 \
  --expected-paper-balanced-projection
finish_stage

FINISHED=$(date +%s)
printf '[overall %d/%d] complete elapsed=%s summary=%s\n' \
  "$COMPLETED_STAGES" "$TOTAL_STAGES" "$(format_seconds "$((FINISHED - PIPELINE_START))")" "$SUMMARY_ROOT/summary_table_pretty.txt"
