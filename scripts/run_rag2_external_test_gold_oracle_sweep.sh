#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
EVALUATOR="$PROJECT/scripts/run_rag2_mcq_eval.py"
LLAMA="/home/user/Uiheon/models/Llama-3-8B-Instruct"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1/all_mcq_source_balanced32_rationale_full_rerank32/candidates/07083d5bac341d9b/candidates.jsonl}"
RATIONALE_ROOT="${RATIONALE_ROOT:-$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1/no_rag_rationales}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT/databases/run_cache/rag2_external_test_gold_oracle_v1}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT/results/rag2_external_test_gold_oracle_v1}"
RAG2_LABELS="${RAG2_LABELS:-}"
HIDDEN_LABELS="${HIDDEN_LABELS:-}"
ANSWER_MODES="${ANSWER_MODES:-free_generation constrained_choice}"
POLICIES="${POLICIES:-rag2 hidden_tau_0 hidden_tau_0p4}"
TOP_K_VALUES="${TOP_K_VALUES:-1 2 4 8 16 32}"
RUN_NO_RAG="${RUN_NO_RAG:-1}"

read -r -a ANSWER_MODE_LIST <<< "$ANSWER_MODES"
read -r -a POLICY_LIST <<< "$POLICIES"
read -r -a TOP_K_LIST <<< "$TOP_K_VALUES"

for MODE in "${ANSWER_MODE_LIST[@]}"; do
  if [[ "$MODE" != "free_generation" && "$MODE" != "constrained_choice" ]]; then
    echo "Unsupported ANSWER_MODES entry: $MODE" >&2
    exit 2
  fi
done
for POLICY in "${POLICY_LIST[@]}"; do
  case "$POLICY" in
    rag2)
      [[ -n "$RAG2_LABELS" && -f "$RAG2_LABELS" ]] || {
        echo "POLICIES includes rag2, but RAG2_LABELS is not a readable file: $RAG2_LABELS" >&2
        exit 2
      }
      ;;
    hidden_tau_0|hidden_tau_0p4)
      [[ -n "$HIDDEN_LABELS" && -f "$HIDDEN_LABELS" ]] || {
        echo "POLICIES includes $POLICY, but HIDDEN_LABELS is not a readable file: $HIDDEN_LABELS" >&2
        exit 2
      }
      ;;
    *)
      echo "Unsupported POLICIES entry: $POLICY" >&2
      exit 2
      ;;
  esac
done
for TOP_K in "${TOP_K_LIST[@]}"; do
  if [[ ! "$TOP_K" =~ ^(1|2|4|8|16|32)$ ]]; then
    echo "Unsupported TOP_K_VALUES entry: $TOP_K" >&2
    exit 2
  fi
done

COMMON=(
  --datasets medmcqa medqa mmlu_anatomy mmlu_clinical_knowledge
             mmlu_college_biology mmlu_college_medicine
             mmlu_medical_genetics mmlu_professional_medicine
  --collection unified --split test
  --benchmark-root "$PROJECT/datasets/benchmark"
  --prompt-profile paper_exact_terminal
  --rationale-artifact-root "$RATIONALE_ROOT"
  --rationale-artifact-policy reuse_only
  --dense-query-mode rationale
  --candidate-cache-source-path "$CANDIDATE_SOURCE"
  --cache-root "$CACHE_ROOT"
  --vector-db-root "$PROJECT/databases/vector_db/RAG_Square"
  --sources pubmed pmc cpg textbooks --candidate-layout source_balanced
  --per-source-top-k 8 --candidate-pool-top-k 32 --rerank-top-k 32
  --query-encoder-path /home/user/Uiheon/models/MedCPT-Query-Encoder
  --cross-encoder-path /home/user/Uiheon/models/MedCPT-Cross-Encoder
  --query-max-length 512 --cross-encoder-max-length 512
  --max-doc-chars 0 --document-packing dynamic_token_budget --document-token-safety-margin 128
  --llm-model-path "$LLAMA" --generation-batch-size 128
  --rationale-max-new-tokens 768 --rationale-length-retry-max-new-tokens 768
  --temperature 0.0 --top-p 1.0 --format-retry-attempts 0
  --gpu-memory-utilization 0.92 --llm-max-model-len 8192
  --gdn-prefill-backend triton --vllm-performance-mode throughput
  --vllm-max-num-seqs 160 --vllm-max-num-batched-tokens 65536
)

EXTRA=()
[[ "${DRY_RUN:-0}" == "1" ]] && EXTRA+=(--dry-run)

run_once() {
  local case_root="$1"
  shift
  if [[ "${DRY_RUN:-0}" != "1" ]] && [[ -d "$case_root" ]] \
     && find "$case_root" -mindepth 2 -maxdepth 2 -type f -name results.jsonl \
          -exec sh -c '[ "$(wc -l < "$1")" -eq 6545 ]' _ {} \; -print -quit | grep -q .; then
    echo "Complete 6,545-row result exists; skipping: $case_root"
    return
  fi
  "$@"
}

for MODE in "${ANSWER_MODE_LIST[@]}"; do
  MODE_ROOT="$RESULTS_ROOT/$MODE"
  MAX_NEW=768
  [[ "$MODE" == "constrained_choice" ]] && MAX_NEW=1
  if [[ "$RUN_NO_RAG" == "1" ]]; then
    run_once "$MODE_ROOT/no_rag/no_rag" "$PYTHON" "$EVALUATOR" "${COMMON[@]}" --case no_rag \
      --answer-decision-mode "$MODE" --max-new-tokens "$MAX_NEW" \
      --results-root "$MODE_ROOT/no_rag" "${EXTRA[@]}"
  fi

  for POLICY in "${POLICY_LIST[@]}"; do
    LABEL_PATH="$HIDDEN_LABELS"
    [[ "$POLICY" == "rag2" ]] && LABEL_PATH="$RAG2_LABELS"
    for TOP_K in "${TOP_K_LIST[@]}"; do
      run_once "$MODE_ROOT/$POLICY/oracle_rag_${POLICY}_top${TOP_K}" "$PYTHON" "$EVALUATOR" "${COMMON[@]}" \
        --case oracle_rag --oracle-policy "$POLICY" --oracle-labels-path "$LABEL_PATH" \
        --filter-rerank-top-k "$TOP_K" --answer-decision-mode "$MODE" \
        --max-new-tokens "$MAX_NEW" --results-root "$MODE_ROOT/$POLICY" "${EXTRA[@]}"
    done
  done
done
