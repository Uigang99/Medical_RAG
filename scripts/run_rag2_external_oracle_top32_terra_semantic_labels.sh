#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
REPO=/home/user/Uiheon/Medical_RAG
MODE=${1:-all}

CACHE_ROOT="$REPO/databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_max32_rationale_answer_rerank128/candidates/521e23c599352822"
RUN_ROOT=/home/user/codex_rag2_outputs/codex_evidence_utility_labels_external_oracle_top32_terra_medium_v1
PREPARED_ROOT="$RUN_ROOT/_prepared_candidates"
LABEL_PARENT="$RUN_ROOT/_annotation"
LABEL_ROOT="$LABEL_PARENT/terra_medium"
PROGRESS_DB="$LABEL_ROOT/progress.sqlite"

DATASETS=(
  medmcqa
  medqa
  mmlu_anatomy
  mmlu_clinical_knowledge
  mmlu_college_biology
  mmlu_college_medicine
  mmlu_medical_genetics
  mmlu_professional_medicine
)

CANDIDATES=()
for dataset in "${DATASETS[@]}"; do
  CANDIDATES+=("$PREPARED_ROOT/candidates/$dataset.jsonl")
done

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

stage() {
  printf '[%s] Overall stage %s/3: %s | overall ETA: %s\n' "$(timestamp)" "$1" "$2" "$3"
}

prepare() {
  stage 1 'join 6,545 benchmark questions with cached rerank Top-32' 'shown by overall/stage progress bars below'
  "$PYTHON" "$REPO/scripts/materialize_rag2_external_oracle_semantic_candidates.py" --log-level INFO prepare \
    --candidate-cache-path "$CACHE_ROOT/candidates.jsonl" \
    --candidate-cache-manifest-path "$CACHE_ROOT/manifest.json" \
    --benchmark-root "$REPO/datasets/benchmark/mcq/unified" \
    --output-root "$PREPARED_ROOT" \
    --datasets "${DATASETS[@]}" \
    --top-k 32 \
    --documents-per-block 8 \
    --resume
}

label() {
  stage 2 'GPT-5.6 Terra-medium semantic annotation of 209,440 external-oracle pairs' 'shown by the live rolling pair-rate/ETA monitor below'
  printf '[%s] Semantic annotation uses the Codex account/service, not CUDA. No GPU is used.\n' "$(timestamp)"
  printf '[%s] Frozen contract: prompt=%s model=%s reasoning=%s documents/block=8 blocks/call=10 pairs/call<=80 web=off workers=8 max_doc_chars=0\n' \
    "$(timestamp)" 'rag2_codex_evidence_utility_prompt_v3_compact_item_index' 'gpt-5.6-terra' 'medium'
  "$PYTHON" "$REPO/scripts/run_rag2_codex_labeling_pilot.py" \
    --candidates-paths "${CANDIDATES[@]}" \
    --output-root "$LABEL_PARENT" \
    --progress-db-path "$PROGRESS_DB" \
    --docs-per-question 8 \
    --no-allow-fewer-documents \
    --questions-per-batch 10 \
    --limit-questions 0 \
    --workers 8 \
    --rebalance-pending-batches \
    --max-attempts 3 \
    --retry-backoff-seconds 60 \
    --retry-jitter-fraction 0.25 \
    --max-worker-restarts 20 \
    --worker-restart-backoff-seconds 30 \
    --worker-start-stagger-seconds 2 \
    --refresh-seconds 10 \
    --variant terra_medium:gpt-5.6-terra:medium \
    --no-enable-web-search \
    --resume \
    --log-level INFO
}

verify() {
  stage 3 'verify exact Top-32 coverage, identities, ranks, and frozen annotation contract' 'shown by overall/stage progress bars below'
  "$PYTHON" "$REPO/scripts/materialize_rag2_external_oracle_semantic_candidates.py" --log-level INFO verify \
    --prepared-root "$PREPARED_ROOT" \
    --label-root "$LABEL_ROOT" \
    --datasets "${DATASETS[@]}" \
    --output-path "$LABEL_ROOT/external_oracle_top32_verification_report.json"
  printf '[%s] Overall 3/3 complete: questions=6,545 pairs=209,440 output=%s\n' "$(timestamp)" "$LABEL_ROOT"
}

case "$MODE" in
  all)
    prepare
    label
    verify
    ;;
  prepare)
    prepare
    ;;
  label)
    label
    ;;
  verify)
    verify
    ;;
  *)
    printf 'Usage: %s [all|prepare|label|verify]\n' "$0" >&2
    exit 2
    ;;
esac
