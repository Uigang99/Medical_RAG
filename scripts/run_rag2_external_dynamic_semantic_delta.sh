#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
REPO=/home/user/Uiheon/Medical_RAG
MODE=${1:-all}

DATA_ROOT="$REPO/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
UNION_ROOT="$DATA_ROOT/external_test_dynamic_topk_rag2_oracle_v1/candidates_topk_union"
EXISTING_ROOT=/home/user/codex_rag2_outputs/codex_evidence_utility_labels_external_oracle_top32_terra_medium_v1/_annotation/terra_medium
RUN_ROOT=/home/user/codex_rag2_outputs/codex_evidence_utility_labels_external_oracle_dynamic_topk_union_terra_medium_v1
PREPARED_ROOT="$RUN_ROOT/_delta_preparation"
PENDING_PARENT="$RUN_ROOT/_pending_annotation"
PENDING_ROOT="$PENDING_PARENT/terra_medium"
FINAL_ROOT="$RUN_ROOT/terra_medium"

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

PENDING_CANDIDATES=()
for dataset in "${DATASETS[@]}"; do
  PENDING_CANDIDATES+=("$PREPARED_ROOT/pending_candidates/$dataset.jsonl")
done

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

stage() {
  printf '[%s] Overall stage %s/3: %s | overall ETA: %s\n' "$(timestamp)" "$1" "$2" "$3"
}

prepare() {
  stage 1 'diff exact dynamic-k union against completed global Top-32 semantic labels' 'shown by overall/stage progress bars below'
  "$PYTHON" "$REPO/scripts/materialize_rag2_external_dynamic_semantic_delta.py" --log-level INFO prepare \
    --candidate-union-root "$UNION_ROOT" \
    --existing-label-root "$EXISTING_ROOT" \
    --output-root "$PREPARED_ROOT" \
    --datasets "${DATASETS[@]}" \
    --max-documents-per-block 8 \
    --resume
}

label() {
  stage 2 'GPT-5.6 Terra-medium annotation of only 2,435 missing dynamic-k pairs' 'shown by live rolling rate/ETA below'
  printf '[%s] Frozen contract: prompt=%s model=%s reasoning=%s max_pairs/call=80 web=off workers=8 max_doc_chars=0\n' \
    "$(timestamp)" 'rag2_codex_evidence_utility_prompt_v3_compact_item_index' 'gpt-5.6-terra' 'medium'
  printf '[%s] This stage uses the Codex service and does not use a GPU.\n' "$(timestamp)"
  "$PYTHON" "$REPO/scripts/run_rag2_codex_labeling_pilot.py" \
    --candidates-paths "${PENDING_CANDIDATES[@]}" \
    --output-root "$PENDING_PARENT" \
    --progress-db-path "$PENDING_ROOT/progress.sqlite" \
    --docs-per-question 8 \
    --allow-fewer-documents \
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

merge() {
  stage 3 'merge reused 209,440 + new 2,435 labels and verify all dynamic-k conditions' 'shown by overall/stage progress bars below'
  "$PYTHON" "$REPO/scripts/materialize_rag2_external_dynamic_semantic_delta.py" --log-level INFO merge \
    --prepared-root "$PREPARED_ROOT" \
    --new-label-root "$PENDING_ROOT" \
    --output-root "$FINAL_ROOT" \
    --datasets "${DATASETS[@]}" \
    --resume
  printf '[%s] Overall 3/3 complete: questions=6,545 pairs=211,875 output=%s\n' "$(timestamp)" "$FINAL_ROOT"
}

case "$MODE" in
  all)
    prepare
    label
    merge
    ;;
  prepare)
    prepare
    ;;
  label)
    label
    ;;
  merge)
    merge
    ;;
  *)
    printf 'Usage: %s [all|prepare|label|merge]\n' "$0" >&2
    exit 2
    ;;
esac
