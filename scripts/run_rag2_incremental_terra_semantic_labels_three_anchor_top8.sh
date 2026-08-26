#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
REPO=/home/user/Uiheon/Medical_RAG
MODE=${1:-all}

NEW_CANDIDATE_ROOT="$REPO/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/candidates/source_balanced32_rerank8_v1"
OLD_LABEL_ROOT=/home/user/codex_rag2_outputs/codex_evidence_utility_labels_top8_terra_medium_v1_resume_from_1255280_a/terra_medium
RUN_ROOT=/home/user/codex_rag2_outputs/codex_evidence_utility_labels_three_anchor_top8_terra_medium_v1_incremental
PREPARED_ROOT="$RUN_ROOT/_incremental_preparation"
PENDING_ROOT="$RUN_ROOT/_pending_annotation"
FINAL_ROOT="$RUN_ROOT/terra_medium"

CANDIDATES=(
  "$NEW_CANDIDATE_ROOT/medmcqa/train/candidates_top8.jsonl"
  "$NEW_CANDIDATE_ROOT/medqa/train/candidates_top8.jsonl"
)
OLD_LABELS=(
  "$OLD_LABEL_ROOT/medmcqa/codex_semantic_labels.jsonl"
  "$OLD_LABEL_ROOT/medqa/codex_semantic_labels.jsonl"
)

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

stage() {
  printf '[%s] Overall stage %s/3: %s | overall ETA: %s\n' "$(timestamp)" "$1" "$2" "$3"
}

prepare() {
  stage 1 'pair-level reuse indexing and pending-candidate materialization' 'measured by the two live progress bars'
  "$PYTHON" "$REPO/scripts/materialize_rag2_incremental_semantic_labels.py" --log-level INFO prepare \
    --candidates-paths "${CANDIDATES[@]}" \
    --existing-label-paths "${OLD_LABELS[@]}" \
    --existing-manifest-path "$OLD_LABEL_ROOT/manifest.json" \
    --output-root "$PREPARED_ROOT" \
    --docs-per-question 8 \
    --sqlite-work-dir /tmp \
    --resume
}

label_pending() {
  stage 2 'GPT-5.6 Terra-medium semantic annotation of pending pairs' 'shown by the live rolling ETA below; merge ETA becomes known in stage 3'
  printf '[%s] Note: semantic annotation uses the Codex service, not CUDA; CUDA_VISIBLE_DEVICES has no effect on this stage.\n' "$(timestamp)"
  printf '[%s] Frozen original contract: prompt=%s model=%s reasoning=%s Top-8 questions/call=10 web=off workers=8 max_doc_chars=0\n' \
    "$(timestamp)" 'rag2_codex_evidence_utility_prompt_v3_compact_item_index' 'gpt-5.6-terra' 'medium'
  "$PYTHON" "$REPO/scripts/run_rag2_codex_labeling_pilot.py" \
    --candidates-paths \
      "$PREPARED_ROOT/pending_candidates/medmcqa.jsonl" \
      "$PREPARED_ROOT/pending_candidates/medqa.jsonl" \
    --output-root "$PENDING_ROOT" \
    --progress-db-path "$PENDING_ROOT/terra_medium/progress.sqlite" \
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
  stage 3 'merge reused and newly generated labels, then verify every Top-8 pair' 'measured by the two live progress bars'
  "$PYTHON" "$REPO/scripts/materialize_rag2_incremental_semantic_labels.py" --log-level INFO merge \
    --prepared-root "$PREPARED_ROOT" \
    --new-label-root "$PENDING_ROOT/terra_medium" \
    --output-root "$FINAL_ROOT" \
    --sqlite-work-dir /tmp \
    --resume
  printf '[%s] Overall 3/3 complete: %s\n' "$(timestamp)" "$FINAL_ROOT"
}

case "$MODE" in
  all)
    prepare
    label_pending
    merge
    ;;
  prepare)
    prepare
    ;;
  label)
    label_pending
    ;;
  merge)
    merge
    ;;
  *)
    printf 'Usage: %s [all|prepare|label|merge]\n' "$0" >&2
    exit 2
    ;;
esac
