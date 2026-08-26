#!/usr/bin/env bash
set -euo pipefail

# Reuse unchanged pair-level semantic labels from the former Top-8 retrieval
# run, label only genuinely new pairs with the unchanged Codex prompt, and
# assemble a complete label file in the new candidate order.

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
CODEX_BIN=${CODEX_BIN:-codex}
MODEL=${MODEL:-gpt-5.6-terra}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
WORKERS=${WORKERS:-8}
PAIR_BUDGET_PER_CALL=${PAIR_BUDGET_PER_CALL:-80}

CANDIDATE_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/candidates/source_balanced32_rerank8_v1"
PRIOR_ROOT=/home/user/codex_rag2_outputs/codex_evidence_utility_labels_top8_terra_medium_v1_resume_from_1255280_a/terra_medium
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/user/codex_rag2_outputs/anchored_top8_semantic_labels_terra_medium_delta_v1}
DELTA_ROOT="$OUTPUT_ROOT/delta"
LABEL_RUNS_ROOT="$OUTPUT_ROOT/label_runs"
FINAL_ROOT="$OUTPUT_ROOT/final"

CANDIDATES=(
  "$CANDIDATE_ROOT/medmcqa/train/candidates_top8.jsonl"
  "$CANDIDATE_ROOT/medqa/train/candidates_top8.jsonl"
)
PRIOR_LABELS=(
  "$PRIOR_ROOT/medmcqa/codex_semantic_labels.jsonl"
  "$PRIOR_ROOT/medqa/codex_semantic_labels.jsonl"
)

for required in "$PYTHON" "$CODEX_BIN" "${CANDIDATES[@]}" "${PRIOR_LABELS[@]}"; do
  if [[ "$required" == */* && ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done
if (( WORKERS < 1 )); then
  echo "WORKERS must be positive" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LABEL_RUNS_ROOT" "$FINAL_ROOT"
active_pids=()
monitor_pid=""
cleanup() {
  local pid
  for pid in "${active_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

echo "[overall 1/10] prepare exact pair reuse and delta buckets (overall ETA unknown)"
"$PYTHON" "$PROJECT/scripts/materialize_rag2_semantic_label_delta.py" \
  --mode prepare \
  --candidates-paths "${CANDIDATES[@]}" \
  --existing-labels-paths "${PRIOR_LABELS[@]}" \
  --delta-root "$DELTA_ROOT" \
  --docs-per-question 8

for k in 1 2 3 4 5 6 7 8; do
  stage=$((k + 1))
  bucket_candidates=()
  for dataset in medmcqa medqa; do
    path="$DELTA_ROOT/$dataset/pending_k${k}.jsonl"
    [[ -s "$path" ]] && bucket_candidates+=("$path")
  done
  if (( ${#bucket_candidates[@]} == 0 )); then
    echo "[overall $stage/10] delta bucket k=$k is empty; skipped"
    continue
  fi

  questions_per_batch=$((PAIR_BUDGET_PER_CALL / k))
  (( questions_per_batch < 1 )) && questions_per_batch=1
  run_root="$LABEL_RUNS_ROOT/k${k}"
  pending_plan="$run_root/pending_plan.json"
  progress_db="$run_root/progress.sqlite"
  mkdir -p "$run_root/logs"

  common_args=(
    --candidates-paths "${bucket_candidates[@]}"
    --output-root "$run_root"
    --docs-per-question "$k"
    --questions-per-batch "$questions_per_batch"
    --max-doc-chars 0
    --codex-bin "$CODEX_BIN"
    --model "$MODEL"
    --model-reasoning-effort "$REASONING_EFFORT"
    --no-enable-web-search
    --max-attempts 3
    --retry-backoff-seconds 30
    --retry-jitter-fraction 0.25
    --resume
  )

  echo "[overall $stage/10] semantic labeling delta bucket k=$k with $WORKERS workers; stage progress/ETA follows"
  "$PYTHON" "$PROJECT/scripts/label_rag2_candidates_with_codex.py" \
    "${common_args[@]}" \
    --pending-plan-path "$pending_plan" \
    --write-pending-plan-only

  active_pids=()
  for ((worker=0; worker<WORKERS; worker++)); do
    "$PYTHON" "$PROJECT/scripts/label_rag2_candidates_with_codex.py" \
      "${common_args[@]}" \
      --worker-count "$WORKERS" \
      --worker-index "$worker" \
      --pending-plan-path "$pending_plan" \
      --progress-db-path "$progress_db" \
      >"$run_root/logs/worker_${worker}.log" 2>&1 &
    active_pids+=("$!")
  done

  "$PYTHON" "$PROJECT/scripts/monitor_rag2_codex_semantic_labels.py" \
    --output-root "$run_root" \
    --refresh-seconds 5 &
  monitor_pid=$!

  worker_failed=0
  for pid in "${active_pids[@]}"; do
    if ! wait "$pid"; then
      worker_failed=1
    fi
  done
  active_pids=()
  if (( worker_failed )); then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    echo "At least one k=$k worker failed. Inspect $run_root/logs and rerun this command; completed batches are retained." >&2
    exit 1
  fi
  wait "$monitor_pid"
  monitor_pid=""

  "$PYTHON" "$PROJECT/scripts/label_rag2_candidates_with_codex.py" \
    "${common_args[@]}" \
    --consolidate-only
done

echo "[overall 10/10] assemble and validate complete semantic labels"
"$PYTHON" "$PROJECT/scripts/materialize_rag2_semantic_label_delta.py" \
  --mode finalize \
  --candidates-paths "${CANDIDATES[@]}" \
  --delta-root "$DELTA_ROOT" \
  --label-runs-root "$LABEL_RUNS_ROOT" \
  --final-output-root "$FINAL_ROOT" \
  --docs-per-question 8

trap - INT TERM EXIT
echo "Semantic labeling complete: $FINAL_ROOT"
