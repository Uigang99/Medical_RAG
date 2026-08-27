#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"
if [[ "${TARGET}" != "all" && "${TARGET}" != "medmcqa" && "${TARGET}" != "medqa" ]]; then
  echo "Usage: $0 [all|medmcqa|medqa]" >&2
  exit 2
fi

PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
BASE="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
PREPARED_ROOT="${BASE}/gold_margin_regression_v1/prepared"
OUTPUT_ROOT="/home/user/Uiheon/models/RAG2-DirectUtilityRegressor-FlanT5-large"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

TRAIN_QUESTIONS="${PILOT_TRAIN_QUESTIONS:-5000}"
EVAL_QUESTIONS="${PILOT_EVAL_QUESTIONS:-1000}"
EPOCHS="${PILOT_EPOCHS:-5}"
TRAIN_BATCH="${TRAIN_DOCUMENTS_PER_BATCH:-128}"
EVAL_BATCH="${EVAL_DOCUMENTS_PER_BATCH:-256}"
UTILITY_THRESHOLD="${UTILITY_THRESHOLD:-0.1}"
EXTREME_SAMPLE_WEIGHT="${EXTREME_SAMPLE_WEIGHT:-2.0}"

if [[ "${TARGET}" == "all" ]]; then
  DATASETS=(medmcqa medqa)
else
  DATASETS=("${TARGET}")
fi

format_seconds() {
  local value="$1"
  printf '%02dh%02dm%02ds' "$((value / 3600))" "$(((value % 3600) / 60))" "$((value % 60))"
}

PIPELINE_START="$(date +%s)"
TOTAL_STAGES="$((1 + ${#DATASETS[@]}))"
echo "[$(date '+%F %T')] Overall pilot 0/${TOTAL_STAGES}: prepared-data check; overall ETA unknown"

if [[ ! -f "${PREPARED_ROOT}/manifest.json" ]]; then
  echo "[$(date '+%F %T')] Stage 1/${TOTAL_STAGES}: materialize continuous-u pointer splits"
  bash "${PROJECT_ROOT}/scripts/run_rag2_margin_regression_prepare.sh"
else
  echo "[$(date '+%F %T')] Stage 1/${TOTAL_STAGES}: prepared continuous-u data already cached"
fi

COMPLETED_DATASETS=0
for DATASET in "${DATASETS[@]}"; do
  STAGE="$((2 + COMPLETED_DATASETS))"
  NOW="$(date +%s)"
  ELAPSED="$((NOW - PIPELINE_START))"
  if [[ "${COMPLETED_DATASETS}" -gt 0 ]]; then
    ETA="$((ELAPSED / COMPLETED_DATASETS * (${#DATASETS[@]} - COMPLETED_DATASETS)))"
    ETA_TEXT="$(format_seconds "${ETA}")"
  else
    ETA_TEXT="unknown until the first dataset rate is measured"
  fi
  echo "[$(date '+%F %T')] Stage ${STAGE}/${TOTAL_STAGES}: ${DATASET} direct-u pilot; overall elapsed=$(format_seconds "${ELAPSED}") overall ETA=${ETA_TEXT}"

  RESUME_ARGS=()
  RESUME_VARIABLE="RESUME_FROM_CHECKPOINT_${DATASET^^}"
  RESUME_VALUE="${!RESUME_VARIABLE:-${RESUME_FROM_CHECKPOINT:-}}"
  if [[ -n "${RESUME_VALUE}" ]]; then
    RESUME_ARGS=(--resume-from-checkpoint "${RESUME_VALUE}")
  fi

  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/train_rag2_margin_regressor.py" \
    --dataset "${DATASET}" \
    --prepared-root "${PREPARED_ROOT}" \
    --model-name-or-path "/home/user/Uiheon/models/Flan-T5-large" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${DATASET}_direct_u_tau${UTILITY_THRESHOLD}_pilot_q${TRAIN_QUESTIONS}_batch${TRAIN_BATCH}_epoch${EPOCHS}" \
    --num-train-epochs "${EPOCHS}" \
    --max-train-questions "${TRAIN_QUESTIONS}" \
    --max-eval-questions "${EVAL_QUESTIONS}" \
    --train-documents-per-batch "${TRAIN_BATCH}" \
    --eval-documents-per-batch "${EVAL_BATCH}" \
    --gradient-accumulation-steps 1 \
    --encoder-learning-rate 1e-5 \
    --head-learning-rate 2e-4 \
    --weight-decay 0.01 \
    --warmup-ratio 0.05 \
    --max-grad-norm 1.0 \
    --huber-delta 0.1 \
    --utility-threshold "${UTILITY_THRESHOLD}" \
    --extreme-sample-weight "${EXTREME_SAMPLE_WEIGHT}" \
    --checkpoint-metric action_macro_f1 \
    --head-hidden-size 256 \
    --dropout 0.1 \
    --trainable-encoder-layers 4 \
    --max-input-tokens 512 \
    --early-stopping-patience 2 \
    --minimum-improvement 1e-4 \
    --trace-shard-cache-size 8 \
    --bf16 \
    --tf32 \
    --no-gradient-checkpointing \
    --show-progress \
    --logging-steps 50 \
    --seed 42 \
    --log-level INFO \
    "${RESUME_ARGS[@]}"

  COMPLETED_DATASETS="$((COMPLETED_DATASETS + 1))"
done

FINISHED="$(date +%s)"
echo "[$(date '+%F %T')] Overall pilot ${TOTAL_STAGES}/${TOTAL_STAGES} complete; elapsed=$(format_seconds "$((FINISHED - PIPELINE_START))") ETA=0s"
