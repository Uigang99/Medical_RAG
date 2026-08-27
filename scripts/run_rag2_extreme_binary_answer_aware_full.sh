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
INPUT_ROOT="${INPUT_ROOT:-${BASE}/extreme_utility_binary_answer_aware_tau0p2_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/user/Uiheon/models/RAG2-ExtremeUtilityBinaryAnswerAware-FlanT5-large}"
RUN_TAG="${RUN_TAG:-tau0p2_four_group_full_v1}"
EXTREME_THRESHOLD="${EXTREME_THRESHOLD:-0.2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

if [[ "${TARGET}" == "all" ]]; then
  DATASETS=(medmcqa medqa)
else
  DATASETS=("${TARGET}")
fi

format_seconds() {
  local value="$1"
  printf '%02dh%02dm%02ds' "$((value / 3600))" "$(((value % 3600) / 60))" "$((value % 60))"
}

latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sort -V \
    | tail -1
}

PIPELINE_START="$(date +%s)"
TOTAL_STAGES="$((1 + ${#DATASETS[@]}))"
COMPLETED_STAGES=0
echo "[$(date '+%F %T')] Overall 0/${TOTAL_STAGES}; stage=prepare complete extreme-only binary inputs; overall ETA unknown"

for DATASET in "${DATASETS[@]}"; do
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_rag2_extreme_binary_filter_inputs.py" \
    --dataset "${DATASET}" \
    --output-root "${INPUT_ROOT}" \
    --extreme-threshold "${EXTREME_THRESHOLD}" \
    --trace-shard-cache-size 8 \
    --checkpoint-interval 10000 \
    --show-progress \
    --log-level INFO
done
COMPLETED_STAGES=1

for DATASET in "${DATASETS[@]}"; do
  if [[ "${DATASET}" == "medmcqa" ]]; then
    EPOCHS="${MEDMCQA_EPOCHS:-5}"
    PATIENCE="${MEDMCQA_EARLY_STOPPING_PATIENCE:-2}"
    WARMUP_STEPS="${MEDMCQA_WARMUP_STEPS:-1000}"
  else
    EPOCHS="${MEDQA_EPOCHS:-10}"
    PATIENCE="${MEDQA_EARLY_STOPPING_PATIENCE:-3}"
    WARMUP_STEPS="${MEDQA_WARMUP_STEPS:-200}"
  fi
  RUN_DIR="${OUTPUT_ROOT}/${DATASET}/${RUN_TAG}"
  if [[ -d "${RUN_DIR}/final_model" && -f "${RUN_DIR}/final_metrics.json" ]]; then
    echo "[$(date '+%F %T')] Reusing completed ${DATASET} model: ${RUN_DIR}/final_model"
    COMPLETED_STAGES="$((COMPLETED_STAGES + 1))"
    continue
  fi
  mkdir -p "${RUN_DIR}"
  RESUME_ARGS=()
  CHECKPOINT_NAME="$(latest_checkpoint "${RUN_DIR}")"
  if [[ -n "${CHECKPOINT_NAME}" ]]; then
    RESUME_ARGS=(--resume-from-checkpoint "${RUN_DIR}/${CHECKPOINT_NAME}")
  fi

  NOW="$(date +%s)"
  ELAPSED="$((NOW - PIPELINE_START))"
  if [[ "${COMPLETED_STAGES}" -gt 1 ]]; then
    ETA_SECONDS="$((ELAPSED / COMPLETED_STAGES * (TOTAL_STAGES - COMPLETED_STAGES)))"
    ETA_TEXT="$(format_seconds "${ETA_SECONDS}")"
  else
    ETA_TEXT="unknown until the first training stage completes"
  fi
  echo "[$(date '+%F %T')] Overall ${COMPLETED_STAGES}/${TOTAL_STAGES}; stage=train ${DATASET} extreme Helpful/Harmful; overall ETA=${ETA_TEXT}"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_rag2_filter_model_paper.py" \
    --dataset "${DATASET}" \
    --split-root "${INPUT_ROOT}" \
    --model-name-or-path /home/user/Uiheon/models/Flan-T5-large \
    --output-root "${OUTPUT_ROOT}" \
    --output-dir "${RUN_DIR}" \
    --run-name "${RUN_TAG}" \
    --label-mode binary \
    --preformatted-input \
    --train-balance-mode four_group_loss \
    --balanced-validation \
    --max-doc-rank 0 \
    --num-train-epochs "${EPOCHS}" \
    --learning-rate 3e-5 \
    --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-32}" \
    --per-device-eval-batch-size "${EVAL_BATCH_SIZE:-64}" \
    --gradient-accumulation-steps 1 \
    --max-seq-length 1024 \
    --overlength-policy drop \
    --max-target-length 8 \
    --weight-decay 0.01 \
    --warmup-steps "${WARMUP_STEPS}" \
    --max-grad-norm 1.0 \
    --preprocessing-num-workers 16 \
    --dataloader-num-workers 8 \
    --dataloader-prefetch-factor 4 \
    --logging-steps 100 \
    --save-total-limit 2 \
    --eval-accumulation-steps 64 \
    --eval-each-epoch \
    --evaluate-final-model \
    --metric-for-best-model macro_f1 \
    --early-stopping-patience "${PATIENCE}" \
    --early-stopping-threshold 1e-4 \
    --bf16 \
    --tf32 \
    --load-model-in-bf16 \
    --no-gradient-checkpointing \
    --seed 42 \
    --log-level INFO \
    "${RESUME_ARGS[@]}"
  COMPLETED_STAGES="$((COMPLETED_STAGES + 1))"
done

FINISHED="$(date +%s)"
echo "[$(date '+%F %T')] Overall ${TOTAL_STAGES}/${TOTAL_STAGES} complete; elapsed=$(format_seconds "$((FINISHED - PIPELINE_START))") ETA=0s"
echo "Final models: ${OUTPUT_ROOT}/{medmcqa,medqa}/${RUN_TAG}/final_model"
