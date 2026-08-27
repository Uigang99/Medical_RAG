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
INPUT_MODE="${INPUT_MODE:-text_only}"
NO_RAG_GENERATION_ROOT="${NO_RAG_GENERATION_ROOT:-${BASE}/train_no_rag_anchored_features_v1/no_rag}"
if [[ "${INPUT_MODE}" != "text_only" && "${INPUT_MODE}" != "text_no_rag_answer" ]]; then
  echo "INPUT_MODE must be text_only or text_no_rag_answer" >&2
  exit 2
fi
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ "${INPUT_MODE}" == "text_no_rag_answer" ]]; then
    OUTPUT_ROOT="/home/user/Uiheon/models/RAG2-ContinuousUtilityNoRAGAnswerCurriculum-FlanT5-large"
  else
    OUTPUT_ROOT="/home/user/Uiheon/models/RAG2-ContinuousUtilityCurriculum-FlanT5-large"
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

EXTREME_THRESHOLD="${EXTREME_THRESHOLD:-0.2}"
DEPLOY_THRESHOLD="${DEPLOY_THRESHOLD:-0.2}"
STAGE1_EPOCHS_MEDMCQA="${STAGE1_EPOCHS_MEDMCQA:-3}"
STAGE1_EPOCHS_MEDQA="${STAGE1_EPOCHS_MEDQA:-8}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-2}"
TRAIN_QUESTIONS_PER_BATCH="${TRAIN_QUESTIONS_PER_BATCH:-16}"
EVAL_DOCUMENTS_PER_BATCH="${EVAL_DOCUMENTS_PER_BATCH:-256}"
if [[ -z "${MAX_INPUT_TOKENS:-}" ]]; then
  if [[ "${INPUT_MODE}" == "text_no_rag_answer" ]]; then
    MAX_INPUT_TOKENS=640
  else
    MAX_INPUT_TOKENS=512
  fi
fi
CALIBRATION_EXTREME_FRACTION="${CALIBRATION_EXTREME_FRACTION:-0.5}"
NEUTRAL_LOSS_WEIGHT="${NEUTRAL_LOSS_WEIGHT:-0.25}"
NEUTRAL_TOLERANCE="${NEUTRAL_TOLERANCE:-0.03}"
STAGE2_MAX_EXTREME_AUROC_DROP="${STAGE2_MAX_EXTREME_AUROC_DROP:-0.02}"
STAGE2_MAX_EXTREME_SIGN_ACCURACY_DROP="${STAGE2_MAX_EXTREME_SIGN_ACCURACY_DROP:-0.02}"
PAIRWISE_LOSS_WEIGHT="${PAIRWISE_LOSS_WEIGHT:-0.05}"
PAIRWISE_MIN_TARGET_GAP="${PAIRWISE_MIN_TARGET_GAP:-0.1}"
PAIRWISE_TEMPERATURE="${PAIRWISE_TEMPERATURE:-0.1}"
MAX_TRAIN_QUESTIONS="${MAX_TRAIN_QUESTIONS:-}"
MAX_EVAL_QUESTIONS="${MAX_EVAL_QUESTIONS:-}"
if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ "${INPUT_MODE}" == "text_no_rag_answer" ]]; then
    RUN_TAG="tau0p2_no_rag_answer_extreme_replay50_neutral25_v1"
  else
    RUN_TAG="tau0p2_extreme_replay50_neutral25_v1"
  fi
fi

if [[ "${TARGET}" == "all" ]]; then
  DATASETS=(medmcqa medqa)
else
  DATASETS=("${TARGET}")
fi

format_seconds() {
  local value="$1"
  printf '%02dh%02dm%02ds' "$((value / 3600))" "$(((value % 3600) / 60))" "$((value % 60))"
}

resume_args_for() {
  local output_dir="$1"
  RESUME_ARGS=()
  if [[ -f "${output_dir}/final_model/margin_regressor_config.json" ]]; then
    return 10
  fi
  if [[ -f "${output_dir}/last_checkpoint/training_state.pt" ]]; then
    RESUME_ARGS=(--resume-from-checkpoint "${output_dir}/last_checkpoint")
    return 0
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "Incomplete run has no resumable checkpoint: ${output_dir}" >&2
    return 2
  fi
  return 0
}

LIMIT_ARGS=()
if [[ -n "${MAX_TRAIN_QUESTIONS}" ]]; then
  LIMIT_ARGS+=(--max-train-questions "${MAX_TRAIN_QUESTIONS}")
fi
if [[ -n "${MAX_EVAL_QUESTIONS}" ]]; then
  LIMIT_ARGS+=(--max-eval-questions "${MAX_EVAL_QUESTIONS}")
fi

PIPELINE_START="$(date +%s)"
TOTAL_STAGES="$((1 + 2 * ${#DATASETS[@]}))"
COMPLETED_STAGES=0
echo "[$(date '+%F %T')] Overall curriculum 0/${TOTAL_STAGES}: prepared-data check; input_mode=${INPUT_MODE}; overall ETA unknown"

if [[ ! -f "${PREPARED_ROOT}/manifest.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/run_rag2_margin_regression_prepare.sh"
else
  echo "[$(date '+%F %T')] Prepared continuous-utility data already cached: ${PREPARED_ROOT}"
fi
COMPLETED_STAGES=1

for DATASET in "${DATASETS[@]}"; do
  if [[ "${DATASET}" == "medmcqa" ]]; then
    STAGE1_EPOCHS="${STAGE1_EPOCHS_MEDMCQA}"
  else
    STAGE1_EPOCHS="${STAGE1_EPOCHS_MEDQA}"
  fi
  RUN_DIR="${OUTPUT_ROOT}/${DATASET}/${RUN_TAG}"
  STAGE1_DIR="${RUN_DIR}/stage1_extreme"
  STAGE2_DIR="${RUN_DIR}/stage2_neutral_calibration"

  NOW="$(date +%s)"
  ELAPSED="$((NOW - PIPELINE_START))"
  if [[ "${COMPLETED_STAGES}" -gt 1 ]]; then
    ETA_SECONDS="$((ELAPSED / COMPLETED_STAGES * (TOTAL_STAGES - COMPLETED_STAGES)))"
    ETA_TEXT="measured $(format_seconds "${ETA_SECONDS}")"
  else
    ETA_TEXT="unknown until a training stage completes"
  fi
  echo "[$(date '+%F %T')] Overall ${COMPLETED_STAGES}/${TOTAL_STAGES}; stage=Stage 1 extreme regression (${DATASET}); overall ETA=${ETA_TEXT}"

  set +e
  resume_args_for "${STAGE1_DIR}"
  RESUME_STATUS=$?
  set -e
  if [[ "${RESUME_STATUS}" -eq 10 ]]; then
    echo "[$(date '+%F %T')] Stage 1 already complete: ${STAGE1_DIR}/final_model"
  elif [[ "${RESUME_STATUS}" -ne 0 ]]; then
    exit "${RESUME_STATUS}"
  else
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_rag2_margin_regressor.py" \
      --dataset "${DATASET}" \
      --prepared-root "${PREPARED_ROOT}" \
      --model-name-or-path /home/user/Uiheon/models/Flan-T5-large \
      --input-mode "${INPUT_MODE}" \
      --no-rag-generation-root "${NO_RAG_GENERATION_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --output-dir "${STAGE1_DIR}" \
      --run-name "${RUN_TAG}_stage1_extreme" \
      --curriculum-stage extreme \
      --num-train-epochs "${STAGE1_EPOCHS}" \
      --train-questions-per-batch "${TRAIN_QUESTIONS_PER_BATCH}" \
      --eval-documents-per-batch "${EVAL_DOCUMENTS_PER_BATCH}" \
      --gradient-accumulation-steps 1 \
      --encoder-learning-rate 1e-5 \
      --head-learning-rate 2e-4 \
      --weight-decay 0.01 \
      --warmup-ratio 0.05 \
      --max-grad-norm 1.0 \
      --huber-delta 0.1 \
      --utility-threshold "${DEPLOY_THRESHOLD}" \
      --extreme-threshold "${EXTREME_THRESHOLD}" \
      --checkpoint-metric extreme_auroc \
      --pairwise-ranking \
      --pairwise-loss-weight "${PAIRWISE_LOSS_WEIGHT}" \
      --pairwise-min-target-gap "${PAIRWISE_MIN_TARGET_GAP}" \
      --pairwise-temperature "${PAIRWISE_TEMPERATURE}" \
      --head-hidden-size 256 \
      --dropout 0.1 \
      --trainable-encoder-layers 4 \
      --max-input-tokens "${MAX_INPUT_TOKENS}" \
      --early-stopping-patience 2 \
      --minimum-improvement 1e-4 \
      --trace-shard-cache-size 8 \
      --bf16 \
      --tf32 \
      --no-gradient-checkpointing \
      --show-progress \
      --logging-steps 25 \
      --seed 42 \
      --log-level INFO \
      "${LIMIT_ARGS[@]}" \
      "${RESUME_ARGS[@]}"
  fi
  COMPLETED_STAGES="$((COMPLETED_STAGES + 1))"

  NOW="$(date +%s)"
  ELAPSED="$((NOW - PIPELINE_START))"
  ETA_SECONDS="$((ELAPSED / COMPLETED_STAGES * (TOTAL_STAGES - COMPLETED_STAGES)))"
  echo "[$(date '+%F %T')] Overall ${COMPLETED_STAGES}/${TOTAL_STAGES}; stage=Stage 2 neutral calibration (${DATASET}); measured overall ETA=$(format_seconds "${ETA_SECONDS}")"

  set +e
  resume_args_for "${STAGE2_DIR}"
  RESUME_STATUS=$?
  set -e
  if [[ "${RESUME_STATUS}" -eq 10 ]]; then
    echo "[$(date '+%F %T')] Stage 2 already complete: ${STAGE2_DIR}/final_model"
  elif [[ "${RESUME_STATUS}" -ne 0 ]]; then
    exit "${RESUME_STATUS}"
  else
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_rag2_margin_regressor.py" \
      --dataset "${DATASET}" \
      --prepared-root "${PREPARED_ROOT}" \
      --model-name-or-path /home/user/Uiheon/models/Flan-T5-large \
      --input-mode "${INPUT_MODE}" \
      --no-rag-generation-root "${NO_RAG_GENERATION_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --output-dir "${STAGE2_DIR}" \
      --run-name "${RUN_TAG}_stage2_neutral" \
      --curriculum-stage calibration \
      --stage1-model "${STAGE1_DIR}/final_model" \
      --num-train-epochs "${STAGE2_EPOCHS}" \
      --train-questions-per-batch "${TRAIN_QUESTIONS_PER_BATCH}" \
      --eval-documents-per-batch "${EVAL_DOCUMENTS_PER_BATCH}" \
      --gradient-accumulation-steps 1 \
      --encoder-learning-rate 2e-6 \
      --head-learning-rate 4e-5 \
      --weight-decay 0.01 \
      --warmup-ratio 0.03 \
      --max-grad-norm 1.0 \
      --huber-delta 0.1 \
      --utility-threshold "${DEPLOY_THRESHOLD}" \
      --extreme-threshold "${EXTREME_THRESHOLD}" \
      --calibration-extreme-fraction "${CALIBRATION_EXTREME_FRACTION}" \
      --neutral-loss-weight "${NEUTRAL_LOSS_WEIGHT}" \
      --neutral-tolerance "${NEUTRAL_TOLERANCE}" \
      --stage2-max-extreme-auroc-drop "${STAGE2_MAX_EXTREME_AUROC_DROP}" \
      --stage2-max-extreme-sign-accuracy-drop "${STAGE2_MAX_EXTREME_SIGN_ACCURACY_DROP}" \
      --checkpoint-metric action_macro_f1 \
      --pairwise-ranking \
      --pairwise-loss-weight "${PAIRWISE_LOSS_WEIGHT}" \
      --pairwise-min-target-gap "${PAIRWISE_MIN_TARGET_GAP}" \
      --pairwise-temperature "${PAIRWISE_TEMPERATURE}" \
      --head-hidden-size 256 \
      --dropout 0.1 \
      --trainable-encoder-layers 4 \
      --max-input-tokens "${MAX_INPUT_TOKENS}" \
      --early-stopping-patience 2 \
      --minimum-improvement 1e-4 \
      --trace-shard-cache-size 8 \
      --bf16 \
      --tf32 \
      --no-gradient-checkpointing \
      --show-progress \
      --logging-steps 25 \
      --seed 42 \
      --log-level INFO \
      "${LIMIT_ARGS[@]}" \
      "${RESUME_ARGS[@]}"
  fi
  COMPLETED_STAGES="$((COMPLETED_STAGES + 1))"
done

FINISHED="$(date +%s)"
echo "[$(date '+%F %T')] Overall curriculum ${TOTAL_STAGES}/${TOTAL_STAGES} complete; elapsed=$(format_seconds "$((FINISHED - PIPELINE_START))") ETA=0s"
echo "Final models are under: ${OUTPUT_ROOT}/{medmcqa,medqa}/${RUN_TAG}/stage2_neutral_calibration/final_model"
