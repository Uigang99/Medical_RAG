#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-medqa}"
if [[ "${DATASET}" != "medqa" && "${DATASET}" != "medmcqa" ]]; then
  echo "Usage: $0 {medqa|medmcqa}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
BASE_DATA="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
DATA_ROOT="${DATA_ROOT:-${BASE_DATA}/semantic_utilization_contrast_pilot_v1}"
MODEL_PATH="${MODEL_PATH:-/home/user/Uiheon/models/Llama-3-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/user/Uiheon/models/RAG2-Semantic-Utilization-LoRA}"
RUN_TAG="${RUN_TAG:-pilot512_v1}"
CONTROL_NAME="${DATASET}_${RUN_TAG}_sft_control"
PROPOSED_NAME="${DATASET}_${RUN_TAG}_semantic_utilization"
COMPARISON_DIR="${OUTPUT_ROOT}/${DATASET}/${DATASET}_${RUN_TAG}_comparison"

MAX_TRAIN_QUESTIONS="${MAX_TRAIN_QUESTIONS:-512}"
MAX_EVAL_QUESTIONS="${MAX_EVAL_QUESTIONS:-128}"
EPOCHS="${EPOCHS:-2}"
PATIENCE="${PATIENCE:-2}"
TRAIN_BATCH="${TRAIN_BATCH:-1}"
EVAL_BATCH="${EVAL_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-6144}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SEED="${SEED:-42}"

workflow_started="$(date +%s)"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
elapsed() {
  local now seconds
  now="$(date +%s)"
  seconds=$((now - workflow_started))
  printf '%02dh%02dm%02ds' $((seconds/3600)) $(((seconds%3600)/60)) $((seconds%60))
}
stage() {
  local index="$1" name="$2" future="$3"
  echo "[$(timestamp)] [overall ${index}/4 | elapsed $(elapsed) | overall ETA ${future}] [${name}]"
}

echo "[$(timestamp)] Semantic-utilization bounded pilot"
echo "  dataset=${DATASET} GPU-visible=${CUDA_VISIBLE_DEVICES:-not-set} train/val/test=${MAX_TRAIN_QUESTIONS}/${MAX_EVAL_QUESTIONS}/${MAX_EVAL_QUESTIONS}"
echo "  model=${MODEL_PATH} epochs=${EPOCHS} batch=${TRAIN_BATCH} grad_accum=${GRAD_ACCUM} max_tokens=${MAX_INPUT_TOKENS}"
echo "  estimated wall time on one H200: 03h-07h when uncached; child stages report measured rolling ETA after calibration"
echo "  comparison: matched SFT control versus semantic-utilization objective; Behavioral utility is not used"

stage 1 "prepare and validate immutable semantic contrast sets" "unknown until training throughput is measured"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_rag2_semantic_utilization_pilot.py" \
  --dataset "${DATASET}" \
  --output-root "${DATA_ROOT}" \
  --max-train-questions "${MAX_TRAIN_QUESTIONS}" \
  --max-eval-questions "${MAX_EVAL_QUESTIONS}" \
  --seed "${SEED}" \
  --resume

common_args=(
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --model-name-or-path "${MODEL_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --train-questions-per-batch "${TRAIN_BATCH}"
  --eval-questions-per-batch "${EVAL_BATCH}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --learning-rate "${LEARNING_RATE}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --seed "${SEED}"
  --device cuda:0
  --dtype bfloat16
  --attn-implementation sdpa
  --resume
)

stage 2 "train matched SFT control" "unknown; active train stage reports measured ETA"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_rag2_semantic_utilization_lora.py" \
  "${common_args[@]}" \
  --run-name "${CONTROL_NAME}" \
  --objective sft_control

stage 3 "train semantic-utilization LoRA" "unknown; active train stage reports measured ETA"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_rag2_semantic_utilization_lora.py" \
  "${common_args[@]}" \
  --run-name "${PROPOSED_NAME}" \
  --objective semantic_utilization

stage 4 "paired internal-test comparison and bootstrap confidence intervals" "under 00h02m"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/compare_rag2_semantic_utilization_pilot.py" \
  --control-dir "${OUTPUT_ROOT}/${DATASET}/${CONTROL_NAME}" \
  --proposed-dir "${OUTPUT_ROOT}/${DATASET}/${PROPOSED_NAME}" \
  --output-dir "${COMPARISON_DIR}" \
  --bootstrap-replicates 2000 \
  --seed "${SEED}"

echo "[$(timestamp)] [overall 4/4 | elapsed $(elapsed) | ETA 00h00m00s] workflow complete"
echo "Comparison: ${COMPARISON_DIR}/COMPARISON.md"
