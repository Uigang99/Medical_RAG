#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "medmcqa" && "$1" != "medqa" ]]; then
  echo "Usage: $0 {medmcqa|medqa}" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
BASE="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
PREPARED_ROOT="${BASE}/gold_margin_regression_v1/prepared"
OUTPUT_ROOT="/home/user/Uiheon/models/RAG2-PairwiseUtilityRanker-FlanT5-large"
INPUT_MODE="${INPUT_MODE:-text_no_rag_answer}"
PAIR_GAP="${PAIR_GAP:-0.1}"
NULL_GAP="${NULL_GAP:-0.1}"

if [[ ! -f "${PREPARED_ROOT}/manifest.json" ]]; then
  echo "Prepared utility pointers are missing: ${PREPARED_ROOT}/manifest.json" >&2
  exit 1
fi

if [[ "${DATASET}" == "medmcqa" ]]; then
  EPOCHS="${EPOCHS_OVERRIDE:-3}"
else
  EPOCHS="${EPOCHS_OVERRIDE:-8}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

LIMIT_ARGS=()
if [[ -n "${MAX_TRAIN_QUESTIONS:-}" ]]; then
  LIMIT_ARGS+=(--max-train-questions "${MAX_TRAIN_QUESTIONS}")
fi
if [[ -n "${MAX_EVAL_QUESTIONS:-}" ]]; then
  LIMIT_ARGS+=(--max-eval-questions "${MAX_EVAL_QUESTIONS}")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS+=(--dry-run)
fi

BALANCE_ARGS=(--balance-no-rag-states)
if [[ "${BALANCE_NO_RAG_STATES:-1}" == "0" ]]; then
  BALANCE_ARGS=(--no-balance-no-rag-states)
fi

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/train_rag2_pairwise_utility_ranker.py" \
  --dataset "${DATASET}" \
  --prepared-root "${PREPARED_ROOT}" \
  --model-name-or-path "/home/user/Uiheon/models/Flan-T5-large" \
  --input-mode "${INPUT_MODE}" \
  --no-rag-generation-root "${BASE}/train_no_rag_anchored_features_v1/no_rag" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${DATASET}_${INPUT_MODE}_pairwise_null_gap${PAIR_GAP}_epoch${EPOCHS}" \
  --num-train-epochs "${EPOCHS}" \
  --train-questions-per-batch "${TRAIN_QUESTIONS_PER_BATCH:-16}" \
  --eval-questions-per-batch "${EVAL_QUESTIONS_PER_BATCH:-32}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --encoder-learning-rate "${ENCODER_LEARNING_RATE:-1e-5}" \
  --head-learning-rate "${HEAD_LEARNING_RATE:-2e-4}" \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --document-pair-min-utility-gap "${PAIR_GAP}" \
  --null-min-absolute-utility "${NULL_GAP}" \
  --pairwise-temperature "${PAIRWISE_TEMPERATURE:-0.1}" \
  --document-pair-loss-weight 1.0 \
  --null-pair-loss-weight 1.0 \
  "${BALANCE_ARGS[@]}" \
  --head-hidden-size 256 \
  --dropout 0.1 \
  --trainable-encoder-layers 4 \
  --max-input-tokens 512 \
  --early-stopping-patience 2 \
  --minimum-improvement 1e-4 \
  --trace-shard-cache-size 8 \
  --bf16 \
  --tf32 \
  --gradient-checkpointing \
  --show-progress \
  --logging-steps 50 \
  --seed 42 \
  --log-level INFO \
  "${LIMIT_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${DRY_RUN_ARGS[@]}"
