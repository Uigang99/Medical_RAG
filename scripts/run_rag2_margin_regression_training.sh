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
OUTPUT_ROOT="/home/user/Uiheon/models/RAG2-MarginRegressor-FlanT5-large"

if [[ ! -f "${PREPARED_ROOT}/manifest.json" ]]; then
  echo "Prepared margin-regression data is missing. Run scripts/run_rag2_margin_regression_prepare.sh first." >&2
  exit 1
fi

if [[ "${DATASET}" == "medmcqa" ]]; then
  EPOCHS="${EPOCHS_OVERRIDE:-3}"
else
  EPOCHS="${EPOCHS_OVERRIDE:-10}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/train_rag2_margin_regressor.py" \
  --dataset "${DATASET}" \
  --prepared-root "${PREPARED_ROOT}" \
  --model-name-or-path "/home/user/Uiheon/models/Flan-T5-large" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${DATASET}_text_only_utility_t1_huber_top4_epoch${EPOCHS}" \
  --num-train-epochs "${EPOCHS}" \
  --train-documents-per-batch 64 \
  --eval-documents-per-batch 128 \
  --gradient-accumulation-steps 2 \
  --encoder-learning-rate 1e-5 \
  --head-learning-rate 2e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --huber-delta 0.1 \
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
  --logging-steps 100 \
  --seed 42 \
  --log-level INFO \
  "${RESUME_ARGS[@]}"
