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
PREPARED_ROOT="${BASE}/shared_gold_margin_regression_v1/prepared"
OUTPUT_ROOT="/home/user/Uiheon/models/RAG2-SharedMarginRegressor-FlanT5-large"

if [[ ! -f "${PREPARED_ROOT}/manifest.json" ]]; then
  echo "Prepared shared-margin data is missing. Run scripts/run_rag2_shared_margin_prepare.sh first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

TRAIN_QUESTIONS="${PILOT_TRAIN_QUESTIONS:-5000}"
EVAL_QUESTIONS="${PILOT_EVAL_QUESTIONS:-1000}"
EPOCHS="${PILOT_EPOCHS:-5}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/train_rag2_shared_margin_regressor.py" \
  --dataset "${DATASET}" \
  --prepared-root "${PREPARED_ROOT}" \
  --model-name-or-path "/home/user/Uiheon/models/Flan-T5-large" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${DATASET}_shared_m0_md_pilot_q${TRAIN_QUESTIONS}_epoch${EPOCHS}" \
  --num-train-epochs "${EPOCHS}" \
  --max-train-questions "${TRAIN_QUESTIONS}" \
  --max-eval-questions "${EVAL_QUESTIONS}" \
  --train-questions-per-batch 4 \
  --eval-questions-per-batch 8 \
  --gradient-accumulation-steps 4 \
  --encoder-learning-rate 1e-5 \
  --head-learning-rate 2e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --huber-delta 0.5 \
  --delta-loss-weight 0.5 \
  --margin-scale 10.0 \
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
  "${RESUME_ARGS[@]}"
