#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "medmcqa" && "$1" != "medqa" ]]; then
  echo "Usage: $0 {medmcqa|medqa}" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
SPLIT_ROOT="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/filter_training_inputs_semantic_top8_four_class_v1"
MODEL_PATH=/home/user/Uiheon/models/Flan-T5-large
OUTPUT_ROOT=/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport

# The semantic binary task is easier than the original four-way task, but the
# four-way validation curves were still improving at their final checkpoints.
# Use a larger upper bound and select/stop on validation macro-F1. MedMCQA has
# 1.17M rows, so five full passes are already substantial; MedQA is small enough
# to allow eight while early stopping protects against overfitting.
if [[ "${DATASET}" == "medmcqa" ]]; then
  DEFAULT_EPOCHS=5
else
  DEFAULT_EPOCHS=8
fi

EPOCHS="${EPOCHS_OVERRIDE:-${DEFAULT_EPOCHS}}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH_OVERRIDE:-1280}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE_OVERRIDE:-2}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD_OVERRIDE:-0.001}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=true

printf '[%s] Semantic support binary training: dataset=%s max_epochs=%s max_length=%s GPU=%s. ' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "${DATASET}" "${EPOCHS}" "${MAX_SEQ_LENGTH}" "${CUDA_VISIBLE_DEVICES}"
printf 'Targets are Direct+Supporting=Helpful and No+Misleading=Not Helpful; Mixed is absent from the source split. '
printf 'Trainer bars report current-stage progress/ETA, and workflow logs identify stages 1/5 through 5/5.\n'

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/train_rag2_filter_model_paper.py" \
  --dataset "${DATASET}" \
  --split-root "${SPLIT_ROOT}" \
  --model-name-or-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${DATASET}_semantic_top8_binary_support_epoch${EPOCHS}_len${MAX_SEQ_LENGTH}_fullpair" \
  --label-mode semantic_binary \
  --train-balance-mode natural \
  --max-doc-rank 8 \
  --num-train-epochs "${EPOCHS}" \
  --learning-rate 3e-5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --overlength-policy drop \
  --max-target-length 30 \
  --weight-decay 0 \
  --warmup-steps 0 \
  --max-grad-norm 0 \
  --metric-for-best-model macro_f1 \
  --eval-each-epoch \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD}" \
  --evaluate-final-model \
  --save-total-limit 1 \
  --preprocessing-num-workers 16 \
  --dataloader-num-workers 16 \
  --dataloader-prefetch-factor 4 \
  --logging-steps 100 \
  --eval-accumulation-steps 32 \
  --bf16 \
  --tf32 \
  --seed 42 \
  --log-level INFO
