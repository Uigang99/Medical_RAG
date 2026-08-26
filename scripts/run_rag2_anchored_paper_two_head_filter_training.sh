#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "medmcqa" && "$1" != "medqa" ]]; then
  echo "Usage: $0 {medmcqa|medqa}" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="/home/user/Uiheon/.venv_vllm/bin/python"
SPLIT_ROOT="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/filter_training_inputs_rag2_paper_reproduction_three_class_v1"
MODEL_PATH="/home/user/Uiheon/models/Flan-T5-large"
OUTPUT_ROOT="/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperReproduction-Anchored-TwoHead"

# Discard increases the train set by 2.26x (MedMCQA) and 2.53x (MedQA).
# These defaults keep the number of examples seen comparable with the prior
# binary reproduction while validation early stopping guards overfitting.
if [[ "${DATASET}" == "medmcqa" ]]; then
  DEFAULT_EPOCHS=3
else
  DEFAULT_EPOCHS=6
fi
EPOCHS="${EPOCHS_OVERRIDE:-${DEFAULT_EPOCHS}}"

# Project-wide hardware contract: physical GPU 1 only.
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/train_rag2_filter_model_two_head.py" \
  --dataset "${DATASET}" \
  --split-root "${SPLIT_ROOT}" \
  --model-name-or-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${DATASET}_rag2_two_head_balanced_epoch${EPOCHS}_len768" \
  --max-doc-rank 8 \
  --num-train-epochs "${EPOCHS}" \
  --learning-rate 3e-5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 768 \
  --overlength-policy drop \
  --doc-stride 128 \
  --sampling-mode hierarchical_balanced \
  --dropout 0.1 \
  --decisive-loss-weight 1.0 \
  --utility-loss-weight 1.0 \
  --weight-decay 0 \
  --warmup-steps 0 \
  --max-grad-norm 0 \
  --metric-for-best-model worst_group_macro_f1 \
  --early-stopping-patience 3 \
  --discard-contamination-limit 0.10 \
  --threshold-min 0.05 \
  --threshold-max 0.95 \
  --threshold-step 0.05 \
  --preprocessing-num-workers 16 \
  --dataloader-num-workers 16 \
  --dataloader-prefetch-factor 4 \
  --logging-steps 100 \
  --save-total-limit 2 \
  --eval-accumulation-steps 32 \
  --bf16 \
  --tf32 \
  --seed 42 \
  --log-level INFO
