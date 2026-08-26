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
OUTPUT_ROOT="/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-PaperReproduction-Anchored-3Class"

# Match the optimizer-update scale of the binary reproduction as closely as
# possible after adding Discard: MedMCQA grows 2.26x (6 -> ~3 epochs), and
# MedQA grows 2.53x (15 -> ~6 epochs).  EPOCHS_OVERRIDE remains available for
# a later controlled extension without changing any other hyperparameter.
if [[ "${DATASET}" == "medmcqa" ]]; then
  DEFAULT_EPOCHS=3
else
  DEFAULT_EPOCHS=6
fi
EPOCHS="${EPOCHS_OVERRIDE:-${DEFAULT_EPOCHS}}"

# Global project contract: pin every requested training job to physical GPU 1.
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=true

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/train_rag2_filter_model_paper.py" \
  --dataset "${DATASET}" \
  --split-root "${SPLIT_ROOT}" \
  --model-name-or-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${DATASET}_rag2_three_class_epoch${EPOCHS}_len512_stride128" \
  --label-mode three_class \
  --max-doc-rank 8 \
  --num-train-epochs "${EPOCHS}" \
  --learning-rate 3e-5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 512 \
  --overlength-policy overflow \
  --doc-stride 128 \
  --max-target-length 30 \
  --weight-decay 0 \
  --warmup-steps 0 \
  --max-grad-norm 0 \
  --metric-for-best-model macro_f1 \
  --eval-each-epoch \
  --evaluate-final-model \
  --save-total-limit 2 \
  --preprocessing-num-workers 16 \
  --dataloader-num-workers 16 \
  --dataloader-prefetch-factor 4 \
  --logging-steps 100 \
  --eval-accumulation-steps 32 \
  --bf16 \
  --tf32 \
  --seed 42 \
  --log-level INFO
