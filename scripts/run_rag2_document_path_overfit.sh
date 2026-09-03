#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
SCRIPT="$PROJECT/scripts/train_rag2_document_path_overfit.py"
DATASET="${1:-medmcqa}"

if [[ "$DATASET" != "medmcqa" ]]; then
  echo "ERROR: this preregistered overfit diagnostic currently targets the completed MedMCQA bounded cache" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_NAME="${RUN_NAME:-medmcqa_document_first_overfit256_v1}"

test -x "$PYTHON_BIN"
test -f "$SCRIPT"
test -f "/home/user/Uiheon/models/Llama-3-8B-Instruct/config.json"
test -f "$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/document_first_bounded_direct_outcomes_v1/medmcqa/training_dataset/manifest.json"

echo "[command plan] MedMCQA 20,000-row audit -> balanced 256 selection -> tokenize -> frozen fidelity -> <=${EPOCHS}-epoch same-set overfit -> report"
echo "[command estimate] H200 GPU 1, eager attention, batch=${BATCH_SIZE}: approximately 35-70 minutes; early stopping may finish sooner"

exec "$PYTHON_BIN" "$SCRIPT" \
  --dataset "$DATASET" \
  --run-name "$RUN_NAME" \
  --questions 256 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout 0 \
  --max-input-tokens 2048 \
  --attn-implementation eager \
  --dtype bfloat16 \
  --device cuda:0 \
  --resume
