#!/usr/bin/env bash
# Bounded PCW source-level Semantic influence learnability pilot.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
SCRIPT="$PROJECT/scripts/train_rag2_pcw_semantic_influence_pilot.py"
LLAMA="/home/user/Uiheon/models/Llama-3-8B-Instruct"
SEMANTIC_MODEL="/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medqa/medqa_semantic_top8_binary_support_epoch8_len1280_fullpair/20260830_170945/final_model"

TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-128}"
VAL_QUESTIONS="${VAL_QUESTIONS:-64}"
TEST_QUESTIONS="${TEST_QUESTIONS:-64}"
MECHANISM_QUESTIONS="${MECHANISM_QUESTIONS:-32}"
OVERFIT_QUESTIONS="${OVERFIT_QUESTIONS:-32}"
OVERFIT_EPOCHS="${OVERFIT_EPOCHS:-20}"
EPOCHS="${EPOCHS:-5}"
RUN_NAME="${RUN_NAME:-medqa_pcw_mixed128_64_64_v4}"

test -x "$PYTHON"
test -f "$SCRIPT"
test -f "$LLAMA/config.json"
test -f "$SEMANTIC_MODEL/config.json"

echo "[command verdict] feasible only as a bounded pilot; full-data training is blocked until all stop gates pass"
echo "[purpose] PCW-isolated document fusion gates can learn Support>non-support causal output influence without gold-answer loss"
echo "[data] MedQA mixed Top-8, disjoint train/val/test=${TRAIN_QUESTIONS}/${VAL_QUESTIONS}/${TEST_QUESTIONS}"
echo "[models] frozen Llama-3-8B + frozen Semantic 2-class Flan-T5; only a small set Router is trained"
echo "[influence] full-vocabulary JSD between PCW full output and one document-channel hard drop"
echo "[workflow] 1/8 preflight -> 2/8 Semantic scores -> 3/8 PCW tokenize/mask -> 4/8 mechanism -> 5/8 frozen influence -> 6/8 overfit -> 7/8 held-out train/val -> 8/8 test/report"
echo "[automatic stop] mechanism failure stops before training; 32-question overfit failure stops before held-out training"
echo "[pilot pass] held-out pair gain >=8%p, non-support influence -20%, Support retention >=90%, output JSD drift <=0.01, bootstrap lower bound >0"
echo "[runtime estimate] H200 GPU 1, empty caches, BF16 eager PCW: approximately 50-90 minutes; calibrated from validated one-question forward/backward throughput"
echo "[storage estimate] approximately 1-3 GiB; no KV cache is persisted"
echo "[resume] all expensive Semantic, mechanism, frozen-baseline, and epoch artifacts are durable; rerun unchanged"

exec "$PYTHON" "$SCRIPT" \
  --dataset medqa \
  --llama-model "$LLAMA" \
  --semantic-model "$SEMANTIC_MODEL" \
  --run-name "$RUN_NAME" \
  --train-questions "$TRAIN_QUESTIONS" \
  --val-questions "$VAL_QUESTIONS" \
  --test-questions "$TEST_QUESTIONS" \
  --mechanism-questions "$MECHANISM_QUESTIONS" \
  --overfit-questions "$OVERFIT_QUESTIONS" \
  --overfit-epochs "$OVERFIT_EPOCHS" \
  --epochs "$EPOCHS" \
  --semantic-batch-size "${SEMANTIC_BATCH_SIZE:-32}" \
  --variant-batch-size "${VARIANT_BATCH_SIZE:-8}" \
  --router-hidden-dim "${ROUTER_HIDDEN_DIM:-64}" \
  --router-heads "${ROUTER_HEADS:-4}" \
  --router-layers "${ROUTER_LAYERS:-2}" \
  --learning-rate "${LEARNING_RATE:-3e-4}" \
  --min-gate "${MIN_GATE:-0.05}" \
  --max-gate "${MAX_GATE:-1.50}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-1000}" \
  --max-input-tokens 4096 \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume "$@"
