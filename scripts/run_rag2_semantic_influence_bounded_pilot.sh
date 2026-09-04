#!/usr/bin/env bash
# Semantic document-influence overfit gate followed by a held-out pilot.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
SCRIPT="$PROJECT/scripts/train_rag2_semantic_influence_pilot.py"
MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"

TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-256}"
VAL_QUESTIONS="${VAL_QUESTIONS:-64}"
TEST_QUESTIONS="${TEST_QUESTIONS:-64}"
OVERFIT_QUESTIONS="${OVERFIT_QUESTIONS:-32}"
OVERFIT_EPOCHS="${OVERFIT_EPOCHS:-12}"
EPOCHS="${EPOCHS:-4}"
RUN_NAME="${RUN_NAME:-medqa_mixed_top8_256_64_64_v1}"

test -x "$PYTHON"
test -f "$SCRIPT"
test -f "$MODEL/config.json"

echo "[command verdict] feasible only as a bounded pilot; full-scale training is blocked until this passes"
echo "[purpose] gold-answer margin 없이 Support 영향력은 보존하면서 Support>non-support 순위를 만들고 non-support 영향력을 낮출 수 있는지 검증"
echo "[supervision] 기존 Semantic label 사용(reference answer 기반 annotation일 수 있음); label과 gold answer는 prompt에 넣지 않음"
echo "[data] MedQA mixed Top-8, disjoint train/val/test=${TRAIN_QUESTIONS}/${VAL_QUESTIONS}/${TEST_QUESTIONS}; document order seeded-randomized"
echo "[model] frozen Llama-3-8B + document-token K/V-only LoRA rank 8; Semantic labels are loss metadata, not prompt tokens"
echo "[workflow] stage 1/7 data -> 2/7 tokenize -> 3/7 frozen baseline -> 4/7 overfit32 -> 5/7 held-out train/val -> 6/7 test -> 7/7 bootstrap/report"
echo "[stop condition] 32-question overfit fails: held-out training is not started"
echo "[test pass] pair-ranking +8%p, non-support influence -20%, Support retention >=90%, full-output JSD drift <=0.01, paired CI lower bound >0"
echo "[runtime estimate] H200 GPU 1, uncached BF16 eager attention: approximately 1.5-3.5 hours; active stage recalibrates ETA"
echo "[resume] prepared data, frozen baselines, and every completed epoch are durable; rerun this command unchanged"

exec "$PYTHON" "$SCRIPT" \
  --dataset medqa \
  --model "$MODEL" \
  --run-name "$RUN_NAME" \
  --train-questions "$TRAIN_QUESTIONS" \
  --val-questions "$VAL_QUESTIONS" \
  --test-questions "$TEST_QUESTIONS" \
  --overfit-questions "$OVERFIT_QUESTIONS" \
  --overfit-epochs "$OVERFIT_EPOCHS" \
  --epochs "$EPOCHS" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout 0 \
  --ranking-margin "${RANKING_MARGIN:-0.02}" \
  --ranking-weight "${RANKING_WEIGHT:-1.0}" \
  --non-support-weight "${NON_SUPPORT_WEIGHT:-0.5}" \
  --support-floor-weight "${SUPPORT_FLOOR_WEIGHT:-1.0}" \
  --full-preservation-weight "${FULL_PRESERVATION_WEIGHT:-1.0}" \
  --max-input-tokens 4096 \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-2000}" \
  --seed 42 \
  --device cuda:0 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --resume
