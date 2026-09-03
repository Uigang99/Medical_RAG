#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
SCRIPT="$PROJECT/scripts/train_rag2_document_path_generalization.py"
DATASET="${1:-medmcqa}"
MODE="${2:-all}"

if [[ "$DATASET" != "medmcqa" ]]; then
  echo "ERROR: this bounded pilot is currently validated for medmcqa only" >&2
  exit 2
fi
if [[ "$MODE" != "all" && "$MODE" != "support_only" && "$MODE" != "non_support_only" ]]; then
  echo "ERROR: mode must be all, support_only, or non_support_only" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-4000}"
VAL_QUESTIONS="${VAL_QUESTIONS:-1000}"
TEST_QUESTIONS="${TEST_QUESTIONS:-1000}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ "$MODE" == "all" ]]; then
  OBJECTIVES=(support_only non_support_only)
else
  OBJECTIVES=("$MODE")
fi

TOTAL=${#OBJECTIVES[@]}
WORKFLOW_START=$(date +%s)
echo "[workflow plan] isolated document-path objectives=${OBJECTIVES[*]} dataset=${DATASET} train=${TRAIN_QUESTIONS} val=${VAL_QUESTIONS} test=${TEST_QUESTIONS} epochs=${EPOCHS} batch=${BATCH_SIZE}"
echo "[workflow estimate] measured joint pilot was 52 minutes; one isolated run is estimated at 25-40 minutes, all=${TOTAL} run(s) at $((25*TOTAL))-$((40*TOTAL)) minutes on one H200"

for INDEX in "${!OBJECTIVES[@]}"; do
  OBJECTIVE="${OBJECTIVES[$INDEX]}"
  NUMBER=$((INDEX + 1))
  RUN_NAME="medmcqa_document_first_${OBJECTIVE}_4k_v1"
  NOW=$(date +%s)
  ELAPSED=$((NOW - WORKFLOW_START))
  echo "[overall ${NUMBER}/${TOTAL} | elapsed $(printf '%02dh%02dm%02ds' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))) | remaining runs=$((TOTAL-NUMBER+1))] starting objective=${OBJECTIVE}"
  "$PYTHON_BIN" "$SCRIPT" \
    --dataset "$DATASET" \
    --objective "$OBJECTIVE" \
    --run-name "$RUN_NAME" \
    --train-questions "$TRAIN_QUESTIONS" \
    --val-questions "$VAL_QUESTIONS" \
    --test-questions "$TEST_QUESTIONS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "${LEARNING_RATE:-2e-4}" \
    --lora-rank "${LORA_RANK:-8}" \
    --lora-alpha "${LORA_ALPHA:-16}" \
    --lora-dropout 0 \
    --max-input-tokens 2048 \
    --attn-implementation eager \
    --gradient-checkpointing \
    --dtype bfloat16 \
    --device cuda:0 \
    --resume
  NOW=$(date +%s)
  ELAPSED=$((NOW - WORKFLOW_START))
  echo "[overall ${NUMBER}/${TOTAL} complete | elapsed $(printf '%02dh%02dm%02ds' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))) | objective=${OBJECTIVE}]"
done

NOW=$(date +%s)
ELAPSED=$((NOW - WORKFLOW_START))
echo "[workflow complete | elapsed $(printf '%02dh%02dm%02ds' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))) | objectives=${OBJECTIVES[*]}]"
