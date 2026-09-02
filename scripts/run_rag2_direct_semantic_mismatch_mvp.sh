#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medmcqa}
MODE=${2:-pilot}
OBJECTIVE=${3:-mismatch}

if [[ "${DATASET}" != "medmcqa" && "${DATASET}" != "medqa" ]]; then
  echo "dataset must be medmcqa or medqa" >&2
  exit 2
fi
if [[ "${MODE}" != "pilot" ]]; then
  echo "Only the preregistered pilot is enabled. Scale only after it passes." >&2
  exit 2
fi
if [[ "${OBJECTIVE}" != "mismatch" && "${OBJECTIVE}" != "question_only" && "${OBJECTIVE}" != "rag_ce" ]]; then
  echo "objective must be mismatch, question_only, or rag_ce" >&2
  exit 2
fi

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
PAIR_ROOT=${PAIR_ROOT:-"$BASE/direct_semantic_mismatch_${MODE}_pairs_v1"}
MODEL_ROOT=${MODEL_ROOT:-/home/user/Uiheon/models/RAG2-Direct-Semantic-Mismatch-LoRA}
LLAMA_MODEL=${LLAMA_MODEL:-/home/user/Uiheon/models/Llama-3-8B-Instruct}

MAX_TRAIN_QUESTIONS=${MAX_TRAIN_QUESTIONS:-4000}
if [[ "${DATASET}" == "medqa" ]]; then
  MAX_EVAL_QUESTIONS=${MAX_EVAL_QUESTIONS:-0}
else
  MAX_EVAL_QUESTIONS=${MAX_EVAL_QUESTIONS:-4000}
fi
EPOCHS=${EPOCHS:-3}

workflow_start=$(date +%s)
announce_stage() {
  local index=$1 total=$2 name=$3
  local elapsed=$(( $(date +%s) - workflow_start ))
  printf '[%s] [overall %s/%s | elapsed %02dh%02dm%02ds | overall ETA unknown] %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$index" "$total" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" "$name"
}

announce_stage 1 2 "preflight, cache joins, and versioned train/validation/test materialization"
"$PYTHON_BIN" "$PROJECT/scripts/prepare_rag2_direct_semantic_mismatch_mvp.py" \
  --dataset "$DATASET" \
  --output-root "$PAIR_ROOT" \
  --max-train-questions "$MAX_TRAIN_QUESTIONS" \
  --max-eval-questions "$MAX_EVAL_QUESTIONS" \
  --train-failure-fraction "${TRAIN_FAILURE_FRACTION:-0.80}" \
  --seed "${SEED:-42}" \
  --resume

announce_stage 2 2 "Llama-3-8B LoRA training objective=${OBJECTIVE}; active bars report current epoch progress/rate/ETA"
run_name="${DATASET}_${MODE}_direct_semantic_mismatch_${OBJECTIVE}_v2"
"$PYTHON_BIN" "$PROJECT/scripts/train_rag2_direct_semantic_mismatch_lora.py" \
  --dataset "$DATASET" \
  --pair-root "$PAIR_ROOT" \
  --model-name-or-path "$LLAMA_MODEL" \
  --output-root "$MODEL_ROOT" \
  --run-name "$run_name" \
  --objective "$OBJECTIVE" \
  --epochs "$EPOCHS" \
  --patience "${PATIENCE:-2}" \
  --train-examples-per-batch "${TRAIN_EXAMPLES_PER_BATCH:-8}" \
  --eval-examples-per-batch "${EVAL_EXAMPLES_PER_BATCH:-32}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --warmup-ratio "${WARMUP_RATIO:-0.03}" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-2048}" \
  --boundary-margin "${BOUNDARY_MARGIN:-0.0}" \
  --gain-margin "${GAIN_MARGIN:-0.5}" \
  --no-rag-preservation-weight "${NO_RAG_PRESERVATION_WEIGHT:-2.0}" \
  --failure-case-weight "${FAILURE_CASE_WEIGHT:-1.0}" \
  --normal-case-weight "${NORMAL_CASE_WEIGHT:-0.1}" \
  --max-no-rag-accuracy-drop "${MAX_NO_RAG_ACCURACY_DROP:-0.005}" \
  --max-destruction-rate-increase "${MAX_DESTRUCTION_RATE_INCREASE:-0.0}" \
  --lora-rank "${LORA_RANK:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --dtype "${DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --device cuda:0 \
  --seed "${SEED:-42}" \
  --resume

elapsed=$(( $(date +%s) - workflow_start ))
printf '[%s] [overall 2/2 complete | elapsed %02dh%02dm%02ds] model=%s/%s/%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "$((elapsed / 3600))" \
  "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
  "$MODEL_ROOT" "$DATASET" "$run_name"
