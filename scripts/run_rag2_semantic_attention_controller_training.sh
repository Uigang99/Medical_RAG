#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medqa}
MODE=${2:-pilot}
ATTENTION_SCOPE=${ATTENTION_SCOPE:-final_choice}

if [[ "${DATASET}" != "medqa" && "${DATASET}" != "medmcqa" ]]; then
  echo "dataset must be medqa or medmcqa" >&2
  exit 2
fi
if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "mode must be pilot or full" >&2
  exit 2
fi
if [[ "${ATTENTION_SCOPE}" != "final_choice" && "${ATTENTION_SCOPE}" != "rationale_wide" ]]; then
  echo "ATTENTION_SCOPE must be final_choice or rationale_wide" >&2
  exit 2
fi

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RUN_TAG=${RUN_TAG:-"${DATASET}_${MODE}_top8_${ATTENTION_SCOPE}_v1"}
# Unbiased rationale generation is independent of the learned attention scope.
# Reuse the already materialized final-choice pilot cache by default.
RATIONALE_CACHE_TAG=${RATIONALE_CACHE_TAG:-"${DATASET}_${MODE}_top8_final_choice_v1"}
RATIONALE_CACHE="$BASE/semantic_attention_controller_v1/$RATIONALE_CACHE_TAG/top8_unbiased_rationales"
FEATURE_DIR="$BASE/semantic_attention_controller_v1/$RUN_TAG/prepared_features"
MODEL_DIR="/home/user/Uiheon/models/RAG2-Semantic-Attention-Controller/$DATASET/$RUN_TAG"
INDEX_PATH="$BASE/semantic_attention_controller_v1/shared_indices/${DATASET}.sqlite"

if [[ "${MODE}" == "pilot" ]]; then
  MAX_GENERATION_QUESTIONS=${MAX_GENERATION_QUESTIONS:-3000}
  MAX_QUESTIONS_PER_SPLIT=${MAX_QUESTIONS_PER_SPLIT:-256}
  EPOCHS=${EPOCHS:-3}
  PARTIAL_FLAG=--allow-partial-rationale-cache
else
  MAX_GENERATION_QUESTIONS=${MAX_GENERATION_QUESTIONS:-0}
  MAX_QUESTIONS_PER_SPLIT=${MAX_QUESTIONS_PER_SPLIT:-0}
  PARTIAL_FLAG=--no-allow-partial-rationale-cache
  if [[ "${DATASET}" == "medqa" ]]; then
    EPOCHS=${EPOCHS:-5}
  else
    EPOCHS=${EPOCHS:-3}
  fi
fi

if [[ "${ATTENTION_SCOPE}" == "rationale_wide" ]]; then
  DEFAULT_QUESTION_BATCH_SIZE=1
  DEFAULT_GRADIENT_ACCUMULATION_STEPS=8
else
  DEFAULT_QUESTION_BATCH_SIZE=32
  DEFAULT_GRADIENT_ACCUMULATION_STEPS=1
fi

workflow_start=$(date +%s)

announce_stage() {
  local stage_index=$1
  local stage_name=$2
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - workflow_start))
  printf '[%s] Workflow %s/3: %s | elapsed=%02dh%02dm%02ds | overall ETA=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$stage_index" "$stage_name" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
    "unknown until the active stage measures throughput"
}

announce_stage 1 "cache unbiased Top-8 rationale + constrained baseline choice"
"$PYTHON_BIN" "$PROJECT/scripts/generate_rag2_top8_baseline_rationales.py" \
  --dataset "$DATASET" \
  --candidate-root "$BASE/candidates/source_balanced32_rerank8_v1" \
  --model-name-or-path /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --output-dir "$RATIONALE_CACHE" \
  --max-questions "$MAX_GENERATION_QUESTIONS" \
  --questions-per-shard "${RATIONALE_QUESTIONS_PER_SHARD:-128}" \
  --generation-batch-size "${RATIONALE_BATCH_SIZE:-64}" \
  --max-new-tokens "${RATIONALE_MAX_NEW_TOKENS:-512}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
  --resume

announce_stage 2 "join full Top-8 labels and cache independent semantic vectors/token spans"
"$PYTHON_BIN" "$PROJECT/scripts/prepare_rag2_semantic_attention_controller.py" \
  --dataset "$DATASET" \
  --rationale-cache "$RATIONALE_CACHE" \
  --output-dir "$FEATURE_DIR" \
  --index-path "$INDEX_PATH" \
  --max-questions-per-split "$MAX_QUESTIONS_PER_SPLIT" \
  --questions-per-shard "${FEATURE_QUESTIONS_PER_SHARD:-128}" \
  --semantic-batch-size "${SEMANTIC_BATCH_SIZE:-64}" \
  --semantic-max-input-length "${SEMANTIC_MAX_INPUT_LENGTH:-2048}" \
  --device cuda:0 \
  "$PARTIAL_FLAG" \
  --resume

announce_stage 3 "train residual document-attention controller with frozen Llama ${ATTENTION_SCOPE} loss"
"$PYTHON_BIN" "$PROJECT/scripts/train_rag2_semantic_attention_controller.py" \
  --dataset "$DATASET" \
  --feature-dir "$FEATURE_DIR" \
  --llm-model /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --output-dir "$MODEL_DIR" \
  --epochs "$EPOCHS" \
  --patience "${PATIENCE:-2}" \
  --question-batch-size "${QUESTION_BATCH_SIZE:-$DEFAULT_QUESTION_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE:-3e-4}" \
  --semantic-layer-start "${SEMANTIC_LAYER_START:-16}" \
  --attention-scope "$ATTENTION_SCOPE" \
  --prior-strength "${PRIOR_STRENGTH:-0.25}" \
  --boundary-epsilon "${BOUNDARY_EPSILON:-0.05}" \
  --ordering-loss-weight "${ORDERING_LOSS_WEIGHT:-0.1}" \
  --anchor-loss-weight "${ANCHOR_LOSS_WEIGHT:-0.001}" \
  --no-rag-group-balance "${NO_RAG_GROUP_BALANCE:-1.0}" \
  --device cuda:0 \
  --resume

workflow_end=$(date +%s)
workflow_elapsed=$((workflow_end - workflow_start))
printf '[%s] Workflow 3/3 complete | elapsed=%02dh%02dm%02ds | model=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$((workflow_elapsed / 3600))" "$(((workflow_elapsed % 3600) / 60))" \
  "$((workflow_elapsed % 60))" "$MODEL_DIR"
