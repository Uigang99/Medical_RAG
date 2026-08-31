#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medqa}
MODE=${2:-memorize}

if [[ "${DATASET}" != "medqa" && "${DATASET}" != "medmcqa" ]]; then
  echo "dataset must be medqa or medmcqa" >&2
  exit 2
fi
if [[ "${MODE}" != "memorize" && "${MODE}" != "pilot" ]]; then
  echo "mode must be memorize or pilot" >&2
  exit 2
fi

boolean_flag() {
  local value=$1
  local enabled=$2
  local disabled=$3
  case "${value}" in
    1|true|TRUE|yes|YES) printf '%s' "${enabled}" ;;
    0|false|FALSE|no|NO) printf '%s' "${disabled}" ;;
    *)
      echo "Expected a boolean value, got: ${value}" >&2
      exit 2
      ;;
  esac
}

RANK_FEATURE_FLAG=$(boolean_flag "${USE_RANK_FEATURE:-1}" --use-rank-feature --no-use-rank-feature)
LENGTH_FEATURE_FLAG=$(boolean_flag "${USE_LENGTH_FEATURE:-1}" --use-length-feature --no-use-length-feature)
SHUFFLE_DOCUMENT_FLAG=$(boolean_flag \
  "${SHUFFLE_DOCUMENTS_DURING_TRAINING:-0}" \
  --shuffle-documents-during-training \
  --no-shuffle-documents-during-training)

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
SOURCE_FEATURE_DIR=${SOURCE_FEATURE_DIR:-"$BASE/semantic_attention_controller_v1/${DATASET}_pilot_top8_rationale_wide_v1/prepared_features"}
RUN_TAG=${RUN_TAG:-"${DATASET}_${MODE}_k8_layers20_28_fixed_rationale_v1"}
FEATURE_DIR=${ATTRIBUTION_FEATURE_DIR:-"$BASE/target_llm_attribution_v1/$RUN_TAG"}
MODEL_DIR=${ATTRIBUTION_MODEL_DIR:-"/home/user/Uiheon/models/RAG2-Target-LLM-Attribution/$DATASET/$RUN_TAG"}

if [[ ! -f "$SOURCE_FEATURE_DIR/preparation_manifest.json" ]]; then
  echo "Missing K-specific source prompt cache: $SOURCE_FEATURE_DIR" >&2
  echo "Set SOURCE_FEATURE_DIR to a prepared cache whose rationale was generated with the same K." >&2
  exit 1
fi

if [[ "${MODE}" == "memorize" ]]; then
  MAX_PREPARED_QUESTIONS=${MAX_PREPARED_QUESTIONS:-64}
  EPOCHS=${EPOCHS:-50}
  PATIENCE=${PATIENCE:-0}
  DROPOUT=${DROPOUT:-0.0}
  WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
  MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-64}
  MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-64}
  MEMORIZATION_FLAG=--memorization-check
  PREPARATION_SPLITS=(--splits train)
else
  MAX_PREPARED_QUESTIONS=${MAX_PREPARED_QUESTIONS:-256}
  EPOCHS=${EPOCHS:-15}
  PATIENCE=${PATIENCE:-4}
  DROPOUT=${DROPOUT:-0.1}
  WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
  MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
  MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
  MEMORIZATION_FLAG=
  PREPARATION_SPLITS=(--splits train val test)
fi

workflow_start=$(date +%s)
announce_stage() {
  local index=$1
  local name=$2
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - workflow_start))
  printf '[%s] Workflow %s/2: %s | elapsed=%02dh%02dm%02ds | overall ETA=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$index" "$name" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
    "unknown until the active stage measures throughput"
}

announce_stage 1 "generate unbiased fixed-rationale LOO teacher and frozen-Llama span features"
"$PYTHON_BIN" "$PROJECT/scripts/prepare_rag2_target_llm_attribution.py" \
  --dataset "$DATASET" \
  --source-feature-dir "$SOURCE_FEATURE_DIR" \
  --llm-model /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --output-dir "$FEATURE_DIR" \
  --layers 20 28 \
  "${PREPARATION_SPLITS[@]}" \
  --max-questions-per-split "$MAX_PREPARED_QUESTIONS" \
  --expected-document-count 8 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --resume

announce_stage 2 "train sequence-aware conditional-removal attribution predictor"
"$PYTHON_BIN" "$PROJECT/scripts/train_rag2_target_llm_attribution.py" \
  --dataset "$DATASET" \
  --feature-dir "$FEATURE_DIR" \
  --output-dir "$MODEL_DIR" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "${BATCH_SIZE:-32}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --weight-decay "$WEIGHT_DECAY" \
  --dropout "$DROPOUT" \
  --model-dim "${MODEL_DIM:-256}" \
  --transformer-layers "${TRANSFORMER_LAYERS:-2}" \
  --attention-heads "${ATTENTION_HEADS:-4}" \
  --feedforward-dim "${FEEDFORWARD_DIM:-1024}" \
  --total-loss-weight "${TOTAL_LOSS_WEIGHT:-1.0}" \
  --share-loss-weight "${SHARE_LOSS_WEIGHT:-0.5}" \
  --set-shift-loss-weight "${SET_SHIFT_LOSS_WEIGHT:-0.5}" \
  --rank-loss-weight "${RANK_LOSS_WEIGHT:-0.1}" \
  --minimum-total-for-share "${MINIMUM_TOTAL_FOR_SHARE:-1e-6}" \
  --minimum-rank-log-ratio "${MINIMUM_RANK_LOG_RATIO:-0.25}" \
  "$RANK_FEATURE_FLAG" \
  "$LENGTH_FEATURE_FLAG" \
  "$SHUFFLE_DOCUMENT_FLAG" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-eval-samples "$MAX_EVAL_SAMPLES" \
  --num-workers "${NUM_WORKERS:-4}" \
  --device cuda:0 \
  $MEMORIZATION_FLAG \
  --resume

workflow_end=$(date +%s)
workflow_elapsed=$((workflow_end - workflow_start))
printf '[%s] Workflow 2/2 complete | elapsed=%02dh%02dm%02ds | model=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$((workflow_elapsed / 3600))" "$(((workflow_elapsed % 3600) / 60))" \
  "$((workflow_elapsed % 60))" "$MODEL_DIR"
