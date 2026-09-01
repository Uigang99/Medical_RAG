#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medqa}
MODE=${2:-pilot}
OBJECTIVE=${3:-all}

if [[ "${DATASET}" != "medqa" && "${DATASET}" != "medmcqa" ]]; then
  echo "dataset must be medqa or medmcqa" >&2
  exit 2
fi
if [[ "${MODE}" != "pilot" && "${MODE}" != "scaled" && "${MODE}" != "full" ]]; then
  echo "mode must be pilot, scaled, or full" >&2
  exit 2
fi
if [[ "${OBJECTIVE}" != "all" && "${OBJECTIVE}" != "compare" && \
      "${OBJECTIVE}" != "proposed_preserved" && "${OBJECTIVE}" != "proposed" && \
      "${OBJECTIVE}" != "question_only" && "${OBJECTIVE}" != "rag_ce" ]]; then
  echo "objective must be all, compare, proposed_preserved, proposed, question_only, or rag_ce" >&2
  exit 2
fi

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
PAIR_ROOT=${PAIR_ROOT:-"$BASE/semantic_behavior_single_document_pairs_${MODE}_v1"}
MODEL_ROOT=${MODEL_ROOT:-/home/user/Uiheon/models/RAG2-SemanticBehavior-LoRA}
LLAMA_MODEL=${LLAMA_MODEL:-/home/user/Uiheon/models/Llama-3-8B-Instruct}
# v1 could fail before its first optimizer step when PyTorch selected cuDNN
# SDPA for a left-padded BF16 batch.  v2 records and uses the safe SDPA policy.
RUN_SUFFIX=${RUN_SUFFIX:-v3}

if [[ "${MODE}" == "pilot" ]]; then
  MAX_TRAIN_PAIRS=${MAX_TRAIN_PAIRS:-3000}
  MAX_EVAL_PAIRS=${MAX_EVAL_PAIRS:-1000}
  EPOCHS=${EPOCHS:-3}
  TRAIN_SELECTION=${TRAIN_SELECTION:-stratified}
elif [[ "${MODE}" == "scaled" ]]; then
  TRAIN_SELECTION=${TRAIN_SELECTION:-natural}
  EPOCHS=${EPOCHS:-3}
  if [[ "${DATASET}" == "medqa" ]]; then
    MAX_TRAIN_PAIRS=${MAX_TRAIN_PAIRS:-0}
    MAX_EVAL_PAIRS=${MAX_EVAL_PAIRS:-0}
  else
    MAX_TRAIN_PAIRS=${MAX_TRAIN_PAIRS:-5000}
    MAX_EVAL_PAIRS=${MAX_EVAL_PAIRS:-1000}
  fi
else
  TRAIN_SELECTION=${TRAIN_SELECTION:-natural}
  MAX_TRAIN_PAIRS=${MAX_TRAIN_PAIRS:-0}
  MAX_EVAL_PAIRS=${MAX_EVAL_PAIRS:-0}
  if [[ "${DATASET}" == "medqa" ]]; then
    EPOCHS=${EPOCHS:-5}
  else
    EPOCHS=${EPOCHS:-3}
  fi
fi

workflow_start=$(date +%s)

announce_stage() {
  local index=$1
  local total=$2
  local name=$3
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - workflow_start))
  printf '[%s] Workflow %s/%s: %s | elapsed=%02dh%02dm%02ds | overall ETA=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$index" "$total" "$name" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
    "reported by the active resumable stage after throughput is measured"
}

if [[ "${OBJECTIVE}" == "all" ]]; then
  OBJECTIVES=(question_only rag_ce proposed proposed_preserved)
elif [[ "${OBJECTIVE}" == "compare" ]]; then
  OBJECTIVES=(proposed proposed_preserved)
else
  OBJECTIVES=("${OBJECTIVE}")
fi
TOTAL_STAGES=$((1 + ${#OBJECTIVES[@]}))

announce_stage 1 "$TOTAL_STAGES" "prepare same-question D+ / D- single-document pairs"
"$PYTHON_BIN" "$PROJECT/scripts/prepare_rag2_semantic_behavior_pairs.py" \
  --dataset "$DATASET" \
  --output-root "$PAIR_ROOT" \
  --max-train-pairs "$MAX_TRAIN_PAIRS" \
  --max-eval-pairs "$MAX_EVAL_PAIRS" \
  --hard-fraction "${HARD_FRACTION:-0.70}" \
  --train-selection "$TRAIN_SELECTION" \
  --violation-threshold "${VIOLATION_THRESHOLD:-0.0}" \
  --seed "${SEED:-42}" \
  --resume

stage=2
for current_objective in "${OBJECTIVES[@]}"; do
  announce_stage "$stage" "$TOTAL_STAGES" \
    "LoRA train objective=${current_objective} (single document per forward)"
  run_name="${DATASET}_${MODE}_semantic_behavior_${current_objective}_${RUN_SUFFIX}"
  if [[ "$current_objective" == "proposed_preserved" ]]; then
    current_negative_weight=${PRESERVED_NEGATIVE_INVARIANCE_WEIGHT:-1.0}
    current_no_rag_weight=${PRESERVED_NO_RAG_PRESERVATION_WEIGHT:-1.0}
  else
    current_negative_weight=${NEGATIVE_INVARIANCE_WEIGHT:-0.1}
    current_no_rag_weight=${NO_RAG_PRESERVATION_WEIGHT:-0.1}
  fi
  "$PYTHON_BIN" "$PROJECT/scripts/train_rag2_semantic_behavior_lora.py" \
    --dataset "$DATASET" \
    --pair-root "$PAIR_ROOT" \
    --model-name-or-path "$LLAMA_MODEL" \
    --output-root "$MODEL_ROOT" \
    --run-name "$run_name" \
    --objective "$current_objective" \
    --epochs "$EPOCHS" \
    --patience "${PATIENCE:-2}" \
    --train-pairs-per-batch "${TRAIN_PAIRS_PER_BATCH:-2}" \
    --eval-pairs-per-batch "${EVAL_PAIRS_PER_BATCH:-4}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --warmup-ratio "${WARMUP_RATIO:-0.03}" \
    --max-input-tokens "${MAX_INPUT_TOKENS:-4096}" \
    --preference-margin "${PREFERENCE_MARGIN:-0.5}" \
    --positive-loss-weight "${POSITIVE_LOSS_WEIGHT:-1.0}" \
    --preference-loss-weight "${PREFERENCE_LOSS_WEIGHT:-1.0}" \
    --negative-invariance-weight "$current_negative_weight" \
    --no-rag-preservation-weight "$current_no_rag_weight" \
    --max-negative-no-rag-answer-change-rate "${MAX_NEGATIVE_NO_RAG_ANSWER_CHANGE_RATE:-0.10}" \
    --max-negative-no-rag-js "${MAX_NEGATIVE_NO_RAG_JS:-0.05}" \
    --max-negative-accuracy-drop "${MAX_NEGATIVE_ACCURACY_DROP:-0.03}" \
    --max-no-rag-accuracy-drop "${MAX_NO_RAG_ACCURACY_DROP:-0.01}" \
    --lora-rank "${LORA_RANK:-16}" \
    --lora-alpha "${LORA_ALPHA:-32}" \
    --dtype "${DTYPE:-bfloat16}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
    --device cuda:0 \
    --seed "${SEED:-42}" \
    --resume
  stage=$((stage + 1))
done

workflow_end=$(date +%s)
elapsed=$((workflow_end - workflow_start))
printf '[%s] Workflow %s/%s complete | elapsed=%02dh%02dm%02ds | outputs=%s/%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "$TOTAL_STAGES" "$TOTAL_STAGES" \
  "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
  "$MODEL_ROOT" "$DATASET"
