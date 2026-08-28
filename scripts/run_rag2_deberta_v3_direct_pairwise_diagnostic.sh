#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}"
TARGET="${2:-all}"
if [[ "${MODE}" != "overfit" && "${MODE}" != "pilot" && "${MODE}" != "both" ]]; then
  echo "Usage: $0 [overfit|pilot|both] [all|medmcqa|medqa]" >&2
  exit 2
fi
if [[ "${TARGET}" != "all" && "${TARGET}" != "medmcqa" && "${TARGET}" != "medqa" ]]; then
  echo "Usage: $0 [overfit|pilot|both] [all|medmcqa|medqa]" >&2
  exit 2
fi

PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
HF_BIN="${HF_BIN:-/home/user/Uiheon/.venv_vllm/bin/hf}"
BASE="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
PREPARED_ROOT="${BASE}/gold_margin_regression_v1/prepared"
MODEL_DIR="${DEBERTA_MODEL_DIR:-/home/user/Uiheon/models/DeBERTa-v3-large}"
OUTPUT_ROOT="${DEBERTA_OUTPUT_ROOT:-/home/user/Uiheon/models/RAG2-DirectPairComparator-DeBERTa-v3-large}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

# Torch 2.11 is built against CUDA 13 in this environment.  The matching
# NVRTC package is installed inside the vLLM virtualenv, but its cu13 library
# directory is not added to the process search path automatically.
NVRTC_CU13_LIB="${NVRTC_CU13_LIB:-/home/user/Uiheon/.venv_vllm/lib/python3.10/site-packages/nvidia/cu13/lib}"
if [[ -f "${NVRTC_CU13_LIB}/libnvrtc-builtins.so.13.0" ]]; then
  export LD_LIBRARY_PATH="${NVRTC_CU13_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ "${MODE}" == "both" ]]; then
  MODES=(overfit pilot)
else
  MODES=("${MODE}")
fi
if [[ "${TARGET}" == "all" ]]; then
  DATASETS=(medmcqa medqa)
else
  DATASETS=("${TARGET}")
fi

format_seconds() {
  local value="$1"
  printf '%02dh%02dm%02ds' "$((value / 3600))" "$(((value % 3600) / 60))" "$((value % 60))"
}

PIPELINE_START="$(date +%s)"
TOTAL_STAGES="$((2 + ${#MODES[@]} * ${#DATASETS[@]}))"
COMPLETED_STAGES=0
echo "[$(date '+%F %T')] Overall DeBERTa-v3 direct-pair diagnostic 0/${TOTAL_STAGES}; current stage=model availability 0/1; stage ETA unknown; overall ETA unknown"

MODEL_WEIGHTS_READY=false
if [[ -f "${MODEL_DIR}/model.safetensors" ]] \
  || [[ -f "${MODEL_DIR}/pytorch_model.bin" ]] \
  || [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] \
  || [[ -f "${MODEL_DIR}/pytorch_model.bin.index.json" ]]; then
  MODEL_WEIGHTS_READY=true
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]] || [[ "${MODEL_WEIGHTS_READY}" != "true" ]]; then
  echo "[$(date '+%F %T')] Stage 1/${TOTAL_STAGES}: download microsoft/deberta-v3-large; Hugging Face reports file progress/ETA"
  "${HF_BIN}" download microsoft/deberta-v3-large --local-dir "${MODEL_DIR}"
else
  echo "[$(date '+%F %T')] Stage 1/${TOTAL_STAGES}: DeBERTa-v3-large already cached (1/1, stage ETA=0s)"
fi
COMPLETED_STAGES=1

if [[ ! -f "${PREPARED_ROOT}/manifest.json" ]]; then
  echo "[$(date '+%F %T')] Stage 2/${TOTAL_STAGES}: materialize utility pointer splits; child pipeline reports progress/ETA"
  bash "${PROJECT_ROOT}/scripts/run_rag2_margin_regression_prepare.sh"
else
  echo "[$(date '+%F %T')] Stage 2/${TOTAL_STAGES}: utility pointer splits already cached (1/1, stage ETA=0s)"
fi
COMPLETED_STAGES=2

for CURRENT_MODE in "${MODES[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    STAGE="$((COMPLETED_STAGES + 1))"
    NOW="$(date +%s)"
    ELAPSED="$((NOW - PIPELINE_START))"
    if [[ "${COMPLETED_STAGES}" -gt 2 ]]; then
      OBSERVED="$((COMPLETED_STAGES - 2))"
      REMAINING="$((TOTAL_STAGES - COMPLETED_STAGES))"
      ETA="$((ELAPSED / OBSERVED * REMAINING))"
      ETA_TEXT="$(format_seconds "${ETA}")"
    else
      ETA_TEXT="unknown until one training stage is measured"
    fi
    echo "[$(date '+%F %T')] Stage ${STAGE}/${TOTAL_STAGES}: ${DATASET} ${CURRENT_MODE}; current stage progress=0%; stage ETA shown by trainer; elapsed=$(format_seconds "${ELAPSED}"); overall ETA=${ETA_TEXT}"

    COMMON_ARGS=(
      --dataset "${DATASET}"
      --prepared-root "${PREPARED_ROOT}"
      --model-name-or-path "${MODEL_DIR}"
      --model-backend sequence_classification
      --no-rag-generation-root "${BASE}/train_no_rag_anchored_features_v1/no_rag"
      --output-root "${OUTPUT_ROOT}"
      --document-pair-min-utility-gap "${PAIR_MIN_GAP:-0.1}"
      --max-input-tokens "${MAX_INPUT_TOKENS:-512}"
      --max-semantic-pairs-per-forward "${MAX_SEMANTIC_PAIRS_PER_FORWARD:-8}"
      --minimum-document-tokens 16
      --trace-shard-cache-size 8
      --bf16
      --tf32
      --no-gradient-checkpointing
      --show-progress
      --seed 42
      --log-level INFO
    )

    if [[ "${CURRENT_MODE}" == "overfit" ]]; then
      TRAIN_QUESTIONS="${OVERFIT_TRAIN_QUESTIONS:-100}"
      EVAL_QUESTIONS="${OVERFIT_EVAL_QUESTIONS:-100}"
      EPOCHS="${OVERFIT_EPOCHS:-30}"
      RUN_NAME="${DATASET}_deberta_v3_direct_pair_overfit_q${TRAIN_QUESTIONS}_epoch${EPOCHS}"
      MODE_ARGS=(
        --run-name "${RUN_NAME}"
        --num-train-epochs "${EPOCHS}"
        --max-train-questions "${TRAIN_QUESTIONS}"
        --max-eval-questions "${EVAL_QUESTIONS}"
        --train-questions-per-batch 2
        --eval-questions-per-batch 4
        --gradient-accumulation-steps 1
        --learning-rate "${OVERFIT_LEARNING_RATE:-2e-5}"
        --weight-decay 0.0
        --warmup-ratio 0.0
        --dropout 0.0
        --evaluate-train
        --checkpoint-split train
        --stop-train-question-macro-accuracy "${OVERFIT_SUCCESS_ACCURACY:-0.95}"
        --early-stopping-patience 30
        --minimum-improvement 1e-4
        --logging-steps 20
      )
    else
      TRAIN_QUESTIONS="${PILOT_TRAIN_QUESTIONS:-5000}"
      EVAL_QUESTIONS="${PILOT_EVAL_QUESTIONS:-1000}"
      EPOCHS="${PILOT_EPOCHS:-3}"
      RUN_NAME="${DATASET}_deberta_v3_direct_pair_pilot_q${TRAIN_QUESTIONS}_epoch${EPOCHS}"
      MODE_ARGS=(
        --run-name "${RUN_NAME}"
        --num-train-epochs "${EPOCHS}"
        --max-train-questions "${TRAIN_QUESTIONS}"
        --max-eval-questions "${EVAL_QUESTIONS}"
        --train-questions-per-batch "${TRAIN_QUESTIONS_PER_BATCH:-4}"
        --eval-questions-per-batch "${EVAL_QUESTIONS_PER_BATCH:-8}"
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
        --learning-rate "${PILOT_LEARNING_RATE:-1e-5}"
        --weight-decay 0.01
        --warmup-ratio 0.06
        --dropout 0.1
        --checkpoint-split validation
        --early-stopping-patience 2
        --minimum-improvement 1e-4
        --logging-steps 25
      )
    fi

    OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/${RUN_NAME}"
    if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
      echo "[$(date '+%F %T')] ${DATASET} ${CURRENT_MODE} already complete: ${OUTPUT_DIR}/summary.json"
    else
      RESUME_ARGS=()
      if [[ -f "${OUTPUT_DIR}/last_checkpoint/training_state.pt" ]]; then
        RESUME_ARGS=(--resume-from-checkpoint "${OUTPUT_DIR}/last_checkpoint")
        echo "[$(date '+%F %T')] Resuming ${DATASET} ${CURRENT_MODE} from ${OUTPUT_DIR}/last_checkpoint"
      fi
      "${PYTHON_BIN}" \
        "${PROJECT_ROOT}/scripts/train_rag2_direct_pairwise_comparator.py" \
        "${COMMON_ARGS[@]}" \
        "${MODE_ARGS[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        "${RESUME_ARGS[@]}"
    fi
    COMPLETED_STAGES="$((COMPLETED_STAGES + 1))"
  done
done

FINISHED="$(date +%s)"
echo "[$(date '+%F %T')] Overall DeBERTa-v3 direct-pair diagnostic ${TOTAL_STAGES}/${TOTAL_STAGES} complete; current stage=complete; elapsed=$(format_seconds "$((FINISHED - PIPELINE_START))"); stage ETA=0s; overall ETA=0s"
