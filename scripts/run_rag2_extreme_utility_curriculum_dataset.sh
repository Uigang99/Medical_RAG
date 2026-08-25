#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "medmcqa" && "$1" != "medqa" ) ]]; then
  echo "Usage: $0 {medmcqa|medqa} [stage1_epochs] [stage2_epochs]" >&2
  exit 2
fi

DATASET="$1"
if [[ "$DATASET" == "medmcqa" ]]; then
  DEFAULT_STAGE1_EPOCHS=5
else
  DEFAULT_STAGE1_EPOCHS=8
fi
STAGE1_EPOCHS="${2:-$DEFAULT_STAGE1_EPOCHS}"
STAGE2_EPOCHS="${3:-2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
PROJECT=/home/user/Uiheon/Medical_RAG
PREPARED="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/hidden_utility_extreme_curriculum_v1/prepared"
RUN=/home/user/Uiheon/models/RAG2-ExtremeUtility-FlanT5-large/$DATASET/block28_prechoice_tau0p4_text_delta_v1
STAGE1="$RUN/stage1_extreme"
STAGE2="$RUN/stage2_neutral"

run_stage1() {
  local resume_args=()
  if [[ -f "$STAGE1/final_model/extreme_utility_config.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 already complete: $STAGE1/final_model"
    return
  elif [[ -f "$STAGE1/last_checkpoint/training_state.pt" ]]; then
    resume_args=(--resume-from-checkpoint "$STAGE1/last_checkpoint")
  elif [[ -e "$STAGE1" ]]; then
    echo "Incomplete Stage-1 directory has no resumable checkpoint: $STAGE1" >&2
    exit 1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training stage 1/2: high-purity extremes ($DATASET, max $STAGE1_EPOCHS epochs)"
  "$PYTHON" "$PROJECT/scripts/train_rag2_extreme_utility_curriculum.py" \
    --dataset "$DATASET" \
    --prepared-root "$PREPARED" \
    --model-name-or-path /home/user/Uiheon/models/Flan-T5-large \
    --output-dir "$STAGE1" \
    --stage extreme \
    --input-mode text_delta \
    --num-train-epochs "$STAGE1_EPOCHS" \
    --documents-per-train-batch 32 \
    --documents-per-eval-batch 64 \
    --gradient-accumulation-steps 2 \
    --pairwise-loss-weight 0.5 \
    --pairwise-temperature 1.0 \
    --max-input-tokens 768 \
    --early-stopping-patience 2 \
    --bf16 \
    --tf32 \
    --gradient-checkpointing \
    "${resume_args[@]}" \
    --log-level INFO
}

run_stage2() {
  local resume_args=()
  if [[ -f "$STAGE2/final_model/extreme_utility_config.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2 already complete: $STAGE2/final_model"
    return
  elif [[ -f "$STAGE2/last_checkpoint/training_state.pt" ]]; then
    resume_args=(--resume-from-checkpoint "$STAGE2/last_checkpoint")
  elif [[ -e "$STAGE2" ]]; then
    echo "Incomplete Stage-2 directory has no resumable checkpoint: $STAGE2" >&2
    exit 1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training stage 2/2: neutral calibration with extreme replay ($DATASET, max $STAGE2_EPOCHS epochs)"
  "$PYTHON" "$PROJECT/scripts/train_rag2_extreme_utility_curriculum.py" \
    --dataset "$DATASET" \
    --prepared-root "$PREPARED" \
    --model-name-or-path /home/user/Uiheon/models/Flan-T5-large \
    --output-dir "$STAGE2" \
    --stage neutral \
    --stage1-model "$STAGE1/final_model" \
    --input-mode text_delta \
    --num-train-epochs "$STAGE2_EPOCHS" \
    --documents-per-train-batch 32 \
    --documents-per-eval-batch 64 \
    --gradient-accumulation-steps 2 \
    --pairwise-loss-weight 0.5 \
    --neutral-loss-weight 0.1 \
    --stage2-max-extreme-auroc-drop 0.01 \
    --max-input-tokens 768 \
    --early-stopping-patience 2 \
    --bf16 \
    --tf32 \
    --gradient-checkpointing \
    "${resume_args[@]}" \
    --log-level INFO
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overall training 0/2 (overall ETA unknown until both stage rates are observed)"
run_stage1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overall training 1/2 (stage 1 complete; stage 2 ETA is shown by its progress bar)"
run_stage2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overall training 2/2 complete: $STAGE2/final_model"
