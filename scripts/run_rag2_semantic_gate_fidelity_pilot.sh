#!/usr/bin/env bash
set -euo pipefail

# Physical GPU selection is controlled by the caller, for example:
# CUDA_VISIBLE_DEVICES=1 bash .../run_rag2_semantic_gate_fidelity_pilot.sh medqa

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medqa}

if [[ "$DATASET" != "medqa" ]]; then
  echo "The cached rationale-wide pilot features currently exist only for medqa." >&2
  exit 2
fi

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RUN_TAG="medqa_pilot_top8_rationale_wide_v1"
FEATURE_DIR="$BASE/semantic_attention_controller_v1/$RUN_TAG/prepared_features"
MODEL_DIR="/home/user/Uiheon/models/RAG2-Semantic-Attention-Controller/medqa/$RUN_TAG"
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT/results/rag2_semantic_gate_fidelity_v1/${RUN_TAG}_n${GATE_AUDIT_SAMPLES:-256}"}

workflow_start=$(date +%s)

announce_stage() {
  local stage_index=$1
  local stage_name=$2
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - workflow_start))
  printf '[%s] Workflow %s/2: %s | elapsed=%02dh%02dm%02ds | overall ETA=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$stage_index" "$stage_name" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
    "unknown until the active stage measures throughput"
}

announce_stage 1 "restore/train rationale-wide semantic document-gate controller"
ATTENTION_SCOPE=rationale_wide \
RUN_TAG="$RUN_TAG" \
RATIONALE_CACHE_TAG=medqa_pilot_top8_final_choice_v1 \
MAX_GENERATION_QUESTIONS=3000 \
MAX_QUESTIONS_PER_SPLIT=256 \
EPOCHS=${GATE_TRAIN_EPOCHS:-3} \
PATIENCE=${GATE_TRAIN_PATIENCE:-2} \
bash "$PROJECT/scripts/run_rag2_semantic_attention_controller_training.sh" medqa pilot

if [[ ! -f "$MODEL_DIR/final_controller.pt" ]]; then
  echo "Controller training did not produce $MODEL_DIR/final_controller.pt" >&2
  exit 1
fi

announce_stage 2 "compare Top-8 gate and attention shares with physical-token LOO influence"
"$PYTHON_BIN" "$PROJECT/scripts/evaluate_rag2_semantic_gate_fidelity.py" \
  --dataset medqa \
  --split test \
  --feature-dir "$FEATURE_DIR" \
  --controller-checkpoint "$MODEL_DIR/final_controller.pt" \
  --llm-model /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "${GATE_AUDIT_SAMPLES:-256}" \
  --sample-seed "${GATE_AUDIT_SEED:-42}" \
  --minimum-total-jsd "${MINIMUM_TOTAL_JSD:-0.000001}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume \
  --log-level INFO

workflow_end=$(date +%s)
workflow_elapsed=$((workflow_end - workflow_start))
printf '[%s] Workflow 2/2 complete | elapsed=%02dh%02dm%02ds | summary=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$((workflow_elapsed / 3600))" "$(((workflow_elapsed % 3600) / 60))" \
  "$((workflow_elapsed % 60))" "$OUTPUT_DIR/summary.md"
