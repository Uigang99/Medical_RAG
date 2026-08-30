#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medqa}
MODE=${2:-pilot}

if [[ "${DATASET}" != "medqa" && "${DATASET}" != "medmcqa" ]]; then
  echo "dataset must be medqa or medmcqa" >&2
  exit 2
fi
if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "mode must be pilot or full" >&2
  exit 2
fi

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RUN_TAG=${RUN_TAG:-"${DATASET}_${MODE}_top8_final_choice_v1"}
FEATURE_DIR="$BASE/semantic_attention_controller_v1/$RUN_TAG/prepared_features"
MODEL_DIR="/home/user/Uiheon/models/RAG2-Semantic-Attention-Controller/$DATASET/$RUN_TAG"
CONTROLLER_PATH="$MODEL_DIR/best_controller.pt"
OUTPUT_DIR="$MODEL_DIR/backend_matched_ablation_v1"

for required in \
  "$FEATURE_DIR/preparation_manifest.json" \
  "$CONTROLLER_PATH"
do
  if [[ ! -f "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 1
  fi
done

printf '[%s] Workflow 1/1: backend-matched zero/prior/learned ablation; progress and ETA follow below\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')"

"$PYTHON_BIN" "$PROJECT/scripts/evaluate_rag2_semantic_attention_controller_ablation.py" \
  --dataset "$DATASET" \
  --feature-dir "$FEATURE_DIR" \
  --controller-path "$CONTROLLER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --splits val test \
  --llm-model /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --question-batch-size "${QUESTION_BATCH_SIZE:-32}" \
  --device cuda:0 \
  --resume

printf '[%s] Workflow 1/1 complete: %s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "$OUTPUT_DIR/summary.json"
