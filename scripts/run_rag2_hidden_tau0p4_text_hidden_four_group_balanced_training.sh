#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "medmcqa" && "$1" != "medqa" ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=1 $0 {medmcqa|medqa}" >&2
  exit 2
fi

DATASET="$1"
PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
PROJECT=/home/user/Uiheon/Medical_RAG
ARTIFACT_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2"
SPLIT_ROOT="$ARTIFACT_ROOT/filter_training_inputs_hidden_utility_top8_tau0p4_binary_v1"
FEATURE_ROOT="$ARTIFACT_ROOT/preanswer_hidden_gold_direction_full_top8_v1/$DATASET"
OUTPUT_ROOT=/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-HiddenUtilityTau0p4-FourGroupBalanced
RUN_NAME="${DATASET}_tau0p4_text_hidden_four_group_balanced_epoch5"

if [[ ! -f "$SPLIT_ROOT/$DATASET/manifest.json" ]]; then
  echo "Missing prepared split: $SPLIT_ROOT/$DATASET/manifest.json" >&2
  exit 1
fi

"$PYTHON" "$PROJECT/scripts/train_rag2_hidden_feature_filter.py" \
  --dataset "$DATASET" \
  --input-mode text_hidden \
  --split-root "$SPLIT_ROOT" \
  --hidden-feature-root "$FEATURE_ROOT" \
  --expected-label-threshold 0.4 \
  --expected-label-mode positive_vs_rest \
  --model-name-or-path /home/user/Uiheon/models/Flan-T5-large \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --train-balance-mode four_group_loss \
  --balanced-validation \
  --num-train-epochs 5 \
  --learning-rate 3e-5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 768 \
  --metric-for-best-model macro_f1 \
  --early-stopping-patience 3 \
  --max-grad-norm 0 \
  --preprocessing-num-workers 16 \
  --logging-steps 100 \
  --save-total-limit 3 \
  --eval-accumulation-steps 32 \
  --bf16 \
  --tf32 \
  --log-level INFO
