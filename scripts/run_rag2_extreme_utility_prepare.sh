#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
PROJECT=/home/user/Uiheon/Medical_RAG
BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RUN="$BASE/hidden_utility_extreme_curriculum_v1"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overall preparation 0/2 (overall ETA unknown until both stage rates are observed)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparation stage 1/2: gold-answer directions at Block 28 / pre_choice"
"$PYTHON" "$PROJECT/scripts/extract_rag2_anchored_gold_directions.py" \
  --no-rag-root "$BASE/train_no_rag_anchored_features_v1" \
  --model-name-or-path /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --output-root "$RUN/gold_directions" \
  --datasets medmcqa medqa \
  --split train \
  --layer 28 \
  --anchor pre_choice \
  --question-batch-size 32 \
  --max-input-tokens 8192 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --resume \
  --log-level INFO

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overall preparation 1/2 (stage 1 complete; stage 2 ETA is shown by its progress bar)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparation stage 2/2: cached hD-h0 scores, threshold audit, and question-disjoint splits"
"$PYTHON" "$PROJECT/scripts/build_rag2_anchored_extreme_utility_dataset.py" \
  --no-rag-root "$BASE/train_no_rag_anchored_features_v1" \
  --document-root "$BASE/document_traces_source_balanced32_rerank8_v1" \
  --direction-root "$RUN/gold_directions" \
  --reference-split-root "$BASE/filter_training_inputs_rag2_paper_reproduction_v1" \
  --output-root "$RUN/prepared" \
  --datasets medmcqa medqa \
  --source-split train \
  --mode all \
  --layer 28 \
  --anchor pre_choice \
  --primary-threshold 0.4 \
  --threshold-grid 0.2 0.3 0.4 0.5 0.6 \
  --neutral-epsilon 0.05 \
  --minimum-purity 0.90 \
  --resume \
  --log-level INFO

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overall preparation 2/2 complete: $RUN/prepared"
