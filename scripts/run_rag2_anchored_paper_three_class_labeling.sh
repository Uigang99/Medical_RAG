#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="/home/user/Uiheon/.venv_vllm/bin/python"
BASE="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/build_rag2_anchored_paper_labels.py" \
  --no-rag-root "${BASE}/train_no_rag_anchored_features_v1" \
  --document-trace-root "${BASE}/document_traces_source_balanced32_rerank8_v1" \
  --output-root "${BASE}/filter_training_inputs_rag2_paper_reproduction_three_class_v1" \
  --datasets medmcqa medqa \
  --source-split train \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --threshold-quantile 0.75 \
  --max-doc-rank 8 \
  --max-doc-chars 0 \
  --training-label-mode three_class \
  --seed 42 \
  --overwrite \
  --show-progress \
  --log-level INFO
