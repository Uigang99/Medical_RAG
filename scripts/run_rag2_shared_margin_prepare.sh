#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
BASE="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/build_rag2_shared_margin_dataset.py" \
  --score-root "${BASE}/gold_margin_utility_source_balanced32_rerank8_v1" \
  --trace-root "${BASE}/document_traces_source_balanced32_rerank8_v1" \
  --reference-split-root "${BASE}/filter_training_inputs_rag2_paper_reproduction_v1" \
  --output-root "${BASE}/shared_gold_margin_regression_v1/prepared" \
  --datasets medmcqa medqa \
  --source-split train \
  --minimum-documents 1 \
  --maximum-documents 8 \
  --exclude-quality-flags \
  --show-progress \
  --log-level INFO
