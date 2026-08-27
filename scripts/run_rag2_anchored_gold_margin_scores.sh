#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
BASE="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"

NO_RAG_FEATURE_ROOT="${NO_RAG_FEATURE_ROOT:-${BASE}/train_no_rag_anchored_features_v1}"
DOCUMENT_FEATURE_ROOT="${DOCUMENT_FEATURE_ROOT:-${BASE}/document_traces_source_balanced32_rerank8_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${BASE}/candidates/source_balanced32_rerank8_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE}/gold_margin_utility_source_balanced32_rerank8_v1}"
TEMPERATURE="${TEMPERATURE:-1.0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Gold-margin pipeline: cached logits -> pair margins -> aggregate audit"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] No LLM/GPU forward pass is required; existing exact A/B/C/D logits are reused."

"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/build_rag2_anchored_gold_margin_scores.py" \
  --no-rag-feature-root "${NO_RAG_FEATURE_ROOT}" \
  --document-feature-root "${DOCUMENT_FEATURE_ROOT}" \
  --candidate-root "${CANDIDATE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --datasets medmcqa medqa \
  --source-split train \
  --temperature "${TEMPERATURE}" \
  --resume \
  --log-level INFO

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Gold-margin pipeline complete: ${OUTPUT_ROOT}"
