#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "medmcqa" && "$1" != "medqa" ]]; then
  echo "Usage: $0 {medmcqa|medqa}" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
ARTIFACT_ROOT="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
CANDIDATES_PATH="${ARTIFACT_ROOT}/candidates/source_balanced32_rerank8_v1/${DATASET}/train/candidates_top8.jsonl"
LABEL_ROOT=/home/user/codex_rag2_outputs/codex_evidence_utility_labels_three_anchor_top8_terra_medium_v1_incremental/terra_medium
REFERENCE_SPLIT_ROOT="${ARTIFACT_ROOT}/filter_training_inputs_rag2_paper_reproduction_three_class_v1"
OUTPUT_ROOT="${ARTIFACT_ROOT}/filter_training_inputs_semantic_top8_four_class_v1"

printf '[%s] Semantic four-class preparation: dataset=%s stage=1/1; live bar reports overall and current-stage ETA.\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "${DATASET}"

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/build_rag2_codex_semantic_filter_inputs.py" \
  --dataset "${DATASET}" \
  --candidates-path "${CANDIDATES_PATH}" \
  --codex-labels-path "${LABEL_ROOT}/${DATASET}/codex_semantic_labels.jsonl" \
  --reference-split-root "${REFERENCE_SPLIT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --training-label-mode semantic_four \
  --top-k 8 \
  --max-doc-chars 0 \
  --sqlite-work-dir /tmp \
  --log-level INFO
