#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON="${BIOREX_PYTHON:-/home/user/Uiheon/.venv_biorex_py312/bin/python}"
MODEL="${BIOREX_MODEL:-/home/user/Uiheon/models/BioREx-BioLinkBERT}"
BIOREX_REPO="${BIOREX_REPO:-/home/user/Uiheon/tools/BioREx}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/datasets/kg/rag2_candidate_medical_kg_mvp_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_candidate_medical_kg_biorex_pilot_v1}"
DOCUMENTS_PER_SOURCE="${DOCUMENTS_PER_SOURCE:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"

if [[ ! -x "${PYTHON}" || ! -s "${MODEL}/config.json" ]]; then
  echo "ERROR: BioREx environment/checkpoint is incomplete." >&2
  echo "Run first: bash ${ROOT}/scripts/setup_rag2_biorex.sh" >&2
  exit 2
fi

echo "[command plan] frozen scispaCy/UMLS entities -> official BioREx pair construction -> BioREx GPU classification -> native-label audit"
echo "[command estimate] after setup, 400 chunks (100/source), H200 GPU 1, batch=${BATCH_SIZE}: first run approximately 3-15 minutes; live ETA calibrates during inference"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=2
export TOKENIZERS_PARALLELISM=true

cd "${ROOT}"
exec "${PYTHON}" scripts/evaluate_rag2_medical_kg_biorex_pilot.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --biorex-repo "${BIOREX_REPO}" \
  --model "${MODEL}" \
  --documents-per-source "${DOCUMENTS_PER_SOURCE}" \
  --batch-size "${BATCH_SIZE}"

