#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
VENV="${KG_VENV:-/home/user/Uiheon/.venv_rag2_kg_mvp_py312}"
MODEL_ROOT="${KG_MODEL_ROOT:-/home/user/Uiheon/models/RAG2-Medical-KG-MVP}"
export KG_MODEL_ROOT="${MODEL_ROOT}"
export SCISPACY_CACHE="${MODEL_ROOT}/scispacy_cache"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
LIMIT_PER_SOURCE="${LIMIT_PER_SOURCE:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/kg/rag2_candidate_medical_kg_mvp_v2}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${ROOT}/databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  PYTHON="${PREFLIGHT_PYTHON:-/home/user/Uiheon/.venv_vllm/bin/python}"
  EXTRA=(--preflight-only)
else
  PYTHON="${VENV}/bin/python"
  EXTRA=()
  if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: KG environment is missing: ${VENV}"
    echo "Run: bash ${ROOT}/scripts/setup_rag2_medical_kg_mvp.sh"
    exit 2
  fi
fi

if [[ "${LIMIT_PER_SOURCE}" == "0" ]]; then
  SCOPE="full 165,642-document union (exact count revalidated at preflight)"
  ESTIMATE="calibrated after the first 32 uncached chunks; expect a long CPU job and use only after the MVP passes"
else
  SCOPE="source-stratified MVP: ${LIMIT_PER_SOURCE} chunks/source"
  ESTIMATE="approximately 5-20 minutes after one-time setup; the live ETA replaces this range after 32 chunks"
fi

echo "[command plan] ${SCOPE}; Top-k union=1,2,4,8,16,32; document text only; local scispaCy+UMLS+relation rules"
echo "[command estimate] ${ESTIMATE}; GPU is not used by en_core_sci_sm; model_root=${MODEL_ROOT}"

cd "${ROOT}"
exec "${PYTHON}" scripts/build_rag2_medical_kg_mvp.py \
  --candidate-root "${CANDIDATE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --top-k 1 2 4 8 16 32 \
  --limit-per-source "${LIMIT_PER_SOURCE}" \
  --seed 42 \
  --linker-threshold "${LINKER_THRESHOLD:-0.85}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --shard-size "${SHARD_SIZE:-50}" \
  "${EXTRA[@]}"
