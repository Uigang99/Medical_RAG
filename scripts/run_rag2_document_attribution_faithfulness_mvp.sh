#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="/home/user/Uiheon/.venv_vllm/bin/python"
DATASET="${1:-medqa}"
ANALYSIS_SPLIT="${ANALYSIS_SPLIT:-train}"
MIXED_QUESTIONS="${MIXED_QUESTIONS:-256}"
ALL_NON_SUPPORT_QUESTIONS="${ALL_NON_SUPPORT_QUESTIONS:-64}"
REMOVAL_BATCH_SIZE="${REMOVAL_BATCH_SIZE:-8}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
SEED="${SEED:-42}"
OUTPUT_VERSION="${OUTPUT_VERSION:-rationale_answer_gradxinput_mixed256_non64_v1}"

BASE="${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
OUTPUT_DIR="${BASE}/document_attribution_faithfulness_mvp_v1/${DATASET}_${ANALYSIS_SPLIT}_${OUTPUT_VERSION}"

echo "[command plan] no-training attribution feasibility MVP"
echo "[purpose] compare document Gradient×Input with the same cached rationale+answer's physical-removal log-likelihood change"
echo "[cohort] dataset=${DATASET} split=${ANALYSIS_SPLIT} mixed=${MIXED_QUESTIONS} all_non_support=${ALL_NON_SUPPORT_QUESTIONS}; each balanced 50:50 on cached response correct/wrong"
echo "[stages] 1/4 select cached traces -> 2/4 exact candidate join + manifest -> 3/4 attribution + 8 removals/question -> 4/4 metrics/report"
echo "[runtime estimate] H200 GPU 1, 320 questions, BF16, removal batch=8: approximately 8-25 minutes; measured 2-question smoke rate was about 1.0 question/s, and stage 3 recalibrates ETA on the real token lengths"
echo "[output] ${OUTPUT_DIR}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_rag2_document_attribution_faithfulness.py" \
  --dataset "${DATASET}" \
  --analysis-split "${ANALYSIS_SPLIT}" \
  --mixed-questions "${MIXED_QUESTIONS}" \
  --all-non-support-questions "${ALL_NON_SUPPORT_QUESTIONS}" \
  --removal-batch-size "${REMOVAL_BATCH_SIZE}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --seed "${SEED}" \
  --model "/home/user/Uiheon/models/Llama-3-8B-Instruct" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --gradient-checkpointing \
  --resume
