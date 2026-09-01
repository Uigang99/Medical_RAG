#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/Uiheon/Medical_RAG
PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${DATASET:-medqa}
GPU_DEVICE=${GPU_DEVICE:-cuda:0}
FEATURE_DIR=${FEATURE_DIR:-${ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/semantic_attention_controller_v1/medqa_attribution_scaling_top8_v1/prepared_features}
INDEX_PATH=${INDEX_PATH:-${ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/semantic_attention_controller_v1/shared_indices/medqa.sqlite}
RUN_ROOT=${RUN_ROOT:-${ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/gold_free_semantic_gate_pilot_v1}
MODEL_ROOT=${MODEL_ROOT:-/home/user/Uiheon/models/RAG2-Gold-Free-Semantic-Gate/medqa}
TEACHER_DIR=${RUN_ROOT}/teachers
OVERFIT_DIR=${MODEL_ROOT}/medqa_top8_gold_free_overfit16_v2
PILOT_DIR=${MODEL_ROOT}/medqa_top8_gold_free_pilot256_v2
EVAL_DIR=${RUN_ROOT}/free_generation_eval128_v2
TRAIN_QUESTIONS=${TRAIN_QUESTIONS:-256}
VAL_QUESTIONS=${VAL_QUESTIONS:-128}
TEST_QUESTIONS=${TEST_QUESTIONS:-128}

started=$(date +%s)
stage_status() {
  local stage=$1
  local label=$2
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - started))
  printf '[overall %s/6 | elapsed %02dh%02dm%02ds | overall ETA unknown until compatible stage rates are measured] %s\n' \
    "${stage}" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) "${label}"
}

for path in "${PYTHON}" "${FEATURE_DIR}" "${INDEX_PATH}" /home/user/Uiheon/models/Llama-3-8B-Instruct; do
  if [[ ! -e "${path}" ]]; then
    echo "Preflight failed: missing ${path}" >&2
    exit 2
  fi
done

stage_status 1 'preflight immutable contracts and selected counts'
"${PYTHON}" "${ROOT}/scripts/cache_rag2_gold_free_semantic_teachers.py" \
  --dataset "${DATASET}" \
  --feature-dir "${FEATURE_DIR}" \
  --index-path "${INDEX_PATH}" \
  --output-dir "${TEACHER_DIR}" \
  --train-questions "${TRAIN_QUESTIONS}" \
  --val-questions "${VAL_QUESTIONS}" \
  --test-questions "${TEST_QUESTIONS}" \
  --plan-only

stage_status 2 'cache frozen-Llama valid-only responses; No-RAG responses are reused'
"${PYTHON}" "${ROOT}/scripts/cache_rag2_gold_free_semantic_teachers.py" \
  --dataset "${DATASET}" \
  --feature-dir "${FEATURE_DIR}" \
  --index-path "${INDEX_PATH}" \
  --output-dir "${TEACHER_DIR}" \
  --train-questions "${TRAIN_QUESTIONS}" \
  --val-questions "${VAL_QUESTIONS}" \
  --test-questions "${TEST_QUESTIONS}" \
  --generation-batch-size 64 \
  --gpu-memory-utilization 0.92 \
  --resume

stage_status 3 'tiny-set overfit test: 16 questions, 5 epochs'
"${PYTHON}" "${ROOT}/scripts/train_rag2_gold_free_semantic_gate.py" \
  --dataset "${DATASET}" \
  --feature-dir "${FEATURE_DIR}" \
  --index-path "${INDEX_PATH}" \
  --teacher-dir "${TEACHER_DIR}" \
  --output-dir "${OVERFIT_DIR}" \
  --mode overfit \
  --overfit-questions 16 \
  --epochs 5 \
  --patience 0 \
  --gradient-accumulation 4 \
  --device "${GPU_DEVICE}" \
  --resume

stage_status 4 'enforce overfit stop condition before held-out pilot'
"${PYTHON}" -c '
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
summary = json.loads(path.read_text())
print("Overfit result: passed={} criterion={}".format(
    summary["passed"], summary["success_criterion"]
))
if not summary["passed"]:
    raise SystemExit("STOP: target/actuator did not pass tiny-set learnability; pilot was not launched")
' "${OVERFIT_DIR}/summary.json"

stage_status 5 'held-out pilot: train=256 validation=128 test=128'
"${PYTHON}" "${ROOT}/scripts/train_rag2_gold_free_semantic_gate.py" \
  --dataset "${DATASET}" \
  --feature-dir "${FEATURE_DIR}" \
  --index-path "${INDEX_PATH}" \
  --teacher-dir "${TEACHER_DIR}" \
  --output-dir "${PILOT_DIR}" \
  --mode pilot \
  --train-questions "${TRAIN_QUESTIONS}" \
  --val-questions "${VAL_QUESTIONS}" \
  --test-questions "${TEST_QUESTIONS}" \
  --epochs 3 \
  --patience 1 \
  --gradient-accumulation 8 \
  --device "${GPU_DEVICE}" \
  --resume

stage_status 6 'evaluation-only free rationale+answer generation: zero gate versus learned gate'
"${PYTHON}" "${ROOT}/scripts/evaluate_rag2_gold_free_semantic_gate_generation.py" \
  --dataset "${DATASET}" \
  --feature-dir "${FEATURE_DIR}" \
  --index-path "${INDEX_PATH}" \
  --teacher-dir "${TEACHER_DIR}" \
  --controller "${PILOT_DIR}/final_controller.pt" \
  --output-dir "${EVAL_DIR}" \
  --test-questions "${TEST_QUESTIONS}" \
  --device "${GPU_DEVICE}" \
  --resume

echo "Completed. Behavior summary: ${PILOT_DIR}/summary.json"
echo "Completed. Free-generation summary: ${EVAL_DIR}/summary.json"
