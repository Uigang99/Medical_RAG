#!/usr/bin/env bash
set -euo pipefail

# Build semantic Top-8 subset plans and generate only the paper-compatible
# rationale + fixed terminal answer traces.  This workflow intentionally does
# not run a separate direct-choice experiment.

PROJECT_ROOT="${PROJECT_ROOT:-/home/user/Uiheon/Medical_RAG}"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/user/Uiheon/models/Llama-3-8B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${DATA_ROOT}/candidates/source_balanced32_rerank8_v1}"
SEMANTIC_LABEL_ROOT="${SEMANTIC_LABEL_ROOT:-/home/user/codex_rag2_outputs/codex_evidence_utility_labels_three_anchor_top8_terra_medium_v1_incremental/terra_medium}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_ROOT}/filter_training_inputs_rag2_paper_reproduction_three_class_v1}"
RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/semantic_subset_rationale_traces_v1}"
PLAN_ROOT="${PLAN_ROOT:-${RUN_ROOT}/subset_plan}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/rationale_traces}"

DATASETS_TEXT="${DATASETS:-medmcqa medqa}"
read -r -a DATASET_ARGS <<< "${DATASETS_TEXT}"

QUESTIONS_PER_SHARD="${QUESTIONS_PER_SHARD:-128}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-64}"
MAX_QUESTIONS_PER_DATASET="${MAX_QUESTIONS_PER_DATASET:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-80}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"

WORKFLOW_START_SECONDS="$(date +%s)"
CURRENT_WORKFLOW_STAGE="initialization"
export PYTHONUNBUFFERED=1

format_duration() {
  local total_seconds="$1"
  printf '%02dh%02dm%02ds' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

workflow_status() {
  local stage="$1"
  local message="$2"
  local now elapsed
  now="$(date +%s)"
  elapsed="$((now - WORKFLOW_START_SECONDS))"
  printf '[overall %s/2 | elapsed %s] %s\n' \
    "${stage}" "$(format_duration "${elapsed}")" "${message}"
}

on_error() {
  local exit_code=$?
  printf '[failed | stage=%s | elapsed %s] exit=%d. Re-run the same command: completed plan/shards resume safely.\n' \
    "${CURRENT_WORKFLOW_STAGE}" \
    "$(format_duration "$(($(date +%s) - WORKFLOW_START_SECONDS))")" "${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

CURRENT_WORKFLOW_STAGE="1/2 semantic subset planning"
workflow_status 1 "materialize and validate semantic subset plan; expectation: plan <2m, tokenizer context preflight about 5-20m, generation roughly 10-16h on one H200 (first completed shards calibrate live ETA)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_rag2_semantic_subset_plan.py" \
  --datasets "${DATASET_ARGS[@]}" \
  --split train \
  --candidate-root "${CANDIDATE_ROOT}" \
  --candidate-file candidates_top8.jsonl \
  --semantic-label-root "${SEMANTIC_LABEL_ROOT}" \
  --semantic-label-file codex_semantic_labels.jsonl \
  --split-root "${SPLIT_ROOT}" \
  --output-root "${PLAN_ROOT}" \
  --top-k 8 \
  --max-questions-per-dataset "${MAX_QUESTIONS_PER_DATASET}" \
  --resume

CURRENT_WORKFLOW_STAGE="2/2 context preflight and rationale+answer generation"
workflow_status 2 "generate rationale + fixed terminal answer for new multi-document subsets; active Python progress reports dataset/shard/subsets/rate/ETA"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/generate_rag2_anchored_semantic_subset_traces.py" \
  --datasets "${DATASET_ARGS[@]}" \
  --split train \
  --plan-root "${PLAN_ROOT}" \
  --candidate-root "${CANDIDATE_ROOT}" \
  --candidate-file candidates_top8.jsonl \
  --model-name-or-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --questions-per-shard "${QUESTIONS_PER_SHARD}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  --resume

trap - ERR
printf '[complete | elapsed %s] plan=%s traces=%s\n' \
  "$(format_duration "$(($(date +%s) - WORKFLOW_START_SECONDS))")" \
  "${PLAN_ROOT}" "${OUTPUT_ROOT}"
