#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
MODEL_PATH="${MODEL_PATH:-/home/user/Uiheon/models/Llama-3-8B-Instruct}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/candidates/source_balanced32_rerank8_v1}"
NO_RAG_FEATURE_ROOT="${NO_RAG_FEATURE_ROOT:-${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/train_no_rag_anchored_features_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/document_traces_source_balanced32_rerank8_v1}"

# GPU 1 is the project default. Inside this isolated process it is cuda:0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline stage 1/2: independent document rationale+answer generation"
"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/generate_rag2_anchored_document_traces.py" \
  --datasets medmcqa medqa \
  --split train \
  --candidate-root "${CANDIDATE_ROOT}" \
  --candidate-file candidates_top8.jsonl \
  --docs-per-question 8 \
  --model-name-or-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --questions-per-shard 128 \
  --generation-batch-size 64 \
  --max-new-tokens 512 \
  --retry-max-new-tokens 768 \
  --temperature 0.0 \
  --top-p 1.0 \
  --gpu-memory-utilization 0.92 \
  --llm-max-model-len 8192 \
  --vllm-performance-mode throughput \
  --vllm-max-num-seqs 80 \
  --vllm-max-num-batched-tokens 65536 \
  --max-doc-chars 0 \
  --resume \
  --log-level INFO

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline stage 2/2: exact replay and anchor feature extraction"
"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/extract_rag2_anchored_document_features.py" \
  --trace-root "${OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --no-rag-feature-root "${NO_RAG_FEATURE_ROOT}" \
  --model-name-or-path "${MODEL_PATH}" \
  --datasets medmcqa medqa \
  --split train \
  --layers 4 12 20 28 31 \
  --batch-size 32 \
  --max-input-tokens 8192 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --minimum-free-space-gib 20 \
  --resume \
  --log-level INFO

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline 2/2 complete: ${OUTPUT_ROOT}"
