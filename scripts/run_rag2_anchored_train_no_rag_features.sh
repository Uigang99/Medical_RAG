#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
MODEL_PATH="${MODEL_PATH:-/home/user/Uiheon/models/Llama-3-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/user/Uiheon/Medical_RAG/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/train_no_rag_anchored_features_v1}"

# The physical GPU is selected by the caller.  GPU 1 is the project default;
# inside the isolated process it is exposed as cuda:0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/generate_rag2_anchored_no_rag_train.py" \
  --datasets medmcqa medqa \
  --split train \
  --benchmark-root "${PROJECT_ROOT}/datasets/benchmark/mcq/unified" \
  --model-name-or-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --questions-per-shard 256 \
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
  --resume \
  --log-level INFO

"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/extract_rag2_anchored_no_rag_features.py" \
  --trace-root "${OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-name-or-path "${MODEL_PATH}" \
  --datasets medmcqa medqa \
  --split train \
  --layers 4 12 20 28 31 \
  --batch-size 32 \
  --max-input-tokens 8192 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --resume \
  --log-level INFO
