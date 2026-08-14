#!/usr/bin/env bash

# Generate the no-RAG rationale+answer traces used to construct RAG2
# pseudo-labels.  This deliberately uses the paper-exact free-response prompt:
# no final-answer template, compact rewrite, or answer-anchored retry is added.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

PROJECT_ROOT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/generate_rag2_no_rag_rationales.py" \
  --datasets medmcqa medqa \
  --split train \
  --collection unified \
  --benchmark-root "${PROJECT_ROOT}/datasets/benchmark" \
  --llm-model-path /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --artifact-root "${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2/no_rag_rationales_train" \
  --results-root "${PROJECT_ROOT}/results/rag2_llama3_paper_exact_free_response_v2/no_rag_generation_train" \
  --run-name medmcqa_medqa_train_paper_exact_free_response \
  --prompt-profile paper_exact \
  --generation-batch-size 32 \
  --max-new-tokens 768 \
  --length-retry-attempts 0 \
  --invalid-retry-attempts 0 \
  --no-retry-invalid \
  --no-retry-quality \
  --no-choice-anchored-retry \
  --temperature 0.0 \
  --top-p 1.0 \
  --gpu-memory-utilization 0.92 \
  --llm-max-model-len 8192 \
  --gdn-prefill-backend triton \
  --vllm-performance-mode throughput \
  --vllm-max-num-seqs 80 \
  --vllm-max-num-batched-tokens 65536 \
  --resume
