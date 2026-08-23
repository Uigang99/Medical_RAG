#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
PROJECT_ROOT="/home/user/Uiheon/Medical_RAG"
NO_RAG_ROOT="${NO_RAG_ROOT:-${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/train_no_rag_anchored_features_v1}"
QUERY_CACHE_ROOT="${QUERY_CACHE_ROOT:-${PROJECT_ROOT}/databases/query_embeddings/medcpt_query_encoder/rag2_paper_compatible_three_anchor_train_v1}"
CANDIDATE_OUTPUT_ROOT="${CANDIDATE_OUTPUT_ROOT:-${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/candidates/source_balanced32_rerank8_v1}"

# Select physical GPU 1 by default.  Once isolated by CUDA_VISIBLE_DEVICES,
# PyTorch and FAISS correctly address that device as logical cuda:0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

exec "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/prepare_rag2_rationale_candidates.py" \
  --datasets medmcqa medqa \
  --split train \
  --collection unified \
  --stages embed retrieve \
  --no-rag-root "${NO_RAG_ROOT}" \
  --query-cache-root "${QUERY_CACHE_ROOT}" \
  --candidate-output-root "${CANDIDATE_OUTPUT_ROOT}" \
  --vector-db-root "${PROJECT_ROOT}/databases/vector_db/RAG_Square" \
  --query-encoder-path /home/user/Uiheon/models/MedCPT-Query-Encoder \
  --cross-encoder-path /home/user/Uiheon/models/MedCPT-Cross-Encoder \
  --embedding-batch-size 1024 \
  --embedding-max-length 512 \
  --embedding-attn-implementation eager \
  --quality-policy technical \
  --sources pubmed pmc cpg textbooks \
  --per-source-top-k 8 \
  --rerank-top-k 8 \
  --no-keep-faiss-indexes-in-memory \
  --retrieval-batch-size 512 \
  --rerank-batch-size 1024 \
  --cross-encoder-max-length 512 \
  --cross-encoder-attn-implementation eager \
  --faiss-gpu-device 0 \
  --faiss-gpu-add-batch-size 100000 \
  --faiss-gpu-temp-memory-mb 512 \
  --metadata-row-cache-size 50000 \
  --resume \
  --log-level INFO \
  "$@"
