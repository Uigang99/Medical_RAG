#!/usr/bin/env bash

# Build RAG2 pseudo-labeling candidates for the strict quality-selected train
# questions only.  The raw no-RAG rationale+answer response is embedded
# unchanged; each question retrieves 10 documents from each of PubMed, PMC,
# CPG, and Textbooks (40 total), then MedCPT cross-encoder reranks them to 32.
# FAISS is deliberately configured below to hold only one physical shard and
# to use bounded staging/search batches, avoiding multi-GiB transient buffers.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

PROJECT_ROOT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_rag2_rationale_candidates.py" \
  --datasets medmcqa medqa \
  --split train \
  --collection unified \
  --stages embed retrieve \
  --no-rag-root "${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2/no_rag_rationales_train" \
  --selection-root "${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2/no_rag_quality_selection_v1" \
  --query-cache-root "${PROJECT_ROOT}/databases/query_embeddings/medcpt_query_encoder/rag2_llama3_8b_paper_exact_train_quality_selected_v1" \
  --candidate-output-root "${PROJECT_ROOT}/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2/candidates/quality_selected_source_balanced40_rerank32_v1" \
  --vector-db-root "${PROJECT_ROOT}/databases/vector_db/RAG_Square" \
  --query-encoder-path /home/user/Uiheon/models/MedCPT-Query-Encoder \
  --cross-encoder-path /home/user/Uiheon/models/MedCPT-Cross-Encoder \
  --embedding-batch-size 1024 \
  --embedding-max-length 512 \
  --embedding-attn-implementation eager \
  --quality-policy conservative \
  --sources pubmed pmc cpg textbooks \
  --per-source-top-k 10 \
  --rerank-top-k 32 \
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
  "$@"
