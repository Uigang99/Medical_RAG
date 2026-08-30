#!/usr/bin/env bash
set -euo pipefail

# Physical GPU selection is controlled by the caller, e.g.
# CUDA_VISIBLE_DEVICES=1 bash .../run_rag2_semantic_attention_mvp_medqa.sh

PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"

MVP_SAMPLES="${MVP_SAMPLES:-256}"
TOP_K="${TOP_K:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEMANTIC_LAYER_START="${SEMANTIC_LAYER_START:-16}"
MAX_SUPPRESSION_FACTOR="${MAX_SUPPRESSION_FACTOR:-4}"
SEMANTIC_TEMPERATURE="${SEMANTIC_TEMPERATURE:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/results/rag2_semantic_attention_suppression_mvp_v1}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Semantic-attention MVP: MedQA n=${MVP_SAMPLES} top-k=${TOP_K} lambdas=0,0.25,0.5,1.0"

"$PYTHON" "$PROJECT/scripts/evaluate_rag2_semantic_attention_mvp.py" \
  --dataset medqa \
  --split test \
  --candidate-cache "$PROJECT/databases/run_cache/rag2_llama3_paper_compatible_three_anchor_v1/all_mcq_paper_balanced_max32_rationale_answer_rerank128/candidates/521e23c599352822/candidates.jsonl" \
  --semantic-filter-model "/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medqa/medqa_semantic_top8_binary_support_epoch8_len1280_fullpair/20260830_170945/final_model" \
  --llm-model "/home/user/Uiheon/models/Llama-3-8B-Instruct" \
  --output-dir "$OUTPUT_DIR" \
  --top-k "$TOP_K" \
  --lambdas 0 0.25 0.5 1.0 \
  --max-samples "$MVP_SAMPLES" \
  --sample-seed 42 \
  --filter-question-batch-size 32 \
  --filter-batch-size 64 \
  --filter-max-input-length 1280 \
  --semantic-temperature "$SEMANTIC_TEMPERATURE" \
  --max-suppression-factor "$MAX_SUPPRESSION_FACTOR" \
  --semantic-layer-start "$SEMANTIC_LAYER_START" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-model-length 8192 \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume \
  --log-level INFO
