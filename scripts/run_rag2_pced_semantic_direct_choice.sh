#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="true"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
SCRIPT="$PROJECT/scripts/evaluate_rag2_pced_direct_choice.py"

TOP_K="${TOP_K:-8}"
GAMMA="${GAMMA:-2.5}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-16}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-64}"
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-128}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$PROJECT/results/rag2_pced_direct_choice_v2/all_mcq_source_balanced32_rerank8_rerank_prior"
  if [[ "$MAX_QUESTIONS" != "0" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_pilot${MAX_QUESTIONS}"
  fi
fi

echo "[workflow plan] stage 1/4 contracts -> stage 2/4 semantic probabilities -> stage 3/4 Llama expert logits -> stage 4/4 paired report"
echo "[comparison] No-RAG | concatenated Top-${TOP_K} Base-RAG | PCED rerank-score prior | PCED semantic-support prior"
echo "[scope] constrained Direct-Choice A/B/C/D; one decoding step; no multi-token expert-switch claim"
echo "[settings] GPU=${CUDA_VISIBLE_DEVICES} top_k=${TOP_K} gamma=${GAMMA} question_batch=${QUESTION_BATCH_SIZE} prompt_batch=${PROMPT_BATCH_SIZE} semantic_batch=${SEMANTIC_BATCH_SIZE} max_questions=${MAX_QUESTIONS}"
if [[ "$MAX_QUESTIONS" == "0" ]]; then
  echo "[command estimate] H200 GPU 1, 6,545 questions, cold semantic/Llama caches: about 25-45 minutes; rerun with complete caches: under 2 minutes"
else
  echo "[command estimate] bounded run; live ETA calibrates separately within each stage"
fi
echo "[resume] identical reruns reuse completed semantic pairs and 128-question Llama score shards"

exec "$PYTHON" "$SCRIPT" \
  --top-k "$TOP_K" \
  --gamma "$GAMMA" \
  --question-batch-size "$QUESTION_BATCH_SIZE" \
  --prompt-batch-size "$PROMPT_BATCH_SIZE" \
  --semantic-batch-size "$SEMANTIC_BATCH_SIZE" \
  --max-questions "$MAX_QUESTIONS" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
