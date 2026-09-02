#!/usr/bin/env bash
# Cache frozen Llama direct-choice logits for train No-RAG and every reranked Top-8 document.
# The Python workflow prints its own per-stage, per-dataset progress, rate, and ETA.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
SCRIPT="$PROJECT/scripts/cache_rag2_direct_choice_train_outcomes.py"
MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
CANDIDATES="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/candidates/source_balanced32_rerank8_v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/direct_choice_single_document_outcomes_source_balanced32_rerank8_v1}"

# 128 is a conservative high-throughput starting point on an H200.  Exact
# outputs do not change with this value; an OOM is automatically retried at a
# smaller batch size.  Override PROMPT_BATCH_SIZE only for operational tuning.
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-128}"
QUESTIONS_PER_SHARD="${QUESTIONS_PER_SHARD:-1024}"

test -x "$PYTHON"
test -f "$SCRIPT"
test -f "$MODEL/config.json"
test -f "$CANDIDATES/medmcqa/train/candidates_top8.jsonl"
test -f "$CANDIDATES/medqa/train/candidates_top8.jsonl"

echo "[overall 0/2 | elapsed 00h00m00s | ETA unknown until the first scoring shard]"
echo "direct-choice train cache: datasets=medmcqa,medqa; questions=192995; single-document pairs=1543960; prompts=1736955"
echo "GPU logical device=${CUDA_VISIBLE_DEVICES}; initial_prompt_batch_size=${PROMPT_BATCH_SIZE}; questions_per_shard=${QUESTIONS_PER_SHARD}"

exec "$PYTHON" "$SCRIPT" \
  --datasets medmcqa medqa \
  --split train \
  --candidate-root "$CANDIDATES" \
  --candidate-file candidates_top8.jsonl \
  --docs-per-question 8 \
  --expected-per-source-top-k 8 \
  --expected-candidate-pool-top-k 32 \
  --model-name-or-path "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --questions-per-shard "$QUESTIONS_PER_SHARD" \
  --prompt-batch-size "$PROMPT_BATCH_SIZE" \
  --max-input-tokens 2048 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --resume \
  --log-level INFO
