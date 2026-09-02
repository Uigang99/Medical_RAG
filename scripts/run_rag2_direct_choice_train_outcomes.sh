#!/usr/bin/env bash
# Cache frozen Llama anchored direct-choice logits for train No-RAG and every reranked Top-8 document.
# Relative to the anchored rationale pipeline, only the rationale block is omitted.
# The Python workflow prints its own per-stage, per-dataset progress, rate, and ETA.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
SCRIPT="$PROJECT/scripts/cache_rag2_direct_choice_train_outcomes.py"
MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
CANDIDATES="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/candidates/source_balanced32_rerank8_v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/anchored_direct_choice_single_document_outcomes_source_balanced32_rerank8_v1}"

# Eager attention is deliberate here. PyTorch 2.11/cuDNN SDPA cannot build an
# execution plan for some of the variable-length left-padded train batches.
# The exact A/B/C/D score contract is unchanged. 64 keeps eager-attention peak
# memory conservative on the H200; CUDA OOM still retries at half batch size.
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-64}"
QUESTIONS_PER_SHARD="${QUESTIONS_PER_SHARD:-1024}"

test -x "$PYTHON"
test -f "$SCRIPT"
test -f "$MODEL/config.json"
test -f "$CANDIDATES/medmcqa/train/candidates_top8.jsonl"
test -f "$CANDIDATES/medqa/train/candidates_top8.jsonl"

echo "[overall 0/2 | elapsed 00h00m00s | ETA unknown until the first scoring shard]"
echo "anchored direct-choice train cache: datasets=medmcqa,medqa; questions=192995; single-document pairs=1543960; prompts=1736955"
echo "comparison contract: same question/options/Documents/chat/final-answer layout as anchored rationale; rationale block omitted"
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
  --attn-implementation eager \
  --resume \
  --log-level INFO
