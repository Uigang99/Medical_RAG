#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
SCRIPT="$PROJECT/scripts/materialize_rag2_document_first_bounded_pilot.py"

DATASET="${1:-medmcqa}"
if [[ "$DATASET" != "medmcqa" && "$DATASET" != "medqa" ]]; then
  echo "Usage: $0 {medmcqa|medqa}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-20000}"
VAL_QUESTIONS="${VAL_QUESTIONS:-4000}"
TEST_QUESTIONS="${TEST_QUESTIONS:-4000}"
SAFETY_QUESTIONS="${SAFETY_QUESTIONS:-4000}"

# MedQA has only 3,872/516/481 same-question Direct-Support/non-support
# semantic pairs.  Its bounded run therefore uses every eligible question and
# a disjoint 481-question safety cohort by default.
if [[ "$DATASET" == "medqa" ]]; then
  TRAIN_QUESTIONS="${TRAIN_QUESTIONS_MEDQA:-3872}"
  VAL_QUESTIONS="${VAL_QUESTIONS_MEDQA:-516}"
  TEST_QUESTIONS="${TEST_QUESTIONS_MEDQA:-481}"
  SAFETY_QUESTIONS="${SAFETY_QUESTIONS_MEDQA:-481}"
fi

PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-64}"
QUESTIONS_PER_SHARD="${QUESTIONS_PER_SHARD:-256}"
EXPECTED_PROMPTS_PER_SECOND="${EXPECTED_PROMPTS_PER_SECOND:-70}"

test -x "$PYTHON_BIN"
test -f "$SCRIPT"
test -f "/home/user/Uiheon/models/Llama-3-8B-Instruct/config.json"

TOTAL_QUESTIONS=$((TRAIN_QUESTIONS + VAL_QUESTIONS + TEST_QUESTIONS + SAFETY_QUESTIONS))
TOTAL_PROMPTS=$((TOTAL_QUESTIONS * 9))
ESTIMATED_SCORING_SECONDS=$((TOTAL_PROMPTS / ${EXPECTED_PROMPTS_PER_SECOND%.*}))
ESTIMATED_TOTAL_SECONDS=$((ESTIMATED_SCORING_SECONDS + 630))
printf '[command plan] dataset=%s questions=%d prompts=%d estimated_wall_time=%02dh%02dm (H200, eager, batch=%s, empty cache)\n' \
  "$DATASET" "$TOTAL_QUESTIONS" "$TOTAL_PROMPTS" \
  "$((ESTIMATED_TOTAL_SECONDS / 3600))" "$(((ESTIMATED_TOTAL_SECONDS % 3600) / 60))" \
  "$PROMPT_BATCH_SIZE"
printf '[command plan] six stages: cohort -> semantic join -> candidate join -> prompt audit -> frozen scoring -> training data\n'

exec "$PYTHON_BIN" "$SCRIPT" \
  --dataset "$DATASET" \
  --train-questions "$TRAIN_QUESTIONS" \
  --val-questions "$VAL_QUESTIONS" \
  --test-questions "$TEST_QUESTIONS" \
  --safety-questions "$SAFETY_QUESTIONS" \
  --questions-per-shard "$QUESTIONS_PER_SHARD" \
  --prompt-batch-size "$PROMPT_BATCH_SIZE" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-2048}" \
  --expected-prompts-per-second "$EXPECTED_PROMPTS_PER_SECOND" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device cuda:0 \
  --resume
