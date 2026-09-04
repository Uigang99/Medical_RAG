#!/usr/bin/env bash
# Direct-Choice document attribution validity audit. This script does not train.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
PRIOR="$BASE/document_attribution_faithfulness_mvp_v1/medqa_train_rationale_answer_gradxinput_mixed256_non64_v1"
OUTPUT="${OUTPUT_DIR:-$BASE/direct_choice_document_attribution_validity_v1/medqa_train_same320_question_first_v1}"

MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
CONTEXT_BATCH_SIZE="${CONTEXT_BATCH_SIZE:-16}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"

test -x "$PYTHON"
test -f "$MODEL/config.json"
test -f "$PRIOR/cohort.jsonl"
test -f "$PRIOR/report.json"

echo "[command purpose] validate whether document attribution preserves within-question relative influence"
echo "[controlled comparison] reuse the exact prior 320-question Top-8 cohort; replace rationale+answer target with one Direct-Choice token only"
echo "[experiment 1] Direct-Choice Gradient×Input ranking vs exact Top-8 leave-one-document-out effect"
echo "[experiment 2] exact removal ranking vs No-RAG singleton-addition ranking and Top/Bottom coalition deletion"
echo "[workflow] stage 1/3 contract preflight -> stage 2/3 GPU attribution/interventions -> stage 3/3 bootstrap/report"
echo "[runtime estimate] H200 GPU 1, 320 questions, BF16, batch=${CONTEXT_BATCH_SIZE}: approximately 8-20 minutes; stage 2 recalibrates ETA from measured throughput"
echo "[scope] question-first anchored Direct-Choice prompt; no rationale; gold and Semantic labels are diagnostics only"
echo "[output] $OUTPUT"

exec "$PYTHON" "$PROJECT/scripts/evaluate_rag2_direct_choice_document_attribution.py" \
  --cohort-file "$PRIOR/cohort.jsonl" \
  --prior-report "$PRIOR/report.json" \
  --model "$MODEL" \
  --output-dir "$OUTPUT" \
  --max-questions "$MAX_QUESTIONS" \
  --context-batch-size "$CONTEXT_BATCH_SIZE" \
  --max-input-tokens 4096 \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --gradient-checkpointing \
  --resume
