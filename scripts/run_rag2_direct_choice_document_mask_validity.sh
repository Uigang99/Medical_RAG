#!/usr/bin/env bash
# No-training validity test for the fixed-position all-layer document mask.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
MODEL="/home/user/Uiheon/models/Llama-3-8B-Instruct"
BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
COHORT="${COHORT_FILE:-$BASE/document_attribution_faithfulness_mvp_v1/medqa_train_rationale_answer_gradxinput_mixed256_non64_v1/cohort.jsonl}"
OUTPUT="${OUTPUT_DIR:-$BASE/direct_choice_document_mask_validity_v1/medqa_train_mixed256_question_first_v1}"

MAX_QUESTIONS="${MAX_QUESTIONS:-256}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
SPEARMAN_THRESHOLD="${SPEARMAN_THRESHOLD:-0.80}"
TOP1_THRESHOLD="${TOP1_THRESHOLD:-0.70}"

test -x "$PYTHON"
test -f "$MODEL/config.json"
test -f "$COHORT"

echo "[command purpose] 학습 전에 hard document mask가 실제 문서 삭제의 영향력 순위를 재현하는지 검증"
echo "[primary reference] Top-8 전체 대비 문서 하나의 mapped token을 물리적으로 삭제한 full-vocabulary JSD"
echo "[proxy] 동일 token을 모든 layer에서 hard mask하고 나머지 position ID를 압축한 full-vocabulary JSD"
echo "[pass criteria] 256문항 평균 within-question Spearman >= ${SPEARMAN_THRESHOLD}, Top-1 document overlap >= ${TOP1_THRESHOLD}"
echo "[workflow] stage 1/3 preflight -> stage 2/3 GPU physical-delete/mask 비교 -> stage 3/3 bootstrap/report"
echo "[runtime estimate] H200 GPU 1, BF16, 256 mixed Top-8 questions: 약 12-30분; stage 2의 실측 속도로 ETA 재계산"
echo "[resume] 질문별 결과를 atomic 저장; 같은 명령 재실행 시 완료 문항을 건너뜀"
echo "[output] $OUTPUT"

exec "$PYTHON" "$PROJECT/scripts/evaluate_rag2_direct_choice_document_mask_validity.py" \
  --cohort-file "$COHORT" \
  --model "$MODEL" \
  --output-dir "$OUTPUT" \
  --max-questions "$MAX_QUESTIONS" \
  --max-input-tokens 4096 \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --spearman-threshold "$SPEARMAN_THRESHOLD" \
  --top1-threshold "$TOP1_THRESHOLD" \
  --seed 42 \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume
