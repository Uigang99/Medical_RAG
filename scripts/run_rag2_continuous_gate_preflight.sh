#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON=${PYTHON:-/home/user/Uiheon/.venv_vllm/bin/python}
BASE=${BASE:-${PROJECT}/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE}/continuous_gate_preflight_top4_internal_val_v1}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT}/results/rag2_continuous_gate_preflight_v1}
LLM_MODEL=${LLM_MODEL:-/home/user/Uiheon/models/Llama-3-8B-Instruct}
MAX_QUESTIONS_PER_DATASET=${MAX_QUESTIONS_PER_DATASET:-500}
GATE_SAMPLES=${GATE_SAMPLES:-128}
TOP_K=${TOP_K:-4}

SEMANTIC_MEDMCQA=${SEMANTIC_MEDMCQA:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medmcqa/medmcqa_semantic_top8_binary_support_epoch5_len1280_fullpair/20260829_212146/final_model}
SEMANTIC_MEDQA=${SEMANTIC_MEDQA:-/home/user/Uiheon/models/RAG2-Filter-FlanT5-large-Semantic-Top8-BinarySupport/medqa/medqa_semantic_top8_binary_support_epoch8_len1280_fullpair/20260830_170945/final_model}
GATE_FEATURES=${GATE_FEATURES:-${BASE}/semantic_attention_controller_v1/medqa_pilot_top8_rationale_wide_v1/prepared_features}

START_SECONDS=${SECONDS}
format_duration() {
  local total=${1}
  printf '%02dh%02dm%02ds' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}
stage() {
  local index=${1}
  local name=${2}
  local elapsed=$((SECONDS - START_SECONDS))
  printf '[overall %s/5 | elapsed %s | overall ETA unknown] stage=%s\n' \
    "${index}" "$(format_duration "${elapsed}")" "${name}"
}
complete() {
  local index=${1}
  local name=${2}
  local stage_start=${3}
  local elapsed=$((SECONDS - stage_start))
  printf '[overall %s/5 complete | stage elapsed %s | total elapsed %s] stage=%s\n' \
    "${index}" "$(format_duration "${elapsed}")" \
    "$(format_duration "$((SECONDS - START_SECONDS))")" "${name}"
}

mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}"
printf 'Continuous-gate preflight plan: internal validation only; datasets=medmcqa,medqa top_k=%s questions/dataset<=%s gate_samples=%s\n' \
  "${TOP_K}" "${MAX_QUESTIONS_PER_DATASET}" "${GATE_SAMPLES}"
printf 'Stages: 1 cohort join; 2 direct-choice exact subsets; 3 semantic/set/answer-mode analysis; 4 rationale-wide gate contract; 5 report\n'
printf 'Resume policy: all expensive stages retain atomic shards/caches; rerunning this command resumes safely.\n'

STAGE_START=${SECONDS}
stage 1 'join question-level validation cohort'
"${PYTHON}" "${PROJECT}/scripts/prepare_rag2_continuous_gate_preflight.py" \
  --datasets medmcqa medqa \
  --source-split train \
  --semantic-split val \
  --candidate-root "${BASE}/candidates/source_balanced32_rerank8_v1" \
  --semantic-root "${BASE}/filter_training_inputs_semantic_top8_four_class_v1" \
  --output-root "${OUTPUT_ROOT}" \
  --top-k "${TOP_K}" \
  --max-questions-per-dataset "${MAX_QUESTIONS_PER_DATASET}" \
  --sample-seed 42 \
  --resume
complete 1 'join question-level validation cohort' "${STAGE_START}"

STAGE_START=${SECONDS}
stage 2 'score every direct-choice Top-k subset with frozen Llama'
"${PYTHON}" "${PROJECT}/scripts/materialize_rag2_semantic_behavioral_subset_oracle.py" \
  --candidate-root "${OUTPUT_ROOT}/candidate_union" \
  --semantic-labels-path "${OUTPUT_ROOT}/semantic_labels/all.jsonl" \
  --model-name-or-path "${LLM_MODEL}" \
  --output-root "${OUTPUT_ROOT}/exact_subset_scores" \
  --datasets medmcqa medqa \
  --split val \
  --top-k "${TOP_K}" \
  --candidate-semantic-labels direct_support supporting_evidence no_evidence misleading_evidence indeterminate_or_mixed \
  --questions-per-shard 64 \
  --max-batch-size 64 \
  --max-batch-tokens 65536 \
  --max-input-tokens 8192 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --resume
complete 2 'score every direct-choice Top-k subset with frozen Llama' "${STAGE_START}"

STAGE_START=${SECONDS}
stage 3 'analyze calibration, set headroom, conditional effects, and answer modes'
"${PYTHON}" "${PROJECT}/scripts/analyze_rag2_continuous_gate_preflight.py" \
  --datasets medmcqa medqa \
  --split val \
  --source-split train \
  --cohort-root "${OUTPUT_ROOT}" \
  --subset-root "${OUTPUT_ROOT}/exact_subset_scores" \
  --output-root "${RESULT_ROOT}/analysis" \
  --semantic-model-medmcqa "${SEMANTIC_MEDMCQA}" \
  --semantic-model-medqa "${SEMANTIC_MEDQA}" \
  --no-rag-trace-root "${BASE}/train_no_rag_anchored_features_v1/trace_shards" \
  --document-trace-root "${BASE}/document_traces_source_balanced32_rerank8_v1/trace_shards" \
  --top-k "${TOP_K}" \
  --semantic-batch-size 64 \
  --semantic-max-input-length 1280 \
  --risk-kappas 0 0.25 0.5 1 2 \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume
complete 3 'analyze calibration, set headroom, conditional effects, and answer modes' "${STAGE_START}"

STAGE_START=${SECONDS}
stage 4 'test rationale-wide gate against physical document removal'
"${PYTHON}" "${PROJECT}/scripts/evaluate_rag2_attention_gate_contract.py" \
  --feature-dir "${GATE_FEATURES}" \
  --dataset medqa \
  --split test \
  --llm-model "${LLM_MODEL}" \
  --output-dir "${RESULT_ROOT}/gate_contract_medqa_rationale_wide" \
  --max-samples "${GATE_SAMPLES}" \
  --sample-seed 42 \
  --gate-factors 0 0.25 0.5 1.5 2 \
  --gate-batch-size 8 \
  --zero-log-bias -20 \
  --attention-scope rationale_wide \
  --semantic-layer-start 16 \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume
complete 4 'test rationale-wide gate against physical document removal' "${STAGE_START}"

STAGE_START=${SECONDS}
stage 5 'write Go/Revise/Stop report'
"${PYTHON}" "${PROJECT}/scripts/summarize_rag2_continuous_gate_preflight.py" \
  --analysis-summary "${RESULT_ROOT}/analysis/analysis_summary.json" \
  --gate-summary "${RESULT_ROOT}/gate_contract_medqa_rationale_wide/gate_contract_summary.json" \
  --output-dir "${RESULT_ROOT}/final"
complete 5 'write Go/Revise/Stop report' "${STAGE_START}"

printf 'Preflight complete | total elapsed %s | report=%s\n' \
  "$(format_duration "$((SECONDS - START_SECONDS))")" "${RESULT_ROOT}/final/preflight_report.md"
