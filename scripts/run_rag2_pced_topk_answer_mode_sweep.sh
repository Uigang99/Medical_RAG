#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="true"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"
TOP_K_VALUES=(1 2 4 8 16 32)
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
GAMMA="${GAMMA:-2.5}"
MAX_RATIONALE_TOKENS="${MAX_RATIONALE_TOKENS:-512}"
ANSWER_RESERVE_TOKENS="${ANSWER_RESERVE_TOKENS:-128}"
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-128}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-64}"
RATIONAL_SHARD_SIZE="${RATIONAL_SHARD_SIZE:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT/results/rag2_pced_topk_answer_mode_sweep_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$PROJECT/databases/run_cache/rag2_pced_dynamic_topk_v1}"
mkdir -p "$OUTPUT_ROOT"

LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/workflow.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

START_EPOCH=$(date +%s)
TOTAL_STAGES=15
CURRENT_STAGE=0

duration() {
  local seconds="$1"
  printf "%02dh%02dm%02ds" "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

begin_stage() {
  CURRENT_STAGE="$1"
  local label="$2"
  local now elapsed percent
  now=$(date +%s)
  elapsed=$((now - START_EPOCH))
  percent=$((100 * (CURRENT_STAGE - 1) / TOTAL_STAGES))
  echo "Overall: ${percent}% [stage ${CURRENT_STAGE}/${TOTAL_STAGES}, elapsed $(duration "$elapsed"), ETA calibrated by active Python stage]"
  echo "Stage ${CURRENT_STAGE}/${TOTAL_STAGES} - ${label}"
}

trap 'code=$?; now=$(date +%s); echo "[workflow FAILED | stage ${CURRENT_STAGE}/${TOTAL_STAGES} | elapsed $(duration $((now-START_EPOCH))) | exit=${code}] durable shards are resumable with the identical command"; exit $code' ERR

echo "[workflow plan] 1 expand the exact Top-8 retrieval query to a 4x32 master | 2 dynamic candidate projection | 3-8 Direct Choice | 9-14 Rationale+Answer | 15 combined report"
echo "[comparison per k] No-RAG | concatenated Base-RAG | PCED rerank prior | PCED semantic-support prior"
echo "[candidate contract] each corpus dense Top-k (4k total) -> existing cross-encoder rerank -> final Top-k"
echo "[Rationale+Answer contract] PCED full-vocabulary fusion at every rationale token; same fixed beta at constrained final answer"
echo "[settings] GPU=${CUDA_VISIBLE_DEVICES} questions=${MAX_QUESTIONS:-0} gamma=${GAMMA} max_rationale_tokens=${MAX_RATIONALE_TOKENS} output=${OUTPUT_ROOT}"
echo "[cold-run estimate] master expansion about 30-70 minutes if absent; Direct sweep about 2-5 hours from measured Top-8 throughput. Rationale+Answer has no compatible prior throughput; expect roughly 12-30 hours, with a measured ETA after the first 16-question shard. Completed shards resume safely."
echo "[memory policy] Direct batches shrink automatically on CUDA OOM; Rationale+Answer processes one question and at most 33 streams at a time; concatenated prompts use equal per-document token caps below 8192 tokens."

DIRECT_MAX_INPUT_TOKENS=$((8192 - MAX_RATIONALE_TOKENS - ANSWER_RESERVE_TOKENS))
if (( DIRECT_MAX_INPUT_TOKENS <= 0 )); then
  echo "ERROR: rationale/answer token reserves leave no prompt budget" >&2
  exit 2
fi

EXACT_CACHE_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1/all_mcq_source_balanced32_rationale_full_rerank32"
EXACT_ARTIFACT_ROOT="$PROJECT/databases/run_cache/rag2_llama3_paper_exact_terminal_v1/no_rag_rationales"
REFERENCE_QUERY_FINGERPRINT="fece4415cb4b1f6dfb9b741a778f56eee2de274e6e9b057dc899437d12e6d305"
REFERENCE_TOP8_CACHE="$EXACT_CACHE_ROOT/candidates/07083d5bac341d9b/candidates.jsonl"

resolve_master() {
  "$PYTHON" - "$EXACT_CACHE_ROOT" "$REFERENCE_QUERY_FINGERPRINT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); fingerprint=sys.argv[2]
matches=[]
for manifest in root.glob("candidates/*/manifest.json"):
    try: value=json.loads(manifest.read_text())
    except Exception: continue
    wanted=(value.get("rows")==6545 and value.get("per_source_top_k")==32
            and value.get("candidate_pool_top_k")==128 and value.get("rerank_top_k")==128
            and value.get("candidate_layout")=="source_balanced"
            and value.get("prompt_profile")=="paper_exact_terminal"
            and value.get("candidate_query_fingerprint")==fingerprint)
    candidate=manifest.parent/"candidates.jsonl"
    if wanted and candidate.is_file(): matches.append(candidate)
if len(matches)>1:
    raise SystemExit("Ambiguous exact-query master candidates: " + ", ".join(map(str,matches)))
if matches: print(matches[0])
PY
}

begin_stage 1 "expand the exact prior Top-8 retrieval query to 4 corpora x 32 and rerank all 128"
MASTER_CANDIDATE_CACHE="${MASTER_CANDIDATE_CACHE:-$(resolve_master)}"
if [[ -z "$MASTER_CANDIDATE_CACHE" ]]; then
  "$PYTHON" "$PROJECT/scripts/run_rag2_mcq_eval.py" \
    --case rerank_rag \
    --datasets medmcqa medqa mmlu_anatomy mmlu_clinical_knowledge mmlu_college_biology mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine \
    --collection unified --split test \
    --prompt-profile paper_exact_terminal --answer-decision-mode constrained_choice \
    --rationale-artifact-root "$EXACT_ARTIFACT_ROOT" --rationale-artifact-policy repair_invalid \
    --dense-query-mode rationale \
    --vector-db-root "$PROJECT/databases/vector_db/RAG_Square" \
    --sources pubmed pmc cpg textbooks --candidate-layout source_balanced \
    --per-source-top-k 32 --candidate-pool-top-k 128 --rerank-top-k 128 \
    --query-encoder-path /home/user/Uiheon/models/MedCPT-Query-Encoder \
    --cross-encoder-path /home/user/Uiheon/models/MedCPT-Cross-Encoder \
    --query-max-length 512 --embedding-batch-size 1024 --retrieval-batch-size 2048 \
    --rerank-batch-size 1024 --cross-encoder-max-length 512 --cross-encoder-attn-implementation eager \
    --faiss-gpu-device 0 --faiss-gpu-use-float16 --faiss-gpu-temp-memory-mb 2048 \
    --max-doc-chars 0 --llm-model-path /home/user/Uiheon/models/Llama-3-8B-Instruct \
    --cache-root "$EXACT_CACHE_ROOT" \
    --results-root "$OUTPUT_ROOT/candidate_expansion_validation" \
    --candidate-cache-only
  MASTER_CANDIDATE_CACHE="$(resolve_master)"
fi
if [[ -z "$MASTER_CANDIDATE_CACHE" || ! -s "$MASTER_CANDIDATE_CACHE" ]]; then
  echo "ERROR: exact-query 4x32/rerank128 master candidate cache was not produced" >&2
  exit 3
fi
echo "[stage 1/15 complete] master_candidate=$MASTER_CANDIDATE_CACHE"

begin_stage 2 "materialize exact dynamic Top-k candidate caches"
"$PYTHON" "$PROJECT/scripts/prepare_rag2_pced_topk_candidates.py" \
  --master-cache "$MASTER_CANDIDATE_CACHE" \
  --reference-top8-cache "$REFERENCE_TOP8_CACHE" \
  --output-root "$CANDIDATE_ROOT" \
  --top-k-values "${TOP_K_VALUES[@]}"

for index in "${!TOP_K_VALUES[@]}"; do
  k="${TOP_K_VALUES[$index]}"
  stage=$((3 + index))
  # Keep this identical across k so the No-RAG numerical baseline is not
  # changed by a different batch shape. Prompt micro-batches still auto-shrink.
  question_batch=8
  begin_stage "$stage" "Direct Choice Top-${k}"
  "$PYTHON" "$PROJECT/scripts/evaluate_rag2_pced_direct_choice_topk.py" \
    --candidate-cache "$CANDIDATE_ROOT/top${k}/candidates.jsonl" \
    --output-dir "$OUTPUT_ROOT/direct_choice/top${k}" \
    --top-k "$k" \
    --gamma "$GAMMA" \
    --question-batch-size "$question_batch" \
    --prompt-batch-size "$PROMPT_BATCH_SIZE" \
    --max-input-tokens "$DIRECT_MAX_INPUT_TOKENS" \
    --semantic-batch-size "$SEMANTIC_BATCH_SIZE" \
    --max-questions "$MAX_QUESTIONS"
done

for index in "${!TOP_K_VALUES[@]}"; do
  k="${TOP_K_VALUES[$index]}"
  stage=$((9 + index))
  begin_stage "$stage" "Rationale+Answer Top-${k}"
  "$PYTHON" "$PROJECT/scripts/evaluate_rag2_pced_rationale_answer.py" \
    --candidate-cache "$CANDIDATE_ROOT/top${k}/candidates.jsonl" \
    --semantic-score-cache "$OUTPUT_ROOT/direct_choice/top${k}/semantic_support_probabilities.jsonl" \
    --output-dir "$OUTPUT_ROOT/rationale_answer/top${k}" \
    --top-k "$k" \
    --gamma "$GAMMA" \
    --max-rationale-tokens "$MAX_RATIONALE_TOKENS" \
    --answer-reserve-tokens "$ANSWER_RESERVE_TOKENS" \
    --shard-size "$RATIONAL_SHARD_SIZE" \
    --max-questions "$MAX_QUESTIONS"
done

begin_stage 15 "combine both answer modes and all Top-k results"
"$PYTHON" "$PROJECT/scripts/summarize_rag2_pced_topk_answer_modes.py" --root "$OUTPUT_ROOT"

END_EPOCH=$(date +%s)
echo "Overall: 100% [stage 15/15 complete, elapsed $(duration $((END_EPOCH-START_EPOCH))), ETA 00h00m00s]"
echo "[workflow complete] report=$OUTPUT_ROOT/combined_summary_table.txt log=$LOG_FILE"
