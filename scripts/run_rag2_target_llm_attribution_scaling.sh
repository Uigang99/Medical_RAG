#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON_BIN=/home/user/Uiheon/.venv_vllm/bin/python
DATASET=${1:-medqa}

if [[ "${DATASET}" != "medqa" ]]; then
  echo "This scaling workflow currently supports medqa only" >&2
  exit 2
fi

read -r -a TRAIN_SIZE_VALUES <<< "${TRAIN_SIZES:-256 512 1024 2048}"
if [[ ${#TRAIN_SIZE_VALUES[@]} -eq 0 ]]; then
  echo "TRAIN_SIZES must contain at least one positive integer" >&2
  exit 2
fi
previous=0
for size in "${TRAIN_SIZE_VALUES[@]}"; do
  if ! [[ "${size}" =~ ^[1-9][0-9]*$ ]] || (( size <= previous )); then
    echo "TRAIN_SIZES must be strictly increasing positive integers" >&2
    exit 2
  fi
  previous=$size
done
MAX_TRAIN_POOL=$previous
EVAL_QUESTIONS=${EVAL_QUESTIONS:-256}
if ! [[ "${EVAL_QUESTIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_QUESTIONS must be a positive integer" >&2
  exit 2
fi

BASE="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
RATIONALE_CACHE="$BASE/semantic_attention_controller_v1/medqa_pilot_top8_final_choice_v1/top8_unbiased_rationales"
SOURCE_FEATURE_DIR=${SOURCE_FEATURE_DIR:-"$BASE/semantic_attention_controller_v1/medqa_attribution_scaling_top8_v1/prepared_features"}
TEACHER_FEATURE_DIR=${TEACHER_FEATURE_DIR:-"$BASE/target_llm_attribution_v1/medqa_scaling_k8_layers20_28_fixed_rationale_v1"}
MODEL_ROOT=${MODEL_ROOT:-"/home/user/Uiheon/models/RAG2-Target-LLM-Attribution/medqa/scaling_k8_content_only_v1"}
INDEX_PATH="$BASE/semantic_attention_controller_v1/shared_indices/medqa.sqlite"

if [[ ! -f "$RATIONALE_CACHE/generation_manifest.json" ]]; then
  echo "Missing rationale cache: $RATIONALE_CACHE" >&2
  exit 1
fi

workflow_start=$(date +%s)
total_stages=$((2 + ${#TRAIN_SIZE_VALUES[@]}))
announce_stage() {
  local index=$1
  local name=$2
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - workflow_start))
  printf '[%s] Scaling workflow %d/%d: %s | elapsed=%02dh%02dm%02ds | overall ETA=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$index" "$total_stages" "$name" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
    "unknown until the active stage measures throughput"
}

announce_stage 1 "prepare the fixed maximum question pool and prompt/span features"
"$PYTHON_BIN" "$PROJECT/scripts/prepare_rag2_semantic_attention_controller.py" \
  --dataset medqa \
  --rationale-cache "$RATIONALE_CACHE" \
  --output-dir "$SOURCE_FEATURE_DIR" \
  --index-path "$INDEX_PATH" \
  --max-questions-per-split "$MAX_TRAIN_POOL" \
  --questions-per-shard "${FEATURE_QUESTIONS_PER_SHARD:-128}" \
  --semantic-batch-size "${SEMANTIC_BATCH_SIZE:-64}" \
  --semantic-max-input-length "${SEMANTIC_MAX_INPUT_LENGTH:-2048}" \
  --device cuda:0 \
  --allow-partial-rationale-cache \
  --resume

announce_stage 2 "materialize unbiased fixed-rationale LOO teachers once for the maximum pool"
"$PYTHON_BIN" "$PROJECT/scripts/prepare_rag2_target_llm_attribution.py" \
  --dataset medqa \
  --source-feature-dir "$SOURCE_FEATURE_DIR" \
  --llm-model /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --output-dir "$TEACHER_FEATURE_DIR" \
  --layers 20 28 \
  --splits train val test \
  --max-questions-per-split "$MAX_TRAIN_POOL" \
  --expected-document-count 8 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --resume

stage=2
for train_size in "${TRAIN_SIZE_VALUES[@]}"; do
  stage=$((stage + 1))
  output_dir="$MODEL_ROOT/train_${train_size}"
  announce_stage "$stage" "train content-only relative attribution predictor on ${train_size} nested questions"
  "$PYTHON_BIN" "$PROJECT/scripts/train_rag2_target_llm_attribution.py" \
    --dataset medqa \
    --feature-dir "$TEACHER_FEATURE_DIR" \
    --output-dir "$output_dir" \
    --epochs "${EPOCHS:-50}" \
    --patience "${PATIENCE:-8}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --learning-rate "${LEARNING_RATE:-2e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.01}" \
    --dropout "${DROPOUT:-0.1}" \
    --model-dim "${MODEL_DIM:-256}" \
    --transformer-layers "${TRANSFORMER_LAYERS:-2}" \
    --attention-heads "${ATTENTION_HEADS:-4}" \
    --feedforward-dim "${FEEDFORWARD_DIM:-1024}" \
    --total-loss-weight 0 \
    --share-loss-weight 1 \
    --set-shift-loss-weight 0 \
    --rank-loss-weight "${RANK_LOSS_WEIGHT:-0.1}" \
    --minimum-total-for-share "${MINIMUM_TOTAL_FOR_SHARE:-1e-6}" \
    --minimum-rank-log-ratio "${MINIMUM_RANK_LOG_RATIO:-0.25}" \
    --no-use-rank-feature \
    --no-use-length-feature \
    --shuffle-documents-during-training \
    --max-train-samples "$train_size" \
    --max-eval-samples "$EVAL_QUESTIONS" \
    --num-workers "${NUM_WORKERS:-4}" \
    --seed "${SEED:-42}" \
    --device cuda:0 \
    --resume
done

"$PYTHON_BIN" - "$MODEL_ROOT" "${TRAIN_SIZE_VALUES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sizes = [int(value) for value in sys.argv[2:]]
print("\ntrain_questions\tbest_epoch\ttest_n\tmeasurable\tspearman\ttop1\tshare_mae\tuniform_mae")
for size in sizes:
    path = root / f"train_{size}" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    test = summary["test"]
    print(
        f"{size}\t{summary['best_epoch']}\t{test['questions']}\t"
        f"{test['measurable_questions']}\t{test['mean_per_question_spearman']:.6f}\t"
        f"{test['top1_accuracy']:.6f}\t{test['share_mae']:.6f}\t"
        f"{test['uniform_share_mae']:.6f}"
    )
PY

workflow_end=$(date +%s)
elapsed=$((workflow_end - workflow_start))
printf '[%s] Scaling workflow complete | elapsed=%02dh%02dm%02ds | models=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
  "$MODEL_ROOT"
