#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
GPU=${GPU:-1}
DATASETS=(medqa medmcqa)
workflow_start=$(date +%s)
# Cost model from the completed MedQA pilot: about 49 minutes for 2,068
# train pairs over three epochs on one H200.  Override when hardware or batch
# settings change.  Pair preparation is already cached and is not included.
EXPECTED_TOTAL_SECONDS=${EXPECTED_TOTAL_SECONDS:-25200}

format_seconds() {
  local value=$1
  printf '%02dh%02dm%02ds' \
    "$((value / 3600))" "$(((value % 3600) / 60))" "$((value % 60))"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scaled preservation comparison plan: datasets=2 objectives/dataset=2"
echo "  MedQA: all eligible training/validation/test pairs"
echo "  MedMCQA: 5,000 train and 1,000 validation/test pairs"
echo "  Objectives: original preference vs stop-gradient + student-No-RAG preservation"
echo "  Estimated wall time: $(format_seconds "$EXPECTED_TOTAL_SECONDS") on one H200; active stages replace this cost estimate with measured ETA"

for index in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$index]}
  now=$(date +%s)
  elapsed=$((now - workflow_start))
  remaining=$((EXPECTED_TOTAL_SECONDS > elapsed ? EXPECTED_TOTAL_SECONDS - elapsed : 0))
  printf '[overall dataset %d/%d | elapsed=%02dh%02dm%02ds | cost-model ETA=%s] starting %s\n' \
    "$((index + 1))" "${#DATASETS[@]}" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
    "$(format_seconds "$remaining")" "$dataset"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$PROJECT/scripts/run_rag2_semantic_behavior_single_document_lora.sh" \
      "$dataset" scaled compare
done

now=$(date +%s)
elapsed=$((now - workflow_start))
printf '[overall dataset 2/2 | elapsed=%02dh%02dm%02ds | ETA=00h00m00s] complete\n' \
  "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
