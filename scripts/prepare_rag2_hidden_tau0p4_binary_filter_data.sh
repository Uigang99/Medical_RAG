#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
PROJECT=/home/user/Uiheon/Medical_RAG
ARTIFACT_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2"
FEATURE_ROOT="$ARTIFACT_ROOT/preanswer_hidden_gold_direction_full_top8_v1"
LABEL_ROOT="$ARTIFACT_ROOT/preanswer_hidden_labels_full_top8_tau0p4_binary_v1"
REFERENCE_SPLIT_ROOT="$ARTIFACT_ROOT/filter_training_inputs_top10_independent_ppl_v2_corrected_nodoc"
OUTPUT_ROOT="$ARTIFACT_ROOT/filter_training_inputs_hidden_utility_top8_tau0p4_binary_v1"

for DATASET in medmcqa medqa; do
  "$PYTHON" "$PROJECT/scripts/materialize_rag2_preanswer_hidden_labels.py" \
    --input-dir "$FEATURE_ROOT/$DATASET" \
    --output-dir "$LABEL_ROOT/$DATASET" \
    --primary-layer layer_28 \
    --neutral-threshold 0.4 \
    --label-mode positive_vs_rest

  "$PYTHON" "$PROJECT/scripts/build_rag2_hidden_filter_inputs.py" \
    --dataset "$DATASET" \
    --hidden-feature-dir "$FEATURE_ROOT/$DATASET" \
    --hidden-label-dir "$LABEL_ROOT/$DATASET" \
    --reference-split-root "$REFERENCE_SPLIT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --expected-primary-layer layer_28 \
    --expected-threshold 0.4 \
    --expected-label-mode positive_vs_rest \
    --log-level INFO
done

echo "Prepared tau=0.4 binary labels and train/val/test inputs under:"
echo "  $OUTPUT_ROOT"
