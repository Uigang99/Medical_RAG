#!/usr/bin/env bash
# Full MedMCQA Direct-Support counterfactual LoRA plus matched ordinary-SFT control.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
SCRIPT="$PROJECT/scripts/train_rag2_evidence_counterfactual_lora.py"
MODEL=/home/user/Uiheon/models/Llama-3-8B-Instruct
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_NAME="${RUN_NAME:-medmcqa_direct_support_stable_all_v5}"

test -x "$PYTHON"
test -f "$SCRIPT"
test -f "$MODEL/config.json"
test -f "$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/evidence_ablation_candidates_strict_v1/manifest.json"
test -f "$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/evidence_sentence_causal_audit_direct_choice_document_first_v2/summary.json"

echo "[command plan] 10 stages: materialize 11,710 stable train pairs -> tokenize -> fidelity -> 128-pair overfit gate -> frozen baseline -> counterfactual LoRA -> held-out test -> matched SFT -> held-out test -> comparison"
echo "[data contract] MedMCQA Direct Support only; cached top-1/top-2 gaps <=0.125 excluded as numerically unstable; dependence_demo=10,675, rescue=1,035; every demo is used once per epoch and rescue is cycled to a 6:2 batch ratio"
echo "[runtime estimate] H200 GPU 1, batch=8, max 4 epochs per arm: approximately 5-9 hours; early stopping may finish sooner; live hierarchical ETA replaces this estimate after calibration"
echo "[resume] prepared JSONL and epoch checkpoints are atomic; rerun this identical command after interruption"

exec "$PYTHON" "$SCRIPT" \
  --dataset medmcqa \
  --run-name "$RUN_NAME" \
  --model-name-or-path "$MODEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --demo-per-batch 6 \
  --rescue-per-batch 2 \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout 0 \
  --pair-improvement 0.5 \
  --answer-weight 1.0 \
  --pair-weight 1.0 \
  --removed-anchor-weight 0.5 \
  --control-consistency-weight 0.1 \
  --minimum-differential-margin 1.0 \
  --maximum-control-word-difference 0.25 \
  --minimum-top1-gap 0.125 \
  --expected-train-pairs 11710 \
  --tiny-demo 96 \
  --tiny-rescue 32 \
  --tiny-epochs 20 \
  --early-stopping-patience 2 \
  --max-input-tokens 2048 \
  --base-logit-tolerance 0.5 \
  --bootstrap-replicates 2000 \
  --attn-implementation eager \
  --gradient-checkpointing \
  --dtype bfloat16 \
  --device cuda:0 \
  --resume \
  --log-level INFO
