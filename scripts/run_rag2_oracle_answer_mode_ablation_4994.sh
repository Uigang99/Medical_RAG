#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
EVALUATOR="$PROJECT/scripts/evaluate_rag2_oracle_label_topk_sweep.py"
SUMMARIZER="$PROJECT/scripts/summarize_rag2_oracle_answer_mode_ablation.py"

RATIONALE_BASE_DIR="${RATIONALE_BASE_DIR:-$PROJECT/results/rag2_oracle_label_topk_4994_v1}"
DIRECT_DIR="${DIRECT_DIR:-$PROJECT/results/rag2_oracle_label_topk_4994_direct_choice_tau0_0p2_0p4_v1}"
RATIONALE_TAU04_DIR="${RATIONALE_TAU04_DIR:-$PROJECT/results/rag2_oracle_label_topk_4994_rationale_hidden_tau0p4_v1}"
COMBINED_DIR="${COMBINED_DIR:-$PROJECT/results/rag2_oracle_answer_mode_ablation_4994_v1}"

RATIONALE_NO_RAG="$RATIONALE_BASE_DIR/no_rag_results.jsonl"
RATIONALE_BASE_SUMMARY="$RATIONALE_BASE_DIR/summary.json"
if [[ ! -f "$RATIONALE_NO_RAG" || ! -f "$RATIONALE_BASE_SUMMARY" ]]; then
  echo "Missing completed rationale baseline run: $RATIONALE_BASE_DIR" >&2
  exit 1
fi

COMMON=(
  --datasets medmcqa medqa
  --source-split train
  --label-split test
  --medmcqa-question-limit 4000
  --medqa-question-limit 1000
  --sample-seed 42
  --top-k-values 1 2 4 8
  --llm-model-path /home/user/Uiheon/models/Llama-3-8B-Instruct
  --generation-batch-size 32
  --max-doc-chars 0
  --temperature 0.0
  --top-p 1.0
  --gpu-memory-utilization 0.92
  --llm-max-model-len 8192
  --gdn-prefill-backend triton
  --vllm-performance-mode throughput
  --vllm-max-num-seqs 80
  --vllm-max-num-batched-tokens 65536
  --resume
  --log-level INFO
)

"$PYTHON" "$EVALUATOR" \
  "${COMMON[@]}" \
  --answer-decision-mode constrained_choice \
  --hidden-thresholds 0 0.2 0.4 \
  --include-rag2 \
  --max-new-tokens 1 \
  --run-dir "$DIRECT_DIR"

"$PYTHON" "$EVALUATOR" \
  "${COMMON[@]}" \
  --answer-decision-mode paper_exact_terminal \
  --hidden-thresholds 0.4 \
  --no-include-rag2 \
  --max-new-tokens 768 \
  --reuse-no-rag-path "$RATIONALE_NO_RAG" \
  --run-dir "$RATIONALE_TAU04_DIR"

"$PYTHON" "$SUMMARIZER" \
  --rationale-base-summary "$RATIONALE_BASE_SUMMARY" \
  --rationale-tau04-summary "$RATIONALE_TAU04_DIR/summary.json" \
  --direct-choice-summary "$DIRECT_DIR/summary.json" \
  --output-dir "$COMBINED_DIR"

echo "Combined comparison: $COMBINED_DIR/summary_table_pretty.txt"
