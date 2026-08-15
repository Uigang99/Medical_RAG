#!/usr/bin/env bash
# Gold-label upper-bound comparison on a deterministic 4,000 MedMCQA + all
# 994 MedQA held-out questions.  No learned filter is loaded.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON="/home/user/Uiheon/.venv_vllm/bin/python"
PROJECT="/home/user/Uiheon/Medical_RAG"

"$PYTHON" "$PROJECT/scripts/evaluate_rag2_oracle_label_topk_sweep.py" \
  --datasets medmcqa medqa \
  --source-split train \
  --label-split test \
  --medmcqa-question-limit 4000 \
  --medqa-question-limit 1000 \
  --sample-seed 42 \
  --top-k-values 1 2 4 8 \
  --hidden-thresholds 0 0.2 \
  --candidate-root "$PROJECT/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2/candidates/quality_selected_source_balanced40_rerank32_v1" \
  --candidate-file candidates_top32.jsonl \
  --rag2-label-root "$PROJECT/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2/filter_training_inputs_top10_independent_ppl_v2_corrected_nodoc" \
  --hidden-label-root "$PROJECT/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2/preanswer_hidden_labels_full_top8_tau0_v1" \
  --llm-model-path /home/user/Uiheon/models/Llama-3-8B-Instruct \
  --run-dir "$PROJECT/results/rag2_oracle_label_topk_4994_v1" \
  --generation-batch-size 32 \
  --max-new-tokens 768 \
  --max-doc-chars 0 \
  --temperature 0.0 \
  --top-p 1.0 \
  --gpu-memory-utilization 0.92 \
  --llm-max-model-len 8192 \
  --gdn-prefill-backend triton \
  --vllm-performance-mode throughput \
  --vllm-max-num-seqs 80 \
  --vllm-max-num-batched-tokens 65536 \
  --enable-prefix-caching \
  --resume \
  "$@"
