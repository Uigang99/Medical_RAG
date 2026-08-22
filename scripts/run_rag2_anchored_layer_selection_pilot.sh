#!/usr/bin/env bash
set -euo pipefail

# FlashInfer sampling tries to JIT-compile through ``ninja`` during vLLM
# warm-up.  The project environment intentionally uses vLLM's native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

PROJECT=/home/user/Uiheon/Medical_RAG
PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
MODEL=/home/user/Uiheon/models/Llama-3-8B-Instruct
ARTIFACT_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1"
PILOT_ROOT="$ARTIFACT_ROOT/layer_selection_pilot_10k_v1"
CANDIDATE_ROOT="$PROJECT/datasets/filtering/rag2/llama3_8b_paper_exact_free_response_v2/candidates/quality_selected_source_balanced40_rerank32_v1"

MAX_PAIRS="${MAX_PAIRS:-10000}"
DOCS_PER_QUESTION="${DOCS_PER_QUESTION:-8}"

"$PYTHON" "$PROJECT/scripts/generate_rag2_anchored_layer_pilot.py" \
  --medmcqa-candidates-path "$CANDIDATE_ROOT/medmcqa/train/candidates_top32.jsonl" \
  --medqa-candidates-path "$CANDIDATE_ROOT/medqa/train/candidates_top32.jsonl" \
  --model-name-or-path "$MODEL" \
  --output-dir "$PILOT_ROOT/traces" \
  --max-pairs "$MAX_PAIRS" \
  --docs-per-question "$DOCS_PER_QUESTION" \
  --selection-seed 42 \
  --questions-per-shard 16 \
  --generation-batch-size 64 \
  --max-new-tokens 512 \
  --retry-max-new-tokens 768 \
  --temperature 0 \
  --top-p 1 \
  --max-doc-chars 0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --llm-max-model-len 8192 \
  --vllm-max-num-seqs 80 \
  --vllm-max-num-batched-tokens 65536 \
  --vllm-performance-mode throughput \
  --resume \
  --log-level INFO

"$PYTHON" "$PROJECT/scripts/extract_rag2_anchored_layer_pilot_features.py" \
  --trace-dir "$PILOT_ROOT/traces" \
  --model-name-or-path "$MODEL" \
  --output-dir "$PILOT_ROOT/features_all_blocks" \
  --question-batch-size 2 \
  --document-batch-size 8 \
  --max-input-tokens 2048 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --resume \
  --log-level INFO

"$PYTHON" "$PROJECT/scripts/analyze_rag2_anchored_layer_pilot.py" \
  --feature-dir "$PILOT_ROOT/features_all_blocks" \
  --output-dir "$PILOT_ROOT/analysis_projection" \
  --primary-score utility_projection \
  --min-subgroup-examples 100 \
  --min-nonzero-c-rate 0.95 \
  --log-level INFO

echo "Layer-selection summary:"
echo "$PILOT_ROOT/analysis_projection/layer_selection_summary.md"
