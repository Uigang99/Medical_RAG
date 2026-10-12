#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING
export HF_HUB_OFFLINE=1
echo "[plan] V3 actionable plans + need-linked evidence; frozen V2 64 questions x 9 = 576; same Llama, documents, decoding and token limit; no answers/choices/KG."
echo "[runtime] H200 GPU1, batch32; new schema has no measured generation rate. First fresh 32-plan batch calibrates ETA; identical command resumes durable batches."
exec "${PYTHON_BIN}" "${PROJECT}/scripts/pilot_rag2_evidence_planning_v3.py" "$@"
