#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/user/Uiheon/Medical_RAG"
PYTHON_BIN="${PYTHON_BIN:-/home/user/Uiheon/.venv_vllm/bin/python}"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING
export HF_HUB_OFFLINE=1
echo "[plan] Llama-3-8B; 64 proportional TEST questions x (8 fixed raw documents + amateur) = 576 planning-only outputs; no answers or KG calls."
echo "[runtime] GPU1 H200, batch32; no compatible measured planning throughput yet. Model loading is separate; first fresh 32-plan batch calibrates ETA. Identical command resumes durable batches."
exec "${PYTHON_BIN}" "${PROJECT}/scripts/pilot_rag2_evidence_planning.py" "$@"
