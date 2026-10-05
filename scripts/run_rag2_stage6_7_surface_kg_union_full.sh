#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/Uiheon/Medical_RAG
PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
SCRIPT="$ROOT/scripts/build_rag2_stage6_7_surface_kg_union_full.py"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[plan] Stage 5 normalized frames -> Stage 6 provenance/surface-edge KG -> Stage 7 canonical connectivity audit"
echo "[compute] CPU/SQLite only; no LLM and no GPU allocation; committed batches resume with the identical command"
exec "$PYTHON" "$SCRIPT" "$@"
