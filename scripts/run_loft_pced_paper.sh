#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_loft_pced_paper.sh preflight|prepare|evaluate|all all|hotpotqa|nq|musique|qampari|quest [extra Python flags]
phase="${1:?phase: preflight|prepare|evaluate|all}"
dataset="${2:?dataset: all|hotpotqa|nq|musique|qampari|quest}"
shift 2
case "$phase" in preflight|prepare|evaluate|all) ;; *) echo "Invalid phase: $phase" >&2; exit 2;; esac
case "$dataset" in
  all) datasets=(hotpotqa nq musique qampari quest) ;;
  hotpotqa|nq|musique|qampari|quest) datasets=("$dataset") ;;
  *) echo "Invalid LOFT dataset: $dataset" >&2; exit 2;;
esac

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export MPLCONFIGDIR=/home/user/Uiheon/Medical_RAG/cache/matplotlib
mkdir -p "$MPLCONFIGDIR"
python=/home/user/Uiheon/Medical_RAG/.venv_pced/bin/python
runner=/home/user/Uiheon/Medical_RAG/scripts/run_loft_pced_paper.py
if [[ ! -x "$python" ]]; then
  echo "Missing PCED environment: $python" >&2
  exit 2
fi

for name in "${datasets[@]}"; do
  if [[ "$phase" == all ]]; then
    "$python" -u "$runner" --phase prepare --dataset "$name" "$@"
    "$python" -u "$runner" --phase evaluate --dataset "$name" "$@"
  else
    "$python" -u "$runner" --phase "$phase" --dataset "$name" "$@"
  fi
done
