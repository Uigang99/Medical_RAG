#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="$task_root/.venv_pced/bin/python"
if [[ ! -x "$task_python" ]]; then
  echo "Missing Python environment: $task_python" >&2
  exit 2
fi
exec "$task_python" -u "$task_root/scripts/evaluate_pced_sentence_pass_800.py" "$@"
