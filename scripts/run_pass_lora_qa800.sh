#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="/home/user/Uiheon/.venv_vllm/bin/python"
export CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export USE_TF=0 USE_FLAX=0 TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=1
if [[ ! -x "$task_python" ]]; then
  echo "Missing Python environment: $task_python" >&2
  exit 2
fi
exec "$task_python" -u "$task_root/scripts/evaluate_pass_lora_qa800.py" "$@"
