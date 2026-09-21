#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/home/user/Uiheon/Medical_RAG"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=ERROR
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
TASK_PYTHON=/home/user/Uiheon/.venv_vllm/bin/python
TASK_SCRIPT="${TASK_ROOT}/scripts/extract_rag2_restored_gemma_propositions.py"
TASK_OUTPUT=$("${TASK_PYTHON}" "${TASK_SCRIPT}" "$@" --resolve-output-dir)
mkdir -p "${TASK_OUTPUT}"
echo '[plan] Frozen Gemma31B/prompt; GPU 1 only; completed 8-document shards are preserved.'
echo '[estimate] Top-8 batch32: measured extrapolation 9.2h; plan 10-12h. Batch16: 13.3h before contingency. OOM retries increase time.'
echo '[resume] On confirmed generation CUDA OOM: split failed group 32 -> 16 -> 8, then restore initial batch. Other failures and batch8 OOM stop safely.'
while true; do
    TASK_LOG=$(mktemp "${TASK_OUTPUT}/attempt_XXXXXXXX.log")
    set +e
    "${TASK_PYTHON}" -u "${TASK_SCRIPT}" "$@" 2>&1 | tee "${TASK_LOG}"
    TASK_RC=${PIPESTATUS[0]}
    set -e
    if (( TASK_RC == 0 )); then exit 0; fi
    if (( TASK_RC == 75 )); then continue; fi
    if (( TASK_RC == 130 || TASK_RC == 143 )); then exit "${TASK_RC}"; fi
    if ! rg -qi 'CUDA out of memory|torch\.(cuda\.)?OutOfMemoryError|CUDA error: out of memory|CUDA_ERROR_OUT_OF_MEMORY' "${TASK_LOG}"; then
        echo "[stopped] Non-OOM error; no automatic retry. Log: ${TASK_LOG}"
        exit "${TASK_RC}"
    fi
    if ! "${TASK_PYTHON}" "${TASK_SCRIPT}" "$@" --reduce-batch-after-oom; then exit "${TASK_RC}"; fi
done
