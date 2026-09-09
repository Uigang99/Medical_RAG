#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon"
REPO="${BIOREX_REPO:-${ROOT}/tools/BioREx}"
VENV="${BIOREX_VENV:-${ROOT}/.venv_biorex_py312}"
MODEL_DIR="${BIOREX_MODEL:-${ROOT}/models/BioREx-BioLinkBERT}"
DOWNLOAD_DIR="${ROOT}/models/BioREx-BioLinkBERT-download"
ARCHIVE="${DOWNLOAD_DIR}/pretrained_model_biolinkbert.zip"
MODEL_URL="https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/pretrained_model_biolinkbert.zip"
EXPECTED_REPO_COMMIT="1f199d8c359eaceb5469538f4daa24aac44c4eae"

echo "[setup plan] 1/4 official BioREx source | 2/4 Python 3.12 environment | 3/4 official checkpoint | 4/4 import/GPU preflight"

if [[ ! -d "${REPO}/.git" ]]; then
  echo "[setup 1/4] cloning official BioREx source"
  git clone https://github.com/ncbi/BioREx.git "${REPO}"
else
  echo "[setup 1/4] official BioREx source already present"
fi
CURRENT_COMMIT="$(git -C "${REPO}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${EXPECTED_REPO_COMMIT}" ]]; then
  echo "ERROR: BioREx source commit differs from the validated contract." >&2
  echo "expected=${EXPECTED_REPO_COMMIT} actual=${CURRENT_COMMIT}" >&2
  exit 2
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[setup 2/4] creating Python 3.12 environment"
  python3.12 -m venv "${VENV}"
fi
echo "[setup 2/4] installing the minimal official-compatible inference stack"
"${VENV}/bin/python" -m pip install --upgrade pip wheel setuptools
"${VENV}/bin/python" -m pip install \
  'tensorflow[and-cuda]==2.18.0' \
  'tf-keras==2.18.0' \
  'transformers==4.47.1' \
  'pandas==2.2.3' \
  'datasets==3.2.0' \
  'sentencepiece!=0.1.92' \
  'protobuf==5.29.2' \
  'scispacy==0.5.5'

if [[ ! -s "${MODEL_DIR}/config.json" ]]; then
  mkdir -p "${DOWNLOAD_DIR}" "${MODEL_DIR}"
  echo "[setup 3/4] resuming official BioREx BioLinkBERT checkpoint download"
  wget --continue --show-progress --output-document="${ARCHIVE}" "${MODEL_URL}"
  unzip -tq "${ARCHIVE}"
  STAGING="$(mktemp -d /tmp/biorex-model.XXXXXX)"
  trap 'rm -rf "${STAGING}"' EXIT
  unzip -q "${ARCHIVE}" -d "${STAGING}"
  FOUND_CONFIG="$(find "${STAGING}" -type f -name config.json -print -quit)"
  if [[ -z "${FOUND_CONFIG}" ]]; then
    echo "ERROR: downloaded archive contains no config.json" >&2
    exit 2
  fi
  FOUND_MODEL="$(dirname "${FOUND_CONFIG}")"
  cp -a "${FOUND_MODEL}/." "${MODEL_DIR}/"
else
  echo "[setup 3/4] official checkpoint already present: ${MODEL_DIR}"
fi

echo "[setup 4/4] checking pinned package versions and TensorFlow GPU visibility"
TF_USE_LEGACY_KERAS=1 CUDA_VISIBLE_DEVICES=1 "${VENV}/bin/python" - <<'PY'
import tensorflow as tf
import transformers
assert tf.__version__.startswith("2.18."), tf.__version__
assert transformers.__version__ == "4.47.1", transformers.__version__
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise SystemExit("ERROR: TensorFlow cannot see GPU 1")
print(f"[setup complete] tensorflow={tf.__version__} transformers={transformers.__version__} GPUs={gpus}")
PY

