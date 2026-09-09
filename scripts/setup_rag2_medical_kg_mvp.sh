#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/Uiheon/Medical_RAG"
VENV="${KG_VENV:-/home/user/Uiheon/.venv_rag2_kg_mvp_py312}"
BOOTSTRAP_VENV="${UV_BOOTSTRAP_VENV:-/home/user/Uiheon/.venv_uv_bootstrap}"
MODEL_ROOT="${KG_MODEL_ROOT:-/home/user/Uiheon/models/RAG2-Medical-KG-MVP}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/user/Uiheon/.cache/uv-rag2-kg}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/home/user/Uiheon/.local/uv-python}"
MODEL_URL="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
MODEL_ARCHIVE="${MODEL_ROOT}/en_core_sci_sm-0.5.4.tar.gz"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
export SCISPACY_CACHE="${MODEL_ROOT}/scispacy_cache"

echo "[setup plan] uv bootstrap -> managed Python 3.12 -> isolated venv -> local model archive -> UMLS cache prewarm"
echo "[setup estimate] 5-25 minutes depending on network; about 2-4 GB under ${MODEL_ROOT} plus the venv"

mkdir -p "${MODEL_ROOT}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

if [[ ! -x "${BOOTSTRAP_VENV}/bin/uv" ]]; then
  python3 -m venv "${BOOTSTRAP_VENV}"
  "${BOOTSTRAP_VENV}/bin/python" -m pip install --upgrade pip uv
fi

"${BOOTSTRAP_VENV}/bin/uv" python install 3.12
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BOOTSTRAP_VENV}/bin/uv" venv --python 3.12 "${VENV}"
fi
if [[ "$("${VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "ERROR: ${VENV} is not a Python 3.12 environment; choose a new KG_VENV path"
  exit 2
fi

"${BOOTSTRAP_VENV}/bin/uv" pip install --python "${VENV}/bin/python" \
  "numpy==1.26.4" "click==8.5.0" "scikit-learn==1.9.0" \
  "spacy==3.7.5" "scispacy==0.5.5" "tqdm>=4.66,<5"

if [[ ! -s "${MODEL_ARCHIVE}" ]]; then
  curl --fail --location --retry 3 --output "${MODEL_ARCHIVE}.partial" "${MODEL_URL}"
  mv "${MODEL_ARCHIVE}.partial" "${MODEL_ARCHIVE}"
fi
"${BOOTSTRAP_VENV}/bin/uv" pip install --python "${VENV}/bin/python" "${MODEL_ARCHIVE}"

cd "${ROOT}"
"${VENV}/bin/python" -c '
try:
    from scripts.build_rag2_medical_kg_mvp import load_pipeline
    nlp, linker = load_pipeline("en_core_sci_sm", "umls", 0.85, 3)
except RuntimeError as exc:
    if "Missing local scispaCy cache" not in str(exc) and "Expected exactly one local UMLS cache" not in str(exc):
        raise
    # First installation only: let the official loader populate SCISPACY_CACHE.
    import spacy
    from scispacy import abbreviation, linking
    bootstrap = spacy.load("en_core_sci_sm")
    bootstrap.add_pipe("abbreviation_detector")
    bootstrap.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": 0.85, "max_entities_per_mention": 3})
    del bootstrap
    nlp, linker = load_pipeline("en_core_sci_sm", "umls", 0.85, 3)
doc = linker(nlp("Aspirin may prevent myocardial infarction."))
import sys
print(f"[setup complete] python={sys.version.split()[0]} model=en_core_sci_sm entities={len(doc.ents)} venv={sys.prefix}")
'

echo "[setup output] venv=${VENV} model_root=${MODEL_ROOT} scispacy_cache=${SCISPACY_CACHE}"
