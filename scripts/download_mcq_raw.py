from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "benchmark" / "mcq" / "raw"

MMLU_MEDICAL_SUBSETS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    allow_patterns: list[str]
    note: str


DATASETS = [
    DatasetSpec(
        name="medqa",
        repo_id="GBaker/MedQA-USMLE-4-options",
        allow_patterns=[
            "README*",
            "dataset_infos.json",
            "*.json",
            "*.jsonl",
            "*.parquet",
            "data/**",
        ],
        note="USMLE-style MedQA, 4-option Hugging Face dataset snapshot.",
    ),
    DatasetSpec(
        name="medmcqa",
        repo_id="openlifescienceai/medmcqa",
        allow_patterns=[
            "README*",
            "dataset_infos.json",
            "*.json",
            "*.jsonl",
            "*.parquet",
            "data/**",
        ],
        note="MedMCQA public Hugging Face dataset snapshot.",
    ),
    DatasetSpec(
        name="mmlu_medical",
        repo_id="cais/mmlu",
        allow_patterns=[
            "README*",
            "dataset_infos.json",
            "hendrycks_test.py",
            *[f"{subset}/**" for subset in MMLU_MEDICAL_SUBSETS],
        ],
        note="Medical MMLU subsets only.",
    ),
]


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _count_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def _relative_files(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def _remove_local_hf_cache(path: Path) -> None:
    shutil.rmtree(path / ".cache", ignore_errors=True)


def _ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _download_one(spec: DatasetSpec, output_dir: Path, overwrite: bool) -> dict:
    target_dir = output_dir / spec.name
    _ensure_clean_dir(target_dir, overwrite)

    local_path = snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=spec.allow_patterns,
    )

    info = HfApi().dataset_info(spec.repo_id)
    local_dir = Path(local_path)
    _remove_local_hf_cache(local_dir)
    return {
        **asdict(spec),
        "revision": info.sha,
        "local_dir": str(local_dir.relative_to(PROJECT_ROOT)),
        "file_count": _count_files(local_dir),
        "size_bytes": _dir_size_bytes(local_dir),
        "files": _relative_files(local_dir),
    }


def _select_specs(names: Iterable[str]) -> list[DatasetSpec]:
    requested = set(names)
    if not requested:
        return DATASETS
    specs_by_name = {spec.name: spec for spec in DATASETS}
    unknown = sorted(requested - set(specs_by_name))
    if unknown:
        raise ValueError(f"Unknown dataset name(s): {', '.join(unknown)}")
    return [specs_by_name[name] for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw MCQ benchmark datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        choices=[spec.name for spec in DATASETS],
        help="Dataset to download. Repeatable. Default: all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing dataset directory before downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "mmlu_medical_subsets": MMLU_MEDICAL_SUBSETS,
        "datasets": [],
    }

    for spec in _select_specs(args.dataset):
        print(f"Downloading {spec.name} from {spec.repo_id} ...", flush=True)
        manifest["datasets"].append(_download_one(spec, output_dir, args.overwrite))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
