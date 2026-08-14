from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_command(args: list[str], timeout: int = 15) -> dict[str, Any]:
    if not shutil.which(args[0]):
        return {"available": False, "command": args, "stdout": "", "stderr": ""}
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return {"available": True, "command": args, "error": repr(exc), "stdout": "", "stderr": ""}
    return {
        "available": True,
        "command": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "visible_index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )

    return {
        "available": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_devices": devices,
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else False,
    }


def _read_small_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size > 5_000_000:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_environment(
    *,
    command: list[str],
    project_root: Path,
    workspace_root: Path,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    package_names = ["numpy", "torch", "transformers", "vllm", "faiss-cpu", "faiss-gpu", "PyYAML"]
    env_names = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TOKENIZERS_PARALLELISM",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONPATH",
    ]
    model_paths = {
        "llm": Path(str(run_config.get("llm_model_path", ""))),
        "cross_encoder": Path(str(run_config.get("cross_encoder_path", ""))),
    }
    model_manifests = {
        name: _read_small_json(path / "download_manifest.json") for name, path in model_paths.items() if str(path)
    }

    vector_db_root = Path(str(run_config.get("vector_db_root", "")))
    query_cache_dir = Path(str(run_config.get("query_cache_dir", "")))

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": str(Path.cwd()),
        "project_root": str(project_root),
        "workspace_root": str(workspace_root),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
            "python_version": sys.version,
        },
        "environment_variables": {name: os.environ.get(name) for name in env_names},
        "packages": {name: _package_version(name) for name in package_names},
        "torch": _torch_info(),
        "nvidia_smi": {
            "list": _run_command(["nvidia-smi", "-L"]),
            "query": _run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
        "artifacts": {
            "benchmark_path": run_config.get("benchmark_path"),
            "query_cache_dir": str(query_cache_dir),
            "query_cache_manifest": _read_small_json(query_cache_dir / "manifest.json"),
            "vector_db_root": str(vector_db_root),
            "vector_db_manifest": _read_small_json(vector_db_root / "manifest.json"),
            "model_manifests": model_manifests,
        },
        "run_config": run_config,
    }


def write_environment_files(output_dir: Path, environment: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Experiment Environment",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Created at | `{environment.get('created_at')}` |",
        f"| Host | `{environment.get('host', {}).get('hostname')}` |",
        f"| Platform | `{environment.get('host', {}).get('platform')}` |",
        f"| Python | `{environment.get('host', {}).get('python_executable')}` |",
        f"| CUDA_VISIBLE_DEVICES | `{environment.get('environment_variables', {}).get('CUDA_VISIBLE_DEVICES')}` |",
        f"| Torch | `{environment.get('torch', {}).get('version')}` |",
        f"| Torch CUDA runtime | `{environment.get('torch', {}).get('cuda_runtime')}` |",
        f"| CUDA available | `{environment.get('torch', {}).get('cuda_available')}` |",
        f"| Visible CUDA devices | `{environment.get('torch', {}).get('cuda_device_count')}` |",
        f"| Transformers | `{environment.get('packages', {}).get('transformers')}` |",
        f"| vLLM | `{environment.get('packages', {}).get('vllm')}` |",
        f"| FAISS CPU | `{environment.get('packages', {}).get('faiss-cpu')}` |",
    ]

    devices = environment.get("torch", {}).get("cuda_devices") or []
    if devices:
        lines.extend(["", "## CUDA Devices", "", "| Visible Index | Name | Memory |", "|---:|---|---:|"])
        for device in devices:
            gib = int(device["total_memory_bytes"]) / (1024**3)
            lines.append(f"| {device['visible_index']} | `{device['name']}` | {gib:.1f} GiB |")

    lines.extend(["", "## Command", "", "```bash", " ".join(environment.get("command", [])), "```", ""])
    (output_dir / "environment.md").write_text("\n".join(lines), encoding="utf-8")

