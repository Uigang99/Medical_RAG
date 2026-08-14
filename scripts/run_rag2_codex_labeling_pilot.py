#!/usr/bin/env python3
"""Run a sequential multi-configuration Codex labelling pilot with live progress.

One foreground controller launches background workers for every model
configuration.  The controller polls the completed batch artifacts and renders
a single aggregate progress view, so the user need not open a second monitor.
Interrupting the controller also stops its workers, while completed batches are
kept; rerunning with --resume safely continues unfinished work.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from monitor_rag2_codex_labeling import render


DEFAULT_VARIANTS = (
    ("terra_xhigh", "gpt-5.6-terra", "xhigh"),
    ("terra_high", "gpt-5.6-terra", "high"),
    ("terra_medium", "gpt-5.6-terra", "medium"),
    ("luna_high", "gpt-5.6-luna", "high"),
)


def parse_variant(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3 or any(not item.strip() for item in parts):
        raise argparse.ArgumentTypeError("--variant must be NAME:MODEL:EFFORT")
    name, model, effort = (item.strip() for item in parts)
    if effort not in {"low", "medium", "high", "xhigh"}:
        raise argparse.ArgumentTypeError("Effort must be low, medium, high, or xhigh")
    return name, model, effort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Codex semantic-labeling comparison pilot with one live aggregate progress display."
    )
    parser.add_argument("--candidates-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--completed-batches-root",
        type=Path,
        default=None,
        help="Optional read-only prior-run root whose validated batch artifacts are reused.",
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--label-script",
        type=Path,
        default=Path(__file__).with_name("label_rag2_candidates_with_codex.py"),
    )
    parser.add_argument(
        "--progress-db-path",
        type=Path,
        default=None,
        help="Optional writable SQLite status DB, useful when resuming artifacts created by another Linux user.",
    )
    parser.add_argument("--docs-per-question", type=int, default=10)
    parser.add_argument("--questions-per-batch", type=int, default=10)
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=200,
        help="Questions per input dataset; zero processes every available question (default: 200).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--rebalance-pending-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the currently unfinished batches and redistribute them evenly across all workers.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=60.0)
    parser.add_argument("--retry-jitter-fraction", type=float, default=0.25)
    parser.add_argument(
        "--max-worker-restarts",
        type=int,
        default=20,
        help="Automatically relaunch a failed worker this many times; zero means unlimited (default: 20).",
    )
    parser.add_argument(
        "--worker-restart-backoff-seconds",
        type=float,
        default=30.0,
        help="Base controller delay before relaunching a failed worker (default: 30).",
    )
    parser.add_argument(
        "--worker-start-stagger-seconds",
        type=float,
        default=2.0,
        help="Stagger initial Codex worker launches to avoid synchronized model-catalog/capacity requests (default: 2).",
    )
    parser.add_argument("--refresh-seconds", type=float, default=10.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-validated-completed-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse completed batches after a transport-only response-schema upgrade, after local validation.",
    )
    parser.add_argument("--enable-web-search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--variant",
        type=parse_variant,
        action="append",
        help="Repeat NAME:MODEL:EFFORT. Defaults to Terra xhigh/high/medium and Luna high.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def worker_command(
    args: argparse.Namespace,
    variant_root: Path,
    model: str,
    effort: str,
    worker_index: int,
    pending_plan_path: Path | None = None,
) -> list[str]:
    command = [
        str(args.python_bin),
        str(args.label_script),
        "--candidates-paths",
        *(str(path) for path in args.candidates_paths),
        "--output-root",
        str(variant_root),
        "--docs-per-question",
        str(args.docs_per_question),
        "--questions-per-batch",
        str(args.questions_per_batch),
        "--limit-questions",
        str(args.limit_questions),
        "--model",
        model,
        "--model-reasoning-effort",
        effort,
        "--worker-count",
        str(args.workers),
        "--worker-index",
        str(worker_index),
        "--max-attempts",
        str(args.max_attempts),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
        "--retry-jitter-fraction",
        str(args.retry_jitter_fraction),
        "--log-level",
        args.log_level,
    ]
    if args.completed_batches_root is not None:
        command.extend(["--completed-batches-root", str(args.completed_batches_root)])
    command.append("--enable-web-search" if args.enable_web_search else "--no-enable-web-search")
    command.append("--resume" if args.resume else "--no-resume")
    command.append(
        "--reuse-validated-completed-batches"
        if args.reuse_validated_completed_batches
        else "--no-reuse-validated-completed-batches"
    )
    if args.progress_db_path is not None:
        command.extend(["--progress-db-path", str(args.progress_db_path)])
    if pending_plan_path is not None:
        command.extend(["--pending-plan-path", str(pending_plan_path)])
    return command


def consolidate_command(args: argparse.Namespace, variant_root: Path, model: str, effort: str) -> list[str]:
    command = worker_command(args, variant_root, model, effort, worker_index=0)
    command.extend(["--consolidate-only"])
    return command


def clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def display(args: argparse.Namespace, names: list[str]) -> None:
    clear_screen()
    expected_pairs_per_run = args.limit_questions * args.docs_per_question * len(args.candidates_paths)
    print(render(args.output_root, names, max_errors=3, expected_pairs_per_run=expected_pairs_per_run), flush=True)
    print("\nCtrl+C stops this controller and its workers; completed batches are retained for --resume.", flush=True)


def run_variant(args: argparse.Namespace, name: str, model: str, effort: str, all_names: list[str]) -> None:
    variant_root = args.output_root / name
    log_dir = variant_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending_plan_path: Path | None = None
    if args.rebalance_pending_batches:
        pending_plan_path = variant_root / "pending_batches_rebalanced.json"
        logging.info("Scanning and freezing unfinished batches for balanced redistribution: %s", pending_plan_path)
        plan_command = worker_command(args, variant_root, model, effort, worker_index=0)
        plan_command.extend(["--pending-plan-path", str(pending_plan_path), "--write-pending-plan-only"])
        planned = subprocess.run(plan_command, check=False)
        if planned.returncode != 0:
            raise RuntimeError(f"Could not build pending-batch plan; return code {planned.returncode}")
    logging.info("Starting %s (%s, reasoning=%s) with %d workers", name, model, effort, args.workers)
    started_at = now_iso()
    start_epoch = time.time()
    processes: dict[int, subprocess.Popen[str]] = {}
    log_handles: dict[int, object] = {}
    restart_counts = {worker_index: 0 for worker_index in range(args.workers)}
    restart_not_before: dict[int, float] = {}
    finished_workers: set[int] = set()

    def launch_worker(worker_index: int) -> None:
        log_handle = log_handles.get(worker_index)
        if log_handle is None:
            log_handle = (log_dir / f"worker_{worker_index}.log").open("a", encoding="utf-8")
            log_handles[worker_index] = log_handle
        process = subprocess.Popen(
            worker_command(args, variant_root, model, effort, worker_index, pending_plan_path),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[worker_index] = process
        logging.info(
            "%s worker %d started (pid=%d, restart=%d)",
            name,
            worker_index,
            process.pid,
            restart_counts[worker_index],
        )

    try:
        for worker_index in range(args.workers):
            launch_worker(worker_index)
            if args.worker_start_stagger_seconds and worker_index + 1 < args.workers:
                time.sleep(args.worker_start_stagger_seconds)

        while len(finished_workers) < args.workers:
            now = time.monotonic()
            for worker_index, process in list(processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del processes[worker_index]
                if returncode == 0:
                    finished_workers.add(worker_index)
                    logging.info("%s worker %d finished successfully", name, worker_index)
                    continue

                restart_counts[worker_index] += 1
                restart_number = restart_counts[worker_index]
                if args.max_worker_restarts and restart_number > args.max_worker_restarts:
                    raise RuntimeError(
                        f"{name} worker {worker_index} exceeded {args.max_worker_restarts} automatic restarts; "
                        f"see {log_dir / f'worker_{worker_index}.log'}"
                    )
                multiplier = min(2 ** (restart_number - 1), 16)
                base_delay = args.worker_restart_backoff_seconds * multiplier
                jitter_unit = ((worker_index + 1) * 37 + restart_number * 17) % 101 / 100.0
                delay = base_delay * (1.0 + args.retry_jitter_fraction * jitter_unit)
                restart_not_before[worker_index] = now + delay
                logging.warning(
                    "%s worker %d exited with code %d; relaunch %d scheduled in %.1fs. "
                    "Validated batches are retained and will be skipped.",
                    name,
                    worker_index,
                    returncode,
                    restart_number,
                    delay,
                )

            now = time.monotonic()
            for worker_index, deadline in list(restart_not_before.items()):
                if now < deadline:
                    continue
                del restart_not_before[worker_index]
                launch_worker(worker_index)

            display(args, all_names)
            time.sleep(args.refresh_seconds)
        display(args, all_names)

        logging.info("All %s workers finished; consolidating valid batch artifacts", name)
        completed = subprocess.run(consolidate_command(args, variant_root, model, effort), check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{name} consolidation failed with return code {completed.returncode}")
        timing = {
            "model": model,
            "reasoning_effort": effort,
            "started_at": started_at,
            "finished_at": now_iso(),
            "wall_seconds": round(time.time() - start_epoch, 3),
        }
        (variant_root / "controller_timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for log_handle in log_handles.values():
            log_handle.close()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit_questions < 0:
        raise ValueError("--limit-questions must be zero (all questions) or a positive integer")
    if args.refresh_seconds <= 0:
        raise ValueError("--refresh-seconds must be positive")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.max_worker_restarts < 0:
        raise ValueError("--max-worker-restarts must be zero or positive")
    if args.retry_backoff_seconds < 0 or args.worker_restart_backoff_seconds < 0:
        raise ValueError("retry and worker-restart backoff values must be non-negative")
    if args.worker_start_stagger_seconds < 0:
        raise ValueError("--worker-start-stagger-seconds must be non-negative")
    if not 0.0 <= args.retry_jitter_fraction <= 1.0:
        raise ValueError("--retry-jitter-fraction must be between 0 and 1")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    variants = args.variant or list(DEFAULT_VARIANTS)
    names = [name for name, _, _ in variants]
    try:
        for name, model, effort in variants:
            run_variant(args, name, model, effort, names)
        display(args, names)
        logging.info("All pilot configurations are complete: %s", args.output_root)
    except KeyboardInterrupt:
        logging.warning("Controller interrupted. Existing workers and completed batch artifacts were left intact.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
