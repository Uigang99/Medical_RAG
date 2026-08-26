#!/usr/bin/env python3
"""Monitor concurrent/sequential Codex semantic-labelling jobs.

The labeler writes one atomically completed JSON artifact per batch, so this
monitor deliberately treats those artifacts (rather than terminal-progress
bars) as the source of truth.  It can therefore be started, stopped, and
restarted at any point without affecting the annotation workers.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = ("terra_xhigh", "terra_high", "terra_medium", "luna_high")
DEFAULT_RATE_WINDOW_BATCHES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor RAG² Codex semantic-labelling progress.")
    parser.add_argument("--root", type=Path, required=True, help="Parent directory containing one directory per run.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Run-directory names in execution order.",
    )
    parser.add_argument("--refresh-seconds", type=float, default=30.0, help="Refresh interval for --watch.")
    parser.add_argument(
        "--expected-pairs-per-run",
        type=int,
        default=0,
        help="Expected pair count for a run directory that has not been created yet (useful for sequential pilots).",
    )
    parser.add_argument("--watch", action=argparse.BooleanOptionalAction, default=True, help="Continuously refresh.")
    parser.add_argument("--clear", action=argparse.BooleanOptionalAction, default=True, help="Clear terminal between refreshes.")
    parser.add_argument("--recent-errors", type=int, default=3, help="Show at most this many non-empty batch errors per run.")
    parser.add_argument(
        "--rate-window-batches",
        type=int,
        default=DEFAULT_RATE_WINDOW_BATCHES,
        help="Use this many most-recent completed batches for current speed/ETA (default: 32).",
    )
    return parser.parse_args()


def parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def seconds_text(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "—"
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


@dataclass
class RunProgress:
    name: str
    root: Path
    exists: bool
    planned_pairs: int = 0
    planned_batches: int = 0
    complete_pairs: int = 0
    complete_batches: int = 0
    status_counts: Counter[str] | None = None
    started_at: datetime | None = None
    latest_completed_at: datetime | None = None
    completed_events: list[tuple[datetime, int, float]] | None = None
    errors: list[str] | None = None
    model: str = ""
    reasoning_effort: str = ""

    @property
    def percent(self) -> float:
        return (100.0 * self.complete_pairs / self.planned_pairs) if self.planned_pairs else 0.0


def inspect_sqlite(path: Path, max_errors: int) -> tuple[Counter[str], list[str], list[datetime]]:
    if not path.is_file():
        return Counter(), [], []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        connection.execute("PRAGMA busy_timeout=2000")
        status_counts = Counter(
            {str(status): int(count) for status, count in connection.execute("SELECT status, COUNT(*) FROM batches GROUP BY status")}
        )
        errors = [
            f"batch {batch_index}: {str(error).replace(chr(10), ' ')[:180]}"
            for batch_index, error in connection.execute(
                "SELECT batch_index, last_error FROM batches "
                "WHERE last_error IS NOT NULL AND last_error != '' ORDER BY batch_index DESC LIMIT ?",
                (max_errors,),
            )
        ]
        timestamps = [
            parsed
            for (value,) in connection.execute("SELECT started_at FROM batches WHERE started_at IS NOT NULL")
            if (parsed := parse_timestamp(value)) is not None
        ]
        connection.close()
        return status_counts, errors, timestamps
    except sqlite3.Error:
        return Counter(), [], []


def inspect_run(name: str, root: Path, max_errors: int) -> RunProgress:
    if not root.is_dir():
        return RunProgress(name=name, root=root, exists=False)

    manifest = read_json(root / "manifest.json") or {}
    datasets = manifest.get("datasets") if isinstance(manifest.get("datasets"), dict) else {}
    planned_pairs = sum(
        int(value.get("pairs", 0))
        for value in datasets.values()
        if isinstance(value, dict) and isinstance(value.get("pairs", 0), int)
    )
    planned_batches = sum(
        int(value.get("planned_batches", 0))
        for value in datasets.values()
        if isinstance(value, dict) and isinstance(value.get("planned_batches", 0), int)
    )

    complete_pairs = 0
    complete_batches = 0
    completed_events: list[tuple[datetime, int, float]] = []
    # A resumed run may use a read-only cache from another Linux account.
    # Count it as completed work, but let a batch written in the current run
    # override a same-index cached artifact.
    artifact_paths: dict[tuple[str, str], Path] = {}
    cache_root_value = manifest.get("completed_batches_root")
    if isinstance(cache_root_value, str) and cache_root_value:
        cache_root = Path(cache_root_value)
        for path in cache_root.glob("batches/*/batch_*.json"):
            artifact_paths[(path.parent.name, path.name)] = path
    for path in root.glob("batches/*/batch_*.json"):
        artifact_paths[(path.parent.name, path.name)] = path
    for path in artifact_paths.values():
        value = read_json(path)
        labels = value.get("labels") if value else None
        if not isinstance(labels, list) or not labels:
            continue
        complete_batches += 1
        complete_pairs += len(labels)
        completed_at = parse_timestamp(value.get("completed_at"))
        if completed_at is not None:
            elapsed_seconds = value.get("elapsed_seconds")
            elapsed = float(elapsed_seconds) if isinstance(elapsed_seconds, (int, float)) else 0.0
            completed_events.append((completed_at, len(labels), max(0.0, elapsed)))

    progress_database_pointer = read_json(root / "active_progress_database.json") or {}
    pointer_path = progress_database_pointer.get("path")
    progress_database_path = Path(pointer_path) if isinstance(pointer_path, str) and pointer_path else root / "progress.sqlite"
    status_counts, errors, started_times = inspect_sqlite(progress_database_path, max_errors)
    manifest_time = parse_timestamp(manifest.get("updated_at"))
    timestamps = started_times + ([manifest_time] if manifest_time else [])
    started_at = min(timestamps) if timestamps else None
    return RunProgress(
        name=name,
        root=root,
        exists=True,
        planned_pairs=planned_pairs,
        planned_batches=planned_batches,
        complete_pairs=complete_pairs,
        complete_batches=complete_batches,
        status_counts=status_counts,
        started_at=started_at,
        latest_completed_at=max((event[0] for event in completed_events), default=None),
        completed_events=completed_events,
        errors=errors,
        model=str(manifest.get("codex_model_request") or ""),
        reasoning_effort=str(manifest.get("codex_reasoning_effort") or ""),
    )


def recent_events(
    events: list[tuple[datetime, int, float]],
    rate_window_batches: int,
) -> list[tuple[datetime, int, float]]:
    """Return a recent rolling completion window, robust to prior slow runs."""
    ordered = sorted(events, key=lambda event: event[0])
    return ordered[-rate_window_batches:]


def run_rate(
    progress: RunProgress,
    now: datetime,
    rate_window_batches: int = DEFAULT_RATE_WINDOW_BATCHES,
) -> tuple[float | None, float | None]:
    """Return rolling completed-pairs/second and ETA.

    A resumable output directory can include an earlier one-worker period,
    failed workers, and an idle gap.  Using the last few dozen atomically
    completed batches estimates the speed of the workers that are active now.
    """
    events = recent_events(progress.completed_events or [], rate_window_batches)
    if not events:
        return None, None
    # Each artifact records its request duration.  Back-projecting the first
    # completion avoids overestimating rate during the first worker wave.
    session_started_at = min(completed_at.timestamp() - elapsed_seconds for completed_at, _, elapsed_seconds in events)
    elapsed = now.timestamp() - session_started_at
    if elapsed <= 0:
        return None, None
    completed_pairs = sum(pair_count for _, pair_count, _ in events)
    rate = completed_pairs / elapsed
    remaining = max(0, progress.planned_pairs - progress.complete_pairs)
    return rate, (remaining / rate if rate > 0 else None)


def status_text(progress: RunProgress) -> str:
    if not progress.exists:
        return "waiting"
    if progress.planned_pairs and progress.complete_pairs >= progress.planned_pairs:
        return "complete"
    counts = progress.status_counts or Counter()
    if counts.get("running", 0):
        return f"running ({counts['running']} batch)"
    if counts.get("failed", 0):
        return f"retry/failed ({counts['failed']})"
    return "starting"


def render(
    root: Path,
    variant_names: list[str],
    max_errors: int,
    expected_pairs_per_run: int = 0,
    rate_window_batches: int = DEFAULT_RATE_WINDOW_BATCHES,
) -> str:
    now = datetime.now().astimezone()
    runs = [inspect_run(name, root / name, max_errors) for name in variant_names]
    expected_pairs = sum(
        run.planned_pairs if run.exists and run.planned_pairs else expected_pairs_per_run
        for run in runs
    )
    complete_pairs = sum(run.complete_pairs for run in runs)
    complete_batches = sum(run.complete_batches for run in runs)
    planned_batches = sum(run.planned_batches for run in runs if run.exists)
    overall_percent = 100.0 * complete_pairs / expected_pairs if expected_pairs else None

    lines = [
        f"RAG² Codex semantic-labelling monitor  |  {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        (
            f"Overall: {complete_pairs:,}/{expected_pairs:,} pairs ({overall_percent:5.1f}%)"
            if overall_percent is not None
            else f"Overall: {complete_pairs:,} completed pairs (planned total initializing)"
        )
        + (f"  |  {complete_batches}/{planned_batches} completed batches" if planned_batches else ""),
        "",
        "run              state              completed              rolling rate   rolling ETA",
        "---------------- ------------------ ---------------------- --------------- -------------------",
    ]
    for run in runs:
        rate, eta = run_rate(run, now, rate_window_batches)
        if not run.exists:
            complete = "not started"
        elif run.planned_pairs:
            complete = f"{run.complete_pairs:,}/{run.planned_pairs:,} ({run.percent:5.1f}%)"
        else:
            complete = f"{run.complete_pairs:,} (plan initializing)"
        rate_text = f"{rate:.3f}" if rate is not None else "—"
        eta_text = seconds_text(eta)
        config = f" {run.model} {run.reasoning_effort}".strip()
        lines.append(f"{run.name:<16} {status_text(run):<18} {complete:<22} {rate_text:>15} {eta_text:>19}{('  [' + config + ']') if config else ''}")
        if run.errors:
            for error in run.errors:
                lines.append(f"  ! {error}")
    lines.extend(
        [
            "",
            f"Rate/ETA use only the latest {rate_window_batches} atomically completed batches; retained work from older/slow runs does not depress the estimate.",
            "Requested configurations run sequentially, so a multi-model overall ETA is unavailable until each model has an observed rate.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.refresh_seconds <= 0:
        raise ValueError("--refresh-seconds must be positive")
    if args.rate_window_batches < 1:
        raise ValueError("--rate-window-batches must be at least 1")
    try:
        while True:
            if args.clear and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(
                render(
                    args.root,
                    list(args.variants),
                    args.recent_errors,
                    args.expected_pairs_per_run,
                    args.rate_window_batches,
                ),
                flush=True,
            )
            if not args.watch:
                return
            time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        print("\nMonitor stopped; labelling workers continue unchanged.")


if __name__ == "__main__":
    main()
