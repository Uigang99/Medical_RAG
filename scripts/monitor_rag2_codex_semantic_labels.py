from __future__ import annotations

"""Show one global progress bar for parallel Codex semantic labelling workers.

Workers write only short batch status transactions to ``progress.sqlite``.  This
monitor sums those transactions across every dataset and worker, so it is safe
to run in a separate terminal while 2/4/8 workers are active.
"""

import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor global progress of parallel Codex semantic labelling.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument(
        "--wait-for-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for manifest/progress.sqlite to appear instead of failing immediately.",
    )
    parser.add_argument("--once", action="store_true", help="Print one global snapshot and exit.")
    return parser.parse_args()


def load_total(manifest_path: Path) -> tuple[int, int]:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = value.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid manifest: {manifest_path}")
    pairs = sum(int(item.get("pairs") or 0) for item in datasets.values() if isinstance(item, dict))
    batches = sum(int(item.get("planned_batches") or 0) for item in datasets.values() if isinstance(item, dict))
    if pairs <= 0 or batches <= 0:
        raise ValueError(f"Manifest has no planned pairs/batches: {manifest_path}")
    return pairs, batches


def status_snapshot(database_path: Path) -> tuple[int, int, int, int, str | None]:
    connection = sqlite3.connect(database_path, timeout=20.0)
    try:
        connection.execute("PRAGMA busy_timeout=20000")
        rows = connection.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(pair_count), 0) FROM batches GROUP BY status"
        ).fetchall()
        first_started = connection.execute("SELECT MIN(started_at) FROM batches WHERE started_at IS NOT NULL").fetchone()[0]
    finally:
        connection.close()
    counts = {str(status): (int(batch_count), int(pair_count)) for status, batch_count, pair_count in rows}
    completed_batches, completed_pairs = counts.get("completed", (0, 0))
    running_batches, _ = counts.get("running", (0, 0))
    failed_batches, _ = counts.get("failed", (0, 0))
    return completed_pairs, completed_batches, running_batches, failed_batches, first_started


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "warming up"
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def seconds_since(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        return max(0.0, (datetime.now().astimezone() - datetime.fromisoformat(timestamp)).total_seconds())
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    if args.refresh_seconds <= 0:
        raise ValueError("--refresh-seconds must be positive")
    manifest_path = args.output_root / "manifest.json"
    database_path = args.output_root / "progress.sqlite"
    while not (manifest_path.is_file() and database_path.is_file()):
        if not args.wait_for_start:
            raise FileNotFoundError(f"Waiting files are missing: {manifest_path}, {database_path}")
        print("Waiting for Codex workers to create progress files...", flush=True)
        time.sleep(min(args.refresh_seconds, 5.0))

    total_pairs, total_batches = load_total(manifest_path)
    completed_pairs, completed_batches, running_batches, failed_batches, first_started = status_snapshot(database_path)
    previous_pairs = completed_pairs
    progress = tqdm(
        total=total_pairs,
        initial=completed_pairs,
        desc="CodexSemanticLabels:global",
        unit="pair",
        dynamic_ncols=True,
        # tqdm's short-term rate estimate swings sharply because workers submit
        # results in 100-pair bursts.  The postfix below instead shows an ETA
        # from all completed work since the first worker began.
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
    )
    try:
        while True:
            completed_pairs, completed_batches, running_batches, failed_batches, first_started = status_snapshot(
                database_path
            )
            if completed_pairs < previous_pairs:
                progress.close()
                progress = tqdm(
                    total=total_pairs,
                    initial=completed_pairs,
                    desc="CodexSemanticLabels:global",
                    unit="pair",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
                )
            else:
                progress.update(completed_pairs - previous_pairs)
            previous_pairs = completed_pairs
            elapsed = seconds_since(first_started)
            rate = completed_pairs / elapsed if elapsed and elapsed > 0 else 0.0
            # Do not claim a numerical ETA until all eight workers have had a
            # chance to complete at least one batch.
            eta = (total_pairs - completed_pairs) / rate if completed_pairs >= 800 and rate > 0 else None
            progress.set_postfix_str(
                f"batches={completed_batches}/{total_batches}, running={running_batches}, "
                f"failed={failed_batches}, avg={rate:.2f} pair/s, ETA={format_duration(eta)}"
            )
            progress.refresh()
            if args.once or completed_pairs >= total_pairs:
                break
            time.sleep(args.refresh_seconds)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
