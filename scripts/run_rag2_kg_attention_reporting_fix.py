#!/usr/bin/env python3
"""Reporting-only compatibility entry point for the frozen attention pilot.

Keep the original hashed training/generation source unchanged so existing
checkpoints and prediction shards remain verifiable. Only convert the summary
gate's NumPy boolean to a JSON-compatible Python boolean.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rag2_kg_attention_pilot as pilot

_original_summarize = pilot.summarize


def summarize(rows, predictions):
    result = _original_summarize(rows, predictions)
    result["exploratory_gate"]["passed"] = bool(result["exploratory_gate"]["passed"])
    return result


pilot.summarize = summarize


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--cached-only" not in argv:
        return pilot.main(argv)
    argv.remove("--cached-only")
    a = pilot.arguments(argv)
    if a.mode != "evaluate" or a.prepare_only or a.preflight_only:
        raise ValueError("--cached-only requires evaluate without preflight/prepare flags")
    progress = pilot.Progress([
        "CPU split/model contracts", "map KG to exact text tokens",
        "reuse cached predictions", "zero-bias parity 16",
        "reuse graph attention MCQ 512", "paired summary",
    ])
    try:
        progress.stage(0, 1, unit="contract")
        progress.log("[plan] CPU-only cached finalization; no model loading, training or generation")
        with pilot.lock_file(a.output_root / ".run.lock"):
            # Check existence before input preparation; evaluate() additionally
            # verifies every shard's IDs, contract and content checksum.
            required = [a.output_root / "predictions" / f"{i:04d}.json" for i in range(32)]
            required += [a.output_root / name for name in ("manifest.json", "ZERO_PARITY.json")]
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise RuntimeError(f"Cached-only finalization refused; missing artifacts: {missing}")
            c = pilot.prepare(a, progress)
            run_hash = pilot.ensure_contract(a.output_root / "manifest.json", c["contract"])
            # All 512 predictions must already exist. evaluate's all_cached
            # branch skips GPU loading and retains the original validation.
            pilot.evaluate(a, c, run_hash, progress)
    except BaseException as exc:
        progress.log(f"[FAILED] cached finalization: {type(exc).__name__}: {exc}; "
                     "original checkpoints/prediction shards retained")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()
