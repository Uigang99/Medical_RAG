#!/usr/bin/env python3
"""V3 prompt/schema on the exact frozen V2 cohort; reuse the existing runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pilot_rag2_evidence_planning as base
from medrag import evidence_planning_v3 as task

ORIGINAL_PARSER = base.args_parser
PREVIOUS = base.ROOT / "results/evidence_planning_test64_llama_v2"


def args_parser():
    parser = ORIGINAL_PARSER()
    parser.description = __doc__
    parser.add_argument("--previous-run", type=Path, default=PREVIOUS)
    return parser


def load_inputs(args):
    previous = args.previous_run
    if previous.resolve() == args.output_dir.resolve():
        raise RuntimeError("V3 must not overwrite the previous pilot; use a new output directory")
    manifest_path = previous / "experiment_manifest.json"
    contract = json.loads(manifest_path.read_text())["contract"]
    cohort = list(base.iter_jsonl(previous / "cohort.jsonl"))
    report = json.loads((previous / "sampling_summary.json").read_text())
    if contract["version"] != "evidence_planning_test64_llama_v2":
        raise RuntimeError("Expected the frozen V2 planning pilot")
    if task.digest(cohort) != contract["cohort_sha256"]:
        raise RuntimeError("Previous cohort differs from its immutable contract")
    if args.questions != len(cohort) or args.questions != contract["questions"] or args.seed != contract["seed"]:
        raise RuntimeError("V3 comparison must preserve previous question count and seed")
    if len({r["sample_id"] for r in cohort}) != len(cohort):
        raise RuntimeError("Duplicate frozen question IDs")
    if any([d["rank"] for d in r["documents"]] != list(range(1, 9))
           or any(not d["text"].strip() for d in r["documents"]) for r in cohort):
        raise RuntimeError("Frozen cohort must contain exactly eight nonempty ordered documents")
    model = contract["identity"]["model"]
    if args.model.resolve() != Path(model["path"]).resolve():
        raise RuntimeError("V3 comparison must use the same Llama checkpoint")
    for filename, key in (("config.json", "config_sha256"), ("tokenizer_config.json", "tokenizer_config_sha256")):
        if base.sha(args.model / filename) != model[key]:
            raise RuntimeError(f"Checkpoint metadata changed: {filename}")
    for weight in model["weights"]:
        if (args.model / weight["name"]).stat().st_size != weight["size"]:
            raise RuntimeError(f"Missing/incomplete checkpoint weight: {weight['name']}")
    for arg in ("batch_size", "max_output_tokens", "max_model_len", "gpu_memory_utilization"):
        if getattr(args, arg) != contract["generation"][arg]:
            raise RuntimeError(f"Keep --{arg.replace('_', '-')}={contract['generation'][arg]} for this controlled comparison")
    identity = copy.deepcopy(contract["identity"])
    identity["planning_revision"] = {
        "previous_run": str(previous.resolve()), "previous_manifest_sha256": base.sha(manifest_path),
        "cohort_reused_verbatim": True,
        "changed_variables": ["planning prompt", "output schema", "structural parser"],
        "interpretation": "Joint revision diagnostic; no individual-change attribution or held-out quality claim",
        "driver_sha256": base.sha(Path(__file__)),
        "inherited_helpers_sha256": base.sha(base.ROOT / "medrag/evidence_planning.py"),
    }
    return cohort, report, identity


def main():
    # The original module and entry point remain unchanged for the V2 run.
    base.task = task
    base.args_parser = args_parser
    base.load_inputs = load_inputs
    base.main()


if __name__ == "__main__":
    main()
