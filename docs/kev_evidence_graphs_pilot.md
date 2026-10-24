# KEV topic / claim / information-role pilot

Purpose: test whether structured decision annotations of the **same fixed Top-8
source propositions** help final QA. This is an exploratory subset of TEST6545,
not an independent final benchmark. No training or new retrieval is performed.
Do not tune labels, thresholds, checkpoints or prompts using its accuracy.

## Execution

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_kev_evidence_graphs_pilot.sh --preflight
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_kev_evidence_graphs_pilot.sh
```

Default: 64 questions, seed 20261024, dataset-proportional sampling independent
of correctness. Original `cohort.jsonl`, not the context-pruned effective cohort,
is used. The prompt order is question → options → original propositions → optional
graph annotations. Empty document slots and missing KG mappings are reported;
they do not remove questions or source propositions.

The default output is
`/home/user/Uiheon/Medical_RAG/results/rag2_kev_evidence_graphs_test64_v1`.
The identical command resumes matching artifacts. Changed contracts need a new
`--output-root`. `--report` recomputes a partial or complete paired summary
without loading models. `--preflight` is CPU-only and writes no run artifacts.

## Scientific contract and stages

1. Recover source proposition and canonical entity IDs from the read-only KG.
   Keep document provenance and original triple surfaces. Gold is stored
   separately and only loaded by reporting, never by model workers.
2. A (topic): KEV chooses YES/NO/UNCERTAIN for proposition ABOUT entity.
   B (claim): KEV judges A⇒B, B⇒A, incompatibility separately. Code assigns
   EQUIVALENT, directed ENTAILS or CONFLICT_CANDIDATE. Conflicting decisions
   are withheld. No transitive closure or component-based equivalence merging.
   C (role): seven fixed roles, multilabel, relative to a particular entity.
   Role edges require an accepted ABOUT edge. Probabilities are not validated
   correctness. The default .70 confidence/.15 margin are **untuned pilot
   abstention settings**, not calibrated medical thresholds.
3. Construction is question-independent for a selected source pair. To bound
   cost, classify at most two nongeneric entities per proposition and propose
   at most two claim neighbors per proposition within each question's Top-8.
   Shared rare entities or lexical Jaccard ≥.25 produce candidates, not labels.
   All unexamined pairs remain unassessed, not negative.
4. Question expert: KEV sees the question stem, not benchmark choices, selects
   relevant entities from at most 16 candidates and requested information roles.
   Code groups up to four propositions around an accepted entity, at most two
   groups. Role coverage and directly predicted duplicate edges affect grouping;
   same-role distinct facts are permitted. No source proposition is deleted.
5. Group expert: KEV selects JOINT / SEPARATE / NOT_RELEVANT / UNCERTAIN.
   No expert writes a new medical conclusion. JOINT/SEPARATE groups plus at most
   24 directly predicted claim relations become advisory ID-based annotations.
6. Llama-3-8B-Instruct runs two matched conditions: **text** vs **graphs**.
   Same evidence order, model, decoding, rationale budget and JSON answer schema.
   Only graph annotations differ. No sentence stopping, correction loop or
   forced number of rationale statements. Both arms use constrained JSON with
   natural rationale and an option-letter answer. Open-ended answer schema is
   supported by the core, but this launcher evaluates the existing MCQ cohort.

No verified local MeSH hierarchy was available. Default A is therefore an
**ABOUT graph, not a MeSH hierarchy experiment**. No parents are invented by KEV.
Optional `--ontology /absolute/path/file.json` accepts externally verified data:

```json
{
  "source": "official ontology release identifier/URL",
  "version": "release",
  "nodes": [{"id": "T1", "label": "topic"}, {"id": "T2", "label": "broader topic"}],
  "mappings": [{"entity_id": "existing KG entity_node_id", "topic_id": "T1"}],
  "broader": [{"child": "T1", "parent": "T2"}]
}
```

Identifiers must already be verified; this loader is not an ontology mapper.
The presence of external hierarchy information is a separate experiment contract.

## Performance and diagnostics

KEV classifies independent requests in length-sorted batches, default 32
decision rows; eight questions in one request count as eight rows. Repeated
proposition/entity and proposition/pair requests are cached. Batch/single
probability parity is checked at startup; failures use single-request forwards.
OOM divides uncommitted batches. A single-request OOM stops with durable caches.
The KEV subprocess exits before vLLM starts: Llama can use 85% of GPU1, with up to
16 concurrent generation requests. No new paid API calls or model downloads.

There is no measured compatible throughput for these new labeling tasks. The
actual sampled request counts are printed by preflight; stage ETA calibrates
from fresh work. Do not extrapolate the old sentence-feedback questions/second
to this graph construction workload. A full-TEST run is not supported by this
bounded launcher.

Outputs: manifest, immutable sampled cohort, common graph, per-question A/B/C
graphs, raw KEV request/probability records, expert requests/groups, paired raw
rationale/answers, machine-readable summary and summary.md. `annotation_audit.jsonl`
provides stratified task/label examples with blank human labels. It does not
claim KEV classification accuracy without independent annotations.

Report paired correct→wrong / wrong→correct, accuracy by dataset and pooled MMLU,
macro-3, parse failures, missing questions, and paired bootstrap uncertainty.
An absolute gain of 3.125 percentage points is the predeclared minimum worthwhile
pilot gain (two net questions at N=64), not a significance threshold or permission
to scale. Inspect annotation correctness and independently confirm on validation.
This comparison identifies the effect of the combined annotation/expert method,
not graph topology in isolation. No historical baseline score is mixed into it.
