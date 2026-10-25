# KEV six-role pilot, 2026-10-25

## Outcome

Five development prompt versions and one frozen confirmation run completed on GPU 1.
Version 2 was selected before confirmation. It is a candidate baseline, **not approved
for unattended full-corpus labeling**. No full-corpus labeling, QA evaluation, training,
or paid API calls were performed.

## Purpose, inputs, operation, outputs

Test whether individual standalone propositions can receive reusable information-role
labels without classifying every proposition pair. This is metadata classification,
not truth verification or proof of a relationship between propositions.

- Input: one original generated proposition, unchanged. No question, answer, options,
  KG, reference labels, source-document text, or neighboring proposition.
- Model: local `kev-9b` adapter and `Qwen3.5-9B-Base`, bfloat16, H200 GPU 1.
- Six separate choice tasks in a KEV request: DEFINE, DESCRIBE, EXPLAIN, ASSESS,
  METHOD, NORM. Each returns YES/NO/UNCERTAIN and probabilities; no free-form rationale.
- Several roles may be YES. NO means that information role is absent, not that the
  proposition is false. UNCERTAIN means unresolved ambiguity, not a negative label.
- Batch size: 48 decision rows. All six runs passed batch/single label parity;
  there were no OOM splits or batch fallbacks. Missing decisions: 0/3,312.

Multi-label design reflects overlapping roles; it has **not** been demonstrated to
be more accurate than a six-way single-label classifier. Neither a single-label
ablation nor source-document augmentation was tested.

## Data and generalization safeguards

Source: `results/rag2_kev_abc_train_input_pilot_v1/cohort.json`, an existing
official-train, source-stratified pair cohort. After exact-text deduplication:
92 development propositions and 92 confirmation propositions. Source documents and
normalized proposition texts do not cross the split. Similar propositions within a
split remain; six decisions on a proposition are not independent samples.

Development sources: textbooks 22, CPG 33, PubMed 13, PMC 24.
Confirmation sources: textbooks 23, CPG 25, PubMed 17, PMC 27.
This is not a uniform random sample of the full corpus. No final-test question or QA
answer was used. Its small size and pair-based selection limit population claims.

Assistant provisional references were written before KEV role outputs. Ambiguous
optional roles were frozen and excluded from quantitative scoring: 29/552 development
decisions and 18/552 confirmation decisions. References were not revised to improve
reported scores. They are **not independently verified human gold**.

Changes were made to general role definitions and presentation, not disease-specific
exceptions, proposition IDs, or memorized target answers. Every version's entire
92-output development set was read qualitatively. Version selection was frozen in
`selection.json` before producing confirmation labels. Confirmation was run once;
its errors were not used for another round of prompt tuning. Any next redesign must
use a new unseen confirmation set, rather than recycle this one as a held-out test.

## Version comparison

All rows below use the same 92 development propositions and 523 unambiguous role
decisions. Precision/recall concern predicted YES against provisional references,
not independently established semantic accuracy. Exact agreement ignores optional
boundary labels. Deltas use v1 as the baseline.

| Version | Change | YES precision | Delta | YES recall | Macro-F1 | Delta | Exact propositions |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | Initial functional definitions | 62.73% | 0.00 pp | 95.28% | 79.27% | 0.00 pp | 40/92 |
| v2 | Clarify role boundaries; exclude mere mention/history from METHOD | 70.63% | +7.90 pp | 95.28% | 83.40% | +4.12 pp | 51/92 |
| v3 | Generic symbolic examples in shared state | 42.15% | -20.58 pp | 88.68% | 64.72% | -14.55 pp | 10/92 |
| v4 | Move the same examples/instructions into question instructions | 57.67% | -5.06 pp | 88.68% | 71.23% | -8.04 pp | 34/92 |
| v5 | No examples; structured include/exclude criteria | 72.36% | +9.62 pp | 83.96% | 77.67% | -1.61 pp | 52/92 |

v5 improved precision slightly over v2 but omitted more valid assessment/definition
roles. v2 had the best balanced macro-F1 and qualitative behavior. No claim of
statistically significant prompt improvement is made from this small selected set.

The controlled v3-to-v4 presentation change reduced DEFINE false positives from
75 to 15 on 91 clear DEFINE decisions. This demonstrates input-format sensitivity;
it does not prove the internal mechanism responsible. Adding more examples or more
instructions did not monotonically improve the classification.

## Frozen v2 confirmation

92 new propositions, 534 unambiguous decisions: 96 true-positive, 29 false-positive,
4 false-negative YES decisions. Precision 76.80%, recall 96.00%, macro-F1 82.23%,
exact agreement 65/92 (70.65%). These are reference-agreement metrics only.

Baseline assigning DESCRIBE alone to every proposition, on the identical 92 items:
precision 62.35%, recall 53.00%, macro-F1 12.80%, exact agreement 52/92 (56.52%).
v2 absolute differences: +14.45 pp precision, +43.00 pp recall, +69.43 pp macro-F1,
and +13 propositions / +14.13 pp exact agreement. Always-NO has zero positive recall
and only 1/92 exact agreement; negative prevalence must not be mistaken for quality.

| Role | Reference positives | Predicted YES scored | Precision | Recall | 95% Wilson precision interval |
|---|---:|---:|---:|---:|---:|
| DEFINE | 7 | 11 | 63.64% | 100.00% | 35.38–84.83% |
| DESCRIBE | 53 | 66 | 80.30% | 100.00% | 69.16–88.11% |
| EXPLAIN | 15 | 18 | 77.78% | 93.33% | 54.78–91.00% |
| ASSESS | 11 | 13 | 69.23% | 81.82% | 42.37–87.32% |
| METHOD | 6 | 9 | 55.56% | 83.33% | 26.66–81.12% |
| NORM | 8 | 8 | 100.00% | 100.00% | 67.56–100.00% |

Intervals are descriptive item-level approximations; near-duplicate propositions
and shared source groups reduce effective independence. Eight successful NORM
positives do not establish corpus-wide reliability. v2 selected UNCERTAIN zero
times on 552 confirmation decisions; an abstain option alone is not a safeguard.

## Qualitative findings

All 92 confirmation predictions were read, not only reference mismatches.

Recurring errors despite new source documents:

- T030P1/P2: drug licensing duration classified as METHOD.
- T054P1: a patient's historical antihypertensive use classified as METHOD.
- T068P2: an observed catheterization result classified as METHOD.
- T028P1: variance explained by a prediction equation classified as causal EXPLAIN.
- T090P1: dye retention in a bacterial definition classified as EXPLAIN without a
  stated mechanism.
- T010P1/P2: relevant vascular-history items classified as DEFINE, while ASSESS missed.
- T014P1/P2: nearly parallel patient/examiner positioning statements receive different
  METHOD decisions.

Not every provisional-reference mismatch is a true error. T046P1 and T094P1 contain
frequency information alongside causation, so DESCRIBE is defensible. T070P1 includes
an explicit ALP threshold, making ASSESS defensible. T036P2 lists category members,
making DEFINE defensible. These findings were recorded separately; neither scores
nor frozen references were retrospectively changed.

Most straightforward anatomy/location, case observations, symptom descriptions,
and explicit simple cause/effect claims were handled sensibly. Remaining failures
involve role boundaries and over-attribution, not malformed output or GPU failure.
The data do not isolate model capacity from taxonomy ambiguity or prove a model
ceiling. They also do not show that propositions are intrinsically worse than source
sentences: that was not compared here.

## Decision and next use

The predeclared gate required pooled positive precision at least 90%, macro-F1 at
least 80%, each role precision at least 85% and recall at least 75%, at least eight
confirmation positives per role, and no recurring qualitative leakage errors.
Confirmation failed the precision, coverage and qualitative requirements. No
threshold was relaxed after observing the result. **Full-corpus expansion is stopped.**

Keep v2 as the fixed candidate baseline, not mandatory graph-routing metadata. The
most useful next design question is whether roles can be expressed as broader,
directly recognizable functions without demanding an artificial separation between
every observation, definition, and explanation. That is an ontology change, not yet
a validated fix. Test any redesign on new document-disjoint data and a broader random
corpus audit; do not repeatedly optimize against this confirmation set.

## Artifacts and validation

Local result root:
`/home/user/Uiheon/Medical_RAG/results/rag2_kev_information_roles_pilot_v1`

- `prompts/v1.json` through `v5.json`: exact prompt specifications.
- `reference_before_kev.json`: immutable preliminary references.
- `v*/development/`: manifests, input cohorts, requests, raw cached probabilities,
  predictions, quantitative summaries, and qualitative iteration reviews.
- `v2/confirmation/`: once-only confirmation and its qualitative audit.
- `selection.json`, `comparison.json`: frozen selection and compact results.
- `runner_executed.py`: exact pre-reporting-fix runner used for these six GPU runs.

Tracked selected prompt: `docs/kev_information_roles_v2.json` (identical contents to
the selected result specification). Entry point:
`scripts/pilot_rag2_kev_information_roles.py`.

Validation: exact 92-item preflight, Python syntax check, targeted unit tests for
multi-label behavior, reference exclusions, no-gold inputs, deduplication and split
leakage, plus actual six GPU runs. A final test exposed division by zero for a
synthetic reference set with no positive roles; the reporting function now returns
null macro-F1 in that case. This does not change any of the reported nonempty-positive
cohort metrics. Historical manifests pin the original runner; use a new output root
for a rerun after source changes rather than overwrite a completed experiment.

Total measured GPU-run wall time, including model loading per run: approximately
131 seconds across six runs, excluding reference preparation and qualitative review.
Per-run times were 14.9–24.5 seconds for 92 propositions / 552 decisions. They are
pilot timings, not a guaranteed full-corpus ETA. No long job remains running.
