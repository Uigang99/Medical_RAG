# One-shot JEV A/B/C proposition-pair quality pilot

## Purpose and fixed setup

Test whether the paid JEV decision model can replace KEV for the same three proposition-pair graph judgments before any full-corpus labeling. The sample is the **same 48 unique, stratified pairs** from the KEV pilot: 24 development and 24 document-disjoint confirmation pairs. It is not a random sample of all candidate pairs. The reference is one assistant's adjudication made before seeing KEV outputs, **not** independently verified clinical ground truth.

Each pair's two unchanged proposition texts and a same-source-document boolean were sent once to the pinned `jev-1.14.0` model. One request contained 22 independent judgments: A topical relationship (one four-way Choice), B bidirectional entailment and contradiction (three Nouls), and C ten information-role relation types with their direction (18 Nouls). Shared instructions forbade importing missing medical facts and treating shared entities or a shared document as proof. B/C `noul >= 0.5` was the frozen positive rule. The prompt, questions, threshold and cohort were not changed after inspecting JEV outputs. One response with rounded Choice probabilities summing to 0.99 required a **local parser tolerance correction only**; its already-saved response was reused, never paid for twice.

The local call ledger contains **48 completed requests, 0 missing/uncertain requests, 148,697 billable input tokens**. At the provider's published $0.042 per million input tokens, the usage-based charge estimate is **$0.006245**; the account invoice itself was not inspected. [Model/price](https://docs.typesafe.ai/models), [API](https://docs.typesafe.ai/api/system-one).

## Paired comparison with selected KEV v7

`A` is macro-F1 across evaluated topical classes. `B` and `C` are *positive-edge* TP/FP/FN against the provisional reference; raw decision agreement is misleading because negatives dominate. Reference-UNCERTAIN decisions were excluded (A 1, B 3, C 6 across all 48), not counted as correct. The identical 24 pairs and reference decisions are used within each split.

| Split (pairs) | Family | Evaluated decisions | KEV v7 quality | JEV quality | JEV TP/FP/FN |
|---|---|---:|---:|---:|---:|
| Development (24) | A | 23 | 0.591 | 0.758 | — |
| Development (24) | B | 69 | 0.500 | 1.000 | 1/0/0 |
| Development (24) | C | 427 | 0.526 | 0.706 | 6/2/3 |
| Confirmation (24) | A | 24 | 0.550 | 0.697 | — |
| Confirmation (24) | B | 72 | 0.500 | 0.000 | 0/0/1 |
| Confirmation (24) | C | 431 | 0.250 | 0.167 | 1/3/7 |

On all 48 pairs, JEV A macro-F1 was 0.720 on 47 evaluated topical decisions; B positive precision/recall was 1/1 and 1/2 respectively (TP 1, FP 0, FN 1) across 141 evaluated decisions; C positive precision/recall was 7/12 = 0.583 and 7/17 = 0.412 (TP 7, FP 5, FN 10) across 858 evaluated decisions. In confirmation, C positive precision was **1/4 = 0.25** and recall **1/8 = 0.125**. These numerators are too small for a stable full-corpus quality estimate. There were zero internally inconsistent B cases, but that does not cure the missed positive entailment.

The predeclared pilot gate from the KEV report was A macro-F1 >= 0.75, B/C positive precision >= 0.85 and recall >= 0.60, and no inconsistent B cases. **JEV failed that gate on confirmation.** Unlike KEV, JEV improved A macro-F1, but it did not give a reliable C graph. No full-corpus JEV labeling was launched.

## Qualitative check and plausible failure mechanism

Correct C detections include P03 (`LIMITS:21`: a diuretic is appropriate only when rest does not relieve edema), P19 and P23 (`CONTRASTS`: explicitly different ages/doses or risks). These show JEV can recognize directly stated comparison/scope links.

The largest weakness is relation recall, especially across documents. On 24 same-document pairs, C had TP/FP/FN = **6/1/4**; on 24 cross-document pairs, **1/4/6**. This is descriptive of the stratified sample, not a population estimate. P40 missed an explicit temporal sequence (`PRECEDES:21`: an infant's bout begins irregularly, then has a second phase). P48 missed the mechanism linking medullary NaCl accumulation to water reabsorption and hyperosmotic urine (`EXPLAINS:21`). Several other missed `EXPLAINS`/`QUALIFIES` examples require combining the two claims rather than checking for a near-verbatim relation. The broad instruction against unstated premises may contribute to this conservatism, but the pilot does **not** isolate that as the cause.

False positives expose a separate boundary: P13 and P14 received bidirectional `EVIDENCE_FOR` edges for near-rephrased anatomical/biochemical facts, although the provisional reference treated repetition/entailment as distinct from evidence. P04's `EVIDENCE_FOR:12` also appears to turn a direct comparison of anesthetic doses into evidence for a different claim. These errors matter because five false C edges among 858 assessed predicates are not a validated full-corpus false-positive rate; the sample was selected to include candidate pairs, not representative negatives.

Reference ambiguity must also be kept visible. In P14 the reference says “cavernous sinus contains the internal carotid artery” entails “the artery passes through the cavernous sinus”; JEV says no. A strict whole-claim entailment reading can reasonably reject that implication. P37's `EXEMPLIFIES:21` was excluded from evaluation as reference-UNCERTAIN. Therefore not every disagreement is demonstrably a JEV error. Rewriting the reference after seeing JEV outputs would bias this comparison; it was not done.

## Decision

The one-shot JEV replacement is **not ready** to label the full candidate-pair set, primarily because positive C relationships did not survive the independent 24-pair check. The next meaningful quality gate is a larger, independently dual-adjudicated challenge set with positive and hard-negative examples for each retained relation, especially cross-document `EXPLAINS`, `QUALIFIES`, `PRECEDES` and `EVIDENCE_FOR`. That is a data-quality recommendation, **not authorization for another paid prompt iteration**. Keep the frozen JEV probabilities and references for later analysis; do not tune a threshold on these 48 pairs and call it held-out success.

Artifacts: `results/rag2_jev_abc_quality_once_v1/manifest.json`, `requests.jsonl`, `raw/`, `cost_ledger.json`, `predictions.jsonl`, split disagreement files and `summary.json`.
