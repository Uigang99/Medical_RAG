# Common proposition-pair candidates (no KEV decisions)

The immutable input is the **unpruned original** TEST6545 Top-8 cohort. Its
283,462 unique proposition IDs, exact claims, document provenance and original
proposition-list positions are retained. Question text/options/gold are not used
or exported. All candidate pairs must co-occur in at least one original Top-8.
No all-corpus Cartesian product, new retrieval, QA generation or training occurs.

## Selection contract

Each proposition proposes up to:

| Channel | Default quota | Eligibility / ranking |
|---|---:|---|
| Shared entity | 1 | Shared canonical entity, non-generic; sum of entity document-IDF, then cosine |
| Direct KG | 1 | Entity endpoints joined by one source triple from the SAME Top-8; endpoint rarity, then cosine |
| Semantic | 2 | MedCPT Article CLS cosine >= 0.70; reserve one slot for an eligible cross-document neighbor |
| Document adjacency | 2 | Previous/next proposition in its original document list; no reconstructed source sentences |

An entity appearing in more than 1% of unique retrieved documents cannot alone
justify a shared-entity proposal: cosine must also pass the semantic floor.
A direct KG bridge must have two non-common endpoints, or pass that floor.
The existing `canonical` identities are used; lexical/local IDs are not silently
promoted to cross-document equivalences. Semantic and adjacency channels retain
access to propositions with missing KG mappings. KG direction/relations are
recorded, NOT interpreted as a logical entailment or propagation rule.

This is a union, not an intersection or mutual-nearest-neighbor rule. Either
endpoint proposing a pair is sufficient. Quotas are outgoing proposals, NOT a
maximum undirected degree. No graph connectivity is forced; isolated propositions
are preserved. One pair is selected once globally but keeps per-question reasons
and bridge witnesses. A bridge witness available for one question must not be
silently imported into another question's evidence.

Quotas, 0.70 cosine and 1% frequency are **untuned starting settings**, not
validated cutoffs or probabilities. Semantic similarity is a cheap candidate
signal, not a certified contradiction/explanation classifier. C relations may
be semantically dissimilar, hence the independent structural/local channels.
Candidates are shared by A/B/C; no A -> B -> C acceptance cascade is performed.

## Encoding and operational boundaries

Frozen local MedCPT Article Encoder runs on physical GPU1 only. Each unique
proposition ID is encoded once (same wording with different IDs retains its
provenance). All content tokens are retained using 512-token BERT windows;
content-length-weighted window CLS means are L2-normalized. This retains token
coverage, **not all cross-window semantic interactions**. Exact original text
remains in the inventory for later KEV decisions.

Encoder batches are length-sorted with a 32,768 padded-token budget and maximum
512 windows; OOM retries only the uncommitted batch at a smaller size. No CPU or
different-encoder fallback changes the contract. The CPU scores small per-Top-8
cosine matrices rather than the 283k-by-283k matrix. KEV/Llama are never loaded.
The encoder is not trained or fine-tuned here.

## Commands

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_common_pair_candidates.sh --preflight
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_common_pair_candidates.sh
```

Preflight is read-only: paths, source hash, schema, first-row layout, encoder
configuration, output contract and free space. It does not scan the whole cohort,
tokenize it or load a model. Execution verifies exact 6,545 / 283,462 counts
before encoding. Full inventory and KG source text consistency failures stop
before GPU allocation.

Default output: `results/rag2_common_pair_candidates_test6545_v1/`.

- `manifest.json`: versioned source/model/code/settings/environment contract.
- `candidates.sqlite3`: durable inventory, KG mapping, global pairs and per-question selection.
- `embeddings.npy`: normalized float32 vectors in inventory index order.
- `propositions.jsonl`: original text and source for every unique proposition.
- `pairs.jsonl`: globally unique pairs; `graphs_to_evaluate=[A,B,C]`; no relation labels.
- `question_candidates.jsonl`: per-question selection reasons, cosine, shared
  entities, local bridge triples, and rejected-pair audit samples.
- `summary.json`: reduction from 8,473,367 unique co-occurring pairs; overlapping
  channel counts, cross-document pairs, isolated/missing KG coverage. Counts
  measure candidate coverage, not relation recall or QA performance.
- `RUNTIME.json`: compatible stage timings. New-stage rates calibrate separately;
  there is no defensible end-to-end ETA before those operations are measured.

Rejected pairs are sampled separately from same-/cross-document strata, recording
population sizes and sampling probabilities. They are **unassessed**, not negative
labels. Later relation-recall audits require independent adjudication, including
rejected pairs; candidate-only KEV judgments do not establish recall. Select
thresholds using train/development relation audits, not TEST answer accuracy.

An exclusive lock prevents simultaneous jobs corrupting one root. Manifest
changes require a new output root; source artifacts are read-only. KG imports
commit every 300 propositions; embeddings every 2,048 propositions; candidate
selection once per question. Identical commands resume these boundaries.
Exports are atomic and can be rebuilt from the durable SQLite database.
