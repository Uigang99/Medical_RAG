# Reusable medical-topic metadata

This is passage classification, not a clinical KG relation extractor, not a
question-need classifier, and not a validated expert-routing policy. The taxonomy
has 21 overlapping topics; the source text remains unchanged.

## Contract and reuse

- Read restored corpus passages, not questions, answer choices, or QA labels.
- Send one passage and 21 independent Noul questions in one JEV request.
- Keep every score. `primary_candidate` is only the highest-scoring candidate,
  not a verified primary label. No multilabel threshold is selected.
- Save source document ID, text hash, taxonomy version, exact definitions/model,
  raw API responses, token usage, and the full score vector.
- Do not change the KG or propagate a document label as the established topic of
  every proposition/entity in it. A proposition can reference its source metadata.
- Share the cache between pilot/full and Train/Test corpus documents. Exact
  title+passage duplicates reuse scores. No fuzzy or semantic deduplication.
- Cache each judgment by model, endpoint, exact state, category ID and question.
  Adding a category requires only that extra judgment per document; changing a
  definition invalidates that category; changing common instructions invalidates
  all affected categories. Adding a genuinely new semantic dimension still needs
  new labeling. Shared-source cache reuse does not make Train/Test independent.

Possible uses: corpus organization, soft retrieval diversification, candidate
expert assignment, and topic-stratified error analysis. These uses need no prior
commitment to a particular expert framework. They do NOT establish QA gains or
clinical applicability. Experts need not correspond one-to-one with categories.

## Run

Environment: `/home/user/Uiheon/.venv_vllm/bin/python`, `httpx` and `python-dotenv`.
The key is read from the existing environment or `/home/user/Uiheon/.env`;
never place a secret in a command or report.

Offline estimate (default input is the 68,247 restored official-Train documents):

```bash
cd /home/user/Uiheon/Medical_RAG
/home/user/Uiheon/.venv_vllm/bin/python scripts/classify_rag2_documents_jev.py --mode estimate
```

Paid pilot, 64 passages, 16 per source corpus:

```bash
/home/user/Uiheon/.venv_vllm/bin/python scripts/classify_rag2_documents_jev.py \
  --mode run --limit 64 --workers 4 --max-api-usd 0.10
```

This is a source-balanced diagnostic, NOT a representative estimate of 21-topic
accuracy. Inspect false additions, missed subjects, overly broad categories and
abstention needs before selecting a multilabel/routing policy. Rare topics may
need extra Train/validation examples. Do not select policies on TEST.

After accepting pilot quality, full Train corpus (same shared cache):

```bash
/home/user/Uiheon/.venv_vllm/bin/python scripts/classify_rag2_documents_jev.py \
  --mode run --limit 0 --workers 4 --max-api-usd 12 \
  --output-root results/rag2_jev_topics21_train_full_v1
```

Use `--source PATH` (repeatable) to select another restored JSONL explicitly.
`--text-field` defaults to `restored_text`; no silent fallback to old chunk text.
`--limit 0` means all unique document IDs, not all benchmark questions.
Outputs are `manifest.json`, `cohort.jsonl`, `labels.jsonl`, and `summary.json`.

## Cost, resumption, and limits

Official price checked 2026-10-20:
[JEV model documentation](https://docs.typesafe.ai/models): $0.042/M input tokens,
output free. This code pins `jev-1.14.0`, not a moving alias.

Estimate uses UTF-8 request bytes / 3..6, not an exact JEV tokenizer. After paid
calls, returned `usage.input_tokens` supplies the price calculation in the ledger.
It is an estimate of charges, not a retrieved account invoice. Existing 32-doc /
17-topic pilot used 47,557 input tokens ($0.001997 at the cited price); the new
taxonomy/input length differs, so don't extrapolate its latency or quality.

At implementation time, offline input-size estimates were:

| Input | Unique API states | Estimated input cost |
| --- | ---: | ---: |
| Source-balanced Train 64 | 64 | $0.0041–0.0083 |
| Restored Train 68,247 docs | 68,152 | $4.40–8.79 |

The local cumulative budget counts completed usage plus conservative reservations
for in-flight/uncertain requests. It is NOT a provider-enforced billing cap.
Reserve-before-send, a single-writer cache lock, no SDK retries, and durable raw
responses reduce accidental repeat charges. On HTTP/network errors, stop and
inspect; `--retry-uncertain` explicitly allows a possibly duplicate charge.
Identical reruns reuse completed calls. Additional budget can be supplied without
invalidating the semantic contract. No Git revision is required for resumption.

No GPU. Wall time depends on account limits and current API latency; progress and
ETA calibrate on fresh requests. Do not infer runtime from nominal model latency.
No paid request is issued by `--mode estimate` or the mock regression tests.
