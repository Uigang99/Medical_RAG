# Question-only PCED: direct label versus rationale + label

## Purpose and fixed inputs

Use the existing Llama-3-8B-Instruct, without LoRA or training, and the immutable
question-only full-corpus `balanced_ce.jsonl` Top-32 retrieval cache. Every
question uses exactly `documents[:8]`, in cached CE rank order. No options,
rationale, gold, new retrieval or additional reranking enters retrieval.
The default evaluation is all 6,545 TEST questions across the eight benchmarks.
This is a baseline implementation, not a TEST-set model/prompt selection run.

## PCED algorithm and declared adaptations

One amateur receives question/options; eight experts each receive the same
question/options and one original chunk (including title). Model weights are
shared but KV caches are independent. The prompt layout is question, options,
evidence. Direct mode uses the existing question-only baseline's prompt wording.

At the first next-token position, compute the mean Jensen–Shannon divergence
between each expert's and the amateur's full-vocabulary distribution (natural
log), following the existing `RationaleAnswerPcedGenerator.jsd_beta` code. Keep
this beta fixed for the entire question in that mode. At every selection use:

`score[k, token] = (1+beta)*expert_logit - beta*amateur_logit + 2.5*log(prior[k])`

Take max over experts for each token, then max over tokens. Do not average
expert probabilities, vote, update priors using history, or use option/gold
feedback. Ties prefer the first token ID and then first CE-ranked expert.

The PCED paper ([v1](https://arxiv.org/html/2601.08670v1)) used BGE retrieval
normalization. This experiment retains MedCPT raw inner-product retrieval.
With the user's approval, normalize IP by min–max over that question's **same
cached 32 documents**, sigmoid raw CE scores, and take the harmonic mean.
Clip normalized inputs/prior with epsilon 1e-8. Equal IP scores receive equal
neutral dense weights (1 before clipping). Do not mislabel raw IP as cosine.
This is a **MedCPT adaptation**, not a paper-exact BGE reproduction.

Direct mode performs one PCED competition restricted to the four space-prefixed
label tokens A/B/C/D. Rationale mode performs full-vocabulary PCED generation,
appending each selected token to **all nine streams**, until a natural
`Final answer:` marker. It then uses the same PCED formula restricted to labels.
On EOS or 512 rationale tokens, do not commit EOS: append the formatting-only
`\nFinal answer:` marker to every stream and select a label. The forced-marker
count and finish reason are recorded. No answer/rationale is rewritten, parsed
heuristically, or regenerated with a new prompt; the cached history is retained.

Changing output instructions necessarily changes initial beta and compute.
Thus the contrast tests the two usable output modes, **not solely the causal
effect of expert switching or shared-token recurrence**. No precomputed document
KV is available; do not transfer the paper's offline-prefix TTFT claims.

## Execution and resource use

One frozen model loads once on GPU 1. Prefill evaluates only last-token logits,
avoiding the enormous full-prefill vocabulary tensor. Several questions and all
their experts are batched. Correct left-padding position IDs are supplied.
During rationale generation, completed questions and their KV are removed;
remaining caches are reused. Length buckets reduce padding. The rationale batch
is estimated from free VRAM and KV size (including maximum decode length), and
OOM retries halve an uncommitted batch without changing tokens or documents.
Direct mode keeps no KV. `--max-batch-questions` caps logical questions, not the
number of experts: 32 questions means 288 streams.

VRAM occupancy alone is not efficiency. Reserve activation/allocator headroom;
never promise 100% compute utilization. No valid new-run H200 throughput exists
yet. `runtime.json` records fresh seconds/question separately per mode; direct
stage ETA is available first, and overall ETA becomes available after rationale
calibration. Do not extrapolate old multi-arm/historical-decoding timings.

## Validation, contracts, outputs

CPU preflight validates the exact 6,545 benchmark/cache identities, all 32 ranks
and score fields, one-token option labels, exact prompt lengths and context
limits **before model loading**. No silent truncation or dropped questions.
Model/tokenizer/input/source hashes, weight identities, prompts, algorithm,
runtime settings and cohorts are fixed in the versioned manifest. Different
contracts require different output directories. `--limit N` is a deterministic
hash-selected smoke cohort, not a tuned/easier subset, and defaults to its own
output directory. Prediction JSON is atomic per question/mode; identical reruns
reuse valid records. An interrupted batch may be recomputed, not deleted.

`summary.json`: exact N, both accuracies by dataset and overall, paired absolute
delta and bootstrap 95% CI, C->C/C->W/W->C/W->W counts and rates, finish reasons,
mean rationale length and expert-switch count. `paired_predictions.jsonl` allows
recomputation. Each raw record retains beta, priors, selected expert ranks,
shared rationale token IDs, document IDs and final-label scores. Gold is read
only for post-generation evaluation, never passed to the decoder.

Reference No-RAG and single-document results are not included as controlled
baselines here: those used different inference paths, and single-document RAG
is not joint Top-8 Base-RAG. This requested comparison contains only the two
PCED output modes; a controlled RAG-effect comparison would need matched new
baselines, not silently reused historical numbers.

No training or full GPU evaluation is started when supplying these commands.

```bash
# Optional exact CPU preflight; no GPU/model loading.
bash /home/user/Uiheon/Medical_RAG/scripts/run_question_only_pced.sh --preflight-only

# Two full conditions, sequentially, one model load, resumable.
bash /home/user/Uiheon/Medical_RAG/scripts/run_question_only_pced.sh
```
