# Official-train proposition-pair attention

Implementation of the agreed bounded architecture, now directly on full
official Train9000 / Val1000 at the user's request. No performance pilot or
tiny-set overfit gate is run by this command. Runtime engineering checks remain.
No 6545-question test evaluation is performed.

## Frozen scientific contract

- Input: `datasets/kg/rag2_official_train10000_top8_h200_throughput_v2`.
  Train MedMCQA6300 + MedQA2700; validation700 +300.
- Model: pristine `models/Llama-3-8B-Instruct`. No old TEST-trained LoRA or
  controller checkpoint. All Llama weights remain frozen.
- All usable exact source propositions from the cached Top8, ordered by
  document rank then original proposition-file order. The prompt uses stable
  `[P1]` identifiers around unmodified claim text. All arms use the same text.
- Input order: Evidence -> question STEM -> options -> assistant `Answer:`.
  Score the single-token valid option labels; no rationale or gold token is
  appended to the model input. This is MCQ answer-selection evaluation, not
  rationale generation or a directly comparable reuse of older LoRA results.
- Controller reads mean+last token representations at layer12 for each complete
  proposition and the question stem. Options and answer labels are never pooled.
- Shared canonical endpoint IDs form proposition adjacency. Role flags are
  receiver/sender `SS,SO,OS,OO`. No lexical merging/new retrieval/global neighbors.
- Controller: 128-dimensional projections, two graph-message rounds, MLP,
  four per-pair outputs bounded by `2*tanh`; head shared; zero output initialization.
- Apply biases to all tokens of proposition pairs in layers13-16 only. Preserve
  causal masking and ordinary attention elsewhere. No direct question/answer
  query bias. The known later question can condition earlier input processing;
  this is full-input conditioning, not strictly left-to-right prefix invariance.
- Only the small controller trains. Exact-text semantic features are not a
  guarantee of faithful semantic reasoning, and learned bias is not a truth label.

## Registered arms

`graph`: receiver/sender/question/neighborhood-conditioned pair readout.

`importance`: same GNN and parameter shapes, but receiver and pair-role readout
features are zeroed. Pairs with the same sender receive the same correction.
This isolates receiver-dependent coupling, **not all use of KG**.

`shuffled`: same pair model, separately trained on deterministic causal,
degree-preserving edge swaps. Text, edge count and role-vector multiset remain.
Some graphs cannot change. `shuffled/SHUFFLE.json` reports actual changed pairs
and unchanged questions; a weak shuffle is not strong structural evidence.

## Objective and selection

Per-question cross-entropy over valid answer-label logits, plus `0.001` times
that question's mean squared pair bias. No external relevance labels or
teacher-generated rationale supervision. The loss backpropagates through
the upper frozen Llama into the controller; freezing does not mean no_grad there.

AdamW lr3e-4, decay0.01, 5%warmup/cosine, clip1, effective batch16, max3epochs,
seed42, BF16 frozen Llama/FP32 controller. Select highest validation Macro2,
then lowest answer CE, then earliest epoch, including the uncorrected baseline.
Macro2 equally weights MedMCQA and MedQA. No MMLU or test selection is used.

The initial TEXT validation uses exactly zero correction with the same block
attention backend. Its accuracy is the uncorrected baseline, but its runtime
includes controller/block overhead and is not ordinary TEXT-RAG latency.

Safety checks compare the manual decoder against the native Llama decoder loop
with independently constructed zero masks and the same chunk schedule. Native
unchunked BF16 hidden/probability differences are recorded separately; native
answer parity must still pass. This separates implementation parity from
shape-dependent kernel rounding without loosening the existing tolerances or
bypassing controller gradients. GPU parity must be checked on the running GPU.

The default runs all three trainable arms sequentially on GPU1. Each arm is
27,000 training example passes and 3,000 validation passes at max3epochs;
the initial 1,000-question TEXT validation is shared. No claim of improvement
is made until the saved validation/test comparisons are actually available.

## Memory, speed and resumability

No second LLM/encoder forward; no full vocabulary logits for every input token.
Ordinary layers use Flash-compatible SDPA. Upper layers use non-reentrant
checkpointing. Controlled blocks are built in query chunks with head-shared
biases, rather than a full batch/head/length/length overlay. Bias-gradient
chunks may require MATH attention and are an actual runtime cost.

Length-grouped effective batches use adaptive microbatches2..16 by length bucket
and target88% of the available device capacity. Confirmed OOM retries the same
entire optimizer window at a lower microbatch, retaining effective batch16.
Cooldown avoids repeatedly retrying an unsafe size; later probes recover larger
sizes. Optimizer-step failures are not retried in-memory. Resume from `last.pt`.

Checkpoints every10 updates and epoch boundaries include controller, optimizer,
scheduler, RNG, exact epoch/cursor, running metrics and adaptive bucket state.
Completed validation files are hash checked and reused. Scientific-contract
changes require a new output root; requested arm subset and reporting frequency
are execution scope, not different learned-model contracts.

No compatible H200 wall-time measurement exists. The first128 fresh training
examples calibrate training ETA; validation is measured separately. Shared
frozen-Llama compute supplies initial duration estimates for pending arms,
which are recalibrated on their own fresh work. All operations remain visible
in the persistent two-line progress footer. Do not infer speed from VRAM alone.

## Commands

Metadata-only exact-path check (no GPU/tokenization/output writes):

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_proposition_kg_attention_training.sh --preflight-only
```

Full training of the three registered arms, no test evaluation:

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_proposition_kg_attention_training.sh
```

If only the proposed graph arm should run first, without changing its settings:

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_proposition_kg_attention_training.sh --arms graph
```

Run the other controls later in the same root:

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_proposition_kg_attention_training.sh --arms importance shuffled
```

`--prepare-only` builds the complete CPU input cache without model loading.
After a trainer/attention-core numerical fix, a new output root may read an old
root's prepared cache with `--reuse-prepared-from <old-output-root>`. The donor
manifest and rows hashes are checked; data-building code, inputs and all
non-code settings must match. Only trainer/core code hashes may differ. No old
training checkpoint is reused, and donor files are never rewritten. Keep this
flag on subsequent resumes of that new root.
Default output: `results/rag2_official_train_proposition_attention_v1/`.
`manifest.json`, `prepared/COMPLETE.json`, `SAFETY.json`, `TEXT_validation.jsonl`,
`{arm}/last.pt`, `{arm}/best.pt`, `{arm}/history.json`, per-epoch validation
predictions, `{arm}/COMPLETE.json`, and `TRAINING_SUMMARY.json` preserve provenance.
Final test evaluation needs a separate frozen evaluator using this exact prompt,
choice scoring, base checkpoint and selected controller; it is not in this command.
