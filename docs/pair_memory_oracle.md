# Token-memory cross-attention: bounded conditional oracle

## What this tests

Can an independently trained MedCPT-to-Llama memory channel improve forced-choice
answers when gold chooses how strongly to use each KG-linked neighbor?

This is **conditional achievable headroom**, not a global optimum, a generalization
test, or proof that an arbitrarily capable future encoder would fail. The bridge
does not yet exist, so it must be fitted on separate questions first. Learning
arbitrary bridge weights or arbitrary residual vectors on diagnostic answers would
not be a valid test of selecting evidence contributions.

## Exact computation

1. Preserve the original pristine Llama-3-8B question/options/proposition prompt,
   order, token spans, choice token IDs, and all source claim text.
2. For each existing KG-linked proposition pair, frozen MedCPT Cross-Encoder BERT
   reads `[question stem] [SEP] [claim A] [claim B]`. Neither options nor gold are
   encoder inputs. Discard the reranking classifier. Keep every A/B token vector.
   Inputs exceeding 512 tokens are never truncated.
3. Cache Llama layers 1–12. At each proposition A token, an added four-head,
   width-256 cross-attention reads neighbor B/C token memories. A null value allows
   attention to read nothing. Original Llama self-attention stays untouched.
4. Split the cross-attention value sum into a tokenwise residual contribution from
   each directed pair, project into Llama's 4096 dimensions, and add at A's tokens:

   `new_A_tokens = original_A_tokens + sum_B gate[A,B] * contribution[A,B]`.

   No mean/CLS representation replaces a proposition. Gate values are shared
   across A's tokens for that pair; they scale the extra value contribution, not
   the original attention logits or a renormalized probability distribution.
   A→B and B→A have independent gates. Known future evidence is available through
   the encoder; generated future tokens and gold are not encoder inputs.
5. Frozen Llama layers 13–32 compute the final four choice logits. Cross-entropy
   on the correct choice trains only shared Q/K/V/O projections. No LoRA, encoder
   finetuning, target-Llama finetuning, or question-local answer vectors.

## Cohorts and stages

- Fixed diagnostic: previously studied official TRAIN128, 90 MedMCQA + 38 MedQA,
  originally 96 correct / 32 wrong, four graphless. It is exploratory, not fresh
  held-out evidence, and is not the final 6545-question test.
- Bridge training: 256 other official-training questions (179/77).
- Bridge selection: 64 official-validation questions (45/19).
- Exact normalized question duplicates are excluded across cohorts. This does
  not certify absence of all semantic paraphrases. Source train/test separation
  relies on the existing official-dataset contract.
- Train/dev selection uses seeded hashes and source proportions, not correctness.
  Graphless train/dev questions and unencodable >512-token pairs are excluded,
  recorded, and replaced before any labels are used for selection. No diagnostic
  question is removed. No retrieval or source KG modification.
- Tiny-eight fit: at most 64 one-question updates; require at least 20% mean CE
  reduction. Discard these fitted parameters. Failure stops the pilot and means
  the bridge's basic optimization is not established, not “KG cannot help.”
- Fresh bridge: two epochs, effective batch eight, AdamW 2e-4. Alternate real and
  degree-rewired memories per question/epoch, so both use a trained channel.
  Select the nonzero trained epoch with lowest mean real/rewired dev CE.
  Compare dev results against text-only; a worse bridge is explicitly reported.
- Freeze the bridge. For each diagnostic base-error question, jointly optimize
  ALL directed pair gates in [0,2], from starts 0 and 1, at most eight Adam updates
  per start (LR .25). Stop on a correct answer. Include the exact text-only no-op.
  Base-correct and graphless cases retain gold no-op without searching.
- Repeat with degree-preserving rewiring, same text, bridge and search budget.
  Report how many connections actually changed; unshufflable graphs are not
  meaningful negative controls.

## Outcomes and limits

Report text-only, fixed bridge, random gates, KG oracle and rewired oracle with
the same 128 denominator. Show CE, correct-answer probability, accuracy, changes,
subgroups and per-question joint actions/replay. The saved raw records permit
independent recomputation.

Preregistered practical screen: KG oracle recovers at least 8/32 errors
(96/128 → at least 104/128, +6.25 percentage points). To motivate specifically
KG-based follow-up, require at least four more recovered questions than rewired
oracle. These are research go/no-go screens, not statistical proof of novelty.

The oracle uses gold for selection and automatically avoids harming the 96
originally correct questions. Its absence of correct→wrong errors is artificial.
It does not establish learnability or a deployable router. Finite search may miss
better actions. Bridge alignment, injection layer, gate range and encoder
compression remain limitations; negative results cannot prove universal
impossibility. Positive results can also exploit answer-sensitive representations,
so rewiring and later gold-free held-out routing tests remain necessary.

## Commands

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_pair_memory_oracle.sh --preflight-only
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_pair_memory_oracle.sh --safety-only
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_pair_memory_oracle.sh
```

GPU1 only. Preflight is CPU-only; safety uses one real long question, checks native
zero parity and both bridge/gate gradients, then exits without retained learning.
It writes measured timing/VRAM to `SAFETY_RUNTIME.json`. Full run writes
`SUMMARY.json`, `per_question.jsonl`, bridge checkpoints and token caches under
`results/rag2_pair_memory_oracle_v1`. Same-command resume preserves source data.
Changing scientific settings/code requires a new output directory. Upper-layer
checkpointing is optional and slower; do not silently change it on resume.

Frozen encoder memories and lower Llama layers are cached; oracle pair
contributions are computed once per question, not on every search step. Encoder
batches start at 32 and halve on OOM. Bridge/oracle use one variable-length
question per forward, gradient accumulation for training. Native upper-layer
SDPA is retained (cuDNN SDPA excluded); no full-model backward weight gradients.
Runtime is calibrated, not inferred from unrelated old attention-bias experiments.
