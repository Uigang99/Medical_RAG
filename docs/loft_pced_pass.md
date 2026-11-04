# Llama PCED dynamic PASS diagnostic

Purpose: test whether per-token local abstention improves expert selection,
without training, CoT, extra decision forwards, or changed retrieval.

The model is the reference Llama-3.1-8B-Instruct. The existing uppercase `PASS`
and space-prefixed ` PASS` are single vocabulary tokens (validated at startup).
`<PASS>` is not used: it tokenizes into multiple IDs.

At every generation step:

1. Evaluate amateur + all 90 expert streams in one batched model forward.
2. Use each expert's raw argmax to detect PASS, before contrastive/prior scores.
3. Sum raw softmax mass of the two PASS IDs. Exclude raw PASS-top1 experts.
4. For surviving experts, add `lambda * log(max(1-pPASS, 1e-6))` to their
   best non-PASS PCED score. Default lambda=1, fixed; no test-set sweep.
5. Broadcast the winning non-PASS token to EVERY stream, including abstainers.
   Exclusion is recomputed next step, so experts can rejoin.
6. If all experts abstain, restore eligibility for this step only while retaining
   penalties. Log this fallback explicitly. EOS is not PASS and remains allowed.

Keep the original first-step-only vocabulary-wise JSD beta, gamma=2.5, CE+BGE
prior, document order, question template, amateur prompt, greedy 32-token output,
and official Subspan EM. PASS rules are appended to expert system instructions.
The comparison therefore includes a matched `prompt_only` arm: same prompt,
PASS output masking, but neither exclusion nor penalty. Existing paper outputs
are reused. Paper-vs-PASS combines prompting and routing changes; prompt-only
vs-PASS isolates routing. Prompt-only is not unrestricted PASS text generation.

Default cohort: 10 questions per dataset selected by deterministic hash(seed,
dataset,qid), 50 total, from the already analyzed LOFT test caches. It is an
exploratory diagnostic, NOT an untouched validation set. At 100 per dataset,
the cohort is all existing 500 questions. No hyperparameter selection is done.

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_loft_pced_pass.sh --dry-run
bash /home/user/Uiheon/Medical_RAG/scripts/run_loft_pced_pass.sh
```

Physical GPU1 only. No GPU inference is performed by `--dry-run`; it checks the
reference/source contracts, local checkpoint shards, exact tokenizer tokens,
prompt lengths without truncation, and compatible resume records. One Llama is
loaded for both arms. Question results and manifests are atomic and resumable.
No previous result or source cache is modified. Use a new output root when
changing lambda or other semantic settings. The cohort-size default roots differ.

Outputs: `manifest.json`, `summary.json`, `summary.csv`, and
`{prompt_only,pass}/predictions/{dataset}/{qid_hash}.json`. Per-step traces record
raw PASS logits/probability, abstainers, actual exclusions, non-PASS candidates,
PCED scores, penalties, winner before/after gating at the SAME prefix, and
all-PASS fallbacks. Gold answers are used only after generation for scoring.

The summary reports per-dataset and micro/macro-5 EM, paired correct/wrong
transitions, descriptive paired-bootstrap intervals, changed predictions and
PASS activity. Within-prefix winner changes are not full counterfactual rollouts.

Runtime: previous direct-PCED records were approximately 3-11 seconds/question
in a different prompt/concurrent-worker setting. Two 50-question arms therefore
suggest minutes to tens of minutes, NOT a measured PASS ETA. Each arm calibrates
its own rate from fresh questions; cached work does not inflate throughput.
