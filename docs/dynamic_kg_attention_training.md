# Dynamic KG attention: bounded development training

This is a feasibility pilot, not an established accuracy improvement. The
previous static overlay trained 3,072 layer/head/type scalars. This experiment
trains a shared, input-conditioned function, not one parameter per KG triple.

## Fixed inputs and target model

- Existing TEST repartition: 2,048 training, 256 validation. Do not load the
  512-question evaluation inputs or 3,729 reserve questions for training.
- Exact cached proposition text, Top-8 scope, ordering, target prompt/answer
  tokens, and source-aligned causal endpoint token pairs are preserved.
- Llama-3-8B-Instruct and the selected medical QA LoRA remain frozen.
- Existing MedCPT Query/Article encoders remain frozen. The query encoder sees
  the question **stem only**. The article encoder sees endpoint names, surface
  predicates and source propositions. Options, gold labels and target answer
  tokens are never controller features.
- All MedCPT content tokens are covered by non-overlapping windows of at most
  510 tokens plus special tokens. Window CLS vectors are cached; a learned
  pooling operation combines them. This avoids truncation, not semantic loss.

## Trainable computation

The default controller has hidden width 128 and two message-passing rounds.
It projects frozen semantic features, pools their windows, and propagates
question-conditioned messages over the current Top-8 graph. Edge messages use
surface relation and source proposition semantics, retaining direction roles.
The final MLP reads both endpoint states, the question, both source propositions,
the relation, and the attention-link type. Its output is one signed bounded
bias per layer/head/link. Only the final output projection starts at zero.

The bias is added to the **same existing endpoint token-pair attention logits**
as the static experiment. There are no new evidence tokens or new connections.
Duplicate contributions to a same-kind token pair are averaged. Independent
kinds are added as before. Full proposition semantics guide the controller,
but the actuator is still endpoint-token attention, not a whole-proposition
gate. Original causal/padding masks remain. The controller can see the entire
known input question/evidence; it cannot see future generated text. Biases are
computed once per input, not recalculated from an evolving rationale.

Two arms use identical parameters, seeds, data, loss and optimization budget:

- `graph`: message sender states come from KG neighbors.
- `local`: replace each sender state by the receiver's own state. Incident
  proposition/relation semantics and KG-based attention positions remain.
  This isolates neighbor-state propagation, **not all KG use**.

Backpropagation follows answer-token CE through frozen Llama computations into
the controller. Freezing weights does not mean disabling autograd through
Llama. No new LoRA, rationale generation, reinforcement learning or gold gate
labels are used. Static versus dynamic additionally changes semantic inputs
and parameter count; it is not a topology-only comparison.

## Bounded execution and checks

1. CPU preflight validates frozen artifacts, source/token alignment on all 2,304
   rows, encoder token coverage and train-only 32-question proportional cohort.
2. Cache frozen semantic windows in resumable 256-text shards; unload encoders.
3. Load frozen Llama once. For each arm, verify zero-bias generation parity on
   16 training questions. Use 32 real training questions for at most 100 updates,
   effective batch 16, checking fixed-cohort CE every 10 updates.
4. Require at least 20% relative reduction in the 32-question CE, finite nonzero
   gradients and measurable question dependence. Graph additionally requires
   sensitivity to neighbor-state propagation. These are engineering gates,
   not generalization/accuracy evidence. Failure stops full training.
5. If both arms pass, **discard the tiny-fit weights**. Train fresh controllers
   for three epochs on 2,048 questions, using 256 validation questions to select
   minimum CE, including the initial zero-bias checkpoint. Never select on 512.
6. Save checkpoints and validation records. Do not run final evaluation.

The learning target is MCQ-supervised and target-model-specific, even though
controller input excludes choices. This pilot cannot establish open-ended QA
generalization or causal correctness of the learned attention scores.

## Commands

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_dynamic_kg_attention_training.sh --preflight-only
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_dynamic_kg_attention_training.sh
```

Optional: `--gate-only` stops after the real 32-question tests. Rerun without that
flag to continue under the same contract. Do not change hyperparameters in a
completed/resumable output directory; use a new `--output-root`.

Default output:
`results/rag2_dynamic_kg_attention_training_v1/`

- `manifest.json`, `PREFLIGHT.json`: immutable experiment contract and checks.
- `features/`: shared frozen encodings, never a trainable semantic model.
- `{local,graph}/tiny32/GATE.json`: feasibility outcome and input sensitivity.
- `{local,graph}/full/{last,best}.pt`: resumable/selected controller only.
- `{local,graph}/full/history.json`: per-epoch train/validation losses.
- `{local,graph}/full/validation_epoch*.jsonl`: individual validation losses.

GPU 1 only. The default is two sequential arms, not parallel GPU jobs. There is
no defensible new wall-clock forecast before calibration; static-training time
is not reused as if throughput were identical. Frozen model activations, not
controller parameter storage, dominate training memory and computation.

## Evaluation after training

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_dynamic_kg_attention_evaluation.sh
```

This does not train or select checkpoints. It requires completed full training
and loads each arm's `full/best.pt`, selected using the 256 validation questions.
The same 512 development questions, evidence, medical LoRA, answer-only greedy
decoding (16 output tokens), parser and Macro-3/Micro definitions are reused.

No-RAG, TEXT and static attention predictions are reused; only dynamic `local`
and `graph` predictions are generated. Zero-bias token parity against cached TEXT
must pass before each new condition. GPU 1, batch 16, OOM batch halving with
restoration on the next shard. No accuracy threshold is tuned on these questions.

Output: `results/rag2_dynamic_kg_attention_evaluation_v1/summary.json`,
`summary_table.json`, and `paired_predictions.jsonl`. Reports include five
conditions, paired C->W/W->C transitions and bootstrap confidence intervals;
`graph_vs_local` is the neighbor-propagation comparison. This remains a reused
development cohort, not an untouched benchmark test.

`--prepare-only` checks inputs on CPU even before training completes.
`--preflight-only` additionally requires valid selected checkpoints. Neither
loads model weights on the GPU or generates answers. Rerunning the ordinary
command resumes durable prediction shards and shared evaluation feature caches.
New evaluation throughput is calibrated, not extrapolated from training time.
