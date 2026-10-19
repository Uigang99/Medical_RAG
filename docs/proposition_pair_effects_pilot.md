# Proposition-pair intervention supervision pilot v1

## Purpose and scope

Test whether a controller can learn input-specific **measured local attention
effects**, rather than converging to four nearly constant layer biases. This
pilot does not update Llama/LoRA, regenerate propositions, retrieve documents,
or use the benchmark TEST questions. It is target-model-specific MCQ answer
selection, not evidence that free-form reasoning improves.

Source: immutable prepared Official Train9000/Val1000 data. Default pilot uses
128 official-train questions (90 MedMCQA/38 MedQA), 32 development questions
(22/10), and 64 held-out development questions (45/19). The latter two come
from disjoint portions of official Val1000, not final benchmark TEST. Selection
is seeded hash order independent of labels, correctness, length and connectivity.
Graphless questions remain. Exact duplicate stems across pilot splits abort.

## Fixed vs changed

Keep source proposition text/order, question/options, tokenizer, causal graph
pairs, first-12-layer frozen features, two-round 128-wide graph controller trunk,
and attention locations (whole proposition blocks, layers 13–16, shared heads).
Only earlier propositions can influence later ones. This is not bidirectional
text attention or supervision of medical relation correctness.

All conditions now use the SAME enabled block-mask backend schedule, target
batch size 1, and FP32 selected-choice lm_head calculation. Therefore this
experiment recalculates its zero baseline; old BF16/native-path scores are NOT
its controlled baseline. cuDNN SDPA is excluded. Before committing measurements,
zero CE is repeated and must differ by <=1e-4 with the same selected answer.

## Measurement / supervision

Select up to four existing causal pairs per question by seeded hash. For each
pair and each of four layers, apply -0.5 and +0.5 separately, with every other
bias zero. Store exact logits, CE, answer probability, source proposition IDs,
pair/layer, baseline/repeat, and pooled features. At most 34 forwards/question,
7,616 measurement forwards for 224 questions; fewer for small/empty graphs.

Target utility = baseline CE minus intervened CE. This is a local, conditional
effect, NOT a timeless label attached to a KG edge. The controller predicts two
utilities per pair/layer; unchanged is explicitly zero. Its action is decrease,
unchanged or increase depending on whether the best predicted utility exceeds
0.01 nats. Differences beneath 0.01 are zeroed only in training targets; raw
measurements are retained. This threshold is an operational pilot threshold,
not an established scientific/medical standard.

Train only the small FP32 controller: question-balanced SmoothL1 loss on
predicted vs measured utility, scaled by max(.02, train-only 90th percentile
absolute utility). No class balancing forces positive/negative effects to
exist. Questions, ordered sender/receiver representations, neighboring KG and
endpoint roles are inputs. Gold labels/options never enter controller features.
Gold is used only to produce train supervision or evaluate development output.

First fit 16 informative TRAIN questions, max 400 updates. Require >=90%
informative action accuracy and <=10% of the zero-action local regret. This is
an overfit/debug gate, not generalization evidence. Start full controller
training afresh afterward; 50 epochs, AdamW 3e-4, weight decay .01, clip norm1,
8 questions/update, no target-LLM backpropagation. Select by minimum Dev32
local regret, including epoch0. Freeze checkpoint before measuring Heldout64.

## Controls and gates

- Zero bias; layer-constant utility predictor fitted only on train; random action
  expectation for local utility. The fixed action controller has no question or
  pair-specific interpretation.
- Signal gate: >=32 informative pair/layer decisions in >=16 train questions,
  with >=16 decisions differing from layer-constant actions. Otherwise stop.
- Primary heldout metric: local action regret = measured best useful action's
  CE improvement minus selected action's CE improvement; average within each
  question, then over questions. Sub-floor effects are zeroed consistently in
  this local metric and the constant comparator; original effects are retained.
  Graphless rows contribute zero and are counted.
- Local pass: >=20% less regret than constants and paired **question-level**
  bootstrap 95% interval for improvement above zero (2,000 fixed-seed resamples).
- Joint check: apply ALL predicted graph-pair actions simultaneously, not just
  the four sampled training pairs. Re-evaluate zero/constant/learned on identical
  Heldout64 prompts. Pass requires mean CE at least .01 below BOTH controls and
  correct count no lower than either. Report Micro, Macro2, per-dataset counts,
  gold probability, paired CE intervals, C/W transitions, and missing count.

Local supervision is around a zero background; joint effects need not add.
Local success with joint failure is evidence to investigate interactions, not
permission to claim QA improvement. This first pilot deliberately does NOT
jointly fine-tune with answer CE. Four-pair sampling also does not demonstrate
same-pair/across-question counterfactual generalization. Small Heldout64 does
not establish a final accuracy gain; even a pass needs independent replication.

## Run / resume

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_proposition_pair_effects_pilot.sh --preflight-only
bash /home/user/Uiheon/Medical_RAG/scripts/run_rag2_proposition_pair_effects_pilot.sh
```

Optional: `--measure-only` stops after Train/Dev measurement and the signal gate;
rerun without this flag to continue under the identical scientific contract.
Different delta, cohort, thresholds, code or model needs a new output root.
An unsuccessful scientific gate writes SUMMARY.json and stops normally, rather
than silently extending training. Do not change thresholds using heldout results.

Expected on idle H200 GPU1: roughly 15–40 minutes, provisional. Prior same-model
forward probes were about 0.09 s/forward, but this cohort's lengths and extra
controller fit are not yet timed. ETA recalibrates after four fresh questions.
Pair measurements resume per completed question, fitting per completed epoch,
and joint evaluation per question/condition. Batch1 deliberately holds numeric
execution constant; this diagnostic does not maximize VRAM occupancy.

Outputs under results/rag2_proposition_pair_effects_pilot_v1: manifest/cohort,
SAFETY.json, measurements/*/*.pt (raw+features), train_dev_interventions.jsonl,
SIGNAL.json, tiny/COMPLETE.json, controller/{history.json,best.pt,last.pt},
heldout_interventions.jsonl, joint_predictions.jsonl, local_heldout.jsonl
(including pair-level predicted effects/actions), paired_heldout.jsonl, SUMMARY.json.
Stopped gates report their actual scope; no final TEST scores are produced.
