# PASS-relative shared-CoT diagnostic v1

## Purpose and scope

Test whether PASS-relative scoring and updated shared reasoning change useful
expert participation. This is a bounded diagnostic, not established improvement.
The default cohort is the **same previously inspected LOFT TEST 50 questions**
(10 each; seed 42) as the direct-PASS pilot. No test-based parameter selection or
untouched-test generalization claim is permitted. No retrieval, training or
model download occurs. Use physical GPU 1 only.

## Controlled arms

All four arms use the same frozen Llama-3.1-8B, cached ordered Top90 passages,
PASS prompt, short-CoT protocol, greedy decoding and budgets.

| Arm | Score and exclusion |
| --- | --- |
| `pced` | Original first-position JSD/contrastive score + gamma 2.5 retrieval prior; mask PASS output, no gate |
| `probability` | `log P_t(v)`; mask PASS output, no gate |
| `pass_static` | `log P_t(v) - log max(P_0(PASS), 1e-4)`; freeze both raw-top1 exclusion and PASS mass at the first reasoning position |
| `pass_dynamic` | `log P_t(v) - log max(P_t(PASS), 1e-4)`; recompute exclusion and mass every token |

PASS mass sums existing tokenizer IDs for `PASS` and ` PASS`; hard exclusion
tests whether **one of these individual tokens** is the raw vocabulary argmax.
Do not register an untrained `<PASS>` token. Ordinary probabilities include
PASS in the original softmax denominator. No retrieval/amateur score is used
in the other three arms. The amateur stream is retained only to keep batching
and same-prefix counterfactual diagnostics identical. There is no extra
classification forward. Each selected token reaches every stream, including
temporarily excluded experts. Static freezing persists across all phases.

All-PASS temporarily restores every expert for that position while retaining
the ratio score. This is logged as forced fallback, not grounded participation.
The floor prevents tiny PASS probabilities receiving arbitrarily high rewards;
it does **not** make these probabilities calibrated evidence judgments.

## Generation protocol

`Reasoning: / Step 1:` -> at most 3 short steps of 32 tokens -> `Answer:` of at
most 32 tokens. Each step is prompted to be one sentence. A newline advances
to the next numbered step. EOS ends reasoning early and requests Answer; its
EOS token is not appended before the forced Answer header. No minimum number
of reasoning sentences is imposed. Max-step, per-step budget, empty field or
unexpected header also moves to Answer and is explicitly recorded. Partial
reasoning is retained, not silently deleted or replaced by cached answers.
One-sentence compliance is prompted, not guaranteed by punctuation heuristics.

Only formatting headers are forced. They are batched with the last pending
shared token into the next cached forward. No re-prefill per sentence, no
separate decision model, no switching to an ALL-document answer model.
Generation quality and step-budget exhaustion must be inspected, not hidden.

Only the final Answer field is scored by the unchanged Subspan EM evaluator;
reasoning matches receive no credit. Empty final answers count as wrong.

## Outputs and interpretation

Default root: `results/loft_llama_pass_ratio_cot_v1_n50/`.
`manifest.json` freezes source hashes, cohort, model/tokenizer, prompts,
algorithms, budgets, environment and code hashes before loading model weights.
Each question/arm is saved atomically; the identical command resumes completed
questions. A different contract requires a new output root. Interrupted
questions restart, while completed questions are preserved.

`summary.json` and `summary.csv` report per-dataset/micro/macro5 accuracy,
empty/budget-limited answers, paired correct-to-wrong/wrong-to-correct changes,
and descriptive paired bootstrap intervals. Historical direct PCED is only an
input cache/reference, not a controlled CoT score comparison.

Per-token traces retain phase and step position, all experts' PASS mass,
raw-best token, ordinary-best token/logprob, actual selected-token logprob,
hard exclusions, floor saturation, fallback, and all four policies' winning
token/expert **at the same actual prefix**. These counterfactual decisions are
not full alternate rollouts. PIDs link back to the original passages/qrels.
Reasoning, forced headers, final answer and stop reasons are all preserved.

Primary diagnostic: dynamic vs probability; dynamic vs static isolates the
temporal gate. Inspect later-step PASS grounding and bridge-document harms,
not merely activation counts. Do not expand if any apparent gain comes only
from changed formatting, fallback or a few familiar test examples. A favorable
diagnostic still needs a separate frozen-config held-out evaluation.

## Run and resource estimate

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_loft_pced_pass_cot.sh
```

Append `--dry-run` for CPU-only source/tokenizer/cache validation without output
writes or model weight loading. Defaults run all four arms sequentially in one
process, loading one Llama and batching its 91 streams, not 91 model copies.
No other GPU is used. Maximum sampled positions are 128/question/arm plus short
forced headers. Compared with the measured direct pilot (~4.3 s/question,
~19 output positions), 60-128 positions suggest roughly 45-120 minutes for
50 x 4 questions on H200; this is a provisional cost model, not measured CoT
throughput. First 5 fresh questions calibrate the estimate; each arm updates
its own measured rate. Cached work is excluded from throughput and pending ETA.

Validation is CPU-only: selector and protocol tests, a tiny untrained CPU Llama
cache-path check, syntax checks and the exact default dry run. These do not
establish actual H200 memory headroom or reasoning quality.
