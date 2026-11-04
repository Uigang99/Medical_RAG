# Bounded dynamic PASS development pilot

This implements training-free per-token abstention in the same Llama forward.
It is a bounded pilot, not evidence of accuracy improvement. Physical GPU 1 only.

## Inputs and controlled comparison

Use all 50 official LOFT development questions (10 each in HotpotQA, NQ,
MuSiQue, Qampari, QuEST). These development questions have previously been used
by other experiments; they are not a new untouched test. ID and exact-question
overlap with the local 500 test questions is checked before tokenization.

Reuse the model-independent full-corpus CE Top90 rankings and BGE-M3 dense/CE
priors in `results/gemma12_pced_comparison_v1/dev`. The directory name refers to
the experiment which prepared them; no Gemma outputs, weights or selected gamma
are used. Each dataset's input hashes, retrieval policy, PIDs/order and score
alignment are checked. No new retrieval, training, download or judge call.

One frozen local Llama-3.1-8B model serves 90 document streams and one amateur.
All arms use identical retrieval, brief reasoning/answer instructions, budgets,
amateur prompt, format masks, parser and metric. Only the expert PASS instruction
and score policy change:

| Arm | PASS instruction | Raw-top1 exclusion | Candidate penalty |
| --- | --- | --- | --- |
| `pced` | no | no | none |
| `prompt_only` | yes | no | none |
| `exclude` | yes | yes | none |
| `bounded` | yes | yes | lambda=4 |

The `pced` baseline uses original PCED scoring with the NEW common generation
protocol. It is not a rerun of the paper's original direct-answer prompt, and
historical TEST accuracies are not matched comparisons for this dev pilot.

## Prompt and scores

Exact prompts are in `medrag/generation/pced_pass_bounded.py` and are copied to
the manifest. Experts ground their current contribution in their own passage,
reassess each shared token, and treat shared reasoning as potentially incorrect.
An incomplete answer document may still support a useful intermediate fact.
The amateur uses background knowledge and the same shared response/format.

Keep the official first-position, vocabulary-wise JSD beta fixed thereafter;
keep gamma=2.5 and the cached retrieval prior. For each candidate token v:

```
base_i(v) = (1+beta_i(v))*z_i(v) - beta_i(v)*z_amateur(v) + 2.5*log(prior_i)
pass_logmass_i = logsumexp(z_i(PASS), z_i(" PASS"))
a_i(v) = sigmoid(pass_logmass_i - z_i(v))
score_i(v) = base_i(v) - 4*a_i(v)
```

This is exactly P(PASS)/(P(PASS)+P(v)), computed in logit space without a
probability floor. Penalties are bounded [0,4]; low PASS never earns an unbounded
bonus. The penalty is token-specific, so it can change an expert's proposal.
These scores are preferences, NOT calibrated evidence correctness probabilities.

The raw argmax after common format masks determines hard abstention; the two
PASS aliases are checked individually for argmax, and summed for soft scoring.
Format masks remove other special tokens and block EOS before a nonempty Answer.
Both the unmasked and format-masked argmax are recorded. PASS never enters the
shared text. All streams, including abstainers, consume every selected token.

If all document experts abstain, they stay excluded: the already-computed
amateur supplies its highest ordinary format-allowed token (winner=-1). This is
explicitly logged as fallback, not grounded document participation. No extra
forward. Without abstention, the highest expert/token score wins; no vote.

## Generation and measurement

Start with `Reasoning:\n`. Natural generation continues through whitespace and
newlines until a line-start `Answer:`. No numbered step or sentence stopping.
At 128 reasoning tokens, force `\nAnswer:` and log the cap; do not discard text.
Answer budget is 32 tokens, identical for all arms. EOS is permitted only after
Answer has an alphanumeric character. Budget exhaustion may still yield an
empty/invalid answer; it is recorded rather than repaired or dropped.

Score only the entire stripped Answer field with the existing Subspan EM.
This retains the historical evaluator's substring/alias limitations. Reasoning
is never scored. No semantic repair, hidden retries, gold routing or answer
substitution. Qrels are retained for analysis only, not treated as per-token
contribution labels.

## Artifacts and pilot criteria

Default root: `results/loft_llama_pass_bounded_dev_v1_n50/`.
The versioned manifest is written before model loading. It freezes input/model/
prompt/code hashes, score and format rules, limits, seeds and runtime versions.
Per-question outputs are atomic; the exact same command resumes finished
questions. Contract changes require a new root, and interrupted questions restart.

`summary.csv` / `summary.json` contain per-dataset accuracy, Micro/Macro-5,
paired gains/harms and descriptive bootstrap intervals, empty answers and cap
counts. Reasoning/answer diagnostics separate PASS rates and fallback frequency.
Each trace retains all experts' ordinary/PCED/bounded proposals, PASS probabilities,
selected-token raw logits/beta/penalty ratios, exclusions and amateur fallback.
Counterfactual score selections use the SAME actual prefix and prompt, not a
different-prompt run or full alternate rollout.

Primary comparison: bounded vs prompt_only. Exclude vs prompt_only isolates
hard gating; bounded vs exclude isolates the soft penalty. At N=50, the numeric
pilot gate requires >=3 additional correct vs prompt_only, no net loss vs pced,
and <=1 empty answer. This small-dev threshold is only a screening criterion;
expansion also requires qualitative evidence that later-token changes improve
grounding, not merely different formatting or amateur fallback. No automatic
hyperparameter search or test-based selection is performed.

## Run and validation

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_loft_pced_pass_bounded.sh
```

Append `--dry-run` for CPU-only source/tokenizer/context/resume validation without
loading model weights, allocating GPU memory or writing experiment results.
Use `--limit-per-dataset 1` for a separate 5-question operational smoke; its
default output root differs and it is not a quality benchmark.

At 50-160 generated positions/question and previously measured ~0.18-0.23 s per
position, 50 questions x4 arms suggests roughly 30-120 minutes on one H200.
This is a cost estimate, not measured throughput for the new prompts/penalty.
The old 21-minute run often ended prematurely and is not a completion-time
promise. First five fresh questions calibrate ETA. 91 streams are batched, one
model is loaded once, and KV caches are retained for the whole question. No
GPU0, per-sentence re-prefill or additional PASS judge forward is used.

Focused CPU tests cover scores, exclusion/fallback, cache broadcast and reentry,
leading newline handling, Answer transition and EOS masks; a tiny untrained CPU
Llama checks the actual HF cache path. These validate mechanics, not task quality
or production H200 execution.
