# Anchored RAG² paper reproduction

This pipeline reproduces the RAG² pseudo-label decision rule and released
Flan-T5 classifier protocol without using hidden-state features.

## Label contract

For every question and one retrieved document, the pipeline uses two
independent Llama-3-8B-Instruct generations:

- no document: answer and generated-rationale PPL;
- one document: answer and generated-rationale PPL.

It computes `Delta-PPL = PPL(no document) - PPL(one document)`.  The threshold
is the 75th percentile of valid Delta-PPL values from train questions only and
is fixed when labeling validation and test questions.  This implements the
paper's largest-25% rule.

| Answer transition | Label |
|---|---|
| Wrong to Correct | Helpful |
| Correct to Wrong | Not Helpful |
| Correct to Correct | Helpful if Delta-PPL >= tau, otherwise Discard |
| Wrong to Wrong | Not Helpful if Delta-PPL >= tau, otherwise Discard |

Discard and invalid rows are retained in audit files but are not used to train
the binary classifier.  The split unit is the question (`sample_id`), so no
question appears in more than one split.

## Materialized data (2026-08-24)

| Dataset | train questions | train Helpful | train Not Helpful | train Discard | binary train pairs | tau |
|---|---:|---:|---:|---:|---:|---:|
| MedMCQA | 146,256 | 300,732 | 217,520 | 651,784 | 518,252 | 0.06985834 |
| MedQA | 8,141 | 14,772 | 11,013 | 39,342 | 25,785 | 0.05357125 |

## Filter training contract

The classifier uses the released RAG² instruction and only the question,
options, and document text.  The targets are the added atomic tokens
`[HELPFUL]` and `[NOT_HELPFUL]`.  The base model is Flan-T5-large.

Paper/public-code settings retained:

- learning rate: `3e-5`;
- per-device batch size: `16`;
- AdamW, linear schedule, zero warmup, zero weight decay;
- maximum input length: `512`;
- overflow stride: `128`;
- no gradient clipping;
- validation after each epoch, with final test evaluation using the checkpoint
  selected by validation accuracy.  Validation does not update model weights.

The paper reports 40 epochs but does not publish the exact pseudo-labeled pair
count, so 40 is not a transferable update budget.  Keeping all examples and
reducing repeated passes preserves source and transition diversity:

- MedMCQA: 6 epochs, about 194,346 optimizer updates before any rare overflow
  expansion;
- MedQA: 15 epochs, about 24,180 optimizer updates before overflow expansion.

These values are also consistent with the best regions in previous local
curves: roughly 1-6 epochs for 500k-600k-pair MedMCQA runs and about 12 epochs
for 25k-33k-pair MedQA runs.  Every epoch is evaluated.  Disk retention is
limited to two checkpoints (the validation-selected best checkpoint is always
preserved), and the final test therefore uses the best validation checkpoint
rather than necessarily the last one.

## Commands

```bash
/home/user/Uiheon/Medical_RAG/scripts/run_rag2_anchored_paper_labeling.sh

/home/user/Uiheon/Medical_RAG/scripts/run_rag2_anchored_paper_filter_training.sh medmcqa
/home/user/Uiheon/Medical_RAG/scripts/run_rag2_anchored_paper_filter_training.sh medqa
```

The training launcher always binds to physical GPU 1.  To run a deliberate
epoch ablation while keeping every other setting fixed, set `EPOCHS_OVERRIDE`.

Primary references:

- [RAG² paper](https://aclanthology.org/2025.naacl-long.635/)
- [Released classifier protocol](https://github.com/dmis-lab/RAG2/tree/main/classifier)
