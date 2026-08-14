# Active Hidden-State RAG2 Research Pipeline

## Research scope

The active method replaces the original RAG2 pseudo-labelling signal based on
answer transitions and rationale perplexity with a pre-answer hidden-state
utility signal. For a question without and with a document, the pipeline
extracts `h0` and `hD`, computes the gold-answer direction `c`, and labels the
document from the projection of `hD - h0` onto `c`. A filter is then trained
with text, hidden features, or both, and evaluated under the same retrieval,
reranking, prompting, and Top-k conditions as the RAG2 baseline.

## Core method code

### Hidden-state extraction, labels, and training

- `scripts/extract_rag2_preanswer_hidden_pilot.py`: fixed pre-answer prompt,
  constrained answer decoding, hidden-state extraction, and gold-direction
  gradient implementation.
- `scripts/extract_rag2_preanswer_hidden_full.py`: resumable full-data feature
  extraction for MedMCQA and MedQA.
- `scripts/materialize_rag2_preanswer_hidden_labels.py`: projection-threshold
  Helpful/Not Helpful labels.
- `scripts/build_rag2_hidden_filter_inputs.py`: leakage-free shared splits for
  text-only, hidden-only, and text+hidden ablations.
- `scripts/train_rag2_hidden_feature_filter.py`: the three filter input modes.
- `medrag/filtering/rag2_preanswer_text_hidden.py`: inference-time hidden-state
  feature and filter implementation.
- `scripts/analyze_rag2_preanswer_hidden_labels.py` and
  `scripts/analyze_rag2_hidden_vs_terra_semantic.py`: label diagnostics.

### RAG2 baseline reproduction

- `scripts/generate_rag2_no_rag_rationales.py`
- `scripts/precompute_rag2_rationale_embeddings.py`
- `scripts/build_rag2_filter_candidates.py`
- `scripts/generate_rag2_document_traces.py`
- `scripts/build_rag2_filter_training_splits.py`
- `scripts/train_rag2_filter_model_paper.py`
- `scripts/audit_rag2_no_rag_quality_selection.py`
- `scripts/audit_rag2_document_traces.py`

These scripts remain necessary for the matched RAG2 baseline even though the
proposed method does not use rationale PPL as its final label.

### Retrieval and final MCQ evaluation

- `scripts/run_rag2_mcq_eval.py`
- `scripts/run_rag2_terminal_all_mcq_unfiltered_topk_sweep.sh`
- `scripts/run_rag2_terminal_all_mcq_topk_sweep.sh`
- `scripts/run_rag2_vs_hidden_terminal_medqa_topk_sweep.sh`
- `medrag/rag2_mcq.py`
- `medrag/retrieval/faiss_retriever.py`
- `medrag/reranking/medcpt_cross_encoder.py`
- `medrag/generation/transformers_generator.py`

The remaining tracked corpus builders and GPT-5.6 semantic-labelling tools are
supporting reproducibility and external label validation, rather than the core
novel method.

## Storage that must be retained

### Base models and active filters

- `/home/user/Uiheon/models/Llama-3-8B-Instruct`
- `/home/user/Uiheon/models/Flan-T5-large`
- `/home/user/Uiheon/models/MedCPT-Article-Encoder`
- `/home/user/Uiheon/models/MedCPT-Query-Encoder`
- `/home/user/Uiheon/models/MedCPT-Cross-Encoder`
- RAG2 corrected baseline `final_model` directories for MedMCQA and MedQA.
- HiddenUtilityTau0 text+hidden final/best models. Until MedMCQA training is
  finalized, retain both its best checkpoint and most recent resume checkpoint.

### Active data and indexes

- `databases/vector_db/RAG_Square`
- `datasets/benchmark/mcq`
- `.../candidates/quality_selected_source_balanced40_rerank32_v1`
- `.../preanswer_hidden_gold_direction_full_top8_v1`
- `.../preanswer_hidden_labels_full_top8_tau0_v1`
- `.../filter_training_inputs_hidden_utility_top8_tau0_v1`
- `.../filter_training_inputs_top10_independent_ppl_v2_corrected_nodoc`
- `.../no_rag_rationales_train` and `.../no_rag_quality_selection_v1`
- `databases/run_cache/rag2_llama3_paper_exact_terminal_v1`
- `results/rag2_llama3_paper_exact_terminal_v1`

The 44GB independent-PPL document traces should be retained only if immediate
RAG2 label regeneration/auditing is important; the compact finalized training
inputs and trained baseline models are sufficient for ordinary comparisons.

## Measured cleanup candidates (2026-08-14)

No files were deleted during this audit.

| Candidate group | Approx. space | Disposition |
|---|---:|---|
| Old window/sentence/attribution model families | 536GB | Remove after confirming those experiments are closed |
| Old window/sentence/attribution datasets | 208GB | Remove |
| Multilayer hidden probe, pilot, old answer-format and Qwen data | 146GB | Remove after retaining compact summaries if desired |
| Hugging Face dataset caches | 68GB | Safe to rebuild; remove first |
| Historical result generations outside the terminal-v1 experiment | about 40GB | Keep summary tables, remove bulky generations |
| Old run caches, query embeddings, and superseded vector DBs | about 15GB | Remove; retain RAG_Square and terminal-v1 cache |
| Raw/unified corpus sources after completed DB build | up to 404GB | Optional; removes local rebuild capability |

The first four groups alone can recover roughly 958GB. Removing historical
results and superseded run caches raises the practical recovery to about 1TB
without touching the active RAG_Square vector database.

## Checkpoint pruning

- The RAG2 baseline `final_model/model.safetensors` files were SHA-256 verified
  to be identical to the selected MedQA checkpoint-8356 and MedMCQA
  checkpoint-38775 model weights. Evaluation scripts now reference
  `final_model`, so the optimizer-bearing checkpoint directories can be
  removed after a final path audit.
- Completed MedQA hidden-filter runs need only their compact `final_model`
  directories for inference. Their Trainer checkpoints are needed only for
  resuming training.
- MedMCQA hidden text+hidden training currently has no finalized model. Keep
  checkpoint-212793 (best at the inspected state) and checkpoint-283724 (latest
  resumable state) until training is completed and a final model is exported.
