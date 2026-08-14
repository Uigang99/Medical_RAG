# Medical-RAG

Research code for medical RAG experiments, including RAG²-style document
filtering, hidden-state utility filtering, attribution-based analyses, and
MCQ benchmark evaluation.

## Repository scope

This repository tracks source code, experiment configurations, tests, and
reproducible run scripts. Local datasets, vector databases, model checkpoints,
run caches, and generated evaluation outputs are deliberately excluded through
`.gitignore` because they are large and environment-specific.

Each material experiment should record its exact command, model paths, corpus
manifest/version, and result directory in a committed script or configuration.

## Active code scope

The tracked scripts cover four reproducible paths:

1. MCQ and RAG-Square corpus preparation plus MedCPT index construction.
2. The current free-response RAG2 pseudo-label and Flan-T5 filter baseline.
3. GPT-5.6 semantic-label analysis and hidden-state utility labelling/filtering.
4. Fixed-terminal, source-balanced MCQ evaluation comparing RAG2 and
   hidden-state filters.

Historical sentence/window/attribution experiments are kept locally but are
not included in Git. The allow-list at the bottom of `.gitignore` defines the
active scripts included in this repository.
