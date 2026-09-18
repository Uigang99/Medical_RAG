# Source-only 1K Statement Graph reference

This is a Codex-curated structural reference, not physician-annotated medical
truth. The source proposition is treated as authoritative. Annotation is done
without viewing v4, v5, or v6 relation predictions.

Each JSONL row uses the immutable source row number `n` and contains:

- `status`: `usable`, `no_core_relation`, `metadata`, or `incomplete`.
- `e`: every graph-worthy core biomedical entity as
  `[exact_surface, entity_type, canonical_name]`. A fourth integer selects a
  later occurrence when the same exact surface occurs multiple times.
- `r`: every relation expressible by the frozen evaluation ontology as
  `[subject_entity_index, relation_type, object_entity_index, polarity]`.
- `x`: explicit graph-worthy relations that the frozen ontology cannot express,
  written in short plain language. These measure ontology insufficiency rather
  than extraction failure.
- `confidence`: `high`, `medium`, or `low`.
- `notes`: ambiguity or adjudication notes.

Entity types and relation types are frozen in
`scripts/validate_rag2_statement_graph_reference.py`. Canonical names normalize
abbreviations, spelling, singular/plural, and obvious aliases. UMLS IDs are not
forced: the primary reference uses local canonical concept IDs, avoiding false
normalization caused by incomplete or ambiguous dictionary coverage.

Primary comparison uses high-confidence rows. Medium/low-confidence rows are
reported separately. Exact relation correctness requires the subject concept,
relation type, object concept, direction, and polarity all to match.
