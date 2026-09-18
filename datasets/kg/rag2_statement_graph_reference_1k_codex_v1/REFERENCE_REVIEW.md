# Reference review record

## Scope

All 1,000 immutable source propositions were read directly.  v4/v5/v6
predictions were not used while selecting entities, normalizing local concept
names, or assigning relations.  The output is a source-faithful structural
reference, not physician-adjudicated medical truth.

## First pass

- Selected all graph-worthy mentions in the eight frozen core entity types.
- Preserved exact source spans and assigned conservative local canonical names.
- Recorded every relation expressible by the frozen ontology, including
  direction and asserted/negated/uncertain polarity.
- Recorded explicit but out-of-ontology facts separately instead of forcing an
  incorrect relation type.
- Marked metadata-only and context-incomplete propositions explicitly.

## Second pass

- Rechecked all 47 low-confidence rows and all 13 context-incomplete rows.
- Audited all 766 relations for valid endpoints, self-edges, duplicates,
  direction, polarity, and basic relation/type compatibility.
- Rechecked composition relations so `PART_OF` points from component to whole.
- Repaired exact-span failures found by the validator; no approximate span is
  accepted by the reference builder.
- Kept 185 medium-confidence rows for sensitivity analysis only.  The primary
  comparison uses the 768 high-confidence propositions.

## Final structural checks

- 1,000/1,000 propositions present in immutable order.
- 2,950 entity mentions and 2,631 local normalized concepts.
- 766 in-ontology relations; 0 self-edges; 0 duplicate edges.
- 713 explicit out-of-ontology relation descriptions retained separately.
- Status: 527 usable, 231 no core relation, 229 metadata, 13 incomplete.

## Limits

This reference is not independent human annotation: both passes were performed
by Codex.  It can test extraction and normalization against the source wording,
but cannot establish that the source is medically true.  Local canonical names
are not authoritative UMLS CUIs, so UMLS evaluation is limited to coverage,
lexical agreement, and repeated-mention clustering consistency.
