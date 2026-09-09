import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rag2_medical_kg_biorex_pilot.py"
SPEC = importlib.util.spec_from_file_location("biorex_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def mention(start, end, entity_id, category, score=0.99, surface="entity"):
    return {
        "start": start,
        "end": end,
        "entity_id": entity_id,
        "category": category,
        "link_score": score,
        "linked": True,
        "surface": surface,
        "canonical_name": surface,
    }


def test_entity_filter_maps_supported_types_and_removes_overlap():
    row = {
        "mentions": [
            mention(0, 7, "UMLS:C1", "DrugOrChemical", surface="aspirin"),
            mention(0, 3, "UMLS:C2", "GeneOrProtein", score=0.91, surface="asp"),
            mention(20, 28, "UMLS:C3", "DiseaseOrDisorder", surface="migraine"),
            mention(30, 35, "UMLS:C4", "Anatomy", surface="brain"),
        ]
    }
    accepted, rejected = MODULE.eligible_mentions(row, 0.90)
    assert [(item["entity_id"], item["biorex_type"]) for item in accepted] == [
        ("UMLS:C1", "ChemicalEntity"),
        ("UMLS:C3", "DiseaseOrPhenotypicFeature"),
    ]
    assert rejected["overlapping_span"] == 1
    assert rejected["unsupported_type:Anatomy"] == 1
    assert MODULE.has_supported_pair(accepted)


def test_softmax_and_primary_scope():
    values = MODULE.softmax([0.0, 1.0, 2.0])
    assert abs(sum(values) - 1.0) < 1e-9
    assert values[2] > values[1] > values[0]
    assert "Association" in MODULE.PRIMARY_LABELS
    assert "None" not in MODULE.PRIMARY_LABELS
