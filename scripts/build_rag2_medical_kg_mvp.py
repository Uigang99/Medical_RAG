#!/usr/bin/env python3
"""Build a source-independent biomedical KG from the exact RAG² Top-k union.

MVP scope:
* documents only (no questions, answers, semantic labels, or utility scores),
* scispaCy entity spans linked to UMLS concepts,
* conservative same-sentence typed rules plus separately marked co-occurrence,
* evidence retained for every assertion,
* deterministic source-stratified sampling, atomic shards, and safe resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from medrag.kg.biomedical_mvp import (
    aggregate_edge_assertions,
    category_from_tuis,
    cooccurrence_assertion,
    infer_relation,
    local_entity_id,
)


RUN_VERSION = "rag2_candidate_medical_kg_mvp_v2"
RELATION_RULE_VERSION = "typed_rules_v2_adjacent_linked_mentions"
SCHEMA_VERSION = 1
SOURCES = ("cpg", "pmc", "pubmed", "textbooks")
DEFAULT_TOP_K = (1, 2, 4, 8, 16, 32)
EXPECTED_INPUT = {
    "candidate_rows": 39_270,
    "document_occurrences": 412_335,
    "unique_documents": 165_642,
    "unique_by_source": {"cpg": 31_365, "pmc": 57_180, "pubmed": 55_881, "textbooks": 21_216},
}


def parse_args() -> argparse.Namespace:
    root = Path("/home/user/Uiheon/Medical_RAG")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=root / "databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "datasets/kg/rag2_candidate_medical_kg_mvp_v2",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K))
    parser.add_argument("--limit-per-source", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacy-model", default="en_core_sci_sm")
    parser.add_argument("--linker-name", default="umls")
    parser.add_argument("--linker-threshold", type=float, default=0.85)
    parser.add_argument("--max-link-candidates", type=int, default=3)
    parser.add_argument("--max-pair-token-distance", type=int, default=40)
    parser.add_argument("--max-sentence-entities", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--audit-mentions", type=int, default=200)
    parser.add_argument("--audit-relations", type=int, default=100)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds == float("inf"):
        return "calibrating"
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TwoLineProgress:
    """Two persistent TTY lines; periodic two-line snapshots otherwise."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        self.started = time.time()
        self.stage_started = self.started
        self.stage_index = 0
        self.stage_total = 1
        self.stage_done = 0
        self.stage_initial = 0
        self.unit = "item"
        self.last_snapshot = 0.0
        self.stage_estimates: list[float | None] = [None] * len(names)
        self.stage_actual: list[float | None] = [None] * len(names)
        self.tty = sys.stderr.isatty()
        self.overall_bar: Any = None
        self.stage_bar: Any = None
        if self.tty:
            from tqdm.auto import tqdm

            self._tqdm = tqdm
            self.overall_bar = tqdm(
                total=1000,
                position=0,
                leave=True,
                dynamic_ncols=True,
                mininterval=1.0,
                bar_format="Overall: unknown [stage 0/{0}, elapsed 00:00:00, ETA calibrating]".format(len(names)),
            )

    def plan(self, details: Sequence[str]) -> None:
        self.log("[workflow plan] " + " | ".join(f"{i+1}/{len(details)} {x}" for i, x in enumerate(details)))

    def start_stage(
        self,
        index: int,
        *,
        total: int,
        initial: int = 0,
        unit: str = "item",
        estimate: float | None = None,
    ) -> None:
        if self.stage_bar is not None:
            self.stage_bar.close()
        self.stage_index = index
        self.stage_total = max(1, int(total))
        self.stage_done = min(max(0, int(initial)), self.stage_total)
        self.stage_initial = self.stage_done
        self.stage_started = time.time()
        self.unit = unit
        self.stage_estimates[index - 1] = estimate
        if self.tty:
            self.stage_bar = self._tqdm(
                total=self.stage_total,
                initial=self.stage_done,
                position=1,
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
                desc=f"Stage {index}/{len(self.names)} - {self.names[index-1]}",
                unit=unit,
            )
        self.render(force=True)

    def _rate_and_eta(self) -> tuple[float | None, float | None]:
        elapsed = max(1e-9, time.time() - self.stage_started)
        live = max(0, self.stage_done - self.stage_initial)
        rate = live / elapsed if live else None
        if self.stage_done >= self.stage_total:
            return rate, 0.0
        if rate:
            return rate, (self.stage_total - self.stage_done) / rate
        estimate = self.stage_estimates[self.stage_index - 1]
        if estimate is None:
            return None, None
        return None, estimate * (1.0 - self.stage_done / self.stage_total)

    def _overall(self, stage_eta: float | None) -> tuple[float | None, float | None]:
        known = all(value is not None for value in self.stage_estimates)
        if not known or stage_eta is None:
            return None, None
        weights = [float(x) for x in self.stage_estimates if x is not None]
        completed = sum(weights[: self.stage_index - 1])
        active = weights[self.stage_index - 1] * self.stage_done / self.stage_total
        total = sum(weights)
        fraction = (completed + active) / max(total, 1e-9)
        future = sum(weights[self.stage_index :])
        return fraction, stage_eta + future

    def render(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_snapshot < (1.0 if self.tty else 30.0):
            return
        self.last_snapshot = now
        rate, stage_eta = self._rate_and_eta()
        fraction, overall_eta = self._overall(stage_eta)
        elapsed = now - self.started
        overall_percent = "unknown" if fraction is None else f"{100*fraction:.0f}%"
        overall_line = (
            f"Overall: {overall_percent} [stage {self.stage_index}/{len(self.names)}, "
            f"elapsed {format_duration(elapsed)}, ETA {format_duration(overall_eta)}]"
        )
        stage_pct = 100 * self.stage_done / self.stage_total
        rate_text = "calibrating" if rate is None else f"{rate:,.1f} {self.unit}/s"
        stage_line = (
            f"Stage {self.stage_index}/{len(self.names)} - {self.names[self.stage_index-1]}: "
            f"{stage_pct:.0f}% {self.stage_done}/{self.stage_total} "
            f"[{rate_text}, ETA {format_duration(stage_eta)}]"
        )
        if self.tty:
            self.overall_bar.n = 0 if fraction is None else int(min(1000, 1000 * fraction))
            self.overall_bar.bar_format = (
                overall_line if fraction is None else
                "Overall: {percentage:3.0f}%|{bar:10}| "
                f"[stage {self.stage_index}/{len(self.names)}, elapsed {format_duration(elapsed)}, ETA {format_duration(overall_eta)}]"
            )
            self.overall_bar.refresh()
            if self.stage_bar is not None:
                self.stage_bar.n = self.stage_done
                self.stage_bar.bar_format = (
                    "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} "
                    f"[{rate_text}, ETA {format_duration(stage_eta)}]"
                )
                self.stage_bar.refresh()
        else:
            print(overall_line, flush=True)
            print(stage_line, flush=True)

    def update(self, amount: int = 1) -> None:
        self.stage_done = min(self.stage_total, self.stage_done + int(amount))
        if self.stage_bar is not None:
            self.stage_bar.n = self.stage_done
        # Extraction calibration supplies the first defensible estimate.
        if self.stage_index == 3 and self.stage_done - self.stage_initial >= 32:
            elapsed = max(1e-9, time.time() - self.stage_started)
            rate = (self.stage_done - self.stage_initial) / elapsed
            self.stage_estimates[2] = self.stage_total / max(rate, 1e-9)
            if self.stage_estimates[3] is None:
                self.stage_estimates[3] = max(5.0, 0.02 * self.stage_total)
        self.render()

    def complete(self, detail: str) -> None:
        self.stage_done = self.stage_total
        self.render(force=True)
        actual = time.time() - self.stage_started
        self.stage_actual[self.stage_index - 1] = actual
        self.stage_estimates[self.stage_index - 1] = actual
        self.log(
            f"[stage {self.stage_index}/{len(self.names)} complete | duration {format_duration(actual)}] {detail}"
        )

    def log(self, message: str) -> None:
        if self.tty and hasattr(self, "_tqdm"):
            self._tqdm.write(message)
        else:
            print(message, flush=True)

    def close(self, *, force_complete: bool = False) -> None:
        if self.stage_bar is not None:
            self.stage_bar.close()
            self.stage_bar = None
        if self.overall_bar is not None:
            if force_complete:
                self.overall_bar.n = self.overall_bar.total
            self.overall_bar.refresh()
            self.overall_bar.close()


def input_paths(candidate_root: Path, top_k: Sequence[int]) -> list[tuple[int, Path]]:
    values: list[tuple[int, Path]] = []
    for k in top_k:
        path = candidate_root / f"top{k}" / "candidates.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing exact Top-{k} candidate cache: {path}")
        values.append((k, path))
    return values


def compact_document(document: Mapping[str, Any], k: int) -> dict[str, Any]:
    stable_id = str(document.get("stable_id") or document.get("corpus_id") or "").strip()
    source = str(document.get("source") or document.get("retrieval_bucket") or "").strip()
    text = str(document.get("text") or "").strip()
    if not stable_id or source not in SOURCES or not text:
        raise ValueError(f"Invalid candidate document: stable_id={stable_id!r} source={source!r} text_len={len(text)}")
    return {
        "stable_id": stable_id,
        "corpus_id": str(document.get("corpus_id") or stable_id),
        "chunk_id": str(document.get("chunk_id") or stable_id),
        "doc_id": str(document.get("doc_id") or ""),
        "source": source,
        "title": str(document.get("title") or "").strip(),
        "text": text,
        "text_sha256": sha256_text(text),
        "top_k_membership": [int(k)],
    }


def scan_union(paths: Sequence[tuple[int, Path]], progress: TwoLineProgress) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    occurrences = 0
    rows = 0
    membership_counts: Counter[int] = Counter()
    for k, path in paths:
        for row in read_jsonl(path):
            rows += 1
            for raw in row.get("reranked_documents", []):
                occurrences += 1
                membership_counts[k] += 1
                document = compact_document(raw, k)
                stable_id = document["stable_id"]
                previous = documents.get(stable_id)
                if previous is None:
                    documents[stable_id] = document
                else:
                    if previous["source"] != document["source"] or previous["text_sha256"] != document["text_sha256"]:
                        raise RuntimeError(f"Stable-ID collision with different source/text: {stable_id}")
                    if k not in previous["top_k_membership"]:
                        previous["top_k_membership"].append(k)
            progress.update(1)
    for document in documents.values():
        document["top_k_membership"].sort()
    source_counts = Counter(row["source"] for row in documents.values())
    union_hash = sha256_text(
        "\n".join(
            f"{stable_id}\t{documents[stable_id]['source']}\t{documents[stable_id]['text_sha256']}\t"
            f"{','.join(map(str, documents[stable_id]['top_k_membership']))}"
            for stable_id in sorted(documents)
        )
    )
    return documents, {
        "candidate_rows": rows,
        "document_occurrences": occurrences,
        "unique_documents": len(documents),
        "unique_by_source": dict(sorted(source_counts.items())),
        "occurrences_by_top_k": {str(k): membership_counts[k] for k, _ in paths},
        "union_sha256": union_hash,
    }


def deterministic_selection(documents: Mapping[str, dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in documents.values():
        grouped[row["source"]].append(row)
    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        rows = grouped[source]
        rows.sort(key=lambda row: sha256_text(f"{seed}|{row['stable_id']}"))
        if limit > 0:
            if len(rows) < limit:
                raise RuntimeError(f"Source {source} has only {len(rows)} unique documents; requested {limit}")
            rows = rows[:limit]
        selected.extend(rows)
    selected.sort(key=lambda row: (row["source"], sha256_text(f"{seed}|{row['stable_id']}")))
    return selected


def analysis_text(document: Mapping[str, Any]) -> tuple[str, int]:
    title = str(document.get("title") or "").strip()
    text = str(document["text"]).strip()
    if not title or text.casefold().startswith(title.casefold()):
        return text, 0
    prefix = title.rstrip(". ") + ". "
    return prefix + text, len(prefix)


def load_pipeline(model_name: str, linker_name: str, threshold: float, max_candidates: int) -> tuple[Any, Any]:
    try:
        import spacy
        import scispacy  # noqa: F401
        from scispacy import abbreviation as _abbreviation  # noqa: F401
        from scispacy import linking as _linking  # noqa: F401
        from scispacy.candidate_generation import (
            CandidateGenerator,
            LinkerPaths,
            load_approximate_nearest_neighbours_index,
        )
        from scispacy.linking import EntityLinker
        from scispacy.linking_utils import UmlsKnowledgeBase
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "scispaCy environment is missing. Run scripts/setup_rag2_medical_kg_mvp.sh first."
        ) from exc
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(f"Missing scispaCy model {model_name!r}; run the setup script first") from exc
    nlp.max_length = max(nlp.max_length, 2_000_000)
    if "abbreviation_detector" not in nlp.pipe_names:
        nlp.add_pipe("abbreviation_detector")
    if linker_name != "umls":
        raise ValueError("The v1 offline contract supports only the UMLS linker")

    cache_root = Path(os.environ.get("SCISPACY_CACHE", "")) / "datasets"
    if not cache_root.is_dir():
        raise RuntimeError(f"Missing local scispaCy cache directory: {cache_root}")

    def one(suffix: str) -> str:
        matches = sorted(path for path in cache_root.iterdir() if path.name.endswith(suffix))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one local UMLS cache file ending {suffix!r}, found {len(matches)}"
            )
        return str(matches[0])

    paths = LinkerPaths(
        ann_index=one(".nmslib_index.bin"),
        tfidf_vectorizer=one(".tfidf_vectorizer.joblib"),
        tfidf_vectors=one(".tfidf_vectors_sparse.npz"),
        concept_aliases_list=one(".concept_aliases.json"),
    )
    kb = UmlsKnowledgeBase(
        file_path=one(".umls_2022_ab_cat0129.jsonl"),
        types_file_path=one(".umls_semantic_type_tree.tsv"),
    )
    candidate_generator = CandidateGenerator(
        ann_index=load_approximate_nearest_neighbours_index(paths, ef_search=200),
        tfidf_vectorizer=joblib.load(paths.tfidf_vectorizer),
        ann_concept_aliases_list=json.loads(Path(paths.concept_aliases_list).read_text(encoding="utf-8")),
        kb=kb,
    )
    linker = EntityLinker(
        nlp=nlp,
        candidate_generator=candidate_generator,
        resolve_abbreviations=True,
        threshold=threshold,
        max_entities_per_mention=max_candidates,
    )
    return nlp, linker


def mention_from_span(span: Any, linker: Any, threshold: float, sentence_index: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for cui, score in list(span._.kb_ents):
        entity = linker.kb.cui_to_entity.get(cui)
        if entity is None:
            continue
        candidates.append(
            {
                "cui": str(cui),
                "score": float(score),
                "canonical_name": str(entity.canonical_name),
                "tuis": sorted(str(value) for value in entity.types),
            }
        )
    chosen = candidates[0] if candidates and candidates[0]["score"] >= threshold else None
    tuis = chosen["tuis"] if chosen else []
    return {
        "surface": span.text,
        "start": int(span.start_char),
        "end": int(span.end_char),
        "token_start": int(span.start),
        "token_end": int(span.end),
        "sentence_index": int(sentence_index),
        "linked": chosen is not None,
        "entity_id": f"UMLS:{chosen['cui']}" if chosen else local_entity_id(span.text),
        "cui": chosen["cui"] if chosen else None,
        "canonical_name": chosen["canonical_name"] if chosen else span.text,
        "tuis": tuis,
        "category": category_from_tuis(tuis),
        "link_score": chosen["score"] if chosen else None,
        "link_candidates": candidates,
    }


def extract_document(
    document: Mapping[str, Any],
    doc: Any,
    linker: Any,
    *,
    threshold: float,
    max_pair_distance: int,
    max_sentence_entities: int,
) -> dict[str, Any]:
    text = doc.text
    sentences = list(doc.sents)
    sentence_for_token: dict[int, int] = {}
    sentence_rows: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        sentence_rows.append(
            {
                "sentence_index": index,
                "start": int(sentence.start_char),
                "end": int(sentence.end_char),
                "text": sentence.text,
            }
        )
        for token_index in range(sentence.start, sentence.end):
            sentence_for_token[token_index] = index

    mentions = [
        mention_from_span(span, linker, threshold, sentence_for_token.get(span.start, -1))
        for span in doc.ents
    ]
    by_sentence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        if mention["sentence_index"] >= 0 and mention["linked"]:
            by_sentence[int(mention["sentence_index"])].append(mention)

    relations: list[dict[str, Any]] = []
    skipped_dense_sentences = 0
    for sentence_index, sentence_mentions in by_sentence.items():
        if len(sentence_mentions) > max_sentence_entities:
            skipped_dense_sentences += 1
            continue
        sentence = sentence_rows[sentence_index]
        sentence_mentions.sort(key=lambda value: (value["start"], value["end"]))
        for left_index, left in enumerate(sentence_mentions):
            for right_index, right in enumerate(
                sentence_mentions[left_index + 1 :], start=left_index + 1
            ):
                if left["entity_id"] == right["entity_id"]:
                    continue
                token_distance = max(0, int(right["token_start"]) - int(left["token_end"]))
                if token_distance > max_pair_distance:
                    continue
                between = text[int(left["end"]) : int(right["start"])]
                # Typed rules are deliberately restricted to adjacent linked
                # mentions.  Otherwise one trigger can spuriously connect
                # every entity pair spanning it.  Non-adjacent pairs remain
                # available as explicitly weaker co-occurrence assertions.
                assertion = (
                    infer_relation(left, right, sentence_text=sentence["text"], between_text=between)
                    if right_index == left_index + 1
                    else None
                )
                relation_kind = "typed" if assertion is not None else "cooccurrence"
                if assertion is None:
                    assertion = cooccurrence_assertion(left, right, sentence_text=sentence["text"])
                row = assertion.to_dict()
                row.update(
                    {
                        "relation_kind": relation_kind,
                        "chunk_id": document["chunk_id"],
                        "stable_id": document["stable_id"],
                        "source": document["source"],
                        "sentence_index": sentence_index,
                        "evidence_sentence": sentence["text"],
                        "evidence_start": sentence["start"],
                        "evidence_end": sentence["end"],
                        "subject_surface": left["surface"] if row["subject_id"] == left["entity_id"] else right["surface"],
                        "object_surface": right["surface"] if row["object_id"] == right["entity_id"] else left["surface"],
                        "token_distance": token_distance,
                    }
                )
                relations.append(row)

    return {
        "stable_id": document["stable_id"],
        "corpus_id": document["corpus_id"],
        "chunk_id": document["chunk_id"],
        "doc_id": document["doc_id"],
        "source": document["source"],
        "title": document["title"],
        "text_sha256": document["text_sha256"],
        "analysis_text_sha256": sha256_text(text),
        "top_k_membership": document["top_k_membership"],
        "sentence_count": len(sentences),
        "mentions": mentions,
        "relations": relations,
        "skipped_dense_sentences": skipped_dense_sentences,
    }


def shard_paths(root: Path, index: int) -> tuple[Path, Path]:
    stem = f"shard_{index:05d}"
    return root / "extraction_shards" / f"{stem}.jsonl", root / "extraction_shards" / f"{stem}.complete.json"


def valid_cached_shard(root: Path, index: int, expected_ids: Sequence[str], contract_hash: str) -> bool:
    data_path, marker_path = shard_paths(root, index)
    if not marker_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "contract_sha256": contract_hash,
        "stable_ids_sha256": sha256_text("\n".join(expected_ids)),
        "row_count": len(expected_ids),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Cached shard contract mismatch: {marker_path}")
    if not data_path.is_file():
        raise RuntimeError(f"Shard completion marker exists without data: {data_path}")
    return True


def flatten_results(root: Path, shard_count: int) -> Iterator[dict[str, Any]]:
    for index in range(shard_count):
        data_path, _ = shard_paths(root, index)
        yield from read_jsonl(data_path)


def finalize(
    root: Path,
    shard_count: int,
    selected_count: int,
    seed: int,
    audit_mentions: int,
    audit_relations: int,
    progress: TwoLineProgress | None = None,
) -> dict[str, Any]:
    mention_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    nodes: dict[str, dict[str, Any]] = {}
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    processed = 0
    for shard_index in range(shard_count):
        data_path, _ = shard_paths(root, shard_index)
        for result in read_jsonl(data_path):
            processed += 1
            stats = source_stats[result["source"]]
            stats["chunks"] += 1
            linked_in_chunk = 0
            typed_in_chunk = 0
            for mention in result["mentions"]:
                flat = {key: value for key, value in mention.items() if key != "link_candidates"}
                flat.update({"stable_id": result["stable_id"], "chunk_id": result["chunk_id"], "source": result["source"]})
                mention_rows.append(flat)
                stats["mentions"] += 1
                if mention["linked"]:
                    stats["linked_mentions"] += 1
                    linked_in_chunk += 1
                    node = nodes.setdefault(
                        mention["entity_id"],
                        {
                            "entity_id": mention["entity_id"],
                            "cui": mention["cui"],
                            "canonical_name": mention["canonical_name"],
                            "category": mention["category"],
                            "tuis": mention["tuis"],
                            "mention_count": 0,
                            "chunk_ids": set(),
                            "source_counts": Counter(),
                            "surface_counts": Counter(),
                        },
                    )
                    node["mention_count"] += 1
                    node["chunk_ids"].add(result["chunk_id"])
                    node["source_counts"][result["source"]] += 1
                    node["surface_counts"][mention["surface"]] += 1
            if linked_in_chunk:
                stats["chunks_with_linked_entity"] += 1
            for relation in result["relations"]:
                relation_rows.append(relation)
                stats["relations"] += 1
                if relation["relation_kind"] == "typed":
                    stats["typed_relations"] += 1
                    typed_in_chunk += 1
                else:
                    stats["cooccurrences"] += 1
            if typed_in_chunk:
                stats["chunks_with_typed_relation"] += 1
        if progress is not None:
            progress.update(1)

    if processed != selected_count:
        raise RuntimeError(f"Finalization saw {processed} chunks, expected {selected_count}")

    node_rows: list[dict[str, Any]] = []
    for entity_id, node in sorted(nodes.items()):
        node_rows.append(
            {
                "entity_id": entity_id,
                "cui": node["cui"],
                "canonical_name": node["canonical_name"],
                "category": node["category"],
                "tuis": node["tuis"],
                "mention_count": node["mention_count"],
                "distinct_chunk_count": len(node["chunk_ids"]),
                "source_counts": dict(sorted(node["source_counts"].items())),
                "surface_forms": [name for name, _ in node["surface_counts"].most_common(20)],
            }
        )
    edge_rows = aggregate_edge_assertions(relation_rows)

    atomic_jsonl(root / "entity_mentions.jsonl", mention_rows)
    atomic_jsonl(root / "relation_assertions.jsonl", relation_rows)
    atomic_jsonl(root / "nodes.jsonl", node_rows)
    atomic_jsonl(root / "edges.jsonl", edge_rows)

    rng = random.Random(seed)
    linked_mentions = [row for row in mention_rows if row["linked"]]
    typed_relations = [row for row in relation_rows if row["relation_kind"] == "typed"]
    sampled_mentions = rng.sample(linked_mentions, min(audit_mentions, len(linked_mentions)))
    sampled_relations = rng.sample(typed_relations, min(audit_relations, len(typed_relations)))
    audit = (
        [{"audit_type": "entity_link", **row} for row in sampled_mentions]
        + [{"audit_type": "typed_relation", **row} for row in sampled_relations]
    )
    rng.shuffle(audit)
    atomic_jsonl(root / "manual_audit_sample.jsonl", audit)

    by_source: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        values = source_stats[source]
        chunks = max(1, values["chunks"])
        mentions = max(1, values["mentions"])
        by_source[source] = {
            **dict(values),
            "linked_mention_rate": values["linked_mentions"] / mentions,
            "linked_entity_chunk_coverage": values["chunks_with_linked_entity"] / chunks,
            "typed_relation_chunk_coverage": values["chunks_with_typed_relation"] / chunks,
        }
    coverages = [by_source[source]["linked_entity_chunk_coverage"] for source in SOURCES]
    automatic_checks = {
        "all_selected_chunks_processed": processed == selected_count,
        "overall_linked_entity_chunk_coverage_ge_0_70": (
            sum(values["chunks_with_linked_entity"] for values in source_stats.values()) / max(1, processed) >= 0.70
        ),
        "every_source_linked_entity_chunk_coverage_ge_0_55": min(coverages) >= 0.55,
        "source_coverage_gap_le_0_25": max(coverages) - min(coverages) <= 0.25,
        "typed_relations_exist": bool(typed_relations),
    }
    summary = {
        "run_version": RUN_VERSION,
        "processed_chunks": processed,
        "entity_mentions": len(mention_rows),
        "linked_entity_mentions": len(linked_mentions),
        "unique_umls_nodes": len(node_rows),
        "relation_assertions": len(relation_rows),
        "typed_relation_assertions": len(typed_relations),
        "cooccurrence_assertions": len(relation_rows) - len(typed_relations),
        "aggregated_edges": len(edge_rows),
        "by_source": by_source,
        "automatic_checks": automatic_checks,
        "automatic_screen_passed": all(automatic_checks.values()),
        "manual_audit_required": {
            "entity_link_precision_target": 0.85,
            "typed_relation_precision_target": 0.75,
            "audit_rows": len(audit),
            "note": "Automatic coverage checks cannot establish KG precision.",
        },
    }
    atomic_json(root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.limit_per_source < 0 or args.shard_size <= 0 or args.batch_size <= 0:
        raise ValueError("limit-per-source must be >=0; shard-size and batch-size must be >0")
    if not 0.0 <= args.linker_threshold <= 1.0:
        raise ValueError("linker-threshold must be in [0,1]")
    top_k = sorted(set(args.top_k))
    paths = input_paths(args.candidate_root, top_k)
    candidate_row_total = sum(1 for _, path in paths for _ in path.open("r", encoding="utf-8"))

    progress = TwoLineProgress(
        [
            "scan exact Top-k caches and select chunks",
            "load biomedical parser and UMLS linker",
            "extract entity and evidence-backed relation shards",
            "aggregate graph and write audit report",
        ]
    )
    progress.plan(
        [
            f"Top-k={top_k}; deterministic union/sample",
            f"{args.spacy_model}+{args.linker_name}",
            f"atomic shards of {args.shard_size}",
            "nodes/edges/coverage/manual-audit sample",
        ]
    )
    progress.start_stage(1, total=candidate_row_total, unit="row", estimate=10.0)
    documents, input_summary = scan_union(paths, progress)
    for key, expected in EXPECTED_INPUT.items():
        if input_summary[key] != expected:
            raise RuntimeError(
                f"Exact candidate-union contract failed for {key}: "
                f"observed={input_summary[key]} expected={expected}"
            )
    selected = deterministic_selection(documents, args.limit_per_source, args.seed)
    selection_hash = sha256_text("\n".join(row["stable_id"] for row in selected))
    progress.complete(
        f"unique={len(documents):,} selected={len(selected):,} source_counts="
        f"{dict(Counter(row['source'] for row in selected))}"
    )

    contract = {
        "run_version": RUN_VERSION,
        "schema_version": SCHEMA_VERSION,
        "candidate_root": str(args.candidate_root.resolve()),
        "candidate_files": [
            {"top_k": k, "path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for k, path in paths
        ],
        "top_k": top_k,
        "candidate_union_sha256": input_summary["union_sha256"],
        "selection_sha256": selection_hash,
        "limit_per_source": args.limit_per_source,
        "seed": args.seed,
        "spacy_model": args.spacy_model,
        "model_storage_root": os.environ.get(
            "KG_MODEL_ROOT", "/home/user/Uiheon/models/RAG2-Medical-KG-MVP"
        ),
        "scispacy_cache": os.environ.get("SCISPACY_CACHE"),
        "linker_name": args.linker_name,
        "linker_threshold": args.linker_threshold,
        "max_link_candidates": args.max_link_candidates,
        "max_pair_token_distance": args.max_pair_token_distance,
        "max_sentence_entities": args.max_sentence_entities,
        "shard_size": args.shard_size,
        "relation_schema": RELATION_RULE_VERSION,
        "uses_questions_answers_or_labels": False,
    }
    contract_hash = sha256_text(canonical_json(contract))
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") != contract_hash:
            raise RuntimeError(
                f"KG MVP contract mismatch at {manifest_path}; use a new versioned output root"
            )
    manifest = {
        "status": "prepared",
        "contract_sha256": contract_hash,
        "contract": contract,
        "input_summary": input_summary,
        "selected_by_source": dict(sorted(Counter(row["source"] for row in selected).items())),
        "hypothesis": (
            "A source-independent local biomedical pipeline can recover normalized entities and "
            "auditable sentence-backed relations from the exact RAG2 candidate-chunk union."
        ),
        "scope_limit": "This MVP measures extraction coverage and audited precision; it does not test downstream QA gains.",
        "reproducibility": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "prompt_parser_generation_settings": "not applicable: corpus-text extraction only",
        },
    }
    atomic_json(args.output_root / "manifest.json", manifest)
    atomic_jsonl(args.output_root / "selected_chunks.jsonl", selected)
    if args.preflight_only:
        progress.close(force_complete=False)
        print(
            f"[preflight complete] unique={len(documents):,} selected={len(selected):,} "
            f"contract={contract_hash[:12]} output={args.output_root}",
            flush=True,
        )
        return

    shard_count = (len(selected) + args.shard_size - 1) // args.shard_size
    cached_chunks = 0
    cached_shards: set[int] = set()
    for index in range(shard_count):
        rows = selected[index * args.shard_size : (index + 1) * args.shard_size]
        ids = [row["stable_id"] for row in rows]
        if valid_cached_shard(args.output_root, index, ids, contract_hash):
            cached_shards.add(index)
            cached_chunks += len(rows)

    remaining = len(selected) - cached_chunks
    progress.start_stage(2, total=1, unit="model", estimate=90.0 if remaining else 0.1)
    if remaining:
        nlp, linker = load_pipeline(
            args.spacy_model, args.linker_name, args.linker_threshold, args.max_link_candidates
        )
    else:
        nlp = linker = None
    progress.update(1)
    progress.complete("loaded local pipeline" if remaining else "all extraction shards cached; model load skipped")

    progress.start_stage(3, total=len(selected), initial=cached_chunks, unit="chunk", estimate=None)
    for index in range(shard_count):
        if index in cached_shards:
            continue
        rows = selected[index * args.shard_size : (index + 1) * args.shard_size]
        texts = [analysis_text(row)[0] for row in rows]
        results: list[dict[str, Any]] = []
        assert nlp is not None and linker is not None
        for document, parsed in zip(
            rows,
            nlp.pipe(texts, batch_size=args.batch_size, n_process=1),
            strict=True,
        ):
            parsed = linker(parsed)
            results.append(
                extract_document(
                    document,
                    parsed,
                    linker,
                    threshold=args.linker_threshold,
                    max_pair_distance=args.max_pair_token_distance,
                    max_sentence_entities=args.max_sentence_entities,
                )
            )
            progress.update(1)
        data_path, marker_path = shard_paths(args.output_root, index)
        atomic_jsonl(data_path, results)
        ids = [row["stable_id"] for row in rows]
        atomic_json(
            marker_path,
            {
                "contract_sha256": contract_hash,
                "stable_ids_sha256": sha256_text("\n".join(ids)),
                "row_count": len(rows),
            },
        )
    progress.complete(f"processed={len(selected):,} cached_at_start={cached_chunks:,} shards={shard_count}")

    progress.start_stage(4, total=shard_count, unit="shard", estimate=max(5.0, 0.02 * len(selected)))
    for index in range(shard_count):
        data_path, marker_path = shard_paths(args.output_root, index)
        if not data_path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"Missing durable extraction shard {index}")
    summary = finalize(
        args.output_root,
        shard_count,
        len(selected),
        args.seed,
        args.audit_mentions,
        args.audit_relations,
        progress,
    )
    manifest["status"] = "complete"
    manifest["summary_path"] = str((args.output_root / "summary.json").resolve())
    atomic_json(manifest_path, manifest)
    atomic_json(
        args.output_root / "COMPLETE.json",
        {"contract_sha256": contract_hash, "processed_chunks": len(selected)},
    )
    progress.complete(
        f"nodes={summary['unique_umls_nodes']:,} typed_assertions={summary['typed_relation_assertions']:,} "
        f"automatic_screen={summary['automatic_screen_passed']} output={args.output_root}"
    )
    progress.close(force_complete=True)
    print(
        f"[workflow complete] chunks={len(selected):,} nodes={summary['unique_umls_nodes']:,} "
        f"typed_relations={summary['typed_relation_assertions']:,} manual_audit_required=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
