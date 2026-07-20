"""Training-data conversion and graph-index construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ontology_pipeline import (
    ALLOWED_RELATIONS,
    Triple,
    build_prompt,
    compact_key,
    relation_counts,
    serialize_triples,
    triples_from_item,
)


DEV_RATIO = 0.2
DEV_SALT = "llms4ol-task-a-baseline-v1"
FOLD_COUNT = 5
FOLD_SALT = "llms4ol-task-a-sft-fold-v1"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_dev_split(item_id: str) -> str:
    digest = hashlib.md5(f"{DEV_SALT}:{item_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "dev" if bucket < DEV_RATIO else "train"


def fold_for_id(item_id: str) -> int:
    digest = hashlib.md5(f"{FOLD_SALT}:{item_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % FOLD_COUNT


def task_a_record(item: Dict[str, Any]) -> Dict[str, Any]:
    triples = triples_from_item(item)
    return {
        "id": item["id"],
        "split": stable_dev_split(item["id"]),
        "fold": fold_for_id(item["id"]),
        "prompt": build_prompt(item),
        "target_json": json.dumps(
            {"primitive-ontology-triples": serialize_triples(triples)},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "relation_counts": relation_counts(triples),
        "gold_triple_count": len(triples),
    }


def coerce_task_b_triple(row: Any) -> Triple | None:
    if not isinstance(row, list) or len(row) != 3:
        return None
    values = tuple(str(value).strip() for value in row)
    if not all(values):
        return None
    return values  # type: ignore[return-value]


def clean_task_b_triples(groups: Sequence[Iterable[Any]]) -> Tuple[List[Triple], Dict[str, int], int]:
    allowed = set(ALLOWED_RELATIONS)
    dropped: Counter[str] = Counter()
    seen = set()
    triples = []
    duplicate_count = 0
    for group in groups:
        for row in group:
            triple = coerce_task_b_triple(row)
            if triple is None:
                dropped["<bad-shape>"] += 1
                continue
            if triple[1] not in allowed:
                dropped[triple[1]] += 1
                continue
            if triple in seen:
                duplicate_count += 1
                continue
            seen.add(triple)
            triples.append(triple)
    return triples, dict(sorted(dropped.items())), duplicate_count


def convert_task_b_items(items: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    dropped_relations: Counter[str] = Counter()
    kept_relations: Counter[str] = Counter()
    duplicate_triples = 0
    empty_targets = 0
    for item in items:
        triples, dropped, duplicate_count = clean_task_b_triples(
            [
                item.get("initial-primitive-ontology-triples") or [],
                item.get("extended-primitive-ontology-triples") or [],
            ]
        )
        dropped_relations.update(dropped)
        kept_relations.update(relation for _, relation, _ in triples)
        duplicate_triples += duplicate_count
        if not triples:
            empty_targets += 1
        item_id = f"task_b_reuse:{item['id']}"
        prompt_item = {
            "id": item_id,
            "context": item["context"],
            "primitive-ontology-triples": serialize_triples(triples),
        }
        rows.append(
            {
                "id": item_id,
                "split": "train",
                "fold": fold_for_id(item_id),
                "source_task": "task_b_reuse_as_task_a_aux",
                "prompt": build_prompt(prompt_item),
                "target_json": json.dumps(
                    {"primitive-ontology-triples": serialize_triples(triples)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "relation_counts": relation_counts(triples),
                "gold_triple_count": len(triples),
            }
        )
    metadata = {
        "task_b_items": len(items),
        "task_b_aux_rows": len(rows),
        "task_b_kept_triples": sum(row["gold_triple_count"] for row in rows),
        "task_b_duplicate_triples_removed": duplicate_triples,
        "task_b_empty_targets": empty_targets,
        "task_b_kept_relation_counts": dict(sorted(kept_relations.items())),
        "task_b_dropped_relation_counts": dict(sorted(dropped_relations.items())),
    }
    return rows, metadata


def require_items(path: Path, task_name: str) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{task_name} data must be a non-empty JSON list: {path}")
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("context"), str):
            raise ValueError(f"{task_name} item {index} has an invalid id or context")
    return payload


def prepare_training_data(task_a_path: Path, task_b_path: Path, work_dir: Path) -> Dict[str, Any]:
    task_a_items = require_items(task_a_path, "Task A")
    task_b_items = require_items(task_b_path, "Task B")
    task_a_rows = [task_a_record(item) for item in task_a_items]
    task_a_dev_rows = [row for row in task_a_rows if row["split"] == "dev"]
    task_b_rows, task_b_metadata = convert_task_b_items(task_b_items)
    mixed_rows = task_a_rows + task_b_rows
    ids = [row["id"] for row in mixed_rows]
    duplicate_ids = [item_id for item_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Mixed training data has a duplicate id: {duplicate_ids[0]}")

    work_dir.mkdir(parents=True, exist_ok=True)
    mixed_path = work_dir / "mixed_train.jsonl"
    dev_path = work_dir / "task_a_dev.jsonl"
    write_jsonl(mixed_path, mixed_rows)
    write_jsonl(dev_path, task_a_dev_rows)
    metadata = {
        "task_a_source": str(task_a_path.resolve()),
        "task_a_sha256": sha256_file(task_a_path),
        "task_a_rows": len(task_a_rows),
        "task_a_dev_rows": len(task_a_dev_rows),
        "task_b_source": str(task_b_path.resolve()),
        "task_b_sha256": sha256_file(task_b_path),
        "mixed_train_rows": len(mixed_rows),
        "allowed_relations": ALLOWED_RELATIONS,
        **task_b_metadata,
    }
    metadata_path = work_dir / "training_data_metadata.json"
    write_json(metadata_path, metadata)
    return {
        "mixed_train": mixed_path,
        "task_a_dev": dev_path,
        "metadata": metadata_path,
        "summary": metadata,
    }


def build_graph_index(task_a_path: Path, output_path: Path) -> Dict[str, Any]:
    items = require_items(task_a_path, "Task A")
    triple_counts: Counter[Triple] = Counter()
    entity_forms: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        for subject, relation, obj in triples_from_item(item):
            triple_counts[(subject, relation, obj)] += 1
            entity_forms[compact_key(subject)][subject] += 1
            entity_forms[compact_key(obj)][obj] += 1
    graph_index = {
        "task_a_source": str(task_a_path.resolve()),
        "task_a_sha256": sha256_file(task_a_path),
        "indexed_items": len(items),
        "triple_counts": [
            {"subject": subject, "relation": relation, "object": obj, "count": count}
            for (subject, relation, obj), count in triple_counts.most_common()
        ],
        "entity_forms": {
            key: dict(forms.most_common())
            for key, forms in entity_forms.items()
        },
    }
    write_json(output_path, graph_index)
    return graph_index
