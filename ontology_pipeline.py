"""Ontology prompt, parsing, graph completion, fusion, and validation utilities."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Triple = Tuple[str, str, str]
PROMPT_VERSION = "task-a-sft-graph-v1"

ALLOWED_RELATIONS = [
    "is-a",
    "instance-of",
    "is defined by",
    "equivalent class",
    "disjoint with",
    "type",
    "exact match",
    "tree view",
    "part_of",
    "has part",
    "database_cross_reference",
    "see also",
    "same as",
    "range",
    "is_conjugate_base_of",
    "is_conjugate_acid_of",
    "derives from",
    "develops_from",
    "broader",
    "domain",
    "located in",
    "has role",
    "term replaced by",
    "positively regulates",
    "regulates",
    "overlaps",
    "ro_0002473",
    "ro_0002220",
]

BAD_ENTITY_KEYS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "group",
    "groups",
    "he",
    "her",
    "his",
    "in",
    "including",
    "is",
    "it",
    "its",
    "la",
    "of",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
    "type",
    "value",
    "category",
    "categories",
    "class",
    "classes",
    "system",
    "systems",
    "taxonomy",
    "taxonomies",
    "domain",
    "domains",
    "field",
    "fields",
    "method",
    "methods",
    "was",
    "were",
    "which",
    "who",
    "whose",
    "with",
    "within",
}

FUSION_BAD_ENTITIES = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "which",
    "which itself",
    "it",
    "its",
    "they",
    "them",
    "one",
    "role",
    "order",
    "realm",
    "key",
    "structural",
    "composition",
    "category",
    "categories",
    "class",
    "classes",
    "group",
    "groups",
    "system",
    "systems",
}

BOUNDARY_WORDS = (
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "where",
    "when",
    "while",
    "as",
    "in",
    "within",
    "across",
    "among",
    "for",
    "to",
    "of",
    "by",
    "with",
    "including",
    "include",
    "includes",
    "contains",
    "contain",
    "encompasses",
    "encompass",
)

SOURCE_WEIGHTS = {"sft": 1.0, "graph_completion": 0.92}
SOURCE_MIN_SCORES = {"sft": 0.5, "graph_completion": 0.86}
RELATION_MIN_SCORES = {
    "default": 0.95,
    "disjoint with": 0.9,
    "equivalent class": 0.9,
    "instance-of": 0.7,
    "is defined by": 0.84,
    "is-a": 0.5,
    "type": 0.9,
}
RELATION_CAPS = {
    "default": 4,
    "disjoint with": 2,
    "equivalent class": 2,
    "has part": 4,
    "instance-of": 32,
    "is defined by": 4,
    "is-a": 32,
    "part_of": 4,
    "see also": 4,
    "type": 2,
}
MAX_TRIPLES_PER_ITEM = 48


@dataclass
class ParseResult:
    rows: List["ScoredTriple"]
    hard_error: str = ""
    soft_errors: Tuple[str, ...] = ()
    raw_length: int = 0
    salvaged_count: int = 0


@dataclass
class ScoredTriple:
    triple: Triple
    source: str
    score: float
    evidence: str = ""
    entity_presence: str = "ontology_prior"


def build_prompt(item: Dict[str, Any]) -> str:
    schema = {"primitive-ontology-triples": [["subject", "relation", "object"]]}
    instructions = [
        "You are constructing a primitive ontology from raw text.",
        "Return only one valid JSON object and no commentary.",
        "Use exactly this key: primitive-ontology-triples.",
        "Each triple must be a 3-item array: [subject, relation, object].",
        "Use only the allowed relations.",
        "Preserve ontology entity spelling and punctuation when it is meaningful.",
        "Do not lowercase all entities.",
        "Create one triple per ontology edge; lists should be expanded into separate triples.",
        "Include taxonomy, instance typing, definition, equivalence, disjointness, and other primitive relations when supported.",
    ]
    payload = {
        "prompt_version": PROMPT_VERSION,
        "instructions": instructions,
        "allowed_relations": ALLOWED_RELATIONS,
        "output_schema": schema,
        "id": item.get("id", ""),
        "context": item["context"],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def apply_chat_template_json(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    enable_thinking: Optional[bool] = False,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return messages[-1]["content"] + "\nJSON:"
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def norm_key(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def unicode_compact_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


def entity_presence(entity: str, context: str) -> float:
    if not entity:
        return 0.0
    if entity in context:
        return 1.0
    if entity.lower() in context.lower():
        return 0.86
    normalized = norm_key(entity)
    compact = compact_key(entity)
    if normalized and normalized in norm_key(context):
        return 0.7
    if compact and compact in compact_key(context):
        return 0.58
    return 0.0


def triples_from_item(item: Dict[str, Any]) -> List[Triple]:
    rows: List[Triple] = []
    for triple in item.get("primitive-ontology-triples", []):
        if isinstance(triple, list) and len(triple) == 3:
            values = tuple(str(value).strip() for value in triple)
            if all(values):
                rows.append(values)  # type: ignore[arg-type]
    return rows


def serialize_triples(triples: Iterable[Triple]) -> List[List[str]]:
    return [[subject, relation, obj] for subject, relation, obj in triples]


def relation_counts(triples: Sequence[Triple]) -> Dict[str, int]:
    return dict(Counter(relation for _, relation, _ in triples).most_common())


def clean_entity(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .,:;[]{}\"'")
    text = re.sub(r"^(?:" + "|".join(BOUNDARY_WORDS) + r")\s+", "", text, flags=re.I)
    text = re.sub(
        r"^.*\b(?:including|such as|include|includes|contains|contain|encompasses|encompass)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^.*\b(?:order|class|category|group)\s+([A-Za-z0-9][A-Za-z0-9_\-/() :]{1,120})$",
        r"\1",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+(?:which|that|where|while|when)\b.*$", "", text, flags=re.I)
    text = re.sub(
        r"\b(?:within|in|for|during|across)\s+(?:the|this|these)?\s*(?:broader|general|wider)?\s*(?:group|category|class|taxonomy|system|database|context|field)\b.*$",
        "",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip(" .,:;[]{}\"'")


def repair_relation(relation: str) -> str:
    relation = relation.strip()
    aliases = {
        "is_a": "is-a",
        "isa": "is-a",
        "subclass_of": "is-a",
        "subclass of": "is-a",
        "subclass": "is-a",
        "subtype": "is-a",
        "type of": "is-a",
        "instance_of": "instance-of",
        "instance of": "instance-of",
        "part of": "part_of",
        "same_as": "same as",
        "equivalent_class": "equivalent class",
        "exact_match": "exact match",
    }
    return aliases.get(relation.lower(), relation)


def is_short_symbol_entity(entity: str, relation: str, other: str) -> bool:
    raw = entity.strip()
    key = unicode_compact_key(raw)
    if len(key) < 1 or len(key) > 2:
        return False
    if not re.fullmatch(r"[A-Za-z]{1,2}", raw):
        return False
    if norm_key(raw) in BAD_ENTITY_KEYS:
        return False
    noise = {"in", "of", "to", "as", "by", "on", "or", "is", "be", "we", "he", "me", "my", "no", "so", "up"}
    if key in noise:
        return False
    if relation not in {"is-a", "instance-of", "type"}:
        return False
    if relation == "is-a" and norm_key(other) in {"atom", "element"}:
        return True
    if relation == "instance-of":
        if unicode_compact_key(raw) == unicode_compact_key(other):
            return True
        if re.fullmatch(r"[A-Z][a-z]?", other.strip()):
            return True
    return False


def valid_ontology_entity(entity: str, allow_short: bool = False) -> bool:
    entity = unicodedata.normalize("NFKC", entity.strip())
    if len(entity) > 180:
        return False
    if len(entity) < 2 and not allow_short:
        return False
    if not unicode_compact_key(entity):
        return False
    key = norm_key(entity)
    if key and key in BAD_ENTITY_KEYS:
        return False
    if len(re.findall(r"\S+", entity)) > 20:
        return False
    return True


def clean_candidate_triple(triple: Triple) -> Optional[Triple]:
    subject = clean_entity(str(triple[0]))
    relation = repair_relation(str(triple[1]))
    obj = clean_entity(str(triple[2]))
    if relation not in set(ALLOWED_RELATIONS):
        return None
    subject_short = is_short_symbol_entity(subject, relation, obj)
    object_short = is_short_symbol_entity(obj, relation, subject)
    if not valid_ontology_entity(subject, subject_short) or not valid_ontology_entity(obj, object_short):
        return None
    if not subject_short and not entity_meets_compact_min(subject, 3):
        return None
    if not object_short and not entity_meets_compact_min(obj, 3):
        return None
    return subject, relation, obj


def entity_meets_compact_min(entity: str, minimum: int) -> bool:
    key = unicode_compact_key(entity)
    required = min(minimum, 2) if any(ord(ch) > 127 for ch in key) else minimum
    return len(key) >= required


def coerce_triple(row: Any) -> Optional[Triple]:
    if isinstance(row, list) and len(row) >= 3:
        return str(row[0]), str(row[1]), str(row[2])
    if isinstance(row, dict):
        subject = row.get("subject", row.get("s"))
        relation = row.get("relation", row.get("predicate", row.get("r")))
        obj = row.get("object", row.get("o"))
        if subject is not None and relation is not None and obj is not None:
            if isinstance(obj, list):
                obj = "; ".join(str(value).strip() for value in obj if str(value).strip())
            return str(subject), str(relation), str(obj)
    return None


def repair_truncated_json(candidate: str) -> str:
    candidate = candidate.strip()
    repairs = []
    for marker in ["]}", "]]", "]"]:
        index = candidate.rfind(marker)
        if index < 0:
            continue
        repaired = candidate[: index + len(marker)]
        if marker == "]]" and not repaired.rstrip().endswith("}"):
            repaired = '{"primitive-ontology-triples":' + repaired + "}"
        elif marker == "]" and candidate.lstrip().startswith("{") and not repaired.rstrip().endswith("}"):
            repaired += "}"
        repairs.append(repaired)
    return repairs[0] if repairs else ""


def extract_json_payload(text: str) -> Tuple[Any, str]:
    candidates: List[str] = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    for left, right in [("{", "}"), ("[", "]")]:
        start = text.find(left)
        end = text.rfind(right)
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
    last_error = "no_json"
    for candidate in candidates:
        try:
            return json.loads(candidate), ""
        except Exception as exc:
            last_error = str(exc)
            repaired = repair_truncated_json(candidate)
            if repaired and repaired != candidate:
                try:
                    return json.loads(repaired), ""
                except Exception as repaired_exc:
                    last_error = str(repaired_exc)
    return None, last_error


def dedupe_scored(rows: Iterable[ScoredTriple]) -> List[ScoredTriple]:
    best: Dict[Triple, ScoredTriple] = {}
    for row in rows:
        if row.triple not in best or row.score > best[row.triple].score:
            best[row.triple] = row
    return list(best.values())


def salvage_model_triples(text: str) -> List[ScoredTriple]:
    decoder = json.JSONDecoder()
    rows: List[ScoredTriple] = []
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(text, index)
        except Exception:
            continue
        triple = coerce_triple(payload)
        cleaned = clean_candidate_triple(triple) if triple else None
        if cleaned:
            rows.append(ScoredTriple(cleaned, "sft", 1.0, "model_json_salvage"))
    return dedupe_scored(rows)


def parse_model_output(text: str) -> ParseResult:
    payload, error = extract_json_payload(text)
    if payload is None:
        salvaged = salvage_model_triples(text)
        if salvaged:
            return ParseResult(
                salvaged,
                hard_error=error,
                soft_errors=("salvaged_truncated_json",),
                raw_length=len(text),
                salvaged_count=len(salvaged),
            )
        return ParseResult([], hard_error=error, raw_length=len(text))
    if isinstance(payload, list):
        triples_payload = payload
    elif isinstance(payload, dict):
        triples_payload = payload.get("primitive-ontology-triples", payload.get("triples", []))
    else:
        return ParseResult([], hard_error="json_not_object_or_list", raw_length=len(text))
    if not isinstance(triples_payload, list):
        return ParseResult([], hard_error="triples_not_list", raw_length=len(text))
    rows: List[ScoredTriple] = []
    diagnostics = []
    for row in triples_payload:
        triple = coerce_triple(row)
        if triple is None:
            diagnostics.append("bad_shape")
            continue
        cleaned = clean_candidate_triple(triple)
        if cleaned is None:
            diagnostics.append("invalid_triple")
            continue
        rows.append(ScoredTriple(cleaned, "sft", 1.0, "model_json"))
    return ParseResult(
        dedupe_scored(rows),
        soft_errors=tuple(sorted(set(diagnostics))),
        raw_length=len(text),
    )


def classify_entity_presence(triple: Triple, context: str) -> str:
    subject, _, obj = triple
    subject_presence = entity_presence(subject, context)
    object_presence = entity_presence(obj, context)
    if subject_presence >= 0.86 and object_presence >= 0.86:
        return "text_span"
    if subject_presence >= 0.58 or object_presence >= 0.58:
        return "normalized_text_span"
    return "ontology_prior"


def build_graph_completion_candidates(
    item: Dict[str, Any],
    graph_index: Dict[str, Any],
    seed_rows: Sequence[ScoredTriple],
) -> List[ScoredTriple]:
    context = item["context"]
    entities = set()
    for row in seed_rows:
        entities.add(row.triple[0])
        entities.add(row.triple[2])
    lowered = context.lower()
    present = []
    for forms in graph_index.get("entity_forms", {}).values():
        for form in forms.keys():
            form = str(form)
            if len(form) >= 3 and form.lower() in lowered:
                present.append(form)
    present = sorted(set(present), key=lambda value: (-len(value), value.lower()))[:16]
    entities.update(present)
    entity_keys = {compact_key(entity) for entity in entities}
    rows = []
    for record in graph_index.get("triple_counts", []):
        triple = (record["subject"], record["relation"], record["object"])
        count = int(record["count"])
        subject, relation, obj = triple
        if subject != obj or relation not in set(ALLOWED_RELATIONS) or count < 1:
            continue
        if compact_key(subject) not in entity_keys and entity_presence(subject, context) < 0.58:
            continue
        score = 0.82 + min(0.16, math.log1p(count) * 0.04)
        cleaned = clean_candidate_triple(triple)
        if cleaned:
            rows.append(
                ScoredTriple(
                    cleaned,
                    "graph_completion",
                    score,
                    f"self_edge_count={count}",
                    classify_entity_presence(cleaned, context),
                )
            )
    return dedupe_scored(rows)


def presence_bonus(triple: Triple, context: str) -> float:
    subject, _, obj = triple
    return 0.04 * entity_presence(subject, context) + 0.04 * entity_presence(obj, context)


def relation_rank_key(row: ScoredTriple) -> Tuple[float, float]:
    priority = {
        "is-a": 0.08,
        "instance-of": 0.06,
        "is defined by": 0.04,
        "equivalent class": 0.03,
        "disjoint with": 0.03,
    }.get(row.triple[1], 0.0)
    presence = {
        "text_span": 0.04,
        "normalized_text_span": 0.02,
        "ontology_prior": 0.0,
    }.get(row.entity_presence, 0.0)
    return row.score + priority + presence, row.score


def fuse_candidates(
    item: Dict[str, Any],
    rows: Sequence[ScoredTriple],
) -> Tuple[List[Triple], List[Dict[str, Any]]]:
    best: Dict[Triple, ScoredTriple] = {}
    rejected: Counter[str] = Counter()
    for row in rows:
        cleaned = clean_candidate_triple(row.triple)
        if cleaned is None:
            rejected["invalid"] += 1
            continue
        if norm_key(cleaned[0]) in FUSION_BAD_ENTITIES or norm_key(cleaned[2]) in FUSION_BAD_ENTITIES:
            rejected["bad_entity"] += 1
            continue
        weighted = row.score * SOURCE_WEIGHTS.get(row.source, 1.0)
        required = max(
            SOURCE_MIN_SCORES.get(row.source, 0.0),
            RELATION_MIN_SCORES.get(cleaned[1], RELATION_MIN_SCORES["default"]),
        )
        if weighted < required:
            rejected["below_threshold"] += 1
            continue
        scored = ScoredTriple(
            cleaned,
            row.source,
            weighted + presence_bonus(cleaned, item["context"]),
            row.evidence,
            classify_entity_presence(cleaned, item["context"]),
        )
        if cleaned not in best or scored.score > best[cleaned].score:
            best[cleaned] = scored

    ranked = sorted(best.values(), key=relation_rank_key, reverse=True)
    accepted = []
    caps: Counter[str] = Counter()
    for row in ranked:
        relation = row.triple[1]
        cap = RELATION_CAPS.get(relation, RELATION_CAPS["default"])
        if caps[relation] >= cap:
            rejected["relation_cap"] += 1
            continue
        accepted.append(row)
        caps[relation] += 1
        if len(accepted) >= MAX_TRIPLES_PER_ITEM:
            break
    log_rows = [
        {
            "triple": list(row.triple),
            "source": row.source,
            "score": row.score,
            "evidence": row.evidence,
            "entity_presence": row.entity_presence,
        }
        for row in accepted
    ]
    log_rows.append({"diagnostics": {"rejected": dict(rejected), "candidate_count": len(rows)}})
    return [row.triple for row in accepted], log_rows


def prediction_payload(
    items: Sequence[Dict[str, Any]],
    triples_by_id: Dict[str, List[Triple]],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "primitive-ontology-triples": serialize_triples(triples_by_id.get(item["id"], [])),
        }
        for item in items
    ]


def validate_prediction_payload(
    payload: Any,
    expected_ids: Optional[Sequence[str]] = None,
    require_order: bool = True,
) -> List[str]:
    if not isinstance(payload, list):
        return ["payload is not a list"]
    errors = []
    seen_ids = set()
    item_ids = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(f"item {index} is not an object")
            continue
        extra = sorted(set(item) - {"id", "primitive-ontology-triples"})
        if extra:
            errors.append(f"item {index} has unexpected keys: {extra}")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"item {index} has no id")
        elif item_id in seen_ids:
            errors.append(f"duplicate id: {item_id}")
        else:
            seen_ids.add(item_id)
            item_ids.append(item_id)
        triples = item.get("primitive-ontology-triples")
        if not isinstance(triples, list):
            errors.append(f"{item_id}: triples is not a list")
            continue
        for triple_index, triple in enumerate(triples):
            if not isinstance(triple, list) or len(triple) != 3:
                errors.append(f"{item_id}: triple {triple_index} is not a length-3 list")
                continue
            if not all(isinstance(value, str) and value for value in triple):
                errors.append(f"{item_id}: triple {triple_index} has an empty or non-string value")
            if triple[1] not in set(ALLOWED_RELATIONS):
                errors.append(f"{item_id}: triple {triple_index} uses an unsupported relation")
    if expected_ids is not None:
        expected = list(expected_ids)
        missing = [item_id for item_id in expected if item_id not in set(item_ids)]
        extra_ids = [item_id for item_id in item_ids if item_id not in set(expected)]
        if missing:
            errors.append(f"missing {len(missing)} expected ids; first missing id={missing[0]}")
        if extra_ids:
            errors.append(f"found {len(extra_ids)} unexpected ids; first unexpected id={extra_ids[0]}")
        if require_order and item_ids != expected:
            errors.append("prediction ids are not in expected order")
    return errors
