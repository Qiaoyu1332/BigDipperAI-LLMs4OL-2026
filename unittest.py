"""Unit tests for the submission's critical data and ontology components."""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path

from data_pipeline import build_graph_index, convert_task_b_items, prepare_training_data
from model_pipeline import sample_weight_for_row
from ontology_pipeline import (
    ALLOWED_RELATIONS,
    ScoredTriple,
    build_graph_completion_candidates,
    build_prompt,
    fuse_candidates,
    parse_model_output,
    validate_prediction_payload,
)


def assert_equal(actual, expected, message: str = "") -> None:
    if actual != expected:
        raise AssertionError(message or f"Expected {expected!r}, got {actual!r}")


def assert_true(value, message: str = "") -> None:
    if not value:
        raise AssertionError(message or f"Expected a truthy value, got {value!r}")


def test_prompt_contract() -> None:
    prompt = json.loads(build_prompt({"id": "x", "context": "Alpha is a class."}))
    assert_equal(prompt["id"], "x")
    assert_equal(prompt["allowed_relations"], ALLOWED_RELATIONS)
    assert_equal(list(prompt["output_schema"]), ["primitive-ontology-triples"])


def test_task_b_conversion() -> None:
    items = [
        {
            "id": "b1",
            "context": "Alpha is a Beta.",
            "initial-primitive-ontology-triples": [["Alpha", "is-a", "Beta"]],
            "extended-primitive-ontology-triples": [
                ["Alpha", "is-a", "Beta"],
                ["Alpha", "unsupported", "Gamma"],
            ],
        }
    ]
    rows, metadata = convert_task_b_items(items)
    assert_equal(len(rows), 1)
    assert_equal(rows[0]["id"], "task_b_reuse:b1")
    assert_equal(rows[0]["gold_triple_count"], 1)
    assert_equal(metadata["task_b_duplicate_triples_removed"], 1)
    assert_equal(metadata["task_b_dropped_relation_counts"]["unsupported"], 1)


def test_relation_weighting() -> None:
    assert_equal(sample_weight_for_row({"relation_counts": {"is-a": 1}}), 1.0)
    assert_equal(sample_weight_for_row({"relation_counts": {"type": 2}}), 3.0)
    assert_equal(
        sample_weight_for_row({"relation_counts": {"is-a": 2, "instance-of": 2}}),
        2.0,
    )


def test_model_output_parsing() -> None:
    text = '{"primitive-ontology-triples":[["Alpha","subclass_of","Beta"],["x","bad","y"]]}'
    result = parse_model_output(text)
    assert_equal([row.triple for row in result.rows], [("Alpha", "is-a", "Beta")])
    assert_true("invalid_triple" in result.soft_errors)


def test_graph_completion_and_fusion() -> None:
    item = {"id": "x", "context": "Carbon is an element."}
    index = {
        "entity_forms": {"carbon": {"Carbon": 3}},
        "triple_counts": [
            {"subject": "Carbon", "relation": "instance-of", "object": "Carbon", "count": 100}
        ],
    }
    seed = [ScoredTriple(("Carbon", "is-a", "element"), "sft", 1.0, "model_json")]
    graph_rows = build_graph_completion_candidates(item, index, seed)
    assert_equal(len(graph_rows), 1)
    triples, log_rows = fuse_candidates(item, seed + graph_rows)
    assert_true(("Carbon", "is-a", "element") in triples)
    assert_true(("Carbon", "instance-of", "Carbon") in triples)
    assert_true("diagnostics" in log_rows[-1])


def test_prediction_validation() -> None:
    payload = [{"id": "x", "primitive-ontology-triples": [["Alpha", "is-a", "Beta"]]}]
    assert_equal(validate_prediction_payload(payload, ["x"], True), [])
    assert_true(validate_prediction_payload(payload, ["y"], True))


def test_file_pipeline() -> None:
    task_a = [
        {
            "id": "a1",
            "context": "Alpha is a Beta.",
            "primitive-ontology-triples": [["Alpha", "is-a", "Beta"]],
        },
        {
            "id": "a2",
            "context": "Gamma is a Delta.",
            "primitive-ontology-triples": [["Gamma", "is-a", "Delta"]],
        },
    ]
    task_b = [
        {
            "id": "b1",
            "context": "Epsilon is a Zeta.",
            "initial-primitive-ontology-triples": [["Epsilon", "is-a", "Zeta"]],
            "extended-primitive-ontology-triples": [],
        }
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task_a_path = root / "task_a.json"
        task_b_path = root / "task_b.json"
        task_a_path.write_text(json.dumps(task_a), encoding="utf-8")
        task_b_path.write_text(json.dumps(task_b), encoding="utf-8")
        prepared = prepare_training_data(task_a_path, task_b_path, root / "work")
        assert_equal(prepared["summary"]["mixed_train_rows"], 3)
        graph = build_graph_index(task_a_path, root / "work" / "graph_index.json")
        assert_equal(graph["indexed_items"], 2)
        assert_equal(graph["triple_counts"][0]["count"], 1)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failures.append(test.__name__)
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    if failures:
        raise SystemExit(f"{len(failures)} test(s) failed: {', '.join(failures)}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
