"""Primary command-line entry point for training and Task A inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from data_pipeline import (
    build_graph_index,
    prepare_training_data,
    require_items,
    write_json,
)
from model_pipeline import DEFAULT_BASE_MODEL, predict_items, train_adapter
from ontology_pipeline import validate_prediction_payload


def external_work_dir(raw_path: str) -> Path:
    project_dir = Path(__file__).resolve().parent
    work_dir = Path(raw_path).expanduser().resolve()
    try:
        work_dir.relative_to(project_dir)
    except ValueError:
        pass
    else:
        raise ValueError("--work-dir must be outside the submission project directory")
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def execute_train(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = external_work_dir(args.work_dir)
    prepared = prepare_training_data(
        Path(args.task_a_train).expanduser().resolve(),
        Path(args.task_b_train).expanduser().resolve(),
        work_dir,
    )
    adapter_dir = work_dir / "adapter"
    result = train_adapter(
        prepared["mixed_train"],
        prepared["task_a_dev"],
        adapter_dir,
        base_model=args.base_model,
        local_files_only=args.local_files_only,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
    )
    summary = {
        "status": "ok",
        "stage": "train",
        "work_dir": str(work_dir),
        "adapter_dir": str(adapter_dir),
        "data": prepared["summary"],
        "training": result,
    }
    write_json(work_dir / "training_summary.json", summary)
    return summary


def execute_predict(args: argparse.Namespace, adapter_override: Path | None = None) -> Dict[str, Any]:
    work_dir = external_work_dir(args.work_dir)
    task_a_train = Path(args.task_a_train).expanduser().resolve()
    test_path = Path(args.test).expanduser().resolve()
    test_items = require_items(test_path, "Task A test")
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max-items must be positive")
        test_items = test_items[: args.max_items]
    graph_index_path = work_dir / "graph_index.json"
    graph_index = build_graph_index(task_a_train, graph_index_path)
    adapter_dir = adapter_override or Path(args.adapter_dir).expanduser().resolve()
    predictions, inference_log = predict_items(
        test_items,
        graph_index,
        adapter_dir,
        base_model=args.base_model,
        local_files_only=args.local_files_only,
    )
    expected_ids = [item["id"] for item in test_items]
    errors = validate_prediction_payload(predictions, expected_ids=expected_ids, require_order=True)
    if errors:
        raise RuntimeError("Prediction validation failed:\n" + "\n".join(errors[:50]))
    prediction_path = work_dir / "task_a_predictions.json"
    log_path = work_dir / "inference_log.json"
    validation_path = work_dir / "validation.json"
    write_json(prediction_path, predictions)
    write_json(log_path, inference_log)
    validation = {
        "status": "ok",
        "items": len(predictions),
        "require_order": True,
        "errors": [],
        "prediction_path": str(prediction_path),
        "test_path": str(test_path),
    }
    write_json(validation_path, validation)
    return {
        "status": "ok",
        "stage": "predict",
        "work_dir": str(work_dir),
        "adapter_dir": str(adapter_dir),
        "predictions": str(prediction_path),
        "inference_log": str(log_path),
        "validation": validation,
        "summary": inference_log["summary"],
    }


def execute_run(args: argparse.Namespace) -> Dict[str, Any]:
    training = execute_train(args)
    prediction = execute_predict(args, Path(training["adapter_dir"]))
    result = {
        "status": "ok",
        "stage": "run",
        "training": training,
        "prediction": prediction,
    }
    write_json(external_work_dir(args.work_dir) / "run_summary.json", result)
    return result


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--local-files-only", action="store_true")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-a-train", required=True)
    parser.add_argument("--task-b-train", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    add_runtime_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a weighted QLoRA ontology generator for LLMs4OL 2026 Task A."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Prepare data, train the adapter, and predict Task A test items.")
    add_training_arguments(run_parser)
    run_parser.add_argument("--test", required=True)
    run_parser.add_argument("--max-items", type=int, default=None)

    train_parser = subparsers.add_parser("train", help="Prepare data and train the adapter.")
    add_training_arguments(train_parser)

    predict_parser = subparsers.add_parser("predict", help="Load an adapter and predict Task A test items.")
    predict_parser.add_argument("--task-a-train", required=True)
    predict_parser.add_argument("--test", required=True)
    predict_parser.add_argument("--adapter-dir", required=True)
    predict_parser.add_argument("--work-dir", required=True)
    predict_parser.add_argument("--max-items", type=int, default=None)
    add_runtime_arguments(predict_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        result = execute_train(args)
    elif args.command == "predict":
        result = execute_predict(args)
    else:
        result = execute_run(args)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
