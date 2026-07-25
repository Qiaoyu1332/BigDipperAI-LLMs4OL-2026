# BigDipperAI at LLMs4OL 2026 Task Flagship

[![Unit tests](https://github.com/Qiaoyu1332/BigDipperAI-LLMs4OL-2026/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/Qiaoyu1332/BigDipperAI-LLMs4OL-2026/actions/workflows/unit-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for the BigDipperAI team's submission to the [LLMs4OL 2026 Task Flagship](https://sites.google.com/view/llms4ol2026/flagship-task). The scripts prepare training data, train a Qwen3-8B QLoRA adapter, generate ontology triples, apply post-processing, and validate the submission JSON.

Challenge data, model weights, trained adapters, prediction files, and organizer reference answers are not included.

## Pipeline

```mermaid
flowchart LR
    A["Task Flagship training data"] --> D["Mixed supervision"]
    B["Task B Reuse data"] --> C["Clean, normalize, filter, convert"] --> D
    D --> E["Relation-aware weighted 4-bit QLoRA"]
    E --> F["Qwen3-8B adapter"]
    G["Task Flagship test input"] --> H["Thinking-off greedy JSON generation"]
    F --> H
    H --> I["JSON repair and bounded retry"]
    A --> J["Training-derived graph index"]
    I --> K["Self-edge completion and score-based fusion"]
    J --> K
    K --> L["Schema, ID, order, and relation validation"]
    L --> M["Submission JSON"]
```

## Installation

- Linux
- Python 3.11
- CUDA-capable NVIDIA GPU suitable for Qwen3-8B 4-bit fine-tuning
- Access to `Qwen/Qwen3-8B`, or a complete local cache

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data format

Obtain the data under the official challenge terms. The commands expect:

- Task Flagship training JSON: `id`, `context`, and `primitive-ontology-triples`.
- Task B Reuse training JSON: `id`, `context`, `initial-primitive-ontology-triples`, and `extended-primitive-ontology-triples`.
- Task Flagship test JSON: `id` and `context`.

[`examples/`](examples/) contains fictional field examples only, not challenge records.

## Run

Run the complete workflow:

```bash
python main.py run \
  --task-a-train /data/train_task_a.json \
  --task-b-train /data/train_task_b.json \
  --test /data/test_task_a_input.json \
  --work-dir /runs/bigdipperai
```

Train only:

```bash
python main.py train \
  --task-a-train /data/train_task_a.json \
  --task-b-train /data/train_task_b.json \
  --work-dir /runs/bigdipperai
```

Predict with an existing adapter:

```bash
python main.py predict \
  --task-a-train /data/train_task_a.json \
  --test /data/test_task_a_input.json \
  --adapter-dir /runs/bigdipperai/adapter \
  --work-dir /runs/bigdipperai-predict
```

Use `--local-files-only` when the base model is fully cached. `--max-train-samples`, `--max-eval-samples`, `--max-steps`, and `--max-items` are smoke-test controls.

## Outputs

The selected work directory contains the trained `adapter/`, `task_a_predictions.json`, `inference_log.json`, `validation.json`, `training_summary.json`, and `run_summary.json`.

## Tests

```bash
python unittest.py
```

The seven tests cover data preparation, prompt construction, relation-aware weighting, parsing, graph completion, fusion, and payload validation.

## Files

- `main.py`: command-line entry point.
- `data_pipeline.py`: data conversion and preparation.
- `model_pipeline.py`: training and generation.
- `ontology_pipeline.py`: parsing, post-processing, and validation.
- `unittest.py`: unit tests.

## License

Source code is released under the [MIT License](LICENSE).
