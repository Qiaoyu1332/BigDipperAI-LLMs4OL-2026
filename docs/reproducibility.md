# Reproducibility record

This document records the software, hardware, runtime, evaluator, and artifact boundaries of the BigDipperAI submission system. It complements the executable commands in the repository README.

## Recorded environment

| Component | Recorded value |
|---|---|
| Operating environment | Linux |
| Python | 3.11.15 |
| PyTorch | 2.12.1+cu130 |
| CUDA runtime | 13.0 |
| Transformers | 5.13.0.dev0 |
| Transformers source commit | `c21da1b58819a597e4a250622819d1f15e14d771` |
| Datasets | 5.0.0 |
| PEFT | 0.19.1 |
| bitsandbytes | 0.49.2 |
| Accelerate | 1.14.0 |
| safetensors | 0.8.0 |
| GPUs | 2 × NVIDIA H100 80GB HBM3 |
| Driver observed during the run | 580.105.08 |
| Compute path | `bfloat16`; selected by the program after the GPUs reported support |

`requirements.txt` lists the pinned direct dependencies for a clean installation. The submission environment used the development build of Transformers at the source commit listed above; reproducing that environment exactly may require installation from that commit instead of a wheel.

## Data boundary

The recorded training input comprised 4,303 official Task Flagship records and 2,774 Task B records produced by the repository's conversion code, for 7,077 supervised records in total. The official test input comprised 2,018 records. Challenge data are not redistributed here.

The mixed training JSONL used for the recorded run had SHA256:

```text
6bd0f29566ab5895a543031079a3f341e159416de68545cbc4409f4b3d161b2b
```

## Training and generation configuration

- Base model: `Qwen/Qwen3-8B`
- Quantization: 4-bit `nf4` with double quantization
- LoRA: `r=32`, `lora_alpha=64`, `lora_dropout=0.05`
- Epochs: 3
- Learning rate: `2e-4`
- Per-device batch size: 1
- Gradient accumulation: 8
- Seeds: 42 for training and data preparation
- Input limit: 3,072 tokens
- Initial output limit: 1,536 tokens
- Retry output limit: 3,072 tokens
- Generation: Qwen3 thinking disabled, `do_sample=false`, `num_beams=1`

The JSON parser, retry trigger, graph completion, source thresholds, relation thresholds, relation caps, 48-triple total cap, and final validation are implemented in the tracked Python modules.

## Runtime boundary

The successful training stage recorded 9,539 seconds, approximately 2 hours 39 minutes. The recorded interval from training completion to validated submission was approximately 4 hours 17 minutes. That interval includes sharded inference on two GPUs, output merging, post-processing, and validation; it is an end-to-end operational interval rather than isolated model execution time.

## Recorded artifact identifiers

These identifiers establish provenance but the corresponding large or restricted artifacts are not distributed in the repository.

| Artifact | SHA256 |
|---|---|
| Training configuration | `d3b0849d5d5c29afd057bd145d3dd578cb75a3c68ff21fcf7ffc6c05b518ecde` |
| Adapter weights | `626efdea6553e0d337936c9ef790822806e9cbeb355bea2e4cd51c247d9c2269` |
| Adapter configuration | `35930ca10f0d2cbb21b06576c05e2632abfca630875c43ef847a9db7a49a8d2e` |
| Final submission JSON | `50d2d452f7bd460494f298a17a47ec95a38f2ecd364090712399ba68c82eef96` |
| Submission manifest | `1197508f778c419f3c09394b1a08c93e42129b18015ef002b747442efea828c6` |

## Official evaluator

The released Task A evaluator was reviewed at LLMs4OL Challenge repository commit:

```text
228049f47c045a72546f9c3870994ddd0e066dc2
```

The evaluator exposes exact, fuzzy, and semantic matching modes and computes Edge F1, Neighborhood Similarity, Taxonomy Similarity, and Graph Similarity. The organizer-held reference outputs are required for official evaluation, so this repository does not report or reconstruct organizer scores.

## Verification checklist

1. Clone the repository and record `git rev-parse HEAD`.
2. Create a Python 3.11 environment and install `requirements.txt`.
3. Run `python unittest.py`; all seven tests should pass without downloading the base model.
4. Obtain the challenge data and Qwen3-8B from their official sources.
5. Run the complete command in the README with an external work directory.
6. Retain the generated summaries, validation report, dependency inventory, command line, hardware inventory, and SHA256 values with the run record.
