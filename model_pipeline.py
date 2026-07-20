"""Weighted QLoRA training and deterministic adapter inference."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data_pipeline import load_jsonl
from ontology_pipeline import (
    ScoredTriple,
    apply_chat_template_json,
    build_graph_completion_candidates,
    build_prompt,
    fuse_candidates,
    parse_model_output,
    prediction_payload,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
MAX_INPUT_TOKENS = 3072
MAX_OUTPUT_TOKENS = 1536
RETRY_OUTPUT_TOKENS = 3072
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LEARNING_RATE = 0.0002
NUM_TRAIN_EPOCHS = 3.0
RELATION_WEIGHTS = {"instance-of": 3.0, "type": 3.0, "default": 1.0}
RELATION_WEIGHT_MAX = 3.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def remove_submission_directory_from_import_path() -> None:
    """Prevent the required unittest.py file from shadowing the standard library."""
    submission_dir = Path(__file__).resolve().parent
    for entry in list(sys.path):
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except OSError:
            continue
        if resolved == submission_dir:
            sys.path.remove(entry)


def relation_counts_from_target(target_json: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    try:
        payload = json.loads(target_json)
    except Exception:
        return counts
    for row in payload.get("primitive-ontology-triples", []):
        if isinstance(row, list) and len(row) == 3:
            relation = str(row[1])
            counts[relation] = counts.get(relation, 0) + 1
    return counts


def sample_weight_for_row(row: Dict[str, Any]) -> float:
    counts = row.get("relation_counts") or relation_counts_from_target(row.get("target_json", ""))
    clean_counts = {
        str(relation): int(count)
        for relation, count in counts.items()
        if count is not None and int(count) > 0
    }
    total = sum(clean_counts.values())
    if total <= 0:
        return 1.0
    weighted = sum(
        count * RELATION_WEIGHTS.get(relation, RELATION_WEIGHTS["default"])
        for relation, count in clean_counts.items()
    )
    return min(RELATION_WEIGHT_MAX, max(1.0, weighted / total))


class CausalJsonCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        max_length = max(len(row["input_ids"]) for row in features)
        input_ids = []
        labels = []
        attention_mask = []
        sample_weight = []
        for row in features:
            padding = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            attention_mask.append(row["attention_mask"] + [0] * padding)
            sample_weight.append(float(row.get("sample_weight", 1.0)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float),
        }


def resolve_resume_checkpoint(output_dir: Path) -> Optional[str]:
    checkpoints = []
    if output_dir.exists():
        for path in output_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except ValueError:
                continue
            checkpoints.append((step, path))
    return str(max(checkpoints, key=lambda item: item[0])[1]) if checkpoints else None


def train_adapter(
    train_jsonl: Path,
    dev_jsonl: Path,
    adapter_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    local_files_only: bool = False,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    remove_submission_directory_from_import_path()
    import torch
    import torch.nn.functional as functional
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )

    train_rows = load_jsonl(train_jsonl)
    eval_rows = load_jsonl(dev_jsonl)
    if max_train_samples is not None:
        train_rows = train_rows[:max_train_samples]
    if max_eval_samples is not None:
        eval_rows = eval_rows[:max_eval_samples]
    if not train_rows or not eval_rows:
        raise ValueError("Training and evaluation data must both contain at least one row")

    def tokenize(row: Dict[str, Any], is_eval: bool) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "Return only valid JSON for the ontology extraction task."},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt_text = apply_chat_template_json(tokenizer, messages, enable_thinking=False)
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = tokenizer(
            row["target_json"] + tokenizer.eos_token,
            truncation=True,
            max_length=MAX_OUTPUT_TOKENS,
            add_special_tokens=False,
        )["input_ids"]
        return {
            "input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "sample_weight": 1.0 if is_eval else sample_weight_for_row(row),
        }

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)
    train_dataset = train_dataset.map(
        lambda row: tokenize(row, False),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda row: tokenize(row, True),
        remove_columns=eval_dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        save_total_limit=3,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_steps=max_steps if max_steps is not None else -1,
        seed=42,
        data_seed=42,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to=[],
        eval_strategy="steps",
        remove_unused_columns=False,
    )

    class WeightedCausalTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
            sample_weight = inputs.pop("sample_weight", None)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            token_loss = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            mask = shift_labels.ne(-100)
            per_sample_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            if sample_weight is None:
                sample_weight = torch.ones_like(per_sample_loss)
            else:
                sample_weight = sample_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
            loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1e-6)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedCausalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalJsonCollator(tokenizer.pad_token_id),
    )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    resume = resolve_resume_checkpoint(adapter_dir)
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return {
        "base_model": base_model,
        "adapter_dir": str(adapter_dir.resolve()),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "global_step": int(train_result.global_step),
        "training_loss": float(train_result.training_loss),
    }


class AdapterGenerator:
    def __init__(self, base_model: str, adapter_dir: Path, local_files_only: bool = False) -> None:
        remove_submission_directory_from_import_path()
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=local_files_only,
            device_map="auto",
            torch_dtype="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            ),
        )
        self.model = PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, item: Dict[str, Any], max_new_tokens: int = MAX_OUTPUT_TOKENS) -> str:
        import torch

        messages = [
            {"role": "system", "content": "Return only valid JSON for the ontology extraction task."},
            {"role": "user", "content": build_prompt(item)},
        ]
        text = apply_chat_template_json(self.tokenizer, messages, enable_thinking=False)
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(next(self.model.parameters()).device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def _attempt_record(attempt: int, result: Any) -> Dict[str, Any]:
    return {
        "attempt": attempt,
        "parse_error": result.hard_error,
        "parse_soft_errors": list(result.soft_errors),
        "raw_length": result.raw_length,
        "salvaged_count": result.salvaged_count,
        "sft_rows": len(result.rows),
    }


def _rows_and_fusion(
    item: Dict[str, Any],
    parse_result: Any,
    graph_index: Dict[str, Any],
) -> Tuple[List[ScoredTriple], List[Tuple[str, str, str]], List[Dict[str, Any]]]:
    rows = list(parse_result.rows)
    rows.extend(build_graph_completion_candidates(item, graph_index, parse_result.rows))
    triples, accepted_log = fuse_candidates(item, rows)
    return rows, triples, accepted_log


def predict_items(
    items: Sequence[Dict[str, Any]],
    graph_index: Dict[str, Any],
    adapter_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    local_files_only: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    generator = AdapterGenerator(base_model, adapter_dir, local_files_only)
    triples_by_id = {}
    logs = []
    for item_index, item in enumerate(items):
        started = time.time()
        raw_text = generator.generate(item)
        result = parse_model_output(raw_text)
        attempts = [_attempt_record(1, result)]
        retry_reason = ""
        if result.hard_error:
            retry_reason = "parse_error"
            retry_text = generator.generate(item, RETRY_OUTPUT_TOKENS)
            retry_result = parse_model_output(retry_text)
            attempts.append(_attempt_record(2, retry_result))
            if retry_result.rows or not retry_result.hard_error:
                raw_text = retry_text
                result = retry_result

        rows, triples, accepted_log = _rows_and_fusion(item, result, graph_index)
        attempts[-1]["final_count"] = len(triples)
        if not triples and not retry_reason:
            retry_reason = "empty_output"
            retry_text = generator.generate(item, RETRY_OUTPUT_TOKENS)
            retry_result = parse_model_output(retry_text)
            retry_rows, retry_triples, retry_log = _rows_and_fusion(item, retry_result, graph_index)
            attempt = _attempt_record(2, retry_result)
            attempt["final_count"] = len(retry_triples)
            attempts.append(attempt)
            if retry_triples or (result.hard_error and not retry_result.hard_error):
                raw_text = retry_text
                result = retry_result
                rows = retry_rows
                triples = retry_triples
                accepted_log = retry_log

        triples_by_id[item["id"]] = triples
        source_counts = Counter(
            row.get("source", "diagnostics")
            for row in accepted_log
            if "triple" in row
        )
        logs.append(
            {
                "id": item["id"],
                "item_index": item_index,
                "final_count": len(triples),
                "candidate_count": len(rows),
                "source_counts": dict(source_counts),
                "parse_error": result.hard_error,
                "parse_soft_errors": list(result.soft_errors),
                "raw_length": result.raw_length,
                "salvaged_count": result.salvaged_count,
                "attempt_count": len(attempts),
                "retry_reason": retry_reason,
                "attempts": attempts,
                "seconds": time.time() - started,
                "triples": accepted_log,
                "raw_text": raw_text,
            }
        )
    return prediction_payload(items, triples_by_id), {
        "items": logs,
        "summary": summarize_logs(logs),
    }


def summarize_logs(logs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not logs:
        return {"items": 0}
    counts = [int(row["final_count"]) for row in logs]
    seconds = [float(row["seconds"]) for row in logs]
    sources: Counter[str] = Counter()
    for row in logs:
        sources.update(row["source_counts"])
    return {
        "items": len(logs),
        "parse_failures": sum(1 for row in logs if row["parse_error"]),
        "parse_soft_failures": sum(1 for row in logs if row["parse_soft_errors"]),
        "retried_items": sum(1 for row in logs if int(row["attempt_count"]) > 1),
        "salvaged_triples": sum(int(row["salvaged_count"]) for row in logs),
        "empty_outputs": sum(1 for count in counts if count == 0),
        "avg_seconds": statistics.mean(seconds),
        "median_seconds": statistics.median(seconds),
        "avg_final_triples": statistics.mean(counts),
        "median_final_triples": statistics.median(counts),
        "final_triples_min": min(counts),
        "final_triples_max": max(counts),
        "final_triples_distribution": dict(sorted(Counter(counts).items())),
        "source_counts": dict(sources.most_common()),
    }
