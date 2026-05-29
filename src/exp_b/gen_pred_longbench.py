#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LONGBENCH_TASKS = ["hotpotqa", "qasper", "gov_report", "repobench-p"]
LONGBENCH_TASK_MAX_NEW_TOKENS = {
    "hotpotqa": 32,
    "qasper": 64,
    "gov_report": 256,
    "repobench-p": 128,
}

LONGBENCH_PROMPTS = {
    "hotpotqa": {
        "prefix": "Answer the question based on the given passages. Only give the final short answer.\n\nPassages:\n",
        "suffix": "\n\nQuestion: {question}\nAnswer:",
    },
    "qasper": {
        "prefix": "Read the following paper excerpt and answer the question concisely. If the answer is not stated in the document, say so directly.\n\nPaper:\n",
        "suffix": "\n\nQuestion: {question}\nAnswer:",
    },
    "gov_report": {
        "prefix": "Write a concise summary of the following government report. Focus on the key findings and conclusions.\n\nReport:\n",
        "suffix": "\n\nSummary:",
    },
    "repobench-p": {
        "prefix": "Complete the code snippet based on the repository context. Only output the missing code continuation, with no explanation.\n\nRepository context:\n",
        "suffix": "\n\nCode prefix:\n{question}\n\nCompletion:",
    },
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_sample_id(example: Dict[str, Any], row_idx: int) -> str:
    for key in ["sample_id", "_id", "id", "index"]:
        value = example.get(key)
        if value is not None:
            return str(value)
    return str(row_idx)


def load_existing_predictions(pred_path: Path) -> List[Dict[str, Any]]:
    if not pred_path.exists():
        return []
    return read_jsonl(pred_path)


def get_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_input_device(model, fallback: str = "cuda:0") -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for device in hf_device_map.values():
            if device in ("cpu", "disk"):
                continue
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            return torch.device(str(device))

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def resolve_task_max_new_tokens(task: str, global_override: int, default_map: Dict[str, int], fallback: int = 128) -> int:
    if global_override > 0:
        return global_override
    return default_map.get(task, fallback)


def render_prompt(tokenizer, raw_prompt: str, apply_chat_template: bool, system_prompt: Optional[str]) -> str:
    if not apply_chat_template:
        return raw_prompt
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("tokenizer.chat_template is None, cannot apply chat template.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": raw_prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    raw_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    apply_chat_template: bool,
    system_prompt: Optional[str],
    model_input_device: torch.device,
) -> Tuple[str, int, int, float]:
    t0 = time.time()
    prompt = render_prompt(tokenizer, raw_prompt, apply_chat_template, system_prompt)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model_input_device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_input_device)

    input_len = int(input_ids.shape[-1])
    do_sample = temperature > 0.0
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["temperature"] = 1.0
        generation_kwargs["top_p"] = 1.0

    gen_ids = model.generate(**generation_kwargs)
    new_ids = gen_ids[0, input_len:]
    pred = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    elapsed = time.time() - t0
    return pred, input_len, int(new_ids.shape[-1]), elapsed


def build_longbench_prompt(
    example: Dict[str, Any],
    task: str,
    tokenizer=None,
    max_input_tokens: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    spec = LONGBENCH_PROMPTS[task]
    question = str(example.get("input", ""))
    context = str(example.get("context", ""))
    suffix = spec["suffix"].format(question=question)
    truncated = False

    if tokenizer is not None and max_input_tokens > 0:
        overhead_ids = tokenizer.encode(spec["prefix"] + suffix, add_special_tokens=False)
        budget = max_input_tokens - len(overhead_ids)
        if budget <= 0:
            raise ValueError(f"max_input_tokens={max_input_tokens} too small for prompt overhead ({len(overhead_ids)} tokens)")
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(ctx_ids) > budget:
            # keep the tail of the context (most relevant for retrieval tasks)
            ctx_ids = ctx_ids[-budget:]
            context = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            truncated = True

    raw_prompt = spec["prefix"] + context + suffix
    return raw_prompt, {
        "benchmark": "longbench",
        "task": task,
        "context_truncated": truncated,
    }


def generate_predictions_for_rows(
    *,
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    pred_file: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    apply_chat_template: bool,
    system_prompt: Optional[str],
    limit: int,
    resume: bool,
    model_input_device: torch.device,
    prompt_builder,
) -> Dict[str, Any]:
    if limit > 0:
        rows = rows[:limit]

    if pred_file.exists() and not resume:
        pred_file.unlink()

    existing_rows = load_existing_predictions(pred_file) if resume else []
    if existing_rows:
        if any("sample_id" not in row for row in existing_rows):
            raise ValueError(f"Existing prediction file lacks sample_id: {pred_file}")
        if len(existing_rows) > len(rows):
            raise ValueError(f"Prediction file has more rows than dataset: {pred_file}")

    reused_count = 0
    generated_count = 0

    for row_idx, ex in enumerate(tqdm(rows, desc=pred_file.stem)):
        sample_id = make_sample_id(ex, row_idx)

        if row_idx < len(existing_rows):
            existing = existing_rows[row_idx]
            if str(existing.get("sample_id")) != sample_id:
                raise ValueError(
                    f"Resume mismatch at row {row_idx} for {pred_file}: "
                    f"existing sample_id={existing.get('sample_id')} current sample_id={sample_id}"
                )
            reused_count += 1
            continue

        raw_prompt, prompt_meta = prompt_builder(ex)
        pred, input_tokens, new_tokens, elapsed = generate_one(
            model=model,
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
            model_input_device=model_input_device,
        )

        out = dict(ex)
        out["sample_id"] = sample_id
        out["pred"] = pred
        out["input_prompt"] = raw_prompt
        out["input_tokens"] = input_tokens
        out["generated_tokens"] = new_tokens
        out["generation_seconds"] = elapsed
        out.update(prompt_meta)

        append_jsonl(pred_file, [out])
        generated_count += 1

    return {
        "pred_file": str(pred_file),
        "num_examples": len(rows),
        "reused_predictions": reused_count,
        "new_predictions": generated_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions for Exp B LongBench tasks.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--longbench_data_root", required=True)
    parser.add_argument("--longbench_tasks", nargs="+", default=LONGBENCH_TASKS)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=0,
        help="Global override. Use 0 to apply task defaults (hotpotqa=32, qasper=64, gov_report=256, repobench-p=128).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--model_device", default="cuda:0")
    parser.add_argument("--attn_implementation", default=None, choices=[None, "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="Truncate context so total input tokens stay within this limit. 0 = no truncation. "
             "Recommended for Mamba on 24GB: 20000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Prevents fragmentation OOM on Mamba with long contexts (PyTorch caching allocator)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": get_dtype(args.dtype),
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(args.model_device)
    model.eval()
    model_input_device = resolve_input_device(model, fallback=args.model_device)
    print(f"Model input device: {model_input_device}")

    longbench_root = Path(args.longbench_data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for task in args.longbench_tasks:
        data_file = longbench_root / f"{task}.jsonl"
        if not data_file.exists():
            print(f"[SKIP] Missing LongBench file: {data_file}")
            continue
        rows = read_jsonl(data_file)
        pred_file = out_root / f"{task}.pred.jsonl"
        print("=" * 80)
        print(f"benchmark: longbench | task={task}")
        print(f"data_file: {data_file}")
        print(f"pred_file: {pred_file}")
        print("=" * 80)
        task_max_new_tokens = resolve_task_max_new_tokens(task, args.max_new_tokens, LONGBENCH_TASK_MAX_NEW_TOKENS, fallback=128)
        result = generate_predictions_for_rows(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            pred_file=pred_file,
            max_new_tokens=task_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            apply_chat_template=args.apply_chat_template,
            system_prompt=args.system_prompt,
            limit=args.limit,
            resume=args.resume,
            model_input_device=model_input_device,
            prompt_builder=lambda ex, task=task: build_longbench_prompt(
                ex, task,
                tokenizer=tokenizer if args.max_input_tokens > 0 else None,
                max_input_tokens=args.max_input_tokens,
            ),
        )
        print(
            f"[DONE] benchmark=longbench task={task} "
            f"num_examples={result['num_examples']} reused={result['reused_predictions']} "
            f"generated={result['new_predictions']}"
        )

    print("=" * 80)
    print(f"Prediction generation finished under: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
