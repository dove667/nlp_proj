#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "length",
        "task",
        "score",
        "num_examples",
        "correct",
        "null_predictions",
        "avg_input_tokens",
        "avg_generated_tokens",
        "total_generation_seconds",
        "data_file",
        "pred_file",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\x00-\x1f]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def answer_in_prediction(answer: str, pred: str) -> bool:
    """
    RULER NIAH retrieval scoring.

    对 niah_single_1 / niah_multikey_1：
    outputs 通常是 magic numbers。
    这里判断 expected answer 是否出现在模型输出中。
    """
    answer_norm = normalize_text(answer)
    pred_norm = normalize_text(pred)

    if not answer_norm:
        return False

    # 数字答案避免 "12" 误匹配 "3124"
    if re.fullmatch(r"[0-9,\s.\-]+", answer_norm):
        pattern = r"(?<!\d)" + re.escape(answer_norm) + r"(?!\d)"
        return re.search(pattern, pred_norm) is not None

    return answer_norm in pred_norm


def score_example(pred: str, outputs: List[str]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for ans in outputs:
        if not answer_in_prediction(str(ans), pred):
            missing.append(str(ans))
    return len(missing) == 0, missing


def get_outputs(example: Dict[str, Any]) -> List[str]:
    outputs = example.get("outputs", None)
    if outputs is None:
        outputs = example.get("output", None)
    if outputs is None:
        outputs = example.get("answers", None)

    if outputs is None:
        return []

    if isinstance(outputs, list):
        return [str(x) for x in outputs]

    return [str(outputs)]


def load_existing_predictions(pred_path: Path) -> Dict[Any, Dict[str, Any]]:
    if not pred_path.exists():
        return {}

    existing: Dict[Any, Dict[str, Any]] = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            existing[row.get("index")] = row
    return existing


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


def render_prompt(
    tokenizer,
    raw_prompt: str,
    apply_chat_template: bool,
    system_prompt: Optional[str],
) -> str:
    if not apply_chat_template:
        return raw_prompt

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "tokenizer.chat_template is None. "
            "Use --no_apply_chat_template / omit --apply_chat_template, "
            "or use a tokenizer directory that contains chat_template."
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": raw_prompt})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


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
) -> Tuple[str, int, int, float]:
    t0 = time.time()

    prompt = render_prompt(
        tokenizer=tokenizer,
        raw_prompt=raw_prompt,
        apply_chat_template=apply_chat_template,
        system_prompt=system_prompt,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

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

    gen_ids = model.generate(**generation_kwargs)

    new_ids = gen_ids[0, input_len:]
    pred = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    elapsed = time.time() - t0
    return pred, input_len, int(new_ids.shape[-1]), elapsed


def evaluate_file(
    model,
    tokenizer,
    data_file: Path,
    pred_file: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    apply_chat_template: bool,
    system_prompt: Optional[str],
    limit: int,
    resume: bool,
) -> Dict[str, Any]:
    rows = read_jsonl(data_file)
    if limit > 0:
        rows = rows[:limit]

    if pred_file.exists() and not resume:
        pred_file.unlink()

    existing = load_existing_predictions(pred_file) if resume else {}

    num_total = 0
    num_correct = 0
    num_null = 0

    generated_count = 0
    total_input_tokens = 0
    total_new_tokens = 0
    total_seconds = 0.0

    for ex in tqdm(rows, desc=f"{data_file.parent.name}/{data_file.name}"):
        idx = ex.get("index")
        outputs = get_outputs(ex)

        if idx in existing:
            pred = str(existing[idx].get("pred", ""))
            correct, _ = score_example(pred, outputs)

            num_total += 1
            num_correct += int(correct)
            num_null += int(len(pred.strip()) == 0)
            continue

        raw_prompt = ex["input"]

        pred, input_tokens, new_tokens, elapsed = generate_one(
            model=model,
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
        )

        correct, missing = score_example(pred, outputs)

        out = dict(ex)
        out["pred"] = pred
        out["correct"] = bool(correct)
        out["missing_outputs"] = missing
        out["input_tokens"] = input_tokens
        out["generated_tokens"] = new_tokens
        out["generation_seconds"] = elapsed

        append_jsonl(pred_file, [out])

        num_total += 1
        num_correct += int(correct)
        num_null += int(len(pred.strip()) == 0)

        generated_count += 1
        total_input_tokens += input_tokens
        total_new_tokens += new_tokens
        total_seconds += elapsed

    score = 100.0 * num_correct / max(num_total, 1)

    return {
        "data_file": str(data_file),
        "pred_file": str(pred_file),
        "num_examples": num_total,
        "correct": num_correct,
        "score": score,
        "null_predictions": num_null,
        "avg_input_tokens": total_input_tokens / max(generated_count, 1),
        "avg_generated_tokens": total_new_tokens / max(generated_count, 1),
        "total_generation_seconds": total_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline HF Transformers evaluation for RULER NIAH retrieval tasks."
    )

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)

    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        required=True,
        help="Example: --lengths 4096 8192 16384",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["niah_single_1", "niah_multikey_1"],
        help="Currently intended for NIAH retrieval tasks.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Wrap RULER base input as a user message using tokenizer.apply_chat_template.",
    )

    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system prompt used only with --apply_chat_template.",
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="bf16",
    )

    parser.add_argument(
        "--device_map",
        default="auto",
        help='Usually "auto" or a CUDA device map.',
    )

    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=[None, "sdpa", "flash_attention_2", "eager"],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug only: evaluate first N examples per file.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing prediction jsonl files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": get_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
        "local_files_only": True,
    }

    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    model.eval()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []

    for length in args.lengths:
        for task in args.tasks:
            data_file = data_root / str(length) / task / "test.jsonl"

            if not data_file.exists():
                print(f"[SKIP] Missing data file: {data_file}")
                continue

            pred_file = out_root / str(length) / f"{task}.pred.jsonl"

            print("=" * 80)
            print(f"length: {length}")
            print(f"task: {task}")
            print(f"data_file: {data_file}")
            print(f"pred_file: {pred_file}")
            print(f"apply_chat_template: {args.apply_chat_template}")
            print("=" * 80)

            result = evaluate_file(
                model=model,
                tokenizer=tokenizer,
                data_file=data_file,
                pred_file=pred_file,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                apply_chat_template=args.apply_chat_template,
                system_prompt=args.system_prompt,
                limit=args.limit,
                resume=args.resume,
            )

            result["length"] = length
            result["task"] = task
            summaries.append(result)

            print(
                f"[RESULT] length={length} task={task} "
                f"score={result['score']:.2f} "
                f"correct={result['correct']}/{result['num_examples']}"
            )

    summary_path = out_root / "summary.csv"
    write_summary_csv(summary_path, summaries)

    print("=" * 80)
    print(f"Saved summary to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()