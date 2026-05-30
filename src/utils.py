import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RULER_REASONING_TASKS = ["vt", "cwe", "fwe"]
LONGBENCH_TASKS = ["hotpotqa", "qasper", "gov_report", "repobench-p"]


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


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


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


def render_prompt(
    tokenizer,
    raw_prompt: str,
    apply_chat_template: bool,
    system_prompt: Optional[str],
) -> str:
    if not apply_chat_template:
        return raw_prompt
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("tokenizer.chat_template is None, cannot apply chat template.")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": raw_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
    model_input_device: torch.device,
) -> Tuple[str, int, int, float]:
    t0 = time.time()
    prompt = render_prompt(
        tokenizer=tokenizer,
        raw_prompt=raw_prompt,
        apply_chat_template=apply_chat_template,
        system_prompt=system_prompt,
    )
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
