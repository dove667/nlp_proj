import os
import time
import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import append_jsonl, read_jsonl

MODEL_PATHS = {
    "llama31": "/data1/zsh/models/Llama-3.1-8B-Instruct",
    "mamba": "/data1/zsh/models/Falcon3-Mamba-7B-Instruct",
}

RESULTS_DIR = Path(__file__).parents[2] / "results" / "exp_c"
PREFILL_CONTEXT_LENS = [4096, 8192, 16384]
CAPACITY_CONTEXT_LENS = [8192, 16384]
CAPACITY_BATCH_SIZES = [1, 2, 4, 8, 16]
DECODE_PROMPT_LEN = 256
DECODE_OUTPUT_LEN = 1024
FILL_TEXT = "The quick brown fox jumps over the lazy dog. "
CONTINUATION_PREFIX_TEMPLATE = (
    "You are completing a continuation-only generation task. "
    "Continue the given prefix directly and keep writing fluent natural text. "
    "Do not answer with meta commentary, bullet points, or explanations. "
    "Do not stop early. Even if the text feels complete, keep extending it naturally until you have produced at least {target_output_len} new tokens. "
    "Do not write an ending marker such as 'The End'.\n\n"
    "Prefix:\n"
)
CONTINUATION_SUFFIX = "\n\nContinuation:\n"


def ensure_offline_env() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_hf_model(model_name: str, device: str) -> tuple[Any, Any]:
    ensure_offline_env()
    model_path = MODEL_PATHS[model_name]
    print(f"Loading {model_name} from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    return model, tokenizer


def unload_hf_model(model: Any, tokenizer: Any) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_fill_ids(tokenizer: Any, target_len: int) -> list[int]:
    unit = tokenizer.encode(FILL_TEXT, add_special_tokens=False)
    tokens: list[int] = []
    while len(tokens) < target_len:
        tokens.extend(unit)
    return tokens[:target_len]


def make_batch_tensors(tokenizer: Any, context_len: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = make_fill_ids(tokenizer, context_len)
    input_ids = torch.tensor([token_ids] * batch_size, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def make_prompt_tensors(tokenizer: Any, prompt_text: str, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device).repeat(batch_size, 1)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device).repeat(batch_size, 1)
    return input_ids, attention_mask


def make_continuation_prompt(tokenizer: Any, target_prompt_len: int, target_output_len: int) -> str:
    prefix_text = CONTINUATION_PREFIX_TEMPLATE.format(target_output_len=target_output_len)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(CONTINUATION_SUFFIX, add_special_tokens=False)
    available = max(target_prompt_len - len(prefix_ids) - len(suffix_ids), 1)
    filler_ids = make_fill_ids(tokenizer, available)
    prompt_ids = prefix_ids + filler_ids + suffix_ids
    if len(prompt_ids) > target_prompt_len:
        prompt_ids = prompt_ids[:target_prompt_len]
    elif len(prompt_ids) < target_prompt_len:
        pad_fill = make_fill_ids(tokenizer, target_prompt_len - len(prompt_ids))
        prompt_ids.extend(pad_fill)
    return tokenizer.decode(prompt_ids, skip_special_tokens=True)


def sync_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_gb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated(device) / 1e9


def hf_generate(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    min_new_tokens: int | None = None,
) -> torch.Tensor:
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if min_new_tokens is not None:
        generation_kwargs["min_new_tokens"] = min_new_tokens
    with torch.no_grad():
        return model.generate(**generation_kwargs)


def time_hf_generate(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    device: str,
    min_new_tokens: int | None = None,
) -> tuple[torch.Tensor, float]:
    sync_cuda(device)
    t0 = time.perf_counter()
    outputs = hf_generate(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
    sync_cuda(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return outputs, elapsed_ms


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    append_jsonl(path, rows)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
