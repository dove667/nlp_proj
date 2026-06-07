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
DECODE_OUTPUT_LEN = 512
FILL_TEXT = "The quick brown fox jumps over the lazy dog. "


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
) -> torch.Tensor:
    with torch.no_grad():
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )


def time_hf_generate(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> tuple[torch.Tensor, float]:
    sync_cuda(device)
    t0 = time.perf_counter()
    outputs = hf_generate(model, tokenizer, input_ids, attention_mask, max_new_tokens)
    sync_cuda(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return outputs, elapsed_ms


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    append_jsonl(path, rows)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
