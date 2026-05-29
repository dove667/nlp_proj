#!/usr/bin/env python3
"""
Exp C — HF Transformers backend benchmark.

Measures TTFT, TPOT, throughput, and peak GPU memory for Llama and Mamba
using HuggingFace Transformers (single GPU, bf16).

Usage:
    python bench_hf.py --model llama31 --context_lens 4096 8192 --batch_sizes 1 4
    python bench_hf.py --model mamba   --context_lens 4096 8192 --batch_sizes 1 4
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATHS = {
    "llama31": "/data1/zsh/models/Llama-3.1-8B-Instruct",
    "mamba": "/data1/zsh/models/Falcon3-Mamba-7B-Instruct",
}

CONTEXT_LENS = [4096, 8192, 16384, 32768]
BATCH_SIZES = [1, 4, 8, 16]
OUTPUT_LEN = 128


def make_input_ids(tokenizer: Any, target_len: int) -> list[int]:
    unit = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ",
        add_special_tokens=False,
    )
    tokens: list[int] = []
    while len(tokens) < target_len:
        tokens.extend(unit)
    return tokens[:target_len]


def bench_one(
    model: Any,
    tokenizer: Any,
    context_len: int,
    batch_size: int,
    output_len: int,
    device: str,
    n_warmup: int,
    n_runs: int,
) -> dict[str, Any]:
    token_ids = make_input_ids(tokenizer, context_len)
    input_ids = torch.tensor([token_ids] * batch_size, device=device)
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs: dict[str, Any] = dict(
        attention_mask=attention_mask,
        max_new_tokens=output_len,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    # Warmup with single request
    with torch.no_grad():
        for _ in range(n_warmup):
            model.generate(input_ids[:1], **{**gen_kwargs, "attention_mask": attention_mask[:1]})
    torch.cuda.synchronize(device)

    ttft_ms_list: list[float] = []
    tpot_ms_list: list[float] = []
    throughput_list: list[float] = []

    for _ in range(n_runs):
        # TTFT: prefill + 1 decode step, single request
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids[:1],
                attention_mask=attention_mask[:1],
                max_new_tokens=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize(device)
        ttft_ms = (time.perf_counter() - t0) * 1000
        ttft_ms_list.append(ttft_ms)

        # Single-request full generation for TPOT
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids[:1], **{**gen_kwargs, "attention_mask": attention_mask[:1]})
        torch.cuda.synchronize(device)
        single_ms = (time.perf_counter() - t0) * 1000
        tpot_ms_list.append((single_ms - ttft_ms) / (output_len - 1))

        # Batch throughput
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        torch.cuda.synchronize(device)
        batch_ms = (time.perf_counter() - t0) * 1000
        actual_out_tokens = (out.shape[1] - input_ids.shape[1]) * batch_size
        throughput_list.append(actual_out_tokens / (batch_ms / 1000))

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "model": "llama31" if "Llama" in str(model.__class__) else "mamba",
        "backend": "hf",
        "context_len": context_len,
        "batch_size": batch_size,
        "output_len": output_len,
        "ttft_ms": sum(ttft_ms_list) / len(ttft_ms_list),
        "tpot_ms": sum(tpot_ms_list) / len(tpot_ms_list),
        "throughput_tokens_per_s": sum(throughput_list) / len(throughput_list),
        "peak_memory_gb": peak_mem_gb,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS), required=True)
    parser.add_argument("--context_lens", type=int, nargs="+", default=CONTEXT_LENS)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--output_len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).parents[2] / "results" / "exp_c",
    )
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    out_path = args.out_dir / f"{args.model}_hf.jsonl"

    print(f"Loading {args.model} from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device).eval()

    total = len(args.context_lens) * len(args.batch_sizes)
    done = 0
    for ctx in args.context_lens:
        for bs in args.batch_sizes:
            done += 1
            print(f"[{done}/{total}] ctx={ctx} bs={bs} ...", flush=True)
            try:
                row = bench_one(
                    model, 
                    tokenizer,
                    context_len=ctx,
                    batch_size=bs,
                    output_len=args.output_len,
                    device=args.device,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs,
                )
                # Fix model name (class name heuristic is fragile; use arg instead)
                row["model"] = args.model
                append_jsonl(out_path, row)
                print(
                    f"  ttft={row['ttft_ms']:.1f}ms  tpot={row['tpot_ms']:.2f}ms  "
                    f"tput={row['throughput_tokens_per_s']:.1f}tok/s  "
                    f"mem={row['peak_memory_gb']:.2f}GB"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM — skipping ctx={ctx} bs={bs}")
                torch.cuda.empty_cache()

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
