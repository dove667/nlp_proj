#!/usr/bin/env python3
"""
Exp C — vLLM backend benchmark (Llama only).

Measures TTFT, TPOT, throughput, and peak GPU memory using vLLM 0.7.x.
TTFT and TPOT are derived from RequestMetrics.first_token_time /
arrival_time / finished_time provided by vLLM's scheduler.

Usage:
    python bench_vllm.py --context_lens 4096 8192 --batch_sizes 1 4
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams

MODEL_PATH = "/data1/zsh/models/Llama-3.1-8B-Instruct"
MODEL_NAME = "llama31"

CONTEXT_LENS = [4096, 8192, 16384, 32768]
BATCH_SIZES = [1, 4, 8, 16]
OUTPUT_LEN = 128

# Dummy text used to fill prompts to the target token length.
_FILL_UNIT = "The quick brown fox jumps over the lazy dog. "


def make_prompt_text(llm: LLM, target_len: int) -> str:
    tokenizer = llm.get_tokenizer()
    unit_ids = tokenizer.encode(_FILL_UNIT, add_special_tokens=False)
    token_ids: list[int] = []
    while len(token_ids) < target_len:
        token_ids.extend(unit_ids)
    token_ids = token_ids[:target_len]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def bench_one(
    llm: LLM,
    context_len: int,
    batch_size: int,
    output_len: int,
    n_warmup: int,
    n_runs: int,
) -> dict[str, Any]:
    prompt = make_prompt_text(llm, context_len)
    prompts = [prompt] * batch_size

    sampling_params = SamplingParams(
        max_tokens=output_len,
        temperature=0,
        ignore_eos=True,
    )
    warmup_params = SamplingParams(max_tokens=1, temperature=0)

    # Warmup
    for _ in range(n_warmup):
        llm.generate([prompt], warmup_params, use_tqdm=False)

    ttft_ms_list: list[float] = []
    tpot_ms_list: list[float] = []
    throughput_list: list[float] = []

    for _ in range(n_runs):
        torch.cuda.reset_peak_memory_stats()
        t_wall_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        t_wall_end = time.perf_counter()

        # Per-request metrics from vLLM scheduler
        ttft_vals: list[float] = []
        tpot_vals: list[float] = []
        total_output_tokens = 0

        for out in outputs:
            m = out.metrics
            n_out = sum(len(o.token_ids) for o in out.outputs)
            total_output_tokens += n_out

            if m is not None and m.first_token_time is not None:
                ttft_vals.append((m.first_token_time - m.arrival_time) * 1000)
                if m.finished_time is not None and n_out > 1:
                    decode_time = (m.finished_time - m.first_token_time) * 1000
                    tpot_vals.append(decode_time / (n_out - 1))

        if ttft_vals:
            ttft_ms_list.append(sum(ttft_vals) / len(ttft_vals))
        if tpot_vals:
            tpot_ms_list.append(sum(tpot_vals) / len(tpot_vals))

        wall_s = t_wall_end - t_wall_start
        throughput_list.append(total_output_tokens / wall_s)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        "model": MODEL_NAME,
        "backend": "vllm",
        "context_len": context_len,
        "batch_size": batch_size,
        "output_len": output_len,
        "ttft_ms": sum(ttft_ms_list) / len(ttft_ms_list) if ttft_ms_list else None,
        "tpot_ms": sum(tpot_ms_list) / len(tpot_ms_list) if tpot_ms_list else None,
        "throughput_tokens_per_s": sum(throughput_list) / len(throughput_list),
        "peak_memory_gb": peak_mem_gb,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_lens", type=int, nargs="+", default=CONTEXT_LENS)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--output_len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).parents[2] / "results" / "exp_c",
    )
    args = parser.parse_args()

    out_path = args.out_dir / "llama31_vllm.jsonl"

    # max_model_len must cover context + output
    max_model_len = max(args.context_lens) + args.output_len

    print(f"Loading {MODEL_NAME} with vLLM (max_model_len={max_model_len}) ...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=False,
    )

    total = len(args.context_lens) * len(args.batch_sizes)
    done = 0
    for ctx in args.context_lens:
        for bs in args.batch_sizes:
            done += 1
            print(f"[{done}/{total}] ctx={ctx} bs={bs} ...", flush=True)
            try:
                row = bench_one(
                    llm,
                    context_len=ctx,
                    batch_size=bs,
                    output_len=args.output_len,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs,
                )
                append_jsonl(out_path, row)
                print(
                    f"  ttft={row['ttft_ms']:.1f}ms  tpot={row['tpot_ms']:.2f}ms  "
                    f"tput={row['throughput_tokens_per_s']:.1f}tok/s  "
                    f"mem={row['peak_memory_gb']:.2f}GB"
                )
            except Exception as e:
                print(f"  ERROR ctx={ctx} bs={bs}: {e}")

    print(f"\nResults saved to {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
