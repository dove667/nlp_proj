import argparse
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from common import (
    CAPACITY_BATCH_SIZES,
    CAPACITY_CONTEXT_LENS,
    DECODE_OUTPUT_LEN,
    RESULTS_DIR,
    hf_generate,
    load_hf_model,
    make_batch_tensors,
    sync_cuda,
    time_hf_generate,
    write_rows,
)

MODEL_PATH = "/data1/zsh/models/Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", choices=["hf", "vllm"], default=["hf", "vllm"])
    parser.add_argument("--context_lens", type=int, nargs="+", default=CAPACITY_CONTEXT_LENS)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=CAPACITY_BATCH_SIZES)
    parser.add_argument("--output_len", type=int, default=DECODE_OUTPUT_LEN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--out_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def make_vllm(llm_max_len: int, gpu_memory_utilization: float) -> LLM:
    return LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=llm_max_len,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=False,
    )


def make_vllm_prompt(llm: LLM, target_len: int) -> str:
    tokenizer = llm.get_tokenizer()
    unit_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False)
    token_ids: list[int] = []
    while len(token_ids) < target_len:
        token_ids.extend(unit_ids)
    return tokenizer.decode(token_ids[:target_len], skip_special_tokens=True)


def run_hf(args: argparse.Namespace, out_path: Path) -> None:
    model, tokenizer = load_hf_model("llama31", args.device)
    for context_len in args.context_lens:
        for batch_size in args.batch_sizes:
            print(f"[HF llama31] ctx={context_len} bs={batch_size} ...", flush=True)
            try:
                input_ids, attention_mask = make_batch_tensors(tokenizer, context_len, batch_size, args.device)
                for _ in range(args.n_warmup):
                    hf_generate(model, tokenizer, input_ids, attention_mask, args.output_len)
                sync_cuda(args.device)

                completion_ms_list: list[float] = []
                requests_per_s_list: list[float] = []
                for _ in range(args.n_runs):
                    _, elapsed_ms = time_hf_generate(
                        model,
                        tokenizer,
                        input_ids,
                        attention_mask,
                        args.output_len,
                        args.device,
                    )
                    completion_ms_list.append(elapsed_ms)
                    requests_per_s_list.append(batch_size / (elapsed_ms / 1000))

                row = {
                    "experiment": "backend_llama",
                    "model": "llama31",
                    "backend": "hf",
                    "context_len": context_len,
                    "batch_size": batch_size,
                    "output_len": args.output_len,
                    "oom": False,
                    "avg_completion_ms": sum(completion_ms_list) / len(completion_ms_list),
                    "requests_per_s": sum(requests_per_s_list) / len(requests_per_s_list),
                }
                write_rows(out_path, [row])
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at ctx={context_len} bs={batch_size}")
                torch.cuda.empty_cache()
                write_rows(
                    out_path,
                    [{
                        "experiment": "backend_llama",
                        "model": "llama31",
                        "backend": "hf",
                        "context_len": context_len,
                        "batch_size": batch_size,
                        "output_len": args.output_len,
                        "oom": True,
                        "avg_completion_ms": None,
                        "requests_per_s": None,
                    }],
                )
                break


def run_vllm(args: argparse.Namespace, out_path: Path) -> None:
    llm = make_vllm(max(args.context_lens) + args.output_len, args.gpu_memory_utilization)
    params = SamplingParams(max_tokens=args.output_len, temperature=0, ignore_eos=True)

    for context_len in args.context_lens:
        prompt = make_vllm_prompt(llm, context_len)
        for batch_size in args.batch_sizes:
            print(f"[vLLM llama31] ctx={context_len} bs={batch_size} ...", flush=True)
            prompts = [prompt] * batch_size
            try:
                for _ in range(args.n_warmup):
                    llm.generate(prompts, params, use_tqdm=False)

                completion_ms_list: list[float] = []
                requests_per_s_list: list[float] = []
                for _ in range(args.n_runs):
                    t0 = time.perf_counter()
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    completion_ms_list.append(elapsed_ms)
                    requests_per_s_list.append(batch_size / (elapsed_ms / 1000))
                    if not outputs:
                        raise RuntimeError("vLLM returned no outputs")

                row = {
                    "experiment": "backend_llama",
                    "model": "llama31",
                    "backend": "vllm",
                    "context_len": context_len,
                    "batch_size": batch_size,
                    "output_len": args.output_len,
                    "oom": False,
                    "avg_completion_ms": sum(completion_ms_list) / len(completion_ms_list),
                    "requests_per_s": sum(requests_per_s_list) / len(requests_per_s_list),
                }
                write_rows(out_path, [row])
            except Exception as exc:
                print(f"  ERROR at ctx={context_len} bs={batch_size}: {exc}")
                write_rows(
                    out_path,
                    [{
                        "experiment": "backend_llama",
                        "model": "llama31",
                        "backend": "vllm",
                        "context_len": context_len,
                        "batch_size": batch_size,
                        "output_len": args.output_len,
                        "oom": True,
                        "avg_completion_ms": None,
                        "requests_per_s": None,
                    }],
                )
                break


def main() -> None:
    args = parse_args()
    out_path = args.out_dir / "backend_llama.jsonl"
    if "hf" in args.backends:
        run_hf(args, out_path)
    if "vllm" in args.backends:
        run_vllm(args, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
