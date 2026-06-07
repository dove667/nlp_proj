import argparse
from pathlib import Path

import torch

from common import (
    MODEL_PATHS,
    PREFILL_CONTEXT_LENS,
    RESULTS_DIR,
    get_peak_memory_gb,
    hf_generate,
    load_hf_model,
    make_batch_tensors,
    reset_peak_memory,
    sync_cuda,
    time_hf_generate,
    unload_hf_model,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PATHS), default=list(MODEL_PATHS))
    parser.add_argument("--context_lens", type=int, nargs="+", default=PREFILL_CONTEXT_LENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--out_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out_dir / "prefill_hf.jsonl"

    for model_name in args.models:
        try:
            model, tokenizer = load_hf_model(model_name, args.device)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM while loading {model_name}; skipping this model")
            torch.cuda.empty_cache()
            continue
        total = len(args.context_lens)
        for idx, context_len in enumerate(args.context_lens, start=1):
            print(f"[{model_name}] [{idx}/{total}] ctx={context_len} ...", flush=True)
            try:
                input_ids, attention_mask = make_batch_tensors(tokenizer, context_len, 1, args.device)

                for _ in range(args.n_warmup):
                    hf_generate(model, tokenizer, input_ids, attention_mask, 1)
                sync_cuda(args.device)

                ttft_ms_list: list[float] = []
                peak_mem_list: list[float] = []
                for _ in range(args.n_runs):
                    _, ttft_ms = time_hf_generate(model, tokenizer, input_ids, attention_mask, 1, args.device)
                    ttft_ms_list.append(ttft_ms)

                    reset_peak_memory(args.device)
                    sync_cuda(args.device)
                    hf_generate(model, tokenizer, input_ids, attention_mask, 1)
                    sync_cuda(args.device)
                    peak = get_peak_memory_gb(args.device)
                    if peak is not None:
                        peak_mem_list.append(peak)

                row = {
                    "experiment": "prefill_hf",
                    "model": model_name,
                    "backend": "hf",
                    "context_len": context_len,
                    "batch_size": 1,
                    "output_len": 1,
                    "ttft_ms": sum(ttft_ms_list) / len(ttft_ms_list),
                    "peak_memory_gb": sum(peak_mem_list) / len(peak_mem_list) if peak_mem_list else None,
                }
                write_rows(out_path, [row])
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM — skipping {model_name} ctx={context_len}")
                torch.cuda.empty_cache()
        unload_hf_model(model, tokenizer)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
