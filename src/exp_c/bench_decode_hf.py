import argparse
from pathlib import Path

from common import (
    DECODE_OUTPUT_LEN,
    DECODE_PROMPT_LEN,
    MODEL_PATHS,
    RESULTS_DIR,
    hf_generate,
    load_hf_model,
    make_batch_tensors,
    sync_cuda,
    time_hf_generate,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PATHS), default=list(MODEL_PATHS))
    parser.add_argument("--prompt_len", type=int, default=DECODE_PROMPT_LEN)
    parser.add_argument("--output_len", type=int, default=DECODE_OUTPUT_LEN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--out_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out_dir / "decode_hf.jsonl"

    for model_name in args.models:
        model, tokenizer = load_hf_model(model_name, args.device)
        print(f"[{model_name}] prompt_len={args.prompt_len} output_len={args.output_len} ...", flush=True)
        input_ids, attention_mask = make_batch_tensors(tokenizer, args.prompt_len, 1, args.device)

        for _ in range(args.n_warmup):
            hf_generate(model, tokenizer, input_ids, attention_mask, args.output_len)
        sync_cuda(args.device)

        ttft_ms_list: list[float] = []
        tpot_ms_list: list[float] = []
        for _ in range(args.n_runs):
            _, ttft_ms = time_hf_generate(model, tokenizer, input_ids, attention_mask, 1, args.device)
            ttft_ms_list.append(ttft_ms)

            _, total_ms = time_hf_generate(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                args.output_len,
                args.device,
            )
            tpot_ms_list.append((total_ms - ttft_ms) / max(args.output_len - 1, 1))

        row = {
            "experiment": "decode_hf",
            "model": model_name,
            "backend": "hf",
            "prompt_len": args.prompt_len,
            "batch_size": 1,
            "output_len": args.output_len,
            "ttft_ms": sum(ttft_ms_list) / len(ttft_ms_list),
            "tpot_ms": sum(tpot_ms_list) / len(tpot_ms_list),
        }
        write_rows(out_path, [row])

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
