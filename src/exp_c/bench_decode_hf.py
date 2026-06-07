import argparse
import subprocess
import sys
from pathlib import Path

import torch

from common import (
    DECODE_OUTPUT_LEN,
    DECODE_PROMPT_LENS,
    MODEL_PATHS,
    RESULTS_DIR,
    hf_generate,
    load_hf_model,
    make_continuation_prompt,
    make_prompt_tensors,
    sync_cuda,
    time_hf_generate,
    unload_hf_model,
    write_rows,
)


def count_new_tokens(outputs, input_ids) -> int:
    return int(outputs.shape[1] - input_ids.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PATHS), default=list(MODEL_PATHS))
    parser.add_argument("--prompt_lens", type=int, nargs="+", default=DECODE_PROMPT_LENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--out_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--_single_model", action="store_true")
    return parser.parse_args()


def run_single_model(args: argparse.Namespace) -> None:
    out_path = args.out_dir / "decode_hf.jsonl"
    for model_name in args.models:
        try:
            model, tokenizer = load_hf_model(model_name, args.device)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM while loading {model_name}; skipping this model")
            torch.cuda.empty_cache()
            continue

        for target_prompt_len in args.prompt_lens:
            prompt_text = make_continuation_prompt(tokenizer, target_prompt_len)
            input_ids, attention_mask = make_prompt_tensors(tokenizer, prompt_text, 1, args.device)
            actual_prompt_len = int(input_ids.shape[-1])
            print(
                f"[{model_name}] prompt_len={actual_prompt_len} target_prompt_len={target_prompt_len} output_len={DECODE_OUTPUT_LEN} ...",
                flush=True,
            )

            for _ in range(args.n_warmup):
                hf_generate(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    DECODE_OUTPUT_LEN,
                )
            sync_cuda(args.device)

            tpot_ms_list: list[float] = []
            actual_new_tokens_list: list[int] = []
            for _ in range(args.n_runs):
                _, first_token_ms = time_hf_generate(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    1,
                    args.device,
                )
                outputs, total_ms = time_hf_generate(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    DECODE_OUTPUT_LEN,
                    args.device,
                )
                actual_new_tokens = count_new_tokens(outputs, input_ids)
                if actual_new_tokens <= 1:
                    continue
                actual_new_tokens_list.append(actual_new_tokens)
                tpot_ms_list.append((total_ms - first_token_ms) / max(actual_new_tokens - 1, 1))

            if not tpot_ms_list:
                print(f"  no usable decode run for prompt_len={target_prompt_len}, skipping")
                continue

            row = {
                "experiment": "decode_hf",
                "model": model_name,
                "backend": "hf",
                "prompt_type": "continuation",
                "prompt_len": actual_prompt_len,
                "target_prompt_len": target_prompt_len,
                "batch_size": 1,
                "output_len": DECODE_OUTPUT_LEN,
                "avg_actual_output_len": sum(actual_new_tokens_list) / len(actual_new_tokens_list),
                "ttft_ms": None,
                "tpot_ms": sum(tpot_ms_list) / len(tpot_ms_list),
            }
            write_rows(out_path, [row])
        unload_hf_model(model, tokenizer)

    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    if args._single_model or len(args.models) == 1:
        run_single_model(args)
        return

    for model_name in args.models:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--models",
            model_name,
            "--device",
            args.device,
            "--n_warmup",
            str(args.n_warmup),
            "--n_runs",
            str(args.n_runs),
            "--out_dir",
            str(args.out_dir),
            "--_single_model",
        ]
        cmd.extend(["--prompt_lens", *[str(x) for x in args.prompt_lens]])
        print(f"Launching subprocess for {model_name} ...", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
