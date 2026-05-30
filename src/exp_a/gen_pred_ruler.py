import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    append_jsonl,
    generate_one,
    get_dtype,
    load_existing_predictions,
    make_sample_id,
    read_jsonl,
    resolve_input_device,
)


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\x00-\x1f]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def answer_in_prediction(answer: str, pred: str) -> bool:
    answer_norm = normalize_text(answer)
    pred_norm = normalize_text(pred)

    if not answer_norm:
        return False

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


def generate_predictions_for_file(
    model,
    tokenizer,
    data_file: Path,
    pred_file: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    apply_chat_template: bool,
    system_prompt: str,
    limit: int,
    resume: bool,
    model_input_device,
) -> Dict[str, Any]:
    rows = read_jsonl(data_file)
    if limit > 0:
        rows = rows[:limit]

    if pred_file.exists() and not resume:
        pred_file.unlink()

    existing_rows = load_existing_predictions(pred_file) if resume else []

    if existing_rows:
        if any("sample_id" not in row for row in existing_rows):
            raise ValueError(
                f"Existing prediction file is from the old buggy resume format: {pred_file}. "
                "Delete it and rerun without --resume once, or regenerate predictions with the new script."
            )
        if len(existing_rows) > len(rows):
            raise ValueError(
                f"Prediction file has more rows than dataset: {pred_file} "
                f"({len(existing_rows)} > {len(rows)})."
            )

    reused_count = 0
    generated_count = 0

    for row_idx, ex in enumerate(tqdm(rows, desc=f"{data_file.parent.name}/{data_file.name}")):
        sample_id = make_sample_id(ex, row_idx)
        outputs = get_outputs(ex)

        if row_idx < len(existing_rows):
            existing = existing_rows[row_idx]
            if str(existing.get("sample_id")) != sample_id:
                raise ValueError(
                    f"Resume mismatch at row {row_idx} for {pred_file}: "
                    f"existing sample_id={existing.get('sample_id')} current sample_id={sample_id}"
                )
            reused_count += 1
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
            model_input_device=model_input_device,
        )

        correct, missing = score_example(pred, outputs)

        out = dict(ex)
        out["sample_id"] = sample_id
        out["pred"] = pred
        out["correct"] = correct
        out["missing_outputs"] = missing
        out["input_tokens"] = input_tokens
        out["generated_tokens"] = new_tokens
        out["generation_seconds"] = elapsed

        append_jsonl(pred_file, [out])
        generated_count += 1

    return {
        "data_file": str(data_file),
        "pred_file": str(pred_file),
        "num_examples": len(rows),
        "reused_predictions": reused_count,
        "new_predictions": generated_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline HF Transformers prediction generation for RULER NIAH retrieval tasks."
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
        help='Use "auto" for sharded loading, or "none" to disable device_map and place the whole model on --model_device.',
    )

    parser.add_argument(
        "--model_device",
        default="cuda:0",
        help='Used when --device_map none. Example: "cuda:0" or "cpu".',
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
        help="Debug only: generate for the first N examples per file.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing prediction jsonl file written by this script version.",
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
        clean_up_tokenization_spaces=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": get_dtype(args.dtype),
        "trust_remote_code": True,
        "local_files_only": True,
    }

    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map

    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    if args.device_map.lower() == "none":
        model = model.to(args.model_device)

    model.eval()
    model_input_device = resolve_input_device(model, fallback=args.model_device)
    print(f"Model input device: {model_input_device}")

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

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

            result = generate_predictions_for_file(
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
                model_input_device=model_input_device,
            )

            print(
                f"[DONE] length={length} task={task} "
                f"num_examples={result['num_examples']} "
                f"reused={result['reused_predictions']} "
                f"generated={result['new_predictions']}"
            )

    print("=" * 80)
    print(f"Prediction generation finished under: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
