import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_LABELS = {
    "hotpotqa": "HotpotQA",
    "qasper": "Qasper",
    "gov_report": "GovReport",
    "repobench-p": "RepoBench-P",
}

METRIC_LABELS = {
    "qa_f1": "QA F1",
    "rouge_l_f1": "Rouge-L F1",
    "exact_match": "Exact Match",
}

MODEL_COLORS = {
    "llama": "#4A90D9",
    "mamba": "#E07B39",
}

BG = "#F7F1E6"
PANEL_BG = "#FBF8F2"
GRID = "#D9D0C3"
TEXT = "#2F241B"
SUBTEXT = "#6C5B4D"
EDGE = "#CBBEAD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Exp B LongBench comparison bars.")
    parser.add_argument("--llama_csv", required=True, help="summary.csv for Llama")
    parser.add_argument("--mamba_csv", required=True, help="summary.csv for Mamba")
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--llama_label", default="Llama-3.1-8B")
    parser.add_argument("--mamba_label", default="Falcon3-Mamba-7B")
    parser.add_argument("--title", default="Exp B LongBench")
    return parser.parse_args()


def load_scores(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {row["task"]: float(row["score"]) for _, row in df.iterrows()}


def main() -> None:
    args = parse_args()

    llama_scores = load_scores(args.llama_csv)
    mamba_scores = load_scores(args.mamba_csv)

    order = [t for t in TASK_LABELS if t in llama_scores or t in mamba_scores]
    n = len(order)
    xs = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 5.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=TEXT, labelsize=11)

    llama_vals = [llama_scores.get(t, 0.0) for t in order]
    mamba_vals = [mamba_scores.get(t, 0.0) for t in order]

    bars_llama = ax.bar(xs - width / 2, llama_vals, width=width,
                        color=MODEL_COLORS["llama"], label=args.llama_label)
    bars_mamba = ax.bar(xs + width / 2, mamba_vals, width=width,
                        color=MODEL_COLORS["mamba"], label=args.mamba_label)

    for bar, val in zip(bars_llama, llama_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9.5, color=TEXT)
    for bar, val in zip(bars_mamba, mamba_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9.5, color=TEXT)

    ax.set_xticks(xs)
    ax.set_xticklabels([TASK_LABELS[t] for t in order], fontsize=11)
    ax.set_ylabel("Score", fontsize=12, color=TEXT)
    ax.legend(fontsize=11, framealpha=0.85, edgecolor=EDGE)

    fig.suptitle(f"{args.title}: Model Comparison", fontsize=22, fontweight="bold", color=TEXT, y=0.97)
    fig.text(
        0.5, 0.90,
        "Evaluated on original LongBench samples  ·  metrics: QA F1 / Rouge-L F1 / Exact Match",
        ha="center", fontsize=10.5, color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.09, right=0.98)

    output_prefix = Path(args.output_prefix)
    out = output_prefix.with_name(output_prefix.name + "_compare.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
