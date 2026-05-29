#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_LABELS = {
    "hotpotqa": "HotpotQA",
    "qasper": "Qasper",
    "gov_report": "GovReport",
    "repobench-p": "RepoBench-P",
}

TASK_COLORS = {
    "hotpotqa": "#F18F01",
    "qasper": "#C73E1D",
    "gov_report": "#8D5A97",
    "repobench-p": "#3D405B",
}

BG = "#F7F1E6"
PANEL_BG = "#FBF8F2"
GRID = "#D9D0C3"
TEXT = "#2F241B"
SUBTEXT = "#6C5B4D"
EDGE = "#CBBEAD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Exp B LongBench summary bars.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--title", default="Exp B LongBench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=TEXT, labelsize=11)

    order = [task for task in TASK_LABELS if task in set(summary_df["task"])]
    xs = list(range(len(order)))
    scores = [float(summary_df.loc[summary_df["task"] == task, "score"].iloc[0]) for task in order]
    colors = [TASK_COLORS[task] for task in order]
    labels = [TASK_LABELS[task] for task in order]

    ax.bar(xs, scores, color=colors, width=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score", fontsize=12, color=TEXT)
    ax.set_title("LongBench", fontsize=18, fontweight="bold", color=TEXT, pad=10)

    for x, score in zip(xs, scores):
        ax.text(x, score, f"{score:.1f}", ha="center", va="bottom", fontsize=10, color=TEXT)

    fig.suptitle(f"{args.title}: Task Summary", fontsize=24, fontweight="bold", color=TEXT, y=0.97)
    fig.text(
        0.5,
        0.90,
        "LongBench is evaluated on original benchmark samples without synthetic length binning or head-tail truncation.",
        ha="center",
        fontsize=11.2,
        color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.09, right=0.98)

    output_prefix = Path(args.output_prefix)
    out = output_prefix.with_name(output_prefix.name + "_summary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved LongBench figure to: {out}")


if __name__ == "__main__":
    main()
