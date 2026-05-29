#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_LABELS = {
    "vt": "RULER VT",
    "cwe": "RULER CWE",
    "fwe": "RULER FWE",
}

TASK_COLORS = {
    "vt": "#0F4C81",
    "cwe": "#2E86AB",
    "fwe": "#58A4B0",
}

BG = "#F7F1E6"
PANEL_BG = "#FBF8F2"
GRID = "#D9D0C3"
TEXT = "#2F241B"
SUBTEXT = "#6C5B4D"
EDGE = "#CBBEAD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Exp B RULER reasoning score curves.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--title", default="Exp B RULER Reasoning")
    return parser.parse_args()


def style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=TEXT, labelsize=11)


def plot_score_figure(summary_df: pd.DataFrame, title: str, output_prefix: Path) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8))
    fig.patch.set_facecolor(BG)

    for task in ["vt", "cwe", "fwe"]:
        task_df = summary_df[summary_df["task"] == task].sort_values("length")
        if task_df.empty:
            continue
        ax.plot(
            task_df["length"],
            task_df["score"],
            marker="o",
            linewidth=2.6,
            markersize=6.5,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
        )
    style_axes(ax)
    ax.set_title("RULER Reasoning", fontsize=18, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel("Context Length", fontsize=12, color=TEXT)
    ax.set_ylabel("Accuracy", fontsize=12, color=TEXT)
    ax.legend(loc="lower left", frameon=True, facecolor=PANEL_BG, edgecolor=EDGE, fontsize=10)

    fig.suptitle(f"{title}: Score Curves", fontsize=24, fontweight="bold", color=TEXT, y=0.97)
    fig.text(
        0.5,
        0.90,
        "RULER reasoning is evaluated as exact-match style accuracy under controlled length scaling.",
        ha="center",
        fontsize=11.5,
        color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.12, right=0.97)

    out = output_prefix.with_name(output_prefix.name + "_scores.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    summary_df["length"] = summary_df["length"].astype(int)

    output_prefix = Path(args.output_prefix)
    score_path = plot_score_figure(summary_df, args.title, output_prefix)
    print(f"Saved score figure to: {score_path}")


if __name__ == "__main__":
    main()
