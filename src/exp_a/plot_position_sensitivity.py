#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


TASK_ORDER = ["niah_single_1", "niah_multikey_1"]
TASK_LABELS = {
    "niah_single_1": "NIAH Single-Key",
    "niah_multikey_1": "NIAH Multi-Key",
}
LENGTH_ORDER = [4096, 8192, 16384, 32768]
BG = "#F7F1E6"
PANEL_BG = "#FBF8F2"
GRID = "#D9D0C3"
TEXT = "#2F241B"
SUBTEXT = "#6C5B4D"
EDGE = "#CBBEAD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create heatmap figure for RULER NIAH position sensitivity."
    )
    parser.add_argument("--position_csv", required=True,
                        help="Path to position_sensitivity_10bins.csv")
    parser.add_argument("--output_prefix", required=True,
                        help="Output prefix, e.g. results/exp_a/llama31/position_sensitivity_llama31")
    parser.add_argument("--title", default="Llama-3.1-8B-Instruct on RULER NIAH",
                        help="Figure title prefix")
    return parser.parse_args()


def _load_data(position_csv: str) -> pd.DataFrame:
    pos_df = pd.read_csv(position_csv)
    pos_df["length"] = pos_df["length"].astype(int)
    pos_df["bucket_id"] = pos_df["bucket_id"].astype(int)
    return pos_df


def _make_heatmap_figure(pos_df: pd.DataFrame, title: str, output_prefix: Path) -> Path:
    fig = plt.figure(figsize=(15.8, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.24)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    fig.patch.set_facecolor(BG)

    cmap = LinearSegmentedColormap.from_list(
        "sandstone",
        ["#FFF7E8", "#F6D28B", "#E89A5B", "#B65434", "#7D3527"],
    )

    lengths_present = sorted(pos_df["length"].unique())
    vmin = min(97.0, float(pos_df["score"].min()))
    vmax = 100.0
    image = None

    for ax, task in zip(axes, TASK_ORDER):
        task_df = pos_df[pos_df["task"] == task]
        pivot = (
            task_df.pivot(index="length", columns="bucket_id", values="score")
            .reindex(index=lengths_present)
        )
        image = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_facecolor(PANEL_BG)
        ax.set_title(f"{TASK_LABELS[task]} Heatmap", fontsize=18, fontweight="bold", color=TEXT, pad=10)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{i * 10}-{(i + 1) * 10}%" for i in pivot.columns], rotation=28, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{idx:,}" for idx in pivot.index])
        ax.tick_params(colors=TEXT, labelsize=11)
        ax.set_xlabel("Position Bucket", fontsize=12, color=TEXT)
        ax.set_ylabel("Context Length", fontsize=12, color=TEXT)
        for spine in ax.spines.values():
            spine.set_color(EDGE)

        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = float(pivot.iat[row_idx, col_idx])
                ax.text(
                    col_idx, row_idx, f"{value:.1f}",
                    ha="center", va="center", fontsize=10.5, color=TEXT,
                    fontweight="bold" if value < 99.95 else "normal",
                )

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Bucket Accuracy (%)", fontsize=12, color=TEXT)
    cbar.ax.tick_params(labelsize=10.5, colors=TEXT)

    fig.suptitle(f"{title}: Heatmap View", fontsize=24, fontweight="bold", color=TEXT, y=0.97)
    fig.text(
        0.5, 0.89,
        "Heatmap view makes the sparse low-score buckets immediately visible without line overlap.",
        ha="center", fontsize=11.5, color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.06, right=0.95)

    output_path = output_prefix.with_name(output_prefix.name + "_heatmap.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    pos_df = _load_data(args.position_csv)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    output_prefix = Path(args.output_prefix)
    heatmap_path = _make_heatmap_figure(pos_df, args.title, output_prefix)
    print(f"Saved heatmap figure to: {heatmap_path}")


if __name__ == "__main__":
    main()
