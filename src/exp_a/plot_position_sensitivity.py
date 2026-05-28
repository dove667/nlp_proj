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
LENGTH_COLORS = {
    4096: "#0F4C81",
    8192: "#2E86AB",
    16384: "#F18F01",
    32768: "#C73E1D",
}
BG = "#F7F1E6"
PANEL_BG = "#FBF8F2"
GRID = "#D9D0C3"
TEXT = "#2F241B"
SUBTEXT = "#6C5B4D"
EDGE = "#CBBEAD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate line and heatmap figures for RULER NIAH position sensitivity."
    )
    parser.add_argument("--summary_csv", required=True, help="Path to summary.csv")
    parser.add_argument(
        "--position_csv",
        required=True,
        help="Path to position_sensitivity_10bins.csv",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Output prefix, e.g. results/exp_a/llama31/position_sensitivity_llama31",
    )
    parser.add_argument(
        "--title",
        default="Llama-3.1-8B-Instruct on RULER NIAH",
        help="Figure title prefix",
    )
    return parser.parse_args()


def _load_data(summary_csv: str, position_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = pd.read_csv(summary_csv)
    pos_df = pd.read_csv(position_csv)
    summary_df["length"] = summary_df["length"].astype(int)
    pos_df["length"] = pos_df["length"].astype(int)
    pos_df["bucket_id"] = pos_df["bucket_id"].astype(int)
    return summary_df, pos_df


def _format_line_axes(ax: plt.Axes, ymin: float, ymax: float) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.8)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.set_xlim(0, 100)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.tick_params(colors=TEXT, labelsize=11)
    ax.set_xlabel("Answer Position in Context (%)", fontsize=12, color=TEXT)
    ax.set_ylabel("Accuracy (%)", fontsize=12, color=TEXT)


def _summary_text(summary_df: pd.DataFrame, task: str) -> str:
    sub = summary_df[summary_df["task"] == task].sort_values("length")
    return "  |  ".join(
        f"{int(row.length):,}: {row.score:.1f}%"
        for row in sub.itertuples()
    )


def _make_line_figure(summary_df: pd.DataFrame, pos_df: pd.DataFrame, title: str, output_prefix: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.3), sharey=True)
    fig.patch.set_facecolor(BG)

    min_score = float(pos_df["score"].min())
    ymin = min(96.8, np.floor((min_score - 0.2) * 10) / 10)
    ymax = 100.15

    for ax, task in zip(axes, TASK_ORDER):
        task_df = pos_df[pos_df["task"] == task].sort_values(["length", "bucket_id"])
        for length in LENGTH_ORDER:
            sub = task_df[task_df["length"] == length]
            if sub.empty:
                continue
            x = sub["avg_relative_position"].to_numpy() * 100.0
            y = sub["score"].to_numpy()
            ax.plot(
                x,
                y,
                marker="o",
                markersize=6.5,
                linewidth=2.6,
                color=LENGTH_COLORS[length],
                label=f"{length:,}",
                alpha=0.96,
            )
        _format_line_axes(ax, ymin, ymax)
        ax.set_title(TASK_LABELS[task], fontsize=18, fontweight="bold", color=TEXT, pad=10)
        ax.text(
            0.02,
            0.04,
            _summary_text(summary_df, task),
            transform=ax.transAxes,
            fontsize=10.5,
            color=SUBTEXT,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#F1E6D2",
                "edgecolor": "#D8C4A3",
            },
        )

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        title="Context Length",
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.03),
        fontsize=11,
        title_fontsize=12,
    )
    legend.get_frame().set_facecolor(PANEL_BG)
    legend.get_frame().set_edgecolor(EDGE)

    fig.suptitle(f"{title}: Line View", fontsize=24, fontweight="bold", color=TEXT, y=0.97)
    fig.text(
        0.5,
        0.90,
        "Bucketed accuracy by answer position. The model stays near-ceiling, with only a mild soft dip on long-context multi-key retrieval.",
        ha="center",
        fontsize=11.5,
        color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.07, right=0.98, wspace=0.18)

    output_path = output_prefix.with_name(output_prefix.name + "_line.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


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

    vmin = min(97.0, float(pos_df["score"].min()))
    vmax = 100.0
    image = None

    for ax, task in zip(axes, TASK_ORDER):
        task_df = pos_df[pos_df["task"] == task]
        pivot = (
            task_df.pivot(index="length", columns="bucket_id", values="score")
            .reindex(index=LENGTH_ORDER)
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
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=TEXT,
                    fontweight="bold" if value < 99.95 else "normal",
                )

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Bucket Accuracy (%)", fontsize=12, color=TEXT)
    cbar.ax.tick_params(labelsize=10.5, colors=TEXT)

    fig.suptitle(f"{title}: Heatmap View", fontsize=24, fontweight="bold", color=TEXT, y=0.97)
    fig.text(
        0.5,
        0.89,
        "Heatmap view makes the sparse low-score buckets immediately visible without line overlap.",
        ha="center",
        fontsize=11.5,
        color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.06, right=0.95)

    output_path = output_prefix.with_name(output_prefix.name + "_heatmap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    summary_df, pos_df = _load_data(args.summary_csv, args.position_csv)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    output_prefix = Path(args.output_prefix)
    line_path = _make_line_figure(summary_df, pos_df, args.title, output_prefix)
    heatmap_path = _make_heatmap_figure(pos_df, args.title, output_prefix)

    print(f"Saved line figure to: {line_path}")
    print(f"Saved heatmap figure to: {heatmap_path}")


if __name__ == "__main__":
    main()
