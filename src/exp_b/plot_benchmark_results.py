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
    "hotpotqa": "HotpotQA",
    "qasper": "Qasper",
    "gov_report": "GovReport",
    "repobench-p": "RepoBench-P",
}

TASK_COLORS = {
    "vt": "#0F4C81",
    "cwe": "#2E86AB",
    "fwe": "#58A4B0",
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
    parser = argparse.ArgumentParser(description="Plot Exp B score curves and decay bars.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--decay_csv", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--title", default="Exp B Reasoning Benchmarks")
    return parser.parse_args()


def style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=TEXT, labelsize=11)


def plot_score_figure(summary_df: pd.DataFrame, title: str, output_prefix: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=False)
    fig.patch.set_facecolor(BG)

    configs = [
        (axes[0], "ruler_reasoning", ["vt", "cwe", "fwe"], "RULER Reasoning"),
        (axes[1], "longbench", ["hotpotqa", "qasper", "gov_report", "repobench-p"], "LongBench"),
    ]

    for ax, benchmark, tasks, subtitle in configs:
        sub = summary_df[summary_df["benchmark"] == benchmark]
        for task in tasks:
            task_df = sub[sub["task"] == task].sort_values("length")
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
        ax.set_title(subtitle, fontsize=18, fontweight="bold", color=TEXT, pad=10)
        ax.set_xlabel("Context Length", fontsize=12, color=TEXT)
        ax.set_ylabel("Score", fontsize=12, color=TEXT)
        ax.legend(loc="lower left", frameon=True, facecolor=PANEL_BG, edgecolor=EDGE, fontsize=10)

    fig.suptitle(f"{title}: Score Curves", fontsize=24, fontweight="bold", color=TEXT, y=0.97)
    fig.text(
        0.5,
        0.90,
        "RULER reasoning uses exact-match style accuracy; LongBench uses task-specific metrics such as QA F1, Rouge-L F1, and exact match.",
        ha="center",
        fontsize=11.5,
        color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.07, right=0.98, wspace=0.18)

    out = output_prefix.with_name(output_prefix.name + "_scores.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_decay_figure(decay_df: pd.DataFrame, title: str, output_prefix: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    fig.patch.set_facecolor(BG)

    configs = [
        (axes[0], "ruler_reasoning", ["vt", "cwe", "fwe"], "RULER Reasoning Decay"),
        (axes[1], "longbench", ["hotpotqa", "qasper", "gov_report", "repobench-p"], "LongBench Decay"),
    ]

    for ax, benchmark, tasks, subtitle in configs:
        sub = decay_df[decay_df["benchmark"] == benchmark].copy()
        style_axes(ax)
        x_positions = []
        x_labels = []
        idx = 0
        for task in tasks:
            task_df = sub[sub["task"] == task].sort_values("to_length")
            for row in task_df.itertuples():
                x_positions.append(idx)
                x_labels.append(f"{TASK_LABELS[task]}\n{int(row.to_length/1024)}K")
                ax.bar(idx, row.delta_score, color=TASK_COLORS[task], width=0.72)
                ax.text(idx, row.delta_score, f"{row.delta_score:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT)
                idx += 1
            idx += 0.6
        ax.axhline(0.0, color=EDGE, linewidth=1.2)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=28, ha="right")
        ax.set_title(subtitle, fontsize=18, fontweight="bold", color=TEXT, pad=10)
        ax.set_xlabel("Task / Target Length", fontsize=12, color=TEXT)
        ax.set_ylabel("Delta vs. shortest length", fontsize=12, color=TEXT)

    fig.suptitle(f"{title}: Decay vs. Shortest Length", fontsize=24, fontweight="bold", color=TEXT, y=0.97)
    fig.text(
        0.5,
        0.90,
        "Negative bars indicate degradation relative to the shortest tested context length within the same task.",
        ha="center",
        fontsize=11.5,
        color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.07, right=0.98, wspace=0.18)

    out = output_prefix.with_name(output_prefix.name + "_decay.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    decay_df = pd.read_csv(args.decay_csv)
    summary_df["length"] = summary_df["length"].astype(int)
    if not decay_df.empty:
        decay_df["to_length"] = decay_df["to_length"].astype(int)

    output_prefix = Path(args.output_prefix)
    score_path = plot_score_figure(summary_df, args.title, output_prefix)
    print(f"Saved score figure to: {score_path}")
    if not decay_df.empty:
        decay_path = plot_decay_figure(decay_df, args.title, output_prefix)
        print(f"Saved decay figure to: {decay_path}")


if __name__ == "__main__":
    main()
