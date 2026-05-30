import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_LABELS = {
    "vt": "VT",
    "cwe": "CWE",
    "fwe": "FWE",
}

TASK_COLORS = {
    "vt": "#0F4C81",
    "cwe": "#2E86AB",
    "fwe": "#58A4B0",
}

MODEL_STYLES = {
    "llama31": {"label": "Llama-3.1-8B-Instruct", "linestyle": "-", "marker": "o"},
    "mamba": {"label": "Falcon3-Mamba-7B-Instruct", "linestyle": "--", "marker": "s"},
}

BG = "#F7F1E6"
PANEL_BG = "#FBF8F2"
GRID = "#D9D0C3"
TEXT = "#2F241B"
SUBTEXT = "#6C5B4D"
EDGE = "#CBBEAD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Exp B RULER reasoning model-comparison score curves.")
    parser.add_argument("--llama_summary_csv", required=True)
    parser.add_argument("--mamba_summary_csv", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--title", default="Exp B RULER Reasoning")
    return parser.parse_args()


def style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=TEXT, labelsize=10)


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["length"] = df["length"].astype(int)
    return df


def plot_score_figure(llama_df: pd.DataFrame, mamba_df: pd.DataFrame, title: str, output_prefix: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.6), sharey=True)
    fig.patch.set_facecolor(BG)

    model_frames = {
        "llama31": llama_df,
        "mamba": mamba_df,
    }

    for ax, task in zip(axes, ["vt", "cwe", "fwe"]):
        style_axes(ax)
        for model_key, frame in model_frames.items():
            sub = frame[frame["task"] == task].sort_values("length")
            if sub.empty:
                continue
            style = MODEL_STYLES[model_key]
            ax.plot(
                sub["length"],
                sub["score"],
                linewidth=2.4,
                markersize=6.0,
                color=TASK_COLORS[task],
                linestyle=style["linestyle"],
                marker=style["marker"],
                label=style["label"],
            )
        ax.set_title(TASK_LABELS[task], fontsize=16, fontweight="bold", color=TEXT, pad=8)
        ax.set_xlabel("Context Length", fontsize=11, color=TEXT)
        ax.set_xticks(sorted(set(llama_df["length"].tolist() + mamba_df["length"].tolist())))
        ax.set_xticklabels([f"{int(x/1024)}K" for x in ax.get_xticks()])

    axes[0].set_ylabel("Accuracy", fontsize=11, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=True,
        facecolor=PANEL_BG,
        edgecolor=EDGE,
        fontsize=9.5,
    )

    fig.suptitle(f"{title}: Model Comparison", fontsize=20, fontweight="bold", color=TEXT, y=0.975)
    fig.text(
        0.5,
        0.885,
        "Each panel keeps the task fixed and compares how the two models degrade as context length grows.",
        ha="center",
        fontsize=10.0,
        color=SUBTEXT,
    )
    fig.subplots_adjust(top=0.76, bottom=0.24, left=0.08, right=0.98, wspace=0.12)

    out = output_prefix.with_name(output_prefix.name + "_model_compare.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    llama_df = load_summary(args.llama_summary_csv)
    mamba_df = load_summary(args.mamba_summary_csv)
    output_prefix = Path(args.output_prefix)
    out = plot_score_figure(llama_df, mamba_df, args.title, output_prefix)
    print(f"Saved score figure to: {out}")


if __name__ == "__main__":
    main()
