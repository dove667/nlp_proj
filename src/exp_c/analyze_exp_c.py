#!/usr/bin/env python3
"""
Exp C — analysis and plotting.

Reads all JSONL result files from results/exp_c/ and generates the three
core figures defined in experiments.md:

  1. TTFT vs context length  (Llama backends + Mamba, batch size = 1)
  2. Peak memory vs context length  (Llama backends + Mamba, batch size = 1)
  3. Throughput vs batch size  (Llama backends, ctx = 8K)

Usage:
    python src/exp_c/analyze_exp_c.py
    python src/exp_c/analyze_exp_c.py --results_dir /path/to/results/exp_c --out_dir /path/to/out
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parents[2] / "results" / "exp_c"

BACKEND_LABELS = {
    "hf": "HF Transformers",
    "vllm": "vLLM",
    "sglang": "SGLang",
}
MODEL_LABELS = {
    "llama31": "Llama-3.1-8B",
    "mamba": "Falcon3-Mamba-7B",
}
SERIES_COLORS = {
    ("llama31", "hf"):     "#4A90D9",
    ("llama31", "vllm"):   "#E07B39",
    ("llama31", "sglang"): "#55A868",
    ("mamba",   "hf"):     "#C44E52",
}
SERIES_MARKERS = {
    ("llama31", "hf"):     "o",
    ("llama31", "vllm"):   "s",
    ("llama31", "sglang"): "^",
    ("mamba",   "hf"):     "D",
}

BG       = "#F7F1E6"
PANEL_BG = "#FBF8F2"
GRID     = "#D9D0C3"
TEXT     = "#2F241B"
SUBTEXT  = "#6C5B4D"
EDGE     = "#CBBEAD"


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No JSONL files found in {results_dir}")
    return pd.DataFrame(rows)


def _series_label(model: str, backend: str) -> str:
    m = MODEL_LABELS.get(model, model)
    b = BACKEND_LABELS.get(backend, backend)
    return f"{m} (HF)" if model == "mamba" else f"{m} — {b}"


def _color(model: str, backend: str) -> str:
    return SERIES_COLORS.get((model, backend), "#888888")


def _marker(model: str, backend: str) -> str:
    return SERIES_MARKERS.get((model, backend), "x")


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=TEXT, labelsize=11)


def _make_figure(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    fig.suptitle(title, fontsize=22, fontweight="bold", color=TEXT, y=0.97)
    fig.text(0.5, 0.90, subtitle, ha="center", fontsize=10.5, color=SUBTEXT)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.11, right=0.98)
    return fig, ax


def _finalize_legend(ax: plt.Axes) -> None:
    leg = ax.legend(fontsize=10, framealpha=0.88, edgecolor=EDGE)
    leg.get_frame().set_facecolor(PANEL_BG)


def plot_ttft_vs_context(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = _make_figure(
        "TTFT vs Context Length",
        "Time to first token at batch size = 1  ·  lower is better",
    )
    sub = df[df["batch_size"] == 1].copy()
    for (model, backend), grp in sub.groupby(["model", "backend"]):
        grp = grp.sort_values("context_len")
        if grp["ttft_ms"].isna().all():
            continue
        ax.plot(
            grp["context_len"] / 1024,
            grp["ttft_ms"],
            label=_series_label(model, backend),
            color=_color(model, backend),
            marker=_marker(model, backend),
            linewidth=2.4,
            markersize=7,
        )
    ax.set_xlabel("Context length (K tokens)", fontsize=12, color=TEXT)
    ax.set_ylabel("TTFT (ms)", fontsize=12, color=TEXT)
    _finalize_legend(ax)
    out = out_dir / "fig1_ttft_vs_context.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_memory_vs_context(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = _make_figure(
        "Peak GPU Memory vs Context Length",
        "Batch size = 1  ·  Mamba has no KV cache but prefill activations grow with depth",
    )
    sub = df[df["batch_size"] == 1].copy()
    for (model, backend), grp in sub.groupby(["model", "backend"]):
        grp = grp.sort_values("context_len")
        ax.plot(
            grp["context_len"] / 1024,
            grp["peak_memory_gb"],
            label=_series_label(model, backend),
            color=_color(model, backend),
            marker=_marker(model, backend),
            linewidth=2.4,
            markersize=7,
        )
    ax.set_xlabel("Context length (K tokens)", fontsize=12, color=TEXT)
    ax.set_ylabel("Peak GPU memory (GB)", fontsize=12, color=TEXT)
    _finalize_legend(ax)
    out = out_dir / "fig2_memory_vs_context.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_throughput_vs_batch(df: pd.DataFrame, out_dir: Path) -> None:
    ctx = 8192
    sub = df[(df["model"] == "llama31") & (df["context_len"] == ctx)].copy()
    if sub.empty:
        ctx = int(df[df["model"] == "llama31"]["context_len"].min())
        sub = df[(df["model"] == "llama31") & (df["context_len"] == ctx)].copy()

    fig, ax = _make_figure(
        "Throughput vs Batch Size — Llama-3.1-8B",
        f"Context length = {ctx // 1024}K  ·  vLLM queues requests when KV pool is exhausted",
    )
    for backend, grp in sub.groupby("backend"):
        grp = grp.sort_values("batch_size")
        ax.plot(
            grp["batch_size"],
            grp["throughput_tokens_per_s"],
            label=_series_label("llama31", backend),
            color=_color("llama31", backend),
            marker=_marker("llama31", backend),
            linewidth=2.4,
            markersize=7,
        )
    ax.set_xlabel("Batch size", fontsize=12, color=TEXT)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12, color=TEXT)
    _finalize_legend(ax)
    out = out_dir / "fig3_throughput_vs_batch.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Summary ===")
    cols = ["model", "backend", "context_len", "batch_size",
            "ttft_ms", "tpot_ms", "throughput_tokens_per_s", "peak_memory_gb"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False, float_format="{:.2f}".format))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory. Defaults to --results_dir.")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    df = load_results(args.results_dir)
    print_summary(df)

    plot_ttft_vs_context(df, out_dir)
    plot_memory_vs_context(df, out_dir)
    plot_throughput_vs_batch(df, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
