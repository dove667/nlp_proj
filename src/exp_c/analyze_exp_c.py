import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR, load_rows

BG = "#FFFFFF"
PANEL_BG = "#FFFFFF"
GRID = "#E2E2E2"
TEXT = "#222222"
EDGE = "#CFCFCF"
SUBTEXT = "#5A5A5A"
COLORS = {
    ("llama31", "hf"): "#4A90D9",
    ("mamba", "hf"): "#C44E52",
    ("llama31", "vllm"): "#E07B39",
}
MARKERS = {
    ("llama31", "hf"): "o",
    ("mamba", "hf"): "D",
    ("llama31", "vllm"): "s",
}
LABELS = {
    ("llama31", "hf"): "Llama-3.1-8B — HF",
    ("mamba", "hf"): "Falcon3-Mamba-7B — HF",
    ("llama31", "vllm"): "Llama-3.1-8B — vLLM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out_dir", type=Path, default=None)
    return parser.parse_args()


def load_optional(results_dir: Path, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(load_rows(path))


def load_optional_csv(results_dir: Path, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def style() -> None:
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})


def make_fig(title: str, subtitle: str, figsize: tuple[float, float] = (9.0, 5.6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=TEXT)
    fig.suptitle(title, fontsize=20, fontweight="bold", color=TEXT, y=0.97)
    fig.text(0.5, 0.90, subtitle, ha="center", fontsize=10.5, color=SUBTEXT)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.11, right=0.98)
    return fig, ax


def add_legend(ax: plt.Axes) -> None:
    legend = ax.legend(fontsize=10, framealpha=0.9, edgecolor=EDGE)
    legend.get_frame().set_facecolor(PANEL_BG)


def plot_prefill(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = make_fig("Exp C1 Prefill TTFT", "HF-only fair comparison: Llama vs Mamba, bs=1")
    for key, group in df.groupby(["model", "backend"]):
        group = group.sort_values("context_len")
        ax.plot(
            group["context_len"] / 1024,
            group["ttft_ms"],
            label=LABELS[key],
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=2.4,
            markersize=7,
        )
    ax.set_xlabel("Context length (K tokens)", color=TEXT)
    ax.set_ylabel("TTFT (ms)", color=TEXT)
    add_legend(ax)
    fig.savefig(out_dir / "c1_prefill_ttft_vs_context.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = make_fig("Exp C1 Prefill Memory", "HF-only fair comparison: Llama vs Mamba, bs=1")
    for key, group in df.groupby(["model", "backend"]):
        group = group.sort_values("context_len")
        ax.plot(
            group["context_len"] / 1024,
            group["peak_memory_gb"],
            label=LABELS[key],
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=2.4,
            markersize=7,
        )
    ax.set_xlabel("Context length (K tokens)", color=TEXT)
    ax.set_ylabel("Peak GPU memory (GB)", color=TEXT)
    add_legend(ax)
    fig.savefig(out_dir / "c1_prefill_memory_vs_context.pdf", bbox_inches="tight")
    plt.close(fig)

    df.sort_values(["model", "context_len"]).to_csv(out_dir / "c1_prefill_summary.csv", index=False)


def plot_decode(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    prompt_len_col = "target_prompt_len" if "target_prompt_len" in df.columns else "prompt_len"
    df = df.copy()
    if "backend" not in df.columns:
        df["backend"] = "hf"
    df = df.dropna(subset=[prompt_len_col, "tpot_ms"])

    fig, ax = make_fig(
        "Exp C2 Decode TPOT vs Prompt Length",
        "HF-only, continuation prompt, bs=1, fixed output length",
    )
    for key, group in df.groupby(["model", "backend"]):
        group = group.sort_values(prompt_len_col)
        ax.plot(
            group[prompt_len_col] / 1024,
            group["tpot_ms"],
            label=LABELS[key],
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=2.4,
            markersize=7,
        )
    ax.set_xlabel("Prompt length (K tokens)", color=TEXT)
    ax.set_ylabel("TPOT (ms/token)", color=TEXT)
    add_legend(ax)
    fig.savefig(out_dir / "c2_decode_tpot_vs_prompt_len.pdf", bbox_inches="tight")
    plt.close(fig)

    df.sort_values(["model", prompt_len_col]).to_csv(out_dir / "c2_decode_summary.csv", index=False)


def plot_backend(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    valid = df[df["oom"] == False].copy()
    for context_len in sorted(valid["context_len"].dropna().unique()):
        sub = valid[valid["context_len"] == context_len].copy()
        fig, ax = make_fig(
            f"Exp C3 Backend Compare at {context_len // 1024}K",
            "Llama HF vs Llama vLLM, sweep batch size",
        )
        for key, group in sub.groupby(["model", "backend"]):
            group = group.sort_values("batch_size")
            ax.plot(
                group["batch_size"],
                group["requests_per_s"],
                label=LABELS[key],
                color=COLORS[key],
                marker=MARKERS[key],
                linewidth=2.4,
                markersize=7,
            )
        ax.set_xlabel("Batch size", color=TEXT)
        ax.set_ylabel("Requests/s", color=TEXT)
        add_legend(ax)
        fig.savefig(out_dir / f"c3_backend_llama_requests_vs_batch_{context_len // 1024}k.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = df.sort_values(["backend", "context_len", "batch_size"]).copy()
    summary.to_csv(out_dir / "c3_backend_llama_summary.csv", index=False)

    hf_rows: list[dict[str, object]] = []
    hf_df = summary[summary["backend"] == "hf"].copy()
    for context_len, group in hf_df.groupby("context_len"):
        ok = group[group["oom"] == False].sort_values("batch_size")
        hf_rows.append(
            {
                "context_len": context_len,
                "max_supported_batch": int(ok["batch_size"].max()) if not ok.empty else 0,
                "oom_encountered": bool((group["oom"] == True).any()),
            }
        )
    pd.DataFrame(hf_rows).sort_values("context_len").to_csv(
        out_dir / "c3_backend_llama_hf_capacity_summary.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    prefill_df = load_optional(args.results_dir, "prefill_hf.jsonl")
    decode_csv_df = load_optional_csv(args.results_dir, "c2_decode_prompt_len_sweep_summary.csv")
    decode_df = decode_csv_df if not decode_csv_df.empty else load_optional(args.results_dir, "decode_hf.jsonl")
    backend_df = load_optional(args.results_dir, "backend_llama.jsonl")

    plot_prefill(prefill_df, out_dir)
    plot_decode(decode_df, out_dir)
    plot_backend(backend_df, out_dir)


if __name__ == "__main__":
    main()
