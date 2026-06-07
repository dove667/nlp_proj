import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


TASK_ORDER = ["niah_single_1", "niah_multikey_1"]
TASK_LABELS = {
    "niah_single_1": "NIAH Single-Key",
    "niah_multikey_1": "NIAH Multi-Key",
}
LENGTH_ORDER = [4096, 8192, 16384, 32768]
BG = "#FFFFFF"
PANEL_BG = "#FFFFFF"
TEXT = "#222222"
SUBTEXT = "#5A5A5A"
EDGE = "#CFCFCF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create heatmap figure(s) for RULER NIAH position sensitivity."
    )
    parser.add_argument(
        "--position_csv",
        required=True,
        nargs="+",
        help="One or more position_sensitivity_10bins.csv files",
    )
    parser.add_argument("--output_prefix", required=True,
                        help="Output prefix, e.g. results/exp_a/llama31/position_sensitivity_llama31")
    parser.add_argument(
        "--model_labels",
        nargs="*",
        default=None,
        help="Optional model labels aligned with --position_csv order",
    )
    return parser.parse_args()


def _load_data(position_csv: str) -> pd.DataFrame:
    pos_df = pd.read_csv(position_csv)
    pos_df["length"] = pos_df["length"].astype(int)
    pos_df["bucket_id"] = pos_df["bucket_id"].astype(int)
    return pos_df


def _resolve_model_labels(position_csvs: list[str], model_labels: list[str] | None) -> list[str]:
    if model_labels is not None and len(model_labels) != len(position_csvs):
        raise ValueError("--model_labels count must match --position_csv count")
    if model_labels is not None:
        return model_labels
    return [Path(csv_path).parent.name for csv_path in position_csvs]


def _make_heatmap_figure(
    pos_dfs: list[pd.DataFrame],
    model_labels: list[str],
    output_prefix: Path,
) -> Path:
    n_models = len(pos_dfs)
    fig_height = 3.0 * n_models + 1.8
    fig = plt.figure(figsize=(12.6, fig_height))
    gs = fig.add_gridspec(n_models, 3, width_ratios=[1, 1, 0.05], wspace=0.18, hspace=0.34)
    cax = fig.add_subplot(gs[0, 2])
    fig.patch.set_facecolor(BG)

    cmap = LinearSegmentedColormap.from_list(
        "sandstone",
        ["#FFF7E8", "#F6D28B", "#E89A5B", "#B65434", "#7D3527"],
    )

    lengths_present = sorted(
        {
            int(length)
            for pos_df in pos_dfs
            for length in pos_df["length"].unique()
            if int(length) in LENGTH_ORDER
        }
    )
    if not lengths_present:
        lengths_present = LENGTH_ORDER

    vmin = min(97.0, min(float(pos_df["score"].min()) for pos_df in pos_dfs))
    vmax = 100.0
    image = None

    for row_idx, (pos_df, model_label) in enumerate(zip(pos_dfs, model_labels)):
        row_axes = [fig.add_subplot(gs[row_idx, 0]), fig.add_subplot(gs[row_idx, 1])]
        if row_idx > 0:
            cax.remove()
            cax = fig.add_subplot(gs[:, 2])

        for col_idx, (ax, task) in enumerate(zip(row_axes, TASK_ORDER)):
            task_df = pos_df[pos_df["task"] == task]
            pivot = (
                task_df.pivot(index="length", columns="bucket_id", values="score")
                .reindex(index=lengths_present)
                .reindex(columns=list(range(10)))
            )
            image = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_facecolor(PANEL_BG)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(
                [f"{i * 10}-{(i + 1) * 10}%" for i in pivot.columns],
                rotation=20,
                ha="right",
            )
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{idx // 1024}K" for idx in pivot.index])
            ax.tick_params(colors=TEXT, labelsize=9.5)
            ax.set_xlabel("Position Bucket", fontsize=11, color=TEXT)
            if col_idx == 0:
                ax.set_ylabel(f"{model_label}\nContext Length", fontsize=11, color=TEXT)
            else:
                ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_color(EDGE)

            for y_idx in range(pivot.shape[0]):
                for x_idx in range(pivot.shape[1]):
                    value = pivot.iat[y_idx, x_idx]
                    if pd.isna(value):
                        continue
                    numeric = float(value)
                    ax.text(
                        x_idx,
                        y_idx,
                        f"{numeric:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8.6,
                        color=TEXT,
                        fontweight="bold" if numeric < 99.95 else "normal",
                    )

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Accuracy (%)", fontsize=11, color=TEXT)
    cbar.ax.tick_params(labelsize=9.5, colors=TEXT)

    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.95)

    output_path = output_prefix.with_name(output_prefix.name + "_heatmap.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    pos_dfs = [_load_data(path) for path in args.position_csv]
    model_labels = _resolve_model_labels(args.position_csv, args.model_labels)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "savefig.facecolor": BG,
        "figure.facecolor": BG,
        "axes.facecolor": PANEL_BG,
    })

    output_prefix = Path(args.output_prefix)
    heatmap_path = _make_heatmap_figure(pos_dfs, model_labels, output_prefix)
    print(f"Saved heatmap figure to: {heatmap_path}")


if __name__ == "__main__":
    main()
