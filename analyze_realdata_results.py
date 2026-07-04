from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = {
    "a9a": Path("runtime_realdata_a9a_sweep_bjb1") / "combined_realdata_summary.csv",
    "w8a": Path("runtime_realdata_w8a_sweep_bjb1") / "combined_realdata_summary.csv",
    "rcv1": Path("runtime_realdata_rcv1_sweep_bjb1") / "combined_realdata_summary.csv",
}

STRATEGIES = [
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
]

DISPLAY = {
    "worker_aware_sparse_flexible": "Worker",
    "rank_aware_sparse_flexible": "Decode",
    "deadline_aware_sparse_flexible": "Deadline",
}

COLORS = {
    "worker_aware_sparse_flexible": "#c44e52",
    "rank_aware_sparse_flexible": "#4c72b0",
    "deadline_aware_sparse_flexible": "#55a868",
}


def main() -> None:
    out_dir = Path("realdata_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = load_combined()
    per_seed = build_per_seed(combined)
    stats = build_stats(per_seed)
    mechanism = build_mechanism(per_seed)

    per_seed.to_csv(out_dir / "realdata_per_seed_improvements.csv", index=False)
    stats.to_csv(out_dir / "realdata_improvement_stats.csv", index=False)
    mechanism.to_csv(out_dir / "realdata_mechanism_deltas.csv", index=False)
    write_report(out_dir / "realdata_diagnostic_report.md", stats, mechanism)

    plot_improvements_with_errorbars(stats, out_dir)
    plot_selected_completed_deltas(mechanism, out_dir)


def load_combined() -> pd.DataFrame:
    frames = []
    for dataset, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["dataset"] = dataset
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_per_seed(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, seed), group in combined.groupby(["dataset", "seed"], sort=False):
        base = group[group["strategy"] == "sparse_flexible_static"].iloc[0]
        for strategy in STRATEGIES:
            row = group[group["strategy"] == strategy].iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "strategy": strategy,
                    "mean_improvement": improvement(
                        base["mean_decode_latency"], row["mean_decode_latency"]
                    ),
                    "p95_improvement": improvement(
                        base["p95_decode_latency"], row["p95_decode_latency"]
                    ),
                    "barrier_improvement": improvement(
                        base["mean_barrier_latency"], row["mean_barrier_latency"]
                    ),
                    "selected_rows_delta": row["mean_selected_rows"]
                    - base["mean_selected_rows"],
                    "completed_rows_delta": row["mean_completed_rows"]
                    - base["mean_completed_rows"],
                    "extra_compute_delta": row["mean_extra_compute"]
                    - base["mean_extra_compute"],
                    "scheduler_ms_delta": 1000.0
                    * (row["mean_scheduler_seconds"] - base["mean_scheduler_seconds"]),
                    "mean_latency_ms": 1000.0 * row["mean_decode_latency"],
                    "p95_latency_ms": 1000.0 * row["p95_decode_latency"],
                    "base_mean_latency_ms": 1000.0 * base["mean_decode_latency"],
                    "base_p95_latency_ms": 1000.0 * base["p95_decode_latency"],
                }
            )
    return pd.DataFrame(rows)


def improvement(base: float, value: float) -> float:
    return (base - value) / base


def build_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        per_seed.groupby(["dataset", "strategy"], sort=False)
        .agg(
            mean_improvement_mean=("mean_improvement", "mean"),
            mean_improvement_std=("mean_improvement", sample_std),
            mean_improvement_min=("mean_improvement", "min"),
            mean_improvement_max=("mean_improvement", "max"),
            p95_improvement_mean=("p95_improvement", "mean"),
            p95_improvement_std=("p95_improvement", sample_std),
            p95_improvement_min=("p95_improvement", "min"),
            p95_improvement_max=("p95_improvement", "max"),
            barrier_improvement_mean=("barrier_improvement", "mean"),
            barrier_improvement_std=("barrier_improvement", sample_std),
        )
        .reset_index()
    )


def build_mechanism(per_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        per_seed.groupby(["dataset", "strategy"], sort=False)
        .agg(
            selected_rows_delta_mean=("selected_rows_delta", "mean"),
            selected_rows_delta_std=("selected_rows_delta", sample_std),
            completed_rows_delta_mean=("completed_rows_delta", "mean"),
            completed_rows_delta_std=("completed_rows_delta", sample_std),
            extra_compute_delta_mean=("extra_compute_delta", "mean"),
            extra_compute_delta_std=("extra_compute_delta", sample_std),
            scheduler_ms_delta_mean=("scheduler_ms_delta", "mean"),
            scheduler_ms_delta_std=("scheduler_ms_delta", sample_std),
        )
        .reset_index()
    )


def sample_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def plot_improvements_with_errorbars(stats: pd.DataFrame, out_dir: Path) -> None:
    datasets = list(DATASETS)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for ax, mean_col, std_col, title in [
        (
            axes[0],
            "mean_improvement_mean",
            "mean_improvement_std",
            "Mean first-decode improvement",
        ),
        (
            axes[1],
            "p95_improvement_mean",
            "p95_improvement_std",
            "P95 first-decode improvement",
        ),
    ]:
        width = 0.22
        x = np.arange(len(datasets))
        for idx, strategy in enumerate(STRATEGIES):
            values = []
            errors = []
            for dataset in datasets:
                row = stats[
                    (stats["dataset"] == dataset) & (stats["strategy"] == strategy)
                ].iloc[0]
                values.append(100.0 * row[mean_col])
                errors.append(100.0 * row[std_col])
            offsets = x + (idx - 1) * width
            ax.bar(
                offsets,
                values,
                width=width,
                color=COLORS[strategy],
                label=DISPLAY[strategy],
                yerr=errors,
                capsize=3,
                linewidth=0.7,
                edgecolor="white",
            )
            for xi, value in zip(offsets, values):
                va = "bottom" if value >= 0 else "top"
                y = value + (2.0 if value >= 0 else -3.0)
                ax.text(xi, y, f"{value:+.1f}", ha="center", va=va, fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, datasets)
        ax.set_ylabel("Improvement over static (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle("Real sparse datasets, 24 workers / 16 shards", fontsize=11)

    png = out_dir / "realdata_improvements_mean_std.png"
    pdf = out_dir / "realdata_improvements_mean_std.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)

    paper_fig = Path("paper") / "socc26" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_fig / "runtime_realdata_improvements_a9a_w8a_rcv1.png", dpi=220)
    fig.savefig(paper_fig / "runtime_realdata_improvements_a9a_w8a_rcv1.pdf")


def plot_selected_completed_deltas(mechanism: pd.DataFrame, out_dir: Path) -> None:
    datasets = list(DATASETS)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for ax, value_col, title in [
        (axes[0], "selected_rows_delta_mean", "Selected rows delta"),
        (axes[1], "completed_rows_delta_mean", "Completed rows delta"),
    ]:
        width = 0.22
        x = np.arange(len(datasets))
        for idx, strategy in enumerate(STRATEGIES):
            values = []
            for dataset in datasets:
                row = mechanism[
                    (mechanism["dataset"] == dataset)
                    & (mechanism["strategy"] == strategy)
                ].iloc[0]
                values.append(row[value_col])
            ax.bar(
                x + (idx - 1) * width,
                values,
                width=width,
                color=COLORS[strategy],
                label=DISPLAY[strategy],
                linewidth=0.7,
                edgecolor="white",
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, datasets)
        ax.set_ylabel("Rows vs. static")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    png = out_dir / "realdata_row_delta_diagnostics.png"
    pdf = out_dir / "realdata_row_delta_diagnostics.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)


def write_report(path: Path, stats: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    lines = [
        "# Real-Data Runtime Diagnostics",
        "",
        "All improvements are measured against `sparse_flexible_static`.",
        "",
        "## Improvement Stability",
        "",
    ]
    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        subset = stats[stats["dataset"] == dataset]
        for _, row in subset.iterrows():
            lines.append(
                "- "
                f"{DISPLAY[row['strategy']]}: mean "
                f"{100 * row['mean_improvement_mean']:+.1f}% +/- "
                f"{100 * row['mean_improvement_std']:.1f}%, p95 "
                f"{100 * row['p95_improvement_mean']:+.1f}% +/- "
                f"{100 * row['p95_improvement_std']:.1f}%"
            )
        lines.append("")

    lines.extend(
        [
            "## Mechanism Summary",
            "",
            "Positive row deltas mean the strategy waits for more rows than static.",
            "",
        ]
    )
    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        subset = mechanism[mechanism["dataset"] == dataset]
        for _, row in subset.iterrows():
            lines.append(
                "- "
                f"{DISPLAY[row['strategy']]}: selected "
                f"{row['selected_rows_delta_mean']:+.2f}, completed "
                f"{row['completed_rows_delta_mean']:+.2f}, extra compute "
                f"{row['extra_compute_delta_mean']:+.2f}, scheduler "
                f"{row['scheduler_ms_delta_mean']:+.2f} ms"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
