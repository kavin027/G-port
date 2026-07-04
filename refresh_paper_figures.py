from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_EN = ROOT / "paper" / "socc26" / "figures"
FIG_ZH = ROOT / "paper" / "socc26_zh" / "figures"

LABELS = {
    "sparse_flexible_static": "Static",
    "worker_aware_sparse_flexible": "Cost-aware",
    "rank_aware_sparse_flexible": "Decode-aware",
    "deadline_aware_sparse_flexible": "Deadline-aware",
    "balanced_rank_aware_sparse_flexible": "Balanced",
}

COLORS = {
    "sparse_flexible_static": "#6b7280",
    "worker_aware_sparse_flexible": "#d97706",
    "rank_aware_sparse_flexible": "#2563eb",
    "deadline_aware_sparse_flexible": "#059669",
    "balanced_rank_aware_sparse_flexible": "#7c3aed",
}


def main() -> None:
    FIG_EN.mkdir(parents=True, exist_ok=True)
    FIG_ZH.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.2,
        }
    )

    plot_hypothesis()
    plot_assignment()
    plot_sparsity()
    plot_runtime_summary()
    plot_tcp_time_to_loss()
    plot_realdata()
    plot_alignment()
    plot_worker_scaling()
    plot_gain_overhead()
    copy_to_zh()


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_EN / name, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def copy_to_zh() -> None:
    for path in FIG_EN.glob("*.png"):
        shutil.copy2(path, FIG_ZH / path.name)


def short_case(label: str) -> str:
    return label.replace("phase_", "").replace("_", "\n")


def plot_hypothesis() -> None:
    path = ROOT / "organized_strengthen_assignment_sparsity_2seeds" / "hypothesis_report.csv"
    df = pd.read_csv(path)
    rows = df[df["scope"] == "ALL"].copy()
    rows["label"] = (
        rows["hypothesis"]
        .str.replace(r"^H\d+\w?\s+", "", regex=True)
        .str.replace("decode-aware ", "decode ")
        .str.replace("deadline-aware ", "deadline ")
        .str.replace("cost-aware ", "cost ")
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    values = 100.0 * rows["mean_latency_improvement"].to_numpy()
    colors = ["#d97706" if v < 0 else "#2563eb" for v in values]
    ax.bar(np.arange(len(rows)), values, color=colors, width=0.68)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_ylabel("Mean gain (%)")
    ax.set_title("Hypothesis-level gains")
    ax.set_xticks(np.arange(len(rows)), rows["label"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.2)
    save(fig, "hypothesis_improvements.png")


def plot_assignment() -> None:
    path = ROOT / "organized_strengthen_assignment_sparsity_2seeds" / "organized_aggregate.csv"
    df = pd.read_csv(path)
    keep = [
        "sparse_flexible_static",
        "worker_aware_sparse_flexible",
        "rank_aware_sparse_flexible",
        "deadline_aware_sparse_flexible",
        "balanced_rank_aware_sparse_flexible",
    ]
    rows = df[(df["suite"] == "assignment") & df["strategy"].isin(keep)].copy()
    cases = list(dict.fromkeys(rows["case"]))
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for strategy in keep:
        sdf = rows[rows["strategy"] == strategy].set_index("case").reindex(cases)
        ax.plot(x, sdf["mean_latency"], marker="o", label=LABELS[strategy], color=COLORS[strategy])
    ax.set_ylabel("Mean latency")
    ax.set_title("Phase-changing heterogeneity")
    ax.set_xticks(x, [short_case(c) for c in cases])
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.2)
    save(fig, "assignment_latency.png")


def plot_sparsity() -> None:
    path = ROOT / "organized_strengthen_assignment_sparsity_2seeds" / "hypothesis_report.csv"
    df = pd.read_csv(path)
    rows = df[(df["hypothesis"] == "H2b decode-aware under sparse inputs") & (df["scope"] != "ALL")].copy()
    rows["density"] = rows["scope"].str.replace("density", "", regex=False).astype(float)
    rows = rows.sort_values("density")
    fig, ax = plt.subplots(figsize=(3.0, 2.25))
    ax.plot(rows["density"], 100.0 * rows["mean_latency_improvement"], marker="o", color="#2563eb")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Input density")
    ax.set_ylabel("Mean gain (%)")
    ax.set_title("Sparse input sensitivity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.2)
    save(fig, "sparsity_worker_aware.png")


def plot_runtime_summary() -> None:
    path = ROOT / "runtime_sweep_highhetero_server" / "aggregate_runtime_summary.csv"
    df = pd.read_csv(path)
    order = [
        "sparse_flexible_static",
        "worker_aware_sparse_flexible",
        "rank_aware_sparse_flexible",
        "deadline_aware_sparse_flexible",
    ]
    rows = df.set_index("strategy").reindex(order)
    x = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.bar(x - width / 2, 1000.0 * rows["mean_decode_latency"], width, label="Mean", color="#60a5fa")
    ax.bar(x + width / 2, 1000.0 * rows["p95_decode_latency"], width, label="p95", color="#1d4ed8")
    ax.set_xticks(x, [LABELS[s] for s in order], rotation=18, ha="right")
    ax.set_ylabel("First-decode (ms)")
    ax.set_title("Multi-process runtime")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.2)
    save(fig, "runtime_latency_summary.png")


def plot_tcp_time_to_loss() -> None:
    path = ROOT / "end_to_end_ci_diagnostics" / "tcp_time_to_loss.csv"
    df = pd.read_csv(path)
    order = [
        "sparse_flexible_static",
        "worker_aware_sparse_flexible",
        "rank_aware_sparse_flexible",
        "deadline_aware_sparse_flexible",
    ]
    grouped = df.groupby("strategy")["barrier_time_to_loss"]
    means = grouped.mean().reindex(order)
    stds = grouped.std().reindex(order).fillna(0.0)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    ax.bar(
        x,
        1000.0 * means.to_numpy(),
        yerr=1000.0 * stds.to_numpy(),
        capsize=3,
        color=[COLORS[s] for s in order],
    )
    ax.set_xticks(x, [LABELS[s] for s in order], rotation=18, ha="right")
    ax.set_ylabel("Barrier time (ms)")
    ax.set_title("Time to 90% loss progress")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.2)
    save(fig, "tcp_time_to_loss.png")


def plot_realdata() -> None:
    path = ROOT / "realdata_diagnostics" / "realdata_improvement_stats.csv"
    df = pd.read_csv(path)
    datasets = ["a9a", "w8a", "rcv1"]
    strategies = [
        "worker_aware_sparse_flexible",
        "rank_aware_sparse_flexible",
        "deadline_aware_sparse_flexible",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.25), sharey=True)
    for ax, mean_col, std_col, title in [
        (axes[0], "mean_improvement_mean", "mean_improvement_std", "Mean"),
        (axes[1], "p95_improvement_mean", "p95_improvement_std", "p95"),
    ]:
        x = np.arange(len(datasets))
        width = 0.24
        for idx, strategy in enumerate(strategies):
            vals = []
            errs = []
            for dataset in datasets:
                row = df[(df["dataset"] == dataset) & (df["strategy"] == strategy)].iloc[0]
                vals.append(100.0 * row[mean_col])
                errs.append(100.0 * row[std_col])
            ax.bar(
                x + (idx - 1) * width,
                vals,
                width=width,
                yerr=errs,
                capsize=2,
                color=COLORS[strategy],
                label=LABELS[strategy],
            )
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks(x, datasets)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Gain (%)")
    axes[1].legend(frameon=False, fontsize=6.7, loc="lower right")
    fig.tight_layout(pad=0.15, w_pad=0.35)
    save(fig, "runtime_realdata_improvements_a9a_w8a_rcv1.png")


def plot_alignment() -> None:
    path = ROOT / "realdata_alignment_diagnostics" / "alignment_mechanism_summary.csv"
    df = pd.read_csv(path)
    datasets = ["w8a", "rcv1"]
    modes = ["aligned", "none", "anti"]
    mode_label = {"aligned": "aligned", "none": "random", "anti": "anti"}
    strategies = ["rank_aware_sparse_flexible", "deadline_aware_sparse_flexible"]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.25), sharey=True)
    for ax, dataset in zip(axes, datasets):
        x = np.arange(len(modes))
        width = 0.34
        for idx, strategy in enumerate(strategies):
            vals = []
            errs = []
            for mode in modes:
                row = df[
                    (df["dataset"] == dataset)
                    & (df["alignment_mode"] == mode)
                    & (df["strategy"] == strategy)
                ].iloc[0]
                vals.append(100.0 * row["p95_improvement"])
                errs.append(100.0 * row["p95_improvement_std"])
            ax.bar(
                x + (idx - 0.5) * width,
                vals,
                width=width,
                yerr=errs,
                capsize=2,
                color=COLORS[strategy],
                label=LABELS[strategy],
            )
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks(x, [mode_label[m] for m in modes], rotation=15, ha="right")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("p95 gain (%)")
    axes[1].legend(frameon=False, fontsize=6.9, loc="lower left")
    fig.tight_layout(pad=0.15, w_pad=0.35)
    save(fig, "controlled_alignment_p95.png")


def plot_worker_scaling() -> None:
    path = ROOT / "runtime_scaling_diagnostics" / "runtime_scaling_diagnostic_summary.csv"
    df = pd.read_csv(path)
    strategies = ["rank_aware_sparse_flexible", "deadline_aware_sparse_flexible"]
    experiments = [("fixed_16_shards", "fixed shards"), ("proportional_shards", "proportional")]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.25), sharey=True)
    for ax, (experiment, title) in zip(axes, experiments):
        sdf = df[df["experiment"] == experiment]
        for strategy in strategies:
            rows = sdf[sdf["strategy"] == strategy].sort_values("workers")
            ax.plot(
                rows["workers"],
                100.0 * rows["p95_decode_improvement"],
                marker="o",
                label=LABELS[strategy],
                color=COLORS[strategy],
            )
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks([8, 16, 24, 32])
        ax.set_title(title)
        ax.set_xlabel("Workers")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("p95 gain (%)")
    axes[1].legend(frameon=False, fontsize=6.9, loc="lower right")
    fig.tight_layout(pad=0.15, w_pad=0.35)
    save(fig, "worker_scaling_p95_improvements.png")


def plot_gain_overhead() -> None:
    path = ROOT / "runtime_scaling_diagnostics" / "runtime_scaling_diagnostic_summary.csv"
    df = pd.read_csv(path)
    rows = df[df["experiment"] == "proportional_shards"]
    strategies = ["rank_aware_sparse_flexible", "deadline_aware_sparse_flexible"]
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for strategy in strategies:
        sdf = rows[rows["strategy"] == strategy].sort_values("workers")
        ax.plot(
            sdf["workers"],
            sdf["p95_latency_saving_ms"],
            marker="o",
            color=COLORS[strategy],
            label=f"{LABELS[strategy]} gain",
        )
        ax.plot(
            sdf["workers"],
            sdf["scheduler_overhead_ms"],
            marker=".",
            linestyle="--",
            color=COLORS[strategy],
            label=f"{LABELS[strategy]} overhead",
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks([8, 16, 24, 32])
    ax.set_xlabel("Workers = shards")
    ax.set_ylabel("Milliseconds")
    ax.set_title("Gain vs scheduling overhead")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.2)
    save(fig, "gain_vs_overhead_scaling.png")


if __name__ == "__main__":
    main()
