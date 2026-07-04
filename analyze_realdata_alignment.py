from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ADAPTIVE = [
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
]

LABELS = {
    "worker_aware_sparse_flexible": "Worker",
    "rank_aware_sparse_flexible": "Decode",
    "deadline_aware_sparse_flexible": "Deadline",
}

COLORS = {
    "aligned": "#4c72b0",
    "none": "#707070",
    "anti": "#c44e52",
}


def main() -> None:
    input_dir = Path("runtime_realdata_alignment_bjb1")
    out_dir = Path("realdata_alignment_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_csv(input_dir / "aggregate_alignment_summary.csv")
    diagnostics = pd.read_csv(input_dir / "alignment_diagnostics.csv")
    mechanism = add_static_deltas(aggregate)
    mechanism.to_csv(out_dir / "alignment_mechanism_summary.csv", index=False)
    diagnostics.to_csv(out_dir / "alignment_diagnostics.csv", index=False)
    plot_alignment_sweep(mechanism, out_dir)
    plot_alignment_mismatch(mechanism, out_dir)
    write_report(out_dir / "alignment_report.md", mechanism, diagnostics)


def add_static_deltas(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in aggregate.groupby(["dataset", "alignment_mode"], sort=False):
        static = group[group["strategy"] == "sparse_flexible_static"].iloc[0]
        for _, row in group.iterrows():
            record = row.to_dict()
            record["completed_rows_delta"] = row["completed_rows"] - static["completed_rows"]
            record["selected_rows_delta"] = row["selected_rows"] - static["selected_rows"]
            record["extra_compute_delta"] = row["extra_compute"] - static["extra_compute"]
            rows.append(record)
    return pd.DataFrame(rows)


def plot_alignment_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["strategy"].isin(["rank_aware_sparse_flexible", "deadline_aware_sparse_flexible"])]
    datasets = sorted(sub["dataset"].unique())
    modes = ["aligned", "none", "anti"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(9.0, 3.4), constrained_layout=True, sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        part = sub[sub["dataset"] == dataset]
        x = np.arange(len(modes))
        width = 0.34
        for idx, strategy in enumerate(["rank_aware_sparse_flexible", "deadline_aware_sparse_flexible"]):
            vals = []
            errs = []
            for mode in modes:
                row = part[(part["alignment_mode"] == mode) & (part["strategy"] == strategy)].iloc[0]
                vals.append(100.0 * row["p95_improvement"])
                errs.append(100.0 * row["p95_improvement_std"])
            offsets = x + (idx - 0.5) * width
            ax.bar(
                offsets,
                vals,
                width=width,
                yerr=errs,
                capsize=3,
                color="#4c72b0" if idx == 0 else "#55a868",
                label=LABELS[strategy],
                edgecolor="white",
                linewidth=0.6,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, modes)
        ax.set_title(dataset)
        ax.set_ylabel("P95 improvement over static (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Controlled decode-speed alignment", fontsize=11)
    save(fig, out_dir / "controlled_alignment_p95")


def plot_alignment_mismatch(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["strategy"].isin(ADAPTIVE)].copy()
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    markers = {
        "worker_aware_sparse_flexible": "o",
        "rank_aware_sparse_flexible": "s",
        "deadline_aware_sparse_flexible": "^",
    }
    for strategy in ADAPTIVE:
        part = sub[sub["strategy"] == strategy]
        ax.scatter(
            part["decode_speed_mismatch"],
            100.0 * part["p95_improvement"],
            label=LABELS[strategy],
            marker=markers[strategy],
            s=55,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
        )
    r = pearson(sub["decode_speed_mismatch"], sub["p95_improvement"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Decode-speed mismatch")
    ax.set_ylabel("P95 improvement over static (%)")
    ax.set_title(f"Tail gain vs. controlled mismatch (r={r:+.2f})")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out_dir / "controlled_alignment_gain_vs_mismatch")


def write_report(path: Path, df: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    sub = df[df["strategy"].isin(ADAPTIVE)]
    lines = [
        "# Controlled Alignment Diagnostics",
        "",
        "The experiment reorders worker speeds so static assignment is aligned, random, or anti-aligned with decode-priority rows.",
        "",
        "## Mismatch Levels",
        "",
    ]
    lines.append(
        diagnostics.groupby(["dataset", "alignment_mode"], sort=False)
        .agg(
            mismatch=("decode_speed_mismatch", "mean"),
            corr=("static_decode_speed_corr", "mean"),
            capacity_cv=("worker_capacity_cv", "mean"),
        )
        .reset_index()
        .to_markdown(index=False)
    )
    lines.extend(["", "## P95 Improvements", ""])
    lines.append(
        sub.pivot_table(
            index=["dataset", "alignment_mode"],
            columns="strategy",
            values="p95_improvement",
            aggfunc="mean",
        )
        .rename(columns=LABELS)
        .mul(100.0)
        .round(1)
        .reset_index()
        .to_markdown(index=False)
    )
    r = pearson(sub["decode_speed_mismatch"], sub["p95_improvement"])
    lines.extend(["", f"Correlation between mismatch and adaptive p95 improvement: r={r:+.3f}."])
    path.write_text("\n".join(lines), encoding="utf-8")


def pearson(a: pd.Series, b: pd.Series) -> float:
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    if len(a) < 2 or a.std(ddof=0) <= 1e-12 or b.std(ddof=0) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    paper_fig = Path("paper") / "socc26" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_fig / stem.with_suffix(".png").name, dpi=220)
    fig.savefig(paper_fig / stem.with_suffix(".pdf").name)


if __name__ == "__main__":
    main()
