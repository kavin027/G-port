from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ADAPTIVE_STRATEGIES = [
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
    "w8a": "#4c72b0",
    "rcv1": "#55a868",
}

MARKERS = {
    "worker_aware_sparse_flexible": "o",
    "rank_aware_sparse_flexible": "s",
    "deadline_aware_sparse_flexible": "^",
}


def main() -> None:
    input_dir = Path("runtime_realdata_sensitivity_bjb1")
    out_dir = Path("realdata_sensitivity_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_csv(input_dir / "aggregate_sensitivity_summary.csv")
    combined = pd.read_csv(input_dir / "combined_sensitivity_summary.csv")

    mechanism = add_static_deltas(aggregate)
    mechanism.to_csv(out_dir / "sensitivity_mechanism_summary.csv", index=False)
    write_report(out_dir / "sensitivity_report.md", mechanism)
    plot_gain_vs_mismatch(mechanism, out_dir)
    plot_slowdown_sensitivity(mechanism, out_dir)
    plot_row_delta_vs_gain(mechanism, out_dir)

    combined.to_csv(out_dir / "combined_sensitivity_summary.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_sensitivity_summary.csv", index=False)


def add_static_deltas(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "straggler_fraction", "straggler_slowdown"]
    for keys, group in aggregate.groupby(group_cols, sort=False):
        static = group[group["strategy"] == "sparse_flexible_static"].iloc[0]
        for _, row in group.iterrows():
            record = row.to_dict()
            record["selected_rows_delta"] = row["selected_rows"] - static["selected_rows"]
            record["completed_rows_delta"] = row["completed_rows"] - static["completed_rows"]
            record["extra_compute_delta"] = row["extra_compute"] - static["extra_compute"]
            record["scheduler_ms_delta"] = 1000.0 * (
                row["scheduler_seconds"] - static["scheduler_seconds"]
            )
            rows.append(record)
    return pd.DataFrame(rows)


def plot_gain_vs_mismatch(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["strategy"].isin(ADAPTIVE_STRATEGIES)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "decode_improvement", "Mean improvement vs. mismatch"),
        (axes[1], "p95_improvement", "P95 improvement vs. mismatch"),
    ]:
        for strategy in ADAPTIVE_STRATEGIES:
            for dataset in sorted(sub["dataset"].unique()):
                part = sub[(sub["strategy"] == strategy) & (sub["dataset"] == dataset)]
                ax.scatter(
                    part["decode_speed_mismatch"],
                    100.0 * part[metric],
                    label=f"{dataset}-{DISPLAY[strategy]}",
                    color=COLORS.get(dataset, "#777777"),
                    marker=MARKERS[strategy],
                    alpha=0.78,
                    s=42,
                    edgecolor="white",
                    linewidth=0.5,
                )
        r = pearson(sub["decode_speed_mismatch"], sub[metric])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Decode-speed mismatch")
        ax.set_ylabel("Improvement over static (%)")
        ax.set_title(f"{title} (r={r:+.2f})")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:6], labels[:6], fontsize=7, frameon=False, ncol=2)
    save(fig, out_dir / "gain_vs_decode_speed_mismatch")


def plot_slowdown_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["strategy"].isin(ADAPTIVE_STRATEGIES)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "decode_improvement", "Mean improvement"),
        (axes[1], "p95_improvement", "P95 improvement"),
    ]:
        for dataset in sorted(sub["dataset"].unique()):
            for strategy in ["rank_aware_sparse_flexible", "deadline_aware_sparse_flexible"]:
                part = (
                    sub[(sub["dataset"] == dataset) & (sub["strategy"] == strategy)]
                    .groupby("straggler_slowdown", as_index=False)
                    .agg(value=(metric, "mean"), mismatch=("decode_speed_mismatch", "mean"))
                    .sort_values("straggler_slowdown")
                )
                label = f"{dataset}-{DISPLAY[strategy]}"
                ax.plot(
                    part["straggler_slowdown"],
                    100.0 * part["value"],
                    marker=MARKERS[strategy],
                    color=COLORS.get(dataset, "#777777"),
                    linestyle="-" if strategy == "rank_aware_sparse_flexible" else "--",
                    label=label,
                )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Straggler slowdown multiplier (larger = less severe)")
        ax.set_ylabel("Improvement over static (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    save(fig, out_dir / "slowdown_sensitivity")


def plot_row_delta_vs_gain(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["strategy"].isin(ADAPTIVE_STRATEGIES)].copy()
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    for strategy in ADAPTIVE_STRATEGIES:
        for dataset in sorted(sub["dataset"].unique()):
            part = sub[(sub["strategy"] == strategy) & (sub["dataset"] == dataset)]
            ax.scatter(
                part["completed_rows_delta"],
                100.0 * part["p95_improvement"],
                label=f"{dataset}-{DISPLAY[strategy]}",
                color=COLORS.get(dataset, "#777777"),
                marker=MARKERS[strategy],
                alpha=0.78,
                s=44,
                edgecolor="white",
                linewidth=0.5,
            )
    r = pearson(sub["completed_rows_delta"], sub["p95_improvement"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Completed rows delta vs. static")
    ax.set_ylabel("P95 improvement over static (%)")
    ax.set_title(f"Tail gain tracks rows avoided (r={r:+.2f})")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    save(fig, out_dir / "p95_gain_vs_completed_rows_delta")


def write_report(path: Path, df: pd.DataFrame) -> None:
    sub = df[df["strategy"].isin(ADAPTIVE_STRATEGIES)].copy()
    lines = [
        "# Real-Data Sensitivity Diagnostics",
        "",
        "The sweep varies straggler fraction and slowdown on w8a and rcv1.",
        "Decode-speed mismatch is the normalized assignment regret of static "
        "task placement relative to pairing high decode-priority rows with "
        "high-capacity workers.",
        "",
        "## Correlations",
        "",
    ]
    for metric, label in [
        ("decode_improvement", "mean improvement"),
        ("p95_improvement", "p95 improvement"),
        ("completed_rows_delta", "completed-row delta"),
    ]:
        if metric == "completed_rows_delta":
            r = pearson(sub["completed_rows_delta"], sub["p95_improvement"])
            lines.append(f"- completed-row delta vs p95 improvement: r={r:+.3f}")
        else:
            r = pearson(sub["decode_speed_mismatch"], sub[metric])
            lines.append(f"- decode-speed mismatch vs {label}: r={r:+.3f}")
    lines.extend(["", "## Best/Worst Parameter Points", ""])
    for dataset in sorted(sub["dataset"].unique()):
        lines.append(f"### {dataset}")
        part = sub[sub["dataset"] == dataset]
        for strategy in ADAPTIVE_STRATEGIES:
            s = part[part["strategy"] == strategy]
            best = s.sort_values("p95_improvement", ascending=False).iloc[0]
            worst = s.sort_values("p95_improvement", ascending=True).iloc[0]
            lines.append(
                "- "
                f"{DISPLAY[strategy]} best p95 {100*best['p95_improvement']:+.1f}% "
                f"at f={best['straggler_fraction']}, s={best['straggler_slowdown']}, "
                f"mismatch={best['decode_speed_mismatch']:.3f}; "
                f"worst {100*worst['p95_improvement']:+.1f}% "
                f"at f={worst['straggler_fraction']}, s={worst['straggler_slowdown']}, "
                f"mismatch={worst['decode_speed_mismatch']:.3f}"
            )
        lines.append("")
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
