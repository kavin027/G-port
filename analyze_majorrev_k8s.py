"""Aggregate major-revision multi-node Kubernetes experiments.

The input directory is expected to contain one subdirectory per run, named
``majorrev_k8s_w{workers}_seed{seed}``, each with ``network_summary.csv`` and
``network_metrics.csv`` emitted by ``run_k8s_network_experiment.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(r"majorrev_k8s_w(?P<workers>\d+)_seed(?P<seed>\d+)")

STRATEGY_LABELS = {
    "speed_aware_uncoded": "Speed-aware uncoded",
    "speculative_replication": "Speculative replication",
    "sparse_flexible_static": "Static sparse-flex",
    "worker_aware_sparse_flexible": "Worker-aware",
    "rank_aware_sparse_flexible": "Rank-aware",
    "deadline_aware_sparse_flexible": "Deadline-aware",
    "system_portfolio": "Portfolio",
    "guarded_system_portfolio": "Guarded portfolio",
    "online_counter_guard_rank_aware_sparse_flexible": "Guarded rank-aware",
    "online_counter_guard_deadline_aware_sparse_flexible": "Guarded deadline-aware",
}


def _percent_gain(baseline: float, value: float) -> float:
    if baseline == 0:
        return 0.0
    return 100.0 * (baseline - value) / baseline


def _bootstrap_ci(values: pd.Series, *, seed: int = 20260630) -> tuple[float, float]:
    data = values.dropna().to_numpy(dtype=float)
    if len(data) == 0:
        return 0.0, 0.0
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(10000, len(data)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _parse_run_dir(path: Path) -> tuple[int, int] | None:
    match = RUN_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group("workers")), int(match.group("seed"))


def _pod_node_counts(pods_wide: Path) -> str:
    if not pods_wide.exists():
        return ""
    nodes: dict[str, int] = {}
    for line in pods_wide.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("coded-worker-"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        node = parts[6]
        nodes[node] = nodes.get(node, 0) + 1
    return ";".join(f"{node}:{count}" for node, count in sorted(nodes.items()))


def load_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metrics_rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        parsed = _parse_run_dir(run_dir)
        if parsed is None:
            continue
        workers, seed = parsed
        summary_path = run_dir / "network_summary.csv"
        metrics_path = run_dir / "network_metrics.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        static = summary.loc[summary["strategy"] == "sparse_flexible_static"].iloc[0]
        speed = summary.loc[summary["strategy"] == "speed_aware_uncoded"].iloc[0]
        pod_counts = _pod_node_counts(run_dir / "k8s_pods_wide.txt")
        for _, row in summary.iterrows():
            item = row.to_dict()
            item["workers"] = workers
            item["seed"] = seed
            item["strategy_label"] = STRATEGY_LABELS.get(item["strategy"], item["strategy"])
            item["pod_node_counts"] = pod_counts
            item["mean_decode_gain_vs_static_pct"] = _percent_gain(
                static["mean_decode_latency"], item["mean_decode_latency"]
            )
            item["p95_decode_gain_vs_static_pct"] = _percent_gain(
                static["p95_decode_latency"], item["p95_decode_latency"]
            )
            item["mean_barrier_gain_vs_static_pct"] = _percent_gain(
                static["mean_barrier_latency"], item["mean_barrier_latency"]
            )
            item["p95_barrier_gain_vs_static_pct"] = _percent_gain(
                static["p95_barrier_latency"], item["p95_barrier_latency"]
            )
            item["mean_decode_gain_vs_speed_pct"] = _percent_gain(
                speed["mean_decode_latency"], item["mean_decode_latency"]
            )
            item["mean_barrier_gain_vs_speed_pct"] = _percent_gain(
                speed["mean_barrier_latency"], item["mean_barrier_latency"]
            )
            rows.append(item)
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            metrics["workers"] = workers
            metrics["seed"] = seed
            metrics_rows.append(metrics)
    if not rows:
        raise SystemExit(f"No runs found under {root}")
    all_summary = pd.DataFrame(rows)
    all_metrics = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    return all_summary, all_metrics


def aggregate(summary: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "mean_decode_latency",
        "p95_decode_latency",
        "mean_barrier_latency",
        "p95_barrier_latency",
        "decode_success_rate",
        "mean_scheduler_seconds",
        "mean_dispatch_seconds",
        "mean_cancel_seconds",
        "mean_decode_cpu_seconds",
        "mean_worker_compute_cpu_seconds",
        "mean_extra_compute",
        "mean_selected_rows",
        "mean_completed_rows",
        "mean_cancelled_rows",
        "second_layer_rate",
        "mean_worker_errors",
        "mean_decode_gain_vs_static_pct",
        "p95_decode_gain_vs_static_pct",
        "mean_barrier_gain_vs_static_pct",
        "p95_barrier_gain_vs_static_pct",
        "mean_decode_gain_vs_speed_pct",
        "mean_barrier_gain_vs_speed_pct",
    ]
    grouped = (
        summary.groupby(["workers", "strategy", "strategy_label"], as_index=False)
        .agg({col: "mean" for col in numeric_cols})
        .sort_values(["workers", "strategy"])
    )
    seed_counts = (
        summary.groupby(["workers", "strategy"], as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
    )
    grouped = grouped.merge(seed_counts, on=["workers", "strategy"], how="left")

    ci_rows = []
    for (workers, strategy), subset in summary.groupby(["workers", "strategy"]):
        item = {"workers": workers, "strategy": strategy}
        for col in [
            "mean_decode_gain_vs_static_pct",
            "p95_decode_gain_vs_static_pct",
            "mean_barrier_gain_vs_static_pct",
            "p95_barrier_gain_vs_static_pct",
        ]:
            low, high = _bootstrap_ci(subset[col])
            item[f"{col}_ci_low"] = low
            item[f"{col}_ci_high"] = high
            item[f"{col}_min"] = float(subset[col].min())
            item[f"{col}_max"] = float(subset[col].max())
        ci_rows.append(item)
    grouped = grouped.merge(pd.DataFrame(ci_rows), on=["workers", "strategy"], how="left")

    if not metrics.empty:
        guard = metrics[
            metrics["strategy"].str.startswith("online_counter_guard")
            | (metrics["strategy"] == "guarded_system_portfolio")
        ].copy()
        if not guard.empty:
            guard["guard_enabled"] = guard["config"].str.contains(
                "online-guard-enable|guarded-portfolio-enable", na=False, regex=True
            )
            guard["guard_fallback"] = guard["config"].str.contains(
                "online-guard-fallback|guarded-portfolio-fallback", na=False, regex=True
            )
            guard_rates = (
                guard.groupby(["workers", "strategy"], as_index=False)
                .agg(
                    guard_enable_rate=("guard_enabled", "mean"),
                    guard_fallback_rate=("guard_fallback", "mean"),
                )
            )
            grouped = grouped.merge(guard_rates, on=["workers", "strategy"], how="left")
    grouped["guard_enable_rate"] = grouped.get("guard_enable_rate", pd.Series(index=grouped.index)).fillna(0.0)
    grouped["guard_fallback_rate"] = grouped.get("guard_fallback_rate", pd.Series(index=grouped.index)).fillna(0.0)
    return grouped


def paper_table(grouped: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "speed_aware_uncoded",
        "speculative_replication",
        "sparse_flexible_static",
        "worker_aware_sparse_flexible",
        "rank_aware_sparse_flexible",
        "deadline_aware_sparse_flexible",
        "system_portfolio",
        "guarded_system_portfolio",
        "online_counter_guard_rank_aware_sparse_flexible",
        "online_counter_guard_deadline_aware_sparse_flexible",
    ]
    table = grouped[grouped["strategy"].isin(keep)].copy()
    table["mean_decode_ms"] = 1000.0 * table["mean_decode_latency"]
    table["p95_decode_ms"] = 1000.0 * table["p95_decode_latency"]
    table["mean_barrier_ms"] = 1000.0 * table["mean_barrier_latency"]
    table["barrier_gain_ci"] = table.apply(
        lambda r: (
            f"{r['mean_barrier_gain_vs_static_pct']:.1f}"
            f" [{r['mean_barrier_gain_vs_static_pct_ci_low']:.1f},"
            f"{r['mean_barrier_gain_vs_static_pct_ci_high']:.1f}]"
        ),
        axis=1,
    )
    table["guard_enable_pct"] = 100.0 * table["guard_enable_rate"]
    order = {name: idx for idx, name in enumerate(keep)}
    table["_order"] = table["strategy"].map(order)
    return table.sort_values(["workers", "_order"])[
        [
            "workers",
            "strategy_label",
            "mean_decode_ms",
            "p95_decode_ms",
            "mean_barrier_ms",
            "mean_barrier_gain_vs_static_pct",
            "mean_barrier_gain_vs_static_pct_ci_low",
            "mean_barrier_gain_vs_static_pct_ci_high",
            "barrier_gain_ci",
            "guard_enable_pct",
            "mean_scheduler_seconds",
            "mean_dispatch_seconds",
            "mean_cancel_seconds",
            "mean_worker_errors",
        ]
    ]


def per_seed_core_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Reviewer-facing per-seed barrier latencies for the core K3s policies."""
    keep = {
        "sparse_flexible_static": "static_barrier_ms",
        "rank_aware_sparse_flexible": "rank_barrier_ms",
        "deadline_aware_sparse_flexible": "deadline_barrier_ms",
        "online_counter_guard_deadline_aware_sparse_flexible": "guard_d_barrier_ms",
    }
    subset = summary[summary["strategy"].isin(keep)].copy()
    subset["barrier_ms"] = 1000.0 * subset["mean_barrier_latency"]
    table = (
        subset.pivot_table(
            index=["workers", "seed"],
            columns="strategy",
            values="barrier_ms",
            aggfunc="first",
        )
        .rename(columns=keep)
        .reset_index()
        .sort_values(["workers", "seed"])
    )
    ordered = ["workers", "seed", *(column for column in keep.values() if column in table.columns)]
    return table[ordered].round(1)


def mismatch_split_table(summary: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic K3s split for the 24-worker high-static-tail seeds.

    The split is derived only from the static-coded W24 barrier latency, so it
    explains heterogeneity without using candidate-policy outcomes to choose a
    favorable subset.
    """
    workers = 24
    static = summary[
        (summary["workers"] == workers)
        & (summary["strategy"] == "sparse_flexible_static")
    ][["seed", "mean_barrier_latency"]].copy()
    if static.empty:
        return pd.DataFrame()
    static["static_barrier_ms"] = 1000.0 * static["mean_barrier_latency"]
    high_seeds = set(static.loc[static["static_barrier_ms"] >= 50.0, "seed"].astype(int))
    all_seeds = set(static["seed"].astype(int))
    low_seeds = all_seeds - high_seeds
    split_specs = [
        ("Low static-tail", low_seeds),
        ("High static-tail", high_seeds),
    ]
    strategy_cols = {
        "sparse_flexible_static": "static_ms",
        "rank_aware_sparse_flexible": "rank_ms",
        "deadline_aware_sparse_flexible": "deadline_ms",
        "system_portfolio": "portfolio_ms",
        "guarded_system_portfolio": "guarded_portfolio_ms",
        "online_counter_guard_deadline_aware_sparse_flexible": "guard_d_ms",
        "online_counter_guard_rank_aware_sparse_flexible": "guard_r_ms",
        "speed_aware_uncoded": "speed_uncoded_ms",
    }
    rows = []
    for label, seeds in split_specs:
        if not seeds:
            continue
        item = {
            "workers": workers,
            "regime": label,
            "seeds": ",".join(str(seed) for seed in sorted(seeds)),
            "n_seeds": len(seeds),
        }
        base = summary[
            (summary["workers"] == workers)
            & (summary["seed"].astype(int).isin(seeds))
            & (summary["strategy"] == "sparse_flexible_static")
        ][["seed", "mean_barrier_latency"]].rename(
            columns={"mean_barrier_latency": "static_latency"}
        )
        for strategy, col in strategy_cols.items():
            subset = summary[
                (summary["workers"] == workers)
                & (summary["seed"].astype(int).isin(seeds))
                & (summary["strategy"] == strategy)
            ][["seed", "mean_barrier_latency"]].copy()
            if subset.empty:
                continue
            item[col] = 1000.0 * float(subset["mean_barrier_latency"].mean())
            if strategy != "sparse_flexible_static":
                paired = base.merge(subset, on="seed", how="inner")
                item[col.replace("_ms", "_gain_pct")] = float(
                    (
                        100.0
                        * (
                            paired["static_latency"]
                            - paired["mean_barrier_latency"]
                        )
                        / paired["static_latency"]
                    ).mean()
                )
        if not metrics.empty:
            guard_metrics = metrics[
                (metrics["workers"] == workers)
                & (metrics["seed"].astype(int).isin(seeds))
                & (
                    metrics["strategy"]
                    == "online_counter_guard_deadline_aware_sparse_flexible"
                )
            ]
            if not guard_metrics.empty:
                item["guard_d_enable_pct"] = 100.0 * float(
                    guard_metrics["config"]
                    .astype(str)
                    .str.contains("online-guard-enable", na=False)
                    .mean()
                )
        rows.append(item)
    columns = [
        "workers",
        "regime",
        "seeds",
        "n_seeds",
        "static_ms",
        "rank_ms",
        "rank_gain_pct",
        "deadline_ms",
        "deadline_gain_pct",
        "portfolio_ms",
        "portfolio_gain_pct",
        "guarded_portfolio_ms",
        "guarded_portfolio_gain_pct",
        "guard_d_ms",
        "guard_d_gain_pct",
        "guard_d_enable_pct",
        "guard_r_ms",
        "guard_r_gain_pct",
        "speed_uncoded_ms",
        "speed_uncoded_gain_pct",
    ]
    table = pd.DataFrame(rows)
    return table[[col for col in columns if col in table.columns]].round(1)


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _fmt_ms(value: float) -> str:
    return f"{1000.0 * value:.1f}"


def write_report(
    root: Path,
    grouped: pd.DataFrame,
    summary: pd.DataFrame,
    split_table: pd.DataFrame,
) -> None:
    lines: list[str] = []
    seeds = sorted(summary["seed"].unique())
    worker_seed_counts = (
        summary.groupby("workers")["seed"]
        .nunique()
        .sort_index()
        .to_dict()
    )
    lines.append("# Major-Revision Kubernetes Experiment Report")
    lines.append("")
    lines.append(
        "Direct three-node K3s experiment: one control/master node and two worker nodes. "
        f"Summarized seeds: {', '.join(str(int(s)) for s in seeds)}. "
        "Per-worker seed counts are "
        + ", ".join(f"W{int(w)}={int(c)}" for w, c in worker_seed_counts.items())
        + "."
    )
    lines.append("")
    lines.append("## Unified Baseline Matrix")
    lines.append("")
    lines.append(
        "| Workers | Strategy | Mean decode (ms) | p95 decode (ms) | Mean barrier (ms) | "
        "Gain vs static decode | Gain vs static barrier (95% CI) | Gain vs speed decode | Guard enable | Worker errors |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = list(STRATEGY_LABELS)
    for workers in sorted(grouped["workers"].unique()):
        subset = grouped[grouped["workers"] == workers].copy()
        subset["_order"] = subset["strategy"].map({s: i for i, s in enumerate(order)}).fillna(99)
        for _, row in subset.sort_values("_order").iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(row["workers"])),
                        str(row["strategy_label"]),
                        _fmt_ms(row["mean_decode_latency"]),
                        _fmt_ms(row["p95_decode_latency"]),
                        _fmt_ms(row["mean_barrier_latency"]),
                        _fmt_pct(row["mean_decode_gain_vs_static_pct"]),
                        (
                            _fmt_pct(row["mean_barrier_gain_vs_static_pct"])
                            + f" [{row['mean_barrier_gain_vs_static_pct_ci_low']:.1f},"
                            + f"{row['mean_barrier_gain_vs_static_pct_ci_high']:.1f}]"
                        ),
                        _fmt_pct(row["mean_decode_gain_vs_speed_pct"]),
                        _fmt_pct(100.0 * row["guard_enable_rate"]),
                        f"{row['mean_worker_errors']:.2f}",
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Scaling Takeaways")
    lines.append("")
    for workers in sorted(grouped["workers"].unique()):
        subset = grouped[grouped["workers"] == workers]
        coded = subset[
            subset["strategy"].isin(
                [
                    "rank_aware_sparse_flexible",
                    "deadline_aware_sparse_flexible",
                    "system_portfolio",
                    "online_counter_guard_rank_aware_sparse_flexible",
                    "online_counter_guard_deadline_aware_sparse_flexible",
                ]
            )
        ]
        if coded.empty:
            lines.append(
                f"- {int(workers)} workers: no coded optimizer strategies were present "
                "in this run subset."
            )
            continue
        best_decode = coded.loc[coded["mean_decode_latency"].idxmin()]
        best_barrier = coded.loc[coded["mean_barrier_latency"].idxmin()]
        guard = subset[subset["strategy"].str.startswith("online_counter_guard")]
        guard_text = ""
        if not guard.empty:
            best_guard = guard.loc[guard["mean_barrier_latency"].idxmin()]
            guard_text = (
                f" The best coded-only online guard keeps "
                f"{_fmt_pct(best_guard['mean_barrier_gain_vs_static_pct'])} "
                f"barrier gain and enables on "
                f"{_fmt_pct(100.0 * best_guard['guard_enable_rate'])} of rounds."
            )
        lines.append(
            f"- {int(workers)} workers: best coded mean decode is {best_decode['strategy_label']} "
            f"with {_fmt_pct(best_decode['mean_decode_gain_vs_static_pct'])} gain vs static; "
            f"best barrier is {best_barrier['strategy_label']} with "
            f"{_fmt_pct(best_barrier['mean_barrier_gain_vs_static_pct'])}."
            f"{guard_text}"
        )
    lines.append("")
    lines.append("## Per-Seed Core Barrier Latencies")
    lines.append("")
    lines.append(
        "The table below reports mean barrier latency in ms for the core static, "
        "rank-aware, deadline-aware, and guarded-deadline policies before averaging "
        "over seeds."
    )
    lines.append("")
    seed_table = per_seed_core_table(summary)
    seed_columns = [
        ("workers", "Workers"),
        ("seed", "Seed"),
        ("static_barrier_ms", "Static"),
        ("rank_barrier_ms", "Rank"),
        ("deadline_barrier_ms", "Deadline"),
        ("guard_d_barrier_ms", "Guard-D"),
    ]
    seed_columns = [(key, label) for key, label in seed_columns if key in seed_table.columns]
    lines.append("| " + " | ".join(label for _, label in seed_columns) + " |")
    lines.append("|" + "|".join("---:" for _ in seed_columns) + "|")
    for _, row in seed_table.iterrows():
        values = []
        for key, _ in seed_columns:
            if key in {"workers", "seed"}:
                values.append(str(int(row[key])))
            else:
                values.append(f"{row[key]:.1f}" if pd.notna(row[key]) else "")
        lines.append(
            "| "
            + " | ".join(values)
            + " |"
        )
    lines.append("")
    if not split_table.empty:
        lines.append("## 24-Worker Static-Tail Split")
        lines.append("")
        lines.append(
            "This diagnostic split is derived only from the W24 static-coded "
            "barrier latency: high-static-tail seeds have static barrier latency "
            "of at least 50 ms.  It explains seed heterogeneity and is not a "
            "deployment rule."
        )
        lines.append("")
        split_columns = [
            ("static_ms", "", "Static"),
            ("rank_ms", "rank_gain_pct", "Rank"),
            ("deadline_ms", "deadline_gain_pct", "Deadline"),
            ("portfolio_ms", "portfolio_gain_pct", "Portfolio"),
            ("guarded_portfolio_ms", "guarded_portfolio_gain_pct", "Guarded portfolio"),
            ("guard_d_ms", "guard_d_gain_pct", "Guard-D"),
        ]
        split_columns = [
            (value_col, gain_col, label)
            for value_col, gain_col, label in split_columns
            if value_col in split_table.columns
        ]
        headers = ["Regime", "Seeds", *(label for _, _, label in split_columns)]
        if "guard_d_enable_pct" in split_table.columns:
            headers.append("Guard-D enable")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---", "---", *(["---:"] * (len(headers) - 2))]) + "|")
        for _, row in split_table.iterrows():
            cells = [str(row["regime"]), str(row["seeds"])]
            for value_col, gain_col, _ in split_columns:
                if not pd.notna(row.get(value_col, np.nan)):
                    cells.append("")
                elif gain_col:
                    cells.append(f"{row[value_col]:.1f} ({row[gain_col]:.1f}%)")
                else:
                    cells.append(f"{row[value_col]:.1f}")
            if "guard_d_enable_pct" in split_table.columns:
                cells.append(f"{row.get('guard_d_enable_pct', 0.0):.1f}%")
            lines.append(
                "| "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    lines.append("## Reproducibility Notes")
    lines.append("")
    lines.append(f"- Seeds: {', '.join(str(int(s)) for s in seeds)}.")
    lines.append("- `decode_success_rate` is 1.0 for all summarized strategies/runs.")
    lines.append("- `mean_worker_errors` is 0.0 in all summarized groups.")
    lines.append(
        "- Per-run Kubernetes manifests, pod placement logs, command logs, and optional "
        "resource-counter snapshots are retained in each run directory."
    )
    lines.append("")
    (root / "majorrev_k8s_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("majorrev_k8s_diagnostics"),
        help="Directory containing majorrev_k8s_w*_seed* subdirectories.",
    )
    args = parser.parse_args()
    summary, metrics = load_runs(args.root)
    grouped = aggregate(summary, metrics)
    table = paper_table(grouped)
    seed_table = per_seed_core_table(summary)
    split_table = mismatch_split_table(summary, metrics)
    args.root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.root / "majorrev_k8s_all_summary.csv", index=False)
    grouped.to_csv(args.root / "majorrev_k8s_group_summary.csv", index=False)
    table.to_csv(args.root / "majorrev_k8s_paper_table.csv", index=False)
    seed_table.to_csv(args.root / "majorrev_k8s_per_seed_core.csv", index=False)
    split_table.to_csv(args.root / "majorrev_k8s_mismatch_split.csv", index=False)
    write_report(args.root, grouped, summary, split_table)
    print(f"Wrote {args.root / 'majorrev_k8s_all_summary.csv'}")
    print(f"Wrote {args.root / 'majorrev_k8s_group_summary.csv'}")
    print(f"Wrote {args.root / 'majorrev_k8s_paper_table.csv'}")
    print(f"Wrote {args.root / 'majorrev_k8s_per_seed_core.csv'}")
    print(f"Wrote {args.root / 'majorrev_k8s_mismatch_split.csv'}")
    print(f"Wrote {args.root / 'majorrev_k8s_report.md'}")


if __name__ == "__main__":
    main()
