"""Aggregate K3s interference/failure stress cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_majorrev_k8s import STRATEGY_LABELS, load_runs


KEEP = [
    "speed_aware_uncoded",
    "sparse_flexible_static",
    "rank_aware_sparse_flexible",
    "system_portfolio",
    "guarded_system_portfolio",
    "online_counter_guard_deadline_aware_sparse_flexible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/coded_k8s_stress_results"))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)
    details = collect_cases(args.root)
    if details.empty:
        raise SystemExit(f"No stress case summaries found under {args.root}")
    summary = summarize(details)
    details.to_csv(out / "k8s_stress_details.csv", index=False)
    summary.to_csv(out / "k8s_stress_summary.csv", index=False)
    write_latex_table(summary, out / "k8s_stress_table.tex")
    write_report(summary, out / "k8s_stress_report.md")
    print(summary.to_string(index=False))
    print(f"Wrote K3s stress diagnostics to {out}")


def collect_cases(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            summary, _ = load_runs(case_dir)
        except SystemExit:
            continue
        summary = summary[summary["strategy"].isin(KEEP)].copy()
        if summary.empty:
            continue
        summary["case"] = case_dir.name
        static = summary[summary["strategy"] == "sparse_flexible_static"][
            ["case", "workers", "seed", "mean_barrier_latency"]
        ].rename(columns={"mean_barrier_latency": "static_barrier_latency"})
        summary = summary.merge(static, on=["case", "workers", "seed"], how="left")
        summary["barrier_gain_vs_static_pct"] = 100.0 * (
            summary["static_barrier_latency"] - summary["mean_barrier_latency"]
        ) / summary["static_barrier_latency"].clip(lower=1e-12)
        frames.append(summary)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    for column in ("mean_worker_recoveries", "mean_reissued_rows"):
        if column not in details.columns:
            details[column] = 0.0
    grouped = (
        details.groupby(["case", "workers", "strategy"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_barrier_ms=("mean_barrier_latency", lambda x: 1000.0 * float(x.mean())),
            p95_barrier_ms=("p95_barrier_latency", lambda x: 1000.0 * float(x.mean())),
            barrier_gain_vs_static_pct=("barrier_gain_vs_static_pct", "mean"),
            decode_success_rate=("decode_success_rate", "mean"),
            worker_errors=("mean_worker_errors", "mean"),
            recoveries=("mean_worker_recoveries", "mean"),
            reissued_rows=("mean_reissued_rows", "mean"),
            dispatch_ms=("mean_dispatch_seconds", lambda x: 1000.0 * float(x.mean())),
            cancel_ms=("mean_cancel_seconds", lambda x: 1000.0 * float(x.mean())),
        )
        .sort_values(["case", "workers", "strategy"])
    )
    grouped["strategy_label"] = grouped["strategy"].map(STRATEGY_LABELS).fillna(grouped["strategy"])
    return grouped


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows: list[str] = []
    preferred = [
        "sparse_flexible_static",
        "rank_aware_sparse_flexible",
        "guarded_system_portfolio",
        "online_counter_guard_deadline_aware_sparse_flexible",
    ]
    subset = summary[summary["strategy"].isin(preferred)].copy()
    order = {strategy: idx for idx, strategy in enumerate(preferred)}
    subset["_order"] = subset["strategy"].map(order)
    for row in subset.sort_values(["case", "workers", "_order"]).itertuples(index=False):
        rows.append(
            "    "
            + " & ".join(
                [
                    str(row.case).replace("_", r"\_"),
                    str(int(row.workers)),
                    str(row.strategy_label),
                    f"{row.mean_barrier_ms:.1f}",
                    f"{row.barrier_gain_vs_static_pct:.1f}\\%",
                    f"{row.worker_errors:.2f}",
                    f"{row.recoveries:.2f}",
                    f"{row.cancel_ms:.1f}",
                ]
            )
            + r" \\"
        )
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{K3s worker-service stress diagnostics. CPU-hog injects best-effort co-located interference; cancel-ack delays worker cancellation acknowledgments. Gains are relative to static sparse-flexible placement within the same case and worker count.}",
        r"  \label{tab:k8s-stress}",
        r"  \begin{tabular}{@{}llrrrrrr@{}}",
        r"    \toprule",
        r"    Case & W & Policy & Barrier & Gain & Worker err. & Recov. & Cancel \\",
        r"         &   &        & (ms) &      & (/round) & (/round) & (ms) \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: pd.DataFrame, path: Path) -> None:
    try:
        table = summary.to_markdown(index=False)
    except ImportError:
        table = "```csv\n" + summary.to_csv(index=False).strip() + "\n```"
    path.write_text(
        "\n".join(
            [
                "# K3s Stress Diagnostic",
                "",
                "This table is a diagnostic of the worker-service path, not a production recovery claim.",
                "",
                table,
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
