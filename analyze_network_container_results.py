from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TCP/container-style runtime results.")
    parser.add_argument("paths", nargs="+", type=Path, help="Result directories containing network_summary.csv.")
    parser.add_argument("--out", type=Path, default=Path("network_container_diagnostics"))
    parser.add_argument(
        "--baseline-strategy",
        default=None,
        help="Optional strategy name for paired per-run improvement diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in args.paths:
        summary_path = path / "network_summary.csv"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        frame["run"] = path.name
        frames.append(frame)
    if not frames:
        raise SystemExit("No network_summary.csv files found.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.out / "combined_network_summary.csv", index=False)

    numeric_cols = [
        "mean_decode_latency",
        "p95_decode_latency",
        "mean_barrier_latency",
        "p95_barrier_latency",
        "decode_latency_improvement_vs_sparse_flexible",
        "p95_decode_latency_improvement_vs_sparse_flexible",
        "barrier_latency_improvement_vs_sparse_flexible",
        "mean_scheduler_seconds",
        "mean_dispatch_seconds",
        "mean_cancel_seconds",
        "mean_network_response_mb",
        "mean_network_response_sleep_seconds",
        "mean_decode_cpu_seconds",
        "mean_worker_compute_cpu_seconds",
        "mean_selected_rows",
        "mean_completed_rows",
        "mean_cancelled_rows",
        "mean_worker_errors",
    ]
    available = [col for col in numeric_cols if col in combined.columns]
    aggregate = combined.groupby("strategy", sort=False)[available].agg(["mean", "std"])
    aggregate.columns = ["_".join(col).rstrip("_") for col in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(args.out / "aggregate_network_summary.csv", index=False)

    baseline_report: list[str] = []
    if args.baseline_strategy is not None:
        baseline_report = _write_baseline_diagnostics(combined, args.out, args.baseline_strategy)

    report = [
        "# TCP/container-style runtime summary",
        "",
        "This experiment runs each logical worker as an isolated TCP service. "
        "On Docker-enabled hosts, the same worker entrypoint can be launched as "
        "one container per worker from `docker/Dockerfile.network-worker`; otherwise "
        "the runtime can fall back to one independent Python TCP service per worker.",
        "",
        "## Aggregate results",
        "",
        "```",
        aggregate.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        *baseline_report,
    ]
    (args.out / "network_report.md").write_text("\n".join(report), encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(f"\nWrote diagnostics to {args.out}")


def _write_baseline_diagnostics(combined: pd.DataFrame, out_dir: Path, baseline_strategy: str) -> list[str]:
    metrics = [
        "mean_decode_latency",
        "p95_decode_latency",
        "mean_barrier_latency",
        "p95_barrier_latency",
    ]
    rows = []
    for run, group in combined.groupby("run", sort=False):
        baseline_rows = group[group["strategy"] == baseline_strategy]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for _, row in group.iterrows():
            record = {"run": run, "strategy": row["strategy"]}
            for metric in metrics:
                record[f"{metric}_ms"] = 1000.0 * float(row[metric])
                record[f"{metric}_improvement_vs_{baseline_strategy}"] = (
                    float(baseline[metric]) - float(row[metric])
                ) / max(float(baseline[metric]), 1e-12)
            for column in [
                "mean_network_response_mb",
                "mean_network_response_sleep_seconds",
                "mean_selected_rows",
                "mean_completed_rows",
                "mean_cancelled_rows",
                "mean_scheduler_seconds",
                "mean_worker_errors",
            ]:
                if column in row:
                    record[column] = row[column]
            rows.append(record)
    if not rows:
        return [
            "## Baseline diagnostics",
            "",
            f"Baseline `{baseline_strategy}` was not present in any run.",
            "",
        ]

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / f"comparison_vs_{baseline_strategy}.csv", index=False)

    aggregate_cols = [column for column in comparison.columns if column not in {"run", "strategy"}]
    aggregate = comparison.groupby("strategy", sort=False)[aggregate_cols].agg(["mean", "std"])
    aggregate.columns = ["_".join(column).rstrip("_") for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(out_dir / f"aggregate_vs_{baseline_strategy}.csv", index=False)

    rng = np.random.default_rng(20260525)
    ci_rows = []
    for strategy, group in comparison.groupby("strategy", sort=False):
        for metric in metrics:
            column = f"{metric}_improvement_vs_{baseline_strategy}"
            values = group[column].to_numpy(dtype=float)
            samples = values[rng.integers(0, values.size, size=(10000, values.size))].mean(axis=1)
            ci_rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "mean": values.mean(),
                    "ci_low": np.quantile(samples, 0.025),
                    "ci_high": np.quantile(samples, 0.975),
                }
            )
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(out_dir / f"bootstrap_ci_vs_{baseline_strategy}.csv", index=False)

    improvement_columns = [
        f"mean_decode_latency_improvement_vs_{baseline_strategy}_mean",
        f"p95_decode_latency_improvement_vs_{baseline_strategy}_mean",
        f"mean_barrier_latency_improvement_vs_{baseline_strategy}_mean",
    ]
    table_columns = [
        "strategy",
        "mean_decode_latency_ms_mean",
        "p95_decode_latency_ms_mean",
        "mean_barrier_latency_ms_mean",
        *improvement_columns,
    ]
    optional_columns = [
        "mean_network_response_sleep_seconds_mean",
        "mean_completed_rows_mean",
        "mean_cancelled_rows_mean",
    ]
    table = aggregate[[column for column in table_columns + optional_columns if column in aggregate.columns]].copy()
    for column in improvement_columns:
        if column in table:
            table[column] *= 100.0
    if "mean_network_response_sleep_seconds_mean" in table:
        table["mean_network_response_sleep_seconds_mean"] *= 1000.0

    ci_table = ci.copy()
    for column in ["mean", "ci_low", "ci_high"]:
        ci_table[column] *= 100.0

    return [
        "## Paired baseline diagnostics",
        "",
        f"Positive gains are relative to `{baseline_strategy}` within the same run.",
        "",
        "```",
        table.to_string(index=False, float_format=lambda value: f"{value:.2f}"),
        "```",
        "",
        "### Bootstrap confidence intervals",
        "",
        "```",
        ci_table.to_string(index=False, float_format=lambda value: f"{value:.2f}"),
        "```",
        "",
    ]


if __name__ == "__main__":
    main()
