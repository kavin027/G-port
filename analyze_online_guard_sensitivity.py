from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE = "sparse_flexible_static"
DEFAULT_CANDIDATES = [
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay causal warm-up guards over Docker iteration logs. The guard "
            "uses only early completed-row counters, then evaluates later "
            "iterations."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("local_docker_container_sweep"),
        help="Directory containing w*_run*/network_metrics.csv files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("online_guard_sensitivity_diagnostics"),
        help="Output directory for CSV, plots, and report.",
    )
    parser.add_argument(
        "--calibration-fractions",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6],
        help="Fractions of each run used for causal warm-up calibration.",
    )
    parser.add_argument(
        "--prefix-tolerances",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0],
        help="Allowed candidate-minus-static completed-row growth during warm-up.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=DEFAULT_CANDIDATES,
        help="Candidate strategies to compare against sparse_flexible_static.",
    )
    parser.add_argument(
        "--min-eval-iterations",
        type=int,
        default=2,
        help="Minimum number of post-calibration iterations required per run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run_rows = _collect_run_rows(args)
    if run_rows.empty:
        raise SystemExit(
            f"No paired {BASELINE} and candidate iteration logs found under {args.root}."
        )

    summary = _summarize(run_rows)
    best = _recommended_setting(summary)

    run_rows.to_csv(args.out / "online_guard_sensitivity_runs.csv", index=False)
    summary.to_csv(args.out / "online_guard_sensitivity_summary.csv", index=False)
    _plot(summary, args.out)
    _write_report(run_rows, summary, best, args.out / "online_guard_sensitivity_report.md")

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nWrote online guard sensitivity diagnostics to {args.out}")


def _collect_run_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(args.root.glob("w*_run*/network_metrics.csv")):
        frame = pd.read_csv(metrics_path)
        run = metrics_path.parent.name
        workers = _parse_workers(run)
        static = frame.loc[frame["strategy"] == BASELINE].sort_values("iteration")
        if static.empty:
            continue
        for candidate in args.candidates:
            cand = frame.loc[frame["strategy"] == candidate].sort_values("iteration")
            if cand.empty:
                continue
            paired = static.merge(
                cand,
                on="iteration",
                suffixes=("_static", "_candidate"),
                validate="one_to_one",
            ).sort_values("iteration")
            if paired.empty:
                continue
            for fraction in args.calibration_fractions:
                calibration_count = int(math.ceil(float(fraction) * len(paired)))
                calibration_count = max(1, min(calibration_count, len(paired) - 1))
                eval_count = len(paired) - calibration_count
                if eval_count < args.min_eval_iterations:
                    continue
                calibration = paired.iloc[:calibration_count]
                evaluation = paired.iloc[calibration_count:]
                prefix_delta = float(
                    calibration["completed_rows_candidate"].mean()
                    - calibration["completed_rows_static"].mean()
                )
                selected_delta = float(
                    calibration["selected_rows_candidate"].mean()
                    - calibration["selected_rows_static"].mean()
                )
                static_p95 = float(np.quantile(evaluation["decode_latency_static"], 0.95))
                candidate_p95 = float(
                    np.quantile(evaluation["decode_latency_candidate"], 0.95)
                )
                always_gain = _gain_pct(static_p95, candidate_p95)
                for tolerance in args.prefix_tolerances:
                    enabled = prefix_delta <= float(tolerance)
                    rows.append(
                        {
                            "run": run,
                            "workers": workers,
                            "candidate": candidate,
                            "calibration_fraction": float(fraction),
                            "calibration_iterations": calibration_count,
                            "eval_iterations": eval_count,
                            "prefix_tolerance_rows": float(tolerance),
                            "calibration_prefix_delta_rows": prefix_delta,
                            "calibration_selected_delta_rows": selected_delta,
                            "enabled": bool(enabled),
                            "static_eval_p95_ms": 1000.0 * static_p95,
                            "candidate_eval_p95_ms": 1000.0 * candidate_p95,
                            "always_on_p95_gain_pct": always_gain,
                            "causal_guard_p95_gain_pct": always_gain if enabled else 0.0,
                            "always_on_negative": always_gain < 0.0,
                            "causal_guard_negative": enabled and always_gain < 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rows.groupby(["candidate", "calibration_fraction", "prefix_tolerance_rows"], sort=True)
        .agg(
            runs=("run", "count"),
            enabled_runs=("enabled", "sum"),
            always_on_mean_p95_gain_pct=("always_on_p95_gain_pct", "mean"),
            causal_guard_mean_p95_gain_pct=("causal_guard_p95_gain_pct", "mean"),
            always_on_median_p95_gain_pct=("always_on_p95_gain_pct", "median"),
            causal_guard_median_p95_gain_pct=("causal_guard_p95_gain_pct", "median"),
            always_on_negative_runs=("always_on_negative", "sum"),
            causal_guard_negative_runs=("causal_guard_negative", "sum"),
            mean_calibration_prefix_delta_rows=("calibration_prefix_delta_rows", "mean"),
            max_calibration_prefix_delta_rows=("calibration_prefix_delta_rows", "max"),
            mean_eval_iterations=("eval_iterations", "mean"),
        )
        .reset_index()
    )
    numeric_int = [
        "runs",
        "enabled_runs",
        "always_on_negative_runs",
        "causal_guard_negative_runs",
    ]
    for column in numeric_int:
        grouped[column] = grouped[column].astype(int)
    return grouped


def _recommended_setting(summary: pd.DataFrame) -> pd.Series:
    rank = summary.loc[
        (summary["candidate"] == "rank_aware_sparse_flexible")
        & (summary["calibration_fraction"] == 0.2)
        & (summary["prefix_tolerance_rows"] == 0.0)
    ]
    if not rank.empty:
        return rank.iloc[0]
    no_negative = summary.loc[summary["causal_guard_negative_runs"] == 0]
    if no_negative.empty:
        return summary.sort_values("causal_guard_mean_p95_gain_pct", ascending=False).iloc[0]
    return no_negative.sort_values(
        ["calibration_fraction", "prefix_tolerance_rows", "causal_guard_mean_p95_gain_pct"],
        ascending=[True, True, False],
    ).iloc[0]


def _plot(summary: pd.DataFrame, out: Path) -> None:
    for candidate, group in summary.groupby("candidate", sort=True):
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for tolerance, sub in group.groupby("prefix_tolerance_rows", sort=True):
            sub = sub.sort_values("calibration_fraction")
            ax.plot(
                sub["calibration_fraction"],
                sub["causal_guard_mean_p95_gain_pct"],
                marker="o",
                linewidth=2.0,
                label=f"tol={tolerance:g} rows",
            )
        always = (
            group.groupby("calibration_fraction", sort=True)["always_on_mean_p95_gain_pct"]
            .mean()
            .reset_index()
            .sort_values("calibration_fraction")
        )
        ax.plot(
            always["calibration_fraction"],
            always["always_on_mean_p95_gain_pct"],
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="always-on",
        )
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.set_xlabel("warm-up fraction")
        ax.set_ylabel("mean post-warm-up p95 gain (%)")
        ax.set_title(_label(candidate))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        stem = candidate.replace("_", "-")
        fig.savefig(out / f"{stem}_online_guard_sensitivity.png", dpi=220)
        fig.savefig(out / f"{stem}_online_guard_sensitivity.pdf")
        plt.close(fig)


def _write_report(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    best: pd.Series,
    path: Path,
) -> None:
    lines = [
        "# Online Guard Sensitivity",
        "",
        "This diagnostic addresses the reviewer concern that the guarded policy is",
        "only an offline replay.  Each row uses only the first calibration window",
        "of a Docker TCP run to decide whether to enable a candidate scheduler;",
        "the p95 latency gain is then computed only on later iterations.  The",
        "replay still runs over logged policy traces, so it is a causal replay",
        "sanity check rather than a deployed online controller.",
        "",
        "## Recommended Conservative Setting",
        "",
        (
            f"- Candidate: `{best['candidate']}`\n"
            f"- Warm-up fraction: {best['calibration_fraction']:.2f}\n"
            f"- Prefix tolerance: {best['prefix_tolerance_rows']:.1f} completed rows\n"
            f"- Enabled runs: {int(best['enabled_runs'])}/{int(best['runs'])}\n"
            f"- Always-on mean p95 gain: {best['always_on_mean_p95_gain_pct']:.1f}%\n"
            f"- Causal-guard mean p95 gain: {best['causal_guard_mean_p95_gain_pct']:.1f}%\n"
            f"- Negative runs after guard: {int(best['causal_guard_negative_runs'])}"
        ),
        "",
        "## Sensitivity Summary",
        "",
        _markdown_table(summary),
        "",
        "## Per-Run Notes",
        "",
    ]
    compact = rows.loc[
        (rows["candidate"] == best["candidate"])
        & (rows["calibration_fraction"] == best["calibration_fraction"])
        & (rows["prefix_tolerance_rows"] == best["prefix_tolerance_rows"])
    ].sort_values(["workers", "run"])
    for _, row in compact.iterrows():
        action = "enable" if bool(row["enabled"]) else "fallback"
        lines.append(
            "- "
            f"{row['run']} ({int(row['workers'])} workers): {action}, "
            f"warm-up prefix delta {row['calibration_prefix_delta_rows']:.2f} rows, "
            f"post-warm-up p95 gain {row['always_on_p95_gain_pct']:.1f}%."
        )
    lines.extend(
        [
            "",
            "## Rebuttal Use",
            "",
            "The guard is best described as a fixed counter rule whose replay is",
            "post-hoc only because policies are run separately.  This sensitivity",
            "sweep maps where the early-window rule holds and where a looser",
            "prefix tolerance admits harmful runs.  It should not be presented as",
            "evidence that the prototype is a complete production cluster scheduler.",
            "",
            "Generated files:",
            "",
            "- `online_guard_sensitivity_runs.csv`",
            "- `online_guard_sensitivity_summary.csv`",
            "- `rank-aware-sparse-flexible_online_guard_sensitivity.png`",
            "- `deadline-aware-sparse-flexible_online_guard_sensitivity.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(summary: pd.DataFrame) -> str:
    columns = [
        "candidate",
        "calibration_fraction",
        "prefix_tolerance_rows",
        "runs",
        "enabled_runs",
        "always_on_mean_p95_gain_pct",
        "causal_guard_mean_p95_gain_pct",
        "always_on_negative_runs",
        "causal_guard_negative_runs",
    ]
    table = summary.loc[:, columns].copy()
    for column in [
        "calibration_fraction",
        "prefix_tolerance_rows",
        "always_on_mean_p95_gain_pct",
        "causal_guard_mean_p95_gain_pct",
    ]:
        table[column] = table[column].map(lambda value: f"{float(value):.1f}")
    return table.to_markdown(index=False)


def _parse_workers(run: str) -> int:
    first = run.split("_", 1)[0]
    return int(first.removeprefix("w"))


def _gain_pct(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return 100.0 * (baseline - candidate) / baseline


def _label(strategy: str) -> str:
    return strategy.replace("_", " ")


if __name__ == "__main__":
    main()
