from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MAIN_METHODS = [
    ("original_sfcl_static", "SFCL"),
    ("rltune_style_selector", "RLTune-sched"),
    ("sailor_style_heterogeneity_aware", "Sailor-sched"),
    ("guarded_system_portfolio", "G-PORT"),
]

ABLATION_METHODS = [
    ("guarded_system_portfolio", "G-PORT"),
    ("sailor_style_heterogeneity_aware", "w/o FD"),
    ("rltune_style_selector", "w/o code"),
    ("online_counter_guard_deadline_aware_sparse_flexible", "w/o portfolio"),
]

RUNTIME_INPUTS = [
    ("TCP", "tcp_summary"),
    ("TCP+stress", "stress_summary"),
    ("K3s", "k3s_summary"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-facing Table 1 and Table 2 CSV/LaTeX files from "
            "collected TCP, TCP+stress, and K3s external-baseline summaries."
        )
    )
    parser.add_argument(
        "--tcp-summary",
        type=Path,
        default=Path("results/external_baselines_tcp_plain_full_best_safe/tcp_external_summary.csv"),
    )
    parser.add_argument(
        "--stress-summary",
        type=Path,
        default=Path(
            "results/external_baselines_network_stress_full_best_safe/tcp_external_summary.csv"
        ),
    )
    parser.add_argument(
        "--k3s-summary",
        type=Path,
        default=Path(
            "results/server_k3s_20260702/coded_k3s_external_full/"
            "external_analysis_rebuild/k3s_external_summary.csv"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("results/paper_reproduction"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = {
        "TCP": load_summary(args.tcp_summary),
        "TCP+stress": load_summary(args.stress_summary),
        "K3s": load_summary(args.k3s_summary),
    }

    table1 = build_main_external_table(summaries)
    table2 = build_ablation_table(summaries)

    table1.to_csv(args.out / "table1_main_external.csv", index=False)
    table2.to_csv(args.out / "table2_gport_ablation.csv", index=False)
    (args.out / "table1_main_external.tex").write_text(
        render_main_external_latex(table1), encoding="utf-8"
    )
    (args.out / "table2_gport_ablation.tex").write_text(
        render_ablation_latex(table2), encoding="utf-8"
    )
    print(f"Wrote paper table artifacts to {args.out}")


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "n_workers",
        "strategy",
        "mean_barrier_ms",
        "p95_barrier_ms",
        "mean_rows_after_decode",
        "decode_success_rate",
        "mean_cancel_ms",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def lookup(frame: pd.DataFrame, strategy: str, workers: int) -> pd.Series:
    rows = frame[(frame["strategy"] == strategy) & (frame["n_workers"] == workers)]
    if rows.empty:
        raise KeyError(f"Missing strategy={strategy!r}, workers={workers}")
    return rows.iloc[0]


def throughput(mean_barrier_ms: float) -> float:
    if mean_barrier_ms <= 0:
        return 0.0
    return 1000.0 / mean_barrier_ms


def build_main_external_table(summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    metrics = [
        ("Mean barrier (ms)", "mean_barrier_ms", False),
        ("P95 barrier (ms)", "p95_barrier_ms", False),
        ("Throughput (iter/s)", "throughput_iter_s", True),
        ("Post-decode rows", "mean_rows_after_decode", False),
        ("Cancel time (ms)", "mean_cancel_ms", False),
        ("Decode success", "decode_success_rate", True),
    ]
    for runtime, frame in summaries.items():
        for workers in [8, 16, 24]:
            values: dict[str, dict[str, float]] = {}
            for strategy, method in MAIN_METHODS:
                row = lookup(frame, strategy, workers)
                values[method] = {
                    "mean_barrier_ms": float(row["mean_barrier_ms"]),
                    "p95_barrier_ms": float(row["p95_barrier_ms"]),
                    "throughput_iter_s": throughput(float(row["mean_barrier_ms"])),
                    "mean_rows_after_decode": float(row["mean_rows_after_decode"]),
                    "mean_cancel_ms": float(row["mean_cancel_ms"]),
                    "decode_success_rate": float(row["decode_success_rate"]),
                }
            for metric_name, metric_key, higher_is_better in metrics:
                method_values = {method: vals[metric_key] for method, vals in values.items()}
                records.append(
                    {
                        "Runtime": runtime,
                        "Workers": workers,
                        "Metric": metric_name,
                        **method_values,
                        "Best": best_method(method_values, higher_is_better),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_ablation_table(summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for runtime, frame in summaries.items():
        for strategy, ablation in ABLATION_METHODS:
            record: dict[str, object] = {
                "Runtime": runtime,
                "Ablation": ablation,
            }
            for workers in [8, 16, 24]:
                row = lookup(frame, strategy, workers)
                record[f"W{workers} mean ms"] = round(float(row["mean_barrier_ms"]), 1)
                record[f"W{workers} post rows"] = round(float(row["mean_rows_after_decode"]), 1)
                record[f"W{workers} cancel ms"] = round(float(row["mean_cancel_ms"]), 1)
            records.append(record)
    return pd.DataFrame.from_records(records)


def best_method(values: dict[str, float], higher_is_better: bool) -> str:
    key = max if higher_is_better else min
    target = key(values.values())
    winners = [method for method, value in values.items() if abs(value - target) < 1e-9]
    return ",".join(winners)


def format_metric(value: float, metric: str) -> str:
    if metric == "Decode success":
        return f"{value:.2f}"
    return f"{value:.1f}"


def render_main_external_latex(table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by build_paper_artifact_tables.py.",
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Main external-baseline comparison.}",
        r"    \label{tab:external-main}",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3.5pt}",
        r"    \begin{tabular}{llrrrrr}",
        r"        \toprule",
        r"        Runtime & Metric & W & SFCL & RLTune-sched & Sailor-sched & G-PORT \\",
        r"        \midrule",
    ]
    current_runtime = None
    for _, row in table.iterrows():
        runtime = row["Runtime"] if row["Runtime"] != current_runtime else ""
        if current_runtime is not None and row["Runtime"] != current_runtime:
            lines.append(r"        \midrule")
        current_runtime = row["Runtime"]
        metric = row["Metric"]
        values = [
            bold_if_best(format_metric(float(row[method]), metric), method, str(row["Best"]))
            for method in ["SFCL", "RLTune-sched", "Sailor-sched", "G-PORT"]
        ]
        lines.append(
            "        "
            + " & ".join([runtime, metric, str(int(row["Workers"])), *values])
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def bold_if_best(value: str, method: str, best: str) -> str:
    return rf"\textbf{{{value}}}" if method in str(best).split(",") else value


def render_ablation_latex(table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by build_paper_artifact_tables.py.",
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{G-PORT ablations across runtimes.}",
        r"    \label{tab:gport-ablation}",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3.5pt}",
        r"    \begin{tabular}{llrrrrrrrrr}",
        r"        \toprule",
        r"        Runtime & Ablation & \multicolumn{3}{c}{W8} & \multicolumn{3}{c}{W16} & \multicolumn{3}{c}{W24} \\",
        r"        \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"        & & Mean & Post & Cancel & Mean & Post & Cancel & Mean & Post & Cancel \\",
        r"        \midrule",
    ]
    current_runtime = None
    for _, row in table.iterrows():
        runtime = row["Runtime"] if row["Runtime"] != current_runtime else ""
        if current_runtime is not None and row["Runtime"] != current_runtime:
            lines.append(r"        \midrule")
        current_runtime = row["Runtime"]
        values = [
            f"{float(row['W8 mean ms']):.1f}",
            f"{float(row['W8 post rows']):.1f}",
            f"{float(row['W8 cancel ms']):.1f}",
            f"{float(row['W16 mean ms']):.1f}",
            f"{float(row['W16 post rows']):.1f}",
            f"{float(row['W16 cancel ms']):.1f}",
            f"{float(row['W24 mean ms']):.1f}",
            f"{float(row['W24 post rows']):.1f}",
            f"{float(row['W24 cancel ms']):.1f}",
        ]
        lines.append(
            "        "
            + " & ".join([runtime, row["Ablation"], *values])
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
