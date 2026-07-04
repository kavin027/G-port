from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = [
    {
        "variant": "Speed only",
        "strategy": "speed_aware_uncoded",
        "uses_speed": "yes",
        "uses_row_cost": "no",
        "uses_prefix": "no",
        "uses_guard": "no",
    },
    {
        "variant": "+ row cost",
        "strategy": "worker_aware_sparse_flexible",
        "uses_speed": "yes",
        "uses_row_cost": "yes",
        "uses_prefix": "no",
        "uses_guard": "no",
    },
    {
        "variant": "+ first-decode prefix",
        "strategy": "rank_aware_sparse_flexible",
        "uses_speed": "yes",
        "uses_row_cost": "yes",
        "uses_prefix": "yes",
        "uses_guard": "no",
    },
    {
        "variant": "+ guard counters",
        "strategy": "guarded_system_portfolio",
        "uses_speed": "yes",
        "uses_row_cost": "yes",
        "uses_prefix": "yes",
        "uses_guard": "yes",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a row-cost/prefix/guard feature ablation from K3s logs."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--label",
        default="static-fallback K3s",
        help="Human-readable source label stored in the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.root)
    if rows.empty:
        raise SystemExit(f"No K3s summary rows found under {args.root}")

    table = build_table(rows, args.label)
    table.to_csv(args.out / "feature_ablation_table.csv", index=False)
    write_latex(table, args.out / "feature_ablation_table.tex")
    write_markdown(table, args.out / "feature_ablation_report.md")
    print(table.to_string(index=False))
    print(f"Wrote feature ablation diagnostics to {args.out}")


def read_rows(root: Path) -> pd.DataFrame:
    path = root / "majorrev_k8s_all_summary.csv"
    if path.exists():
        frame = pd.read_csv(path)
        if {"workers", "seed", "strategy", "mean_barrier_latency"}.issubset(frame.columns):
            return frame

    frames: list[pd.DataFrame] = []
    for summary in sorted(root.rglob("network_summary.csv")):
        run = parse_run_dir(summary.parent.name)
        if run is None:
            continue
        frame = pd.read_csv(summary)
        frame["workers"] = run[0]
        frame["seed"] = run[1]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_run_dir(name: str) -> tuple[int, int] | None:
    marker = "majorrev_k8s_w"
    if not name.startswith(marker):
        return None
    try:
        rest = name[len(marker) :]
        workers_text, seed_text = rest.split("_seed", 1)
        return int(workers_text), int(seed_text)
    except ValueError:
        return None


def build_table(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = rows.copy()
    rows["mean_barrier_ms"] = 1000.0 * rows["mean_barrier_latency"].astype(float)
    if "mean_extra_compute" not in rows.columns:
        rows["mean_extra_compute"] = np.nan
    if "mean_cancel_seconds" not in rows.columns:
        rows["mean_cancel_seconds"] = np.nan
    rows["mean_cancel_ms"] = 1000.0 * rows["mean_cancel_seconds"].astype(float)
    rows["decode_success_rate"] = rows.get("decode_success_rate", pd.Series(1.0, index=rows.index))

    output: list[dict[str, object]] = []
    for spec in VARIANTS:
        strategy = spec["strategy"]
        subset = rows[rows["strategy"] == strategy]
        if subset.empty:
            continue
        item: dict[str, object] = {
            "source": label,
            **spec,
            "avg_extra_compute": float(subset["mean_extra_compute"].mean()),
            "avg_cancel_ms": float(subset["mean_cancel_ms"].mean()),
            "decode_success_rate": float(subset["decode_success_rate"].mean()),
        }
        regressions = 0
        for workers in (8, 16, 24):
            worker_subset = subset[subset["workers"] == workers]
            base = rows[
                (rows["workers"] == workers)
                & (rows["strategy"].isin(["sparse_flexible_static", "original_sfcl_static"]))
            ][["seed", "mean_barrier_ms"]].rename(columns={"mean_barrier_ms": "baseline_ms"})
            if worker_subset.empty or base.empty:
                item[f"w{workers}_gain_pct"] = np.nan
                item[f"w{workers}_mean_ms"] = np.nan
                continue
            paired = worker_subset[["seed", "mean_barrier_ms"]].merge(base, on="seed", how="left")
            gain = 100.0 * (paired["baseline_ms"] - paired["mean_barrier_ms"]) / paired["baseline_ms"]
            item[f"w{workers}_gain_pct"] = float(gain.mean())
            item[f"w{workers}_mean_ms"] = float(worker_subset["mean_barrier_ms"].mean())
            regressions += int((paired["mean_barrier_ms"] > paired["baseline_ms"]).sum())
        item["regressions"] = int(regressions)
        output.append(item)
    return pd.DataFrame(output)


def write_latex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{Feature ablation on K3s logs.}",
        r"  \label{tab:feature-ablation}",
        r"  \begin{tabular}{@{}lccccrrrr@{}}",
        r"    \toprule",
        r"    Variant & Speed & Cost & Prefix & Guard & W8 & W16 & W24 & Reg. \\",
        r"    \midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            "    "
            + f"{row['variant']} & {row['uses_speed']} & {row['uses_row_cost']} "
            + f"& {row['uses_prefix']} & {row['uses_guard']} "
            + f"& {row['w8_gain_pct']:.1f}\\% & {row['w16_gain_pct']:.1f}\\% "
            + f"& {row['w24_gain_pct']:.1f}\\% & {int(row['regressions'])} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    path.write_text(
        "# Feature Ablation\n\n"
        "This diagnostic reuses paired K3s worker-service logs and attributes "
        "the performance changes to feature visibility rather than to a new "
        "scheduler implementation.\n\n"
        + table.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
