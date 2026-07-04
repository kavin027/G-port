from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_META: dict[str, dict[str, str]] = {
    "original_sfcl_static": {
        "method": "Original-SFCL",
        "source": "Sparse-flexible coded learning",
        "artifact": "same-runtime reimplementation",
        "uses_first_decode_features": "No",
        "notes": "static sparse-flexible placement",
    },
    "sparse_flexible_static": {
        "method": "SparseFlexibleStatic",
        "source": "Sparse-flexible coded learning",
        "artifact": "same-runtime reimplementation",
        "uses_first_decode_features": "No",
        "notes": "legacy static baseline name",
    },
    "rltune_style_selector": {
        "method": "RLTune-style",
        "source": "SoCC'25 RLTune",
        "artifact": "public artifact; runtime adaptation",
        "uses_first_decode_features": "Indirect",
        "notes": "portfolio over uncoded, replicated, static-coded, and rank-coded arms",
    },
    "sailor_style_heterogeneity_aware": {
        "method": "Sailor-style",
        "source": "SOSP'25 Sailor",
        "artifact": "public artifact; runtime adaptation",
        "uses_first_decode_features": "No",
        "notes": "heterogeneity-aware throughput and load model",
    },
    "guarded_system_portfolio": {
        "method": "Guarded portfolio",
        "source": "this paper",
        "artifact": "ours",
        "uses_first_decode_features": "Guarded",
        "notes": "online guard over a system portfolio",
    },
    "guarded_system_portfolio_best_safe": {
        "method": "Guarded portfolio-BS",
        "source": "this paper",
        "artifact": "ours",
        "uses_first_decode_features": "Guarded",
        "notes": "online guard with fixed best-safe fallback",
    },
    "online_counter_guard_deadline_aware_sparse_flexible": {
        "method": "Guard-D",
        "source": "this paper",
        "artifact": "ours",
        "uses_first_decode_features": "Guarded",
        "notes": "coded-only guard; static sparse-flexible fallback",
    },
    "straggler_whatif_diagnostic": {
        "method": "What-if diagnostic",
        "source": "OSDI'25 StragglerAnalysis-style",
        "artifact": "public artifact; log replay diagnostic",
        "uses_first_decode_features": "Offline only",
        "notes": "not a deployable scheduler",
    },
}

TABLE_ORDER = [
    "original_sfcl_static",
    "rltune_style_selector",
    "sailor_style_heterogeneity_aware",
    "guarded_system_portfolio",
    "guarded_system_portfolio_best_safe",
    "online_counter_guard_deadline_aware_sparse_flexible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize external-style baselines run in the worker-service runtime."
    )
    parser.add_argument("--root", type=Path, default=Path("results/external_baselines"))
    parser.add_argument("--out", type=Path, default=Path("results/external_baselines"))
    parser.add_argument(
        "--best-safe-root",
        type=Path,
        default=None,
        help=(
            "Optional independent K3s run with --portfolio-fallback best_safe. "
            "Its guarded portfolio row is appended to the external matrix as "
            "Guarded portfolio-BS using the run's own static baseline."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(args.root)
    if metrics.empty:
        raise SystemExit(f"No network_metrics.csv files found under {args.root}")

    per_seed = per_seed_summary(metrics)
    per_seed.to_csv(args.out / "per_seed_external_results.csv", index=False)

    summary = external_summary(metrics)
    k3s_summary = summary[summary["external_mode"] == "k3s"]
    tcp_summary = summary[summary["external_mode"] != "k3s"]
    k3s_summary.to_csv(args.out / "k3s_external_summary.csv", index=False)
    tcp_summary.to_csv(args.out / "tcp_external_summary.csv", index=False)

    matrix = external_matrix(summary, per_seed)
    if args.best_safe_root is not None:
        best_safe_metrics = read_metrics(args.best_safe_root)
        if best_safe_metrics.empty:
            raise SystemExit(
                f"No network_metrics.csv files found under {args.best_safe_root}"
            )
        best_safe_per_seed = per_seed_summary(best_safe_metrics)
        best_safe_summary = external_summary(best_safe_metrics)
        best_safe_per_seed.to_csv(
            args.out / "best_safe_per_seed_external_results.csv", index=False
        )
        best_safe_summary.to_csv(args.out / "best_safe_k3s_summary.csv", index=False)
        best_safe_matrix = external_matrix(best_safe_summary, best_safe_per_seed)
        best_safe_row = best_safe_matrix[
            (best_safe_matrix["external_mode"] == "k3s")
            & (best_safe_matrix["strategy"] == "guarded_system_portfolio")
        ].copy()
        if best_safe_row.empty:
            raise SystemExit(
                "Best-safe root did not contain guarded_system_portfolio metrics"
            )
        best_safe_row["strategy"] = "guarded_system_portfolio_best_safe"
        for key, value in METHOD_META["guarded_system_portfolio_best_safe"].items():
            column = {
                "uses_first_decode_features": "uses_first_decode_features",
            }.get(key, key)
            best_safe_row[column] = value
        matrix = pd.concat([matrix, best_safe_row], ignore_index=True)
        matrix["sort_key"] = matrix["strategy"].map(
            {strategy: index for index, strategy in enumerate(TABLE_ORDER)}
        ).fillna(len(TABLE_ORDER))
        matrix = matrix.sort_values(["external_mode", "sort_key"]).drop(columns=["sort_key"])
    matrix.to_csv(args.out / "external_baseline_matrix.csv", index=False)

    overhead = external_overhead_summary(metrics, per_seed)
    overhead.to_csv(args.out / "external_overhead_summary.csv", index=False)

    arms = selected_arm_distribution(metrics)
    arms.to_csv(args.out / "external_arm_distribution.csv", index=False)

    whatif = whatif_diagnostics(metrics)
    whatif.to_csv(args.out / "whatif_diagnostics.csv", index=False)
    write_latex_table(summary, whatif, args.out / "external_baselines_table.tex")
    write_matrix_latex_table(matrix, args.out / "external_baseline_matrix_table.tex")
    write_taxonomy_latex_table(args.out / "external_baseline_taxonomy_table.tex")

    print(f"Wrote external baseline summaries to {args.out}")
    print(summary.to_string(index=False))


def read_metrics(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("network_metrics.csv")):
        frame = pd.read_csv(path)
        metadata = read_metadata(path.parent)
        parsed = parse_run_path(path.parent)
        metadata = {**parsed, **metadata}
        if "external_mode" not in metadata:
            metadata["external_mode"] = metadata.get("mode", "tcp")
        if "seed" not in metadata and "seed" in frame.columns:
            metadata["seed"] = int(frame["seed"].iloc[0])
        if "n_workers" not in metadata and "n_workers" in frame.columns:
            metadata["n_workers"] = int(frame["n_workers"].iloc[0])
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, sort_keys=True)
            frame[key] = value
        frame["run_dir"] = str(path.parent)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "external_baseline_run.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_path(run_dir: Path) -> dict[str, Any]:
    match = re.search(r"(?P<mode>[A-Za-z0-9_]+)_w(?P<workers>\d+)_seed(?P<seed>\d+)", run_dir.name)
    if not match:
        k3s_match = re.search(r"majorrev_k8s_w(?P<workers>\d+)_seed(?P<seed>\d+)", run_dir.name)
        if not k3s_match:
            return {}
        return {
            "external_mode": "k3s",
            "n_workers": int(k3s_match.group("workers")),
            "seed": int(k3s_match.group("seed")),
        }
    mode = match.group("mode")
    if mode == "majorrev_k8s":
        mode = "k3s"
    return {
        "external_mode": mode,
        "n_workers": int(match.group("workers")),
        "seed": int(match.group("seed")),
    }


def per_seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["external_mode", "n_workers", "seed", "strategy"], sort=False)
    rows = grouped.agg(
        mean_barrier_ms=("barrier_latency", lambda x: 1000.0 * x.mean()),
        p95_barrier_ms=("barrier_latency", lambda x: 1000.0 * x.quantile(0.95)),
        mean_decode_ms=("decode_latency", lambda x: 1000.0 * x.mean()),
        mean_extra_compute=("extra_compute", "mean"),
        mean_rows_after_decode=("rows_after_decode", "mean"),
        decode_success_rate=("decode_success", "mean"),
        mean_dispatch_ms=("dispatch_seconds", lambda x: 1000.0 * x.mean()),
        mean_cancel_ms=("cancel_seconds", lambda x: 1000.0 * x.mean()),
    ).reset_index()
    return add_method_metadata(rows)


def external_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["external_mode", "n_workers", "strategy"], sort=False)
    rows = grouped.agg(
        mean_barrier_ms=("barrier_latency", lambda x: 1000.0 * x.mean()),
        p95_barrier_ms=("barrier_latency", lambda x: 1000.0 * x.quantile(0.95)),
        mean_decode_ms=("decode_latency", lambda x: 1000.0 * x.mean()),
        p95_decode_ms=("decode_latency", lambda x: 1000.0 * x.quantile(0.95)),
        mean_extra_compute=("extra_compute", "mean"),
        mean_rows_after_decode=("rows_after_decode", "mean"),
        decode_success_rate=("decode_success", "mean"),
        mean_dispatch_ms=("dispatch_seconds", lambda x: 1000.0 * x.mean()),
        mean_cancel_ms=("cancel_seconds", lambda x: 1000.0 * x.mean()),
        mean_network_response_mb=("network_response_bytes", lambda x: x.mean() / 1_000_000.0),
    ).reset_index()
    rows = add_method_metadata(rows)
    rows["gain_vs_original_percent"] = np.nan
    for (mode, workers), idx in rows.groupby(["external_mode", "n_workers"]).groups.items():
        block = rows.loc[idx]
        baseline = baseline_barrier(block)
        if np.isfinite(baseline) and baseline > 0:
            rows.loc[idx, "gain_vs_original_percent"] = (
                100.0 * (baseline - rows.loc[idx, "mean_barrier_ms"]) / baseline
            )
    return rows


def external_matrix(summary: pd.DataFrame, per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mode, strategy), block in summary.groupby(["external_mode", "strategy"], sort=False):
        item: dict[str, Any] = {"external_mode": mode, "strategy": strategy}
        meta = METHOD_META.get(strategy, {})
        for key in ("method", "source", "artifact", "uses_first_decode_features", "notes"):
            item[key] = meta.get(key, strategy)
        for workers in (8, 16, 24):
            worker_block = block[block["n_workers"] == workers]
            if worker_block.empty:
                item[f"w{workers}_mean_barrier_ms"] = np.nan
                item[f"w{workers}_p95_barrier_ms"] = np.nan
                item[f"w{workers}_gain_percent"] = np.nan
                item[f"w{workers}_regressions"] = np.nan
                continue
            record = worker_block.iloc[0]
            item[f"w{workers}_mean_barrier_ms"] = float(record["mean_barrier_ms"])
            item[f"w{workers}_p95_barrier_ms"] = float(record["p95_barrier_ms"])
            item[f"w{workers}_gain_percent"] = float(record["gain_vs_original_percent"])

            seed_block = per_seed[
                (per_seed["external_mode"] == mode)
                & (per_seed["n_workers"] == workers)
                & (per_seed["strategy"] == strategy)
            ]
            base_block = per_seed[
                (per_seed["external_mode"] == mode)
                & (per_seed["n_workers"] == workers)
                & (per_seed["strategy"].isin(["original_sfcl_static", "sparse_flexible_static"]))
            ][["seed", "mean_barrier_ms"]].rename(columns={"mean_barrier_ms": "baseline_ms"})
            if seed_block.empty or base_block.empty:
                item[f"w{workers}_regressions"] = np.nan
            else:
                joined = seed_block.merge(base_block, on="seed", how="left")
                item[f"w{workers}_regressions"] = int(
                    (joined["mean_barrier_ms"] > joined["baseline_ms"]).sum()
                )
        item["avg_p95_barrier_ms"] = float(block["p95_barrier_ms"].mean())
        item["avg_extra_compute"] = float(block["mean_extra_compute"].mean())
        item["avg_cancel_ms"] = float(block["mean_cancel_ms"].mean())
        rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["sort_key"] = result["strategy"].map(
        {strategy: index for index, strategy in enumerate(TABLE_ORDER)}
    ).fillna(len(TABLE_ORDER))
    return result.sort_values(["external_mode", "sort_key"]).drop(columns=["sort_key"])


def external_overhead_summary(metrics: pd.DataFrame, per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = metrics.groupby(["external_mode", "strategy"], sort=False).agg(
        mean_extra_compute=("extra_compute", "mean"),
        mean_dispatch_ms=("dispatch_seconds", lambda x: 1000.0 * x.mean()),
        mean_cancel_ms=("cancel_seconds", lambda x: 1000.0 * x.mean()),
        mean_rows_after_decode=("rows_after_decode", "mean"),
        mean_selected_rows=("selected_rows", "mean"),
        mean_completed_rows=("completed_rows", "mean"),
    ).reset_index()
    base = per_seed[
        per_seed["strategy"].isin(["original_sfcl_static", "sparse_flexible_static"])
    ][["external_mode", "n_workers", "seed", "mean_barrier_ms"]].rename(
        columns={"mean_barrier_ms": "baseline_ms"}
    )
    merged = per_seed.merge(base, on=["external_mode", "n_workers", "seed"], how="left")
    merged["regression"] = merged["mean_barrier_ms"] > merged["baseline_ms"]
    regressions = merged.groupby(["external_mode", "strategy"], sort=False).agg(
        regression_count=("regression", "sum"),
        regression_cases=("regression", "count"),
    ).reset_index()
    rows = rows.merge(regressions, on=["external_mode", "strategy"], how="left")
    return add_method_metadata(rows)


def selected_arm_distribution(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    frame["selected_arm"] = frame.apply(selected_arm, axis=1)
    rows = frame.groupby(["external_mode", "strategy", "selected_arm"], sort=False).agg(
        rounds=("iteration", "count")
    ).reset_index()
    totals = rows.groupby(["external_mode", "strategy"])["rounds"].transform("sum")
    rows["round_fraction"] = rows["rounds"] / totals.clip(lower=1)
    return add_method_metadata(rows)


def selected_arm(row: pd.Series) -> str:
    strategy = str(row["strategy"])
    config = str(row.get("config", ""))
    if strategy == "rltune_style_selector":
        match = re.search(r"rltune-style-([^:]+):", config)
        return match.group(1) if match else "unknown"
    if strategy == "sailor_style_heterogeneity_aware":
        match = re.search(r"sailor-style-([^:]+):", config)
        return match.group(1) if match else "unknown"
    if strategy == "guarded_system_portfolio":
        match = re.search(r"guarded-portfolio-(?:enable|fallback)-([^:]+):", config)
        return match.group(1) if match else "unknown"
    if strategy == "online_counter_guard_deadline_aware_sparse_flexible":
        match = re.search(r"online-guard-(enable|fallback)-", config)
        return match.group(1) if match else "unknown"
    return "fixed"


def baseline_barrier(block: pd.DataFrame) -> float:
    for strategy in ("original_sfcl_static", "sparse_flexible_static"):
        values = block.loc[block["strategy"] == strategy, "mean_barrier_ms"]
        if not values.empty:
            return float(values.iloc[0])
    return float("nan")


def add_method_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for key in ("method", "source", "artifact", "uses_first_decode_features", "notes"):
        rows[key] = rows["strategy"].map(lambda name: METHOD_META.get(name, {}).get(key, name))
    return rows


def whatif_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics.copy()
    for column in ("dispatch_seconds", "cancel_seconds", "network_response_sleep_seconds"):
        if column not in rows.columns:
            rows[column] = 0.0
    tail_after_decode = (
        rows["barrier_latency"]
        - rows["decode_latency"]
        - rows["dispatch_seconds"]
        - rows["cancel_seconds"]
    ).clip(lower=0.0)
    slow_fraction = rows["slow_workers"] / rows["n_workers"].clip(lower=1)
    no_straggler_factor = (1.0 - 0.45 * slow_fraction.clip(upper=0.75)).clip(lower=0.50)
    rows["actual_barrier_ms"] = 1000.0 * rows["barrier_latency"]
    cancel_free_seconds = np.maximum(
        rows["decode_latency"] + rows["dispatch_seconds"], rows["barrier_latency"] - rows["cancel_seconds"]
    )
    no_straggler_seconds = np.maximum(
        rows["decode_latency"] * no_straggler_factor,
        rows["dispatch_seconds"] + rows["decode_cpu_seconds"] + rows["network_response_sleep_seconds"],
    )
    fastest_safe_seconds = np.maximum(
        rows["decode_latency"] + rows["dispatch_seconds"],
        rows["barrier_latency"] - 0.50 * tail_after_decode - rows["cancel_seconds"],
    )
    rows["cancel_free_barrier_ms"] = 1000.0 * np.minimum(
        rows["barrier_latency"], cancel_free_seconds
    )
    rows["no_straggler_barrier_ms"] = 1000.0 * np.minimum(
        rows["barrier_latency"], no_straggler_seconds
    )
    rows["fastest_safe_barrier_ms"] = 1000.0 * np.minimum(
        rows["barrier_latency"], fastest_safe_seconds
    )
    rows["removable_tail_fraction"] = (
        (rows["actual_barrier_ms"] - rows["fastest_safe_barrier_ms"])
        / rows["actual_barrier_ms"].clip(lower=1e-12)
    ).clip(lower=0.0, upper=1.0)
    rows["dominant_bottleneck"] = rows.apply(dominant_bottleneck, axis=1)
    grouped = rows.groupby(["external_mode", "n_workers", "seed", "strategy"], sort=False)
    result = grouped.agg(
        actual_barrier_ms=("actual_barrier_ms", "mean"),
        no_straggler_barrier_ms=("no_straggler_barrier_ms", "mean"),
        cancel_free_barrier_ms=("cancel_free_barrier_ms", "mean"),
        fastest_safe_barrier_ms=("fastest_safe_barrier_ms", "mean"),
        removable_tail_fraction=("removable_tail_fraction", "mean"),
        dominant_bottleneck=("dominant_bottleneck", mode_value),
    ).reset_index()
    return add_method_metadata(result)


def dominant_bottleneck(row: pd.Series) -> str:
    values = {
        "dispatch": float(row.get("dispatch_seconds", 0.0)),
        "cancel": float(row.get("cancel_seconds", 0.0)),
        "network": float(row.get("network_response_sleep_seconds", 0.0)),
        "post-decode tail": max(
            0.0,
            float(row["barrier_latency"])
            - float(row["decode_latency"])
            - float(row.get("dispatch_seconds", 0.0))
            - float(row.get("cancel_seconds", 0.0)),
        ),
    }
    return max(values.items(), key=lambda item: item[1])[0]


def mode_value(values: pd.Series) -> str:
    modes = values.mode()
    if modes.empty:
        return ""
    return str(modes.iloc[0])


def write_latex_table(summary: pd.DataFrame, whatif: pd.DataFrame, out_path: Path) -> None:
    tcp = summary[summary["external_mode"] != "k3s"].copy()
    if tcp.empty:
        tcp = summary.copy()
    rows: list[dict[str, Any]] = []
    for strategy in TABLE_ORDER:
        block = tcp[tcp["strategy"] == strategy]
        if block.empty:
            continue
        weighted = block.mean(numeric_only=True)
        meta = METHOD_META[strategy]
        rows.append(
            {
                "method": meta["method"],
                "source": meta["source"],
                "first_decode": meta["uses_first_decode_features"],
                "mean": float(weighted["mean_barrier_ms"]),
                "p95": float(weighted["p95_barrier_ms"]),
                "extra": float(weighted["mean_extra_compute"]),
                "notes": meta["notes"],
            }
        )

    whatif_source = whatif[
        (whatif["external_mode"] != "k3s")
        & (whatif["strategy"].isin(["original_sfcl_static", "sparse_flexible_static"]))
    ]
    if not whatif_source.empty:
        weighted = whatif_source.mean(numeric_only=True)
        rows.append(
            {
                "method": "What-if diagnostic",
                "source": "OSDI'25 style",
                "first_decode": "Offline only",
                "mean": float(weighted["fastest_safe_barrier_ms"]),
                "p95": float(whatif_source["fastest_safe_barrier_ms"].quantile(0.95)),
                "extra": float("nan"),
                "notes": "replay upper bound, not a scheduler",
            }
        )

    lines = [
        "% Auto-generated by analyze_external_baselines.py",
        "\\begin{tabular}{@{}llllrrl@{}}",
        "\\toprule",
        "Method & Source & Artifact & FD feat. & Mean & p95 & Notes \\\\",
        "\\midrule",
    ]
    for row in rows:
        artifact = METHOD_META.get(_strategy_from_method(row["method"]), {}).get("artifact", "log replay")
        mean = f"{row['mean']:.1f}"
        p95 = f"{row['p95']:.1f}"
        note = latex_escape(str(row["notes"]))
        lines.append(
            f"{latex_escape(row['method'])} & {latex_escape(row['source'])} & "
            f"{latex_escape(artifact)} & {latex_escape(row['first_decode'])} & "
            f"{mean} & {p95} & {note} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_latex_table(matrix: pd.DataFrame, out_path: Path) -> None:
    if matrix.empty:
        out_path.write_text("% No external matrix available.\n", encoding="utf-8")
        return
    mode = "k3s" if "k3s" in set(matrix["external_mode"]) else str(matrix["external_mode"].iloc[0])
    table = matrix[matrix["external_mode"] == mode]
    table = table[table["strategy"].isin(TABLE_ORDER)]
    lines = [
        "% Auto-generated by analyze_external_baselines.py",
        "\\begin{tabular}{@{}lclcccrrrr@{}}",
        "\\toprule",
        "Method & FD & W8 m/g & W16 m/g & W24 m/g & p95 & Extra & Cancel & Reg. \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        regressions = int(
            sum(
                value
                for value in (
                    row.get("w8_regressions", 0),
                    row.get("w16_regressions", 0),
                    row.get("w24_regressions", 0),
                )
                if pd.notna(value)
            )
        )
        lines.append(
            f"{latex_escape(str(row['method']))} & "
            f"{latex_escape(str(row['uses_first_decode_features']))} & "
            f"{row['w8_mean_barrier_ms']:.1f}/{row['w8_gain_percent']:+.1f}\\% & "
            f"{row['w16_mean_barrier_ms']:.1f}/{row['w16_gain_percent']:+.1f}\\% & "
            f"{row['w24_mean_barrier_ms']:.1f}/{row['w24_gain_percent']:+.1f}\\% & "
            f"{row['avg_p95_barrier_ms']:.1f} & {row['avg_extra_compute']:.2f} & "
            f"{row['avg_cancel_ms']:.1f} & "
            f"{regressions} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_taxonomy_latex_table(out_path: Path) -> None:
    rows = [
        ("Original-SFCL", "TIT 2025", "paper", "coded worker", "No", "Yes", "No", "Online", "fixed placement"),
        ("RLTune-style", "SoCC'25", "public", "portfolio", "Yes", "No", "Indirect", "Online", "same-runtime adapter"),
        ("Sailor-style", "SOSP'25", "public", "training scheduler", "Yes", "Yes", "No", "Online", "same-runtime adapter"),
        ("Cuckoo-style", "SoCC'25", "public", "job packing", "Yes", "No", "No", "Not run", "Related Work only"),
        ("What-if diagnostic", "OSDI'25", "public", "trace analysis", "N/A", "N/A", "Offline", "Offline", "upper-bound replay"),
        ("Guarded portfolio", "this paper", "ours", "worker service", "Yes", "Yes", "Guarded", "Online", "deployed controller"),
        ("Guard-D", "this paper", "ours", "coded worker", "Yes", "Yes", "Guarded", "Online", "coded-only controller"),
    ]
    lines = [
        "% Auto-generated by analyze_external_baselines.py",
        "\\begin{tabular}{@{}lllllllll@{}}",
        "\\toprule",
        "Method & Source & Code & Layer & Het. & Cost & FD feat. & Mode & Fair mode \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _strategy_from_method(method: str) -> str:
    for strategy, meta in METHOD_META.items():
        if meta["method"] == method:
            return strategy
    return method


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


if __name__ == "__main__":
    main()
