"""Replay performance-mode fallback choices from K3s per-round logs.

The deployed guarded portfolio in ``run_k8s_network_experiment.py`` uses a
static-coded fallback.  This script keeps the enabled guarded-portfolio rounds
unchanged and changes only rounds whose config records ``fallback-static``.  It
then compares three fallback choices:

* static fallback: the deployed guarded-portfolio trace;
* speed fallback: replace fallback rounds with speed-aware uncoded latency;
* best-safe replay: replace fallback rounds with the cheaper observed latency
  between speed-aware uncoded and static sparse-flexible placement in the
  paired round.

The replay is diagnostic rather than a new controller run.  It gives a paired
upper bound for a safe-baseline fallback over already collected K3s traces; the
online runtime separately exposes ``--portfolio-fallback best_safe`` to choose
the safe fallback before a round from predictor features.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(r"majorrev_k8s_w(?P<workers>\d+)_seed(?P<seed>\d+)")


POLICY_LABELS = {
    "static": "Static coded",
    "speed_aware_uncoded": "Speed uncoded",
    "portfolio": "Portfolio",
    "guard_static_fb": "Guarded portfolio, static fallback",
    "guard_speed_fb": "Guarded portfolio, speed fallback",
    "guard_best_safe_fb": "Guarded portfolio, best-safe replay",
}


def _percent_gain(baseline: float, value: float) -> float:
    if baseline == 0:
        return 0.0
    return 100.0 * (baseline - value) / baseline


def _bootstrap_ci(values: pd.Series, *, seed: int = 20260701) -> tuple[float, float]:
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


def load_replay(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        parsed = _parse_run_dir(run_dir)
        if parsed is None:
            continue
        workers, seed = parsed
        metrics_path = run_dir / "network_metrics.csv"
        if not metrics_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        required = {
            "sparse_flexible_static": "static",
            "speed_aware_uncoded": "speed",
            "system_portfolio": "portfolio",
            "guarded_system_portfolio": "guard",
        }
        pieces: list[pd.DataFrame] = []
        missing = []
        for strategy, label in required.items():
            subset = metrics.loc[metrics["strategy"] == strategy].copy()
            if subset.empty:
                missing.append(strategy)
                continue
            subset = subset[["iteration", "barrier_latency", "config"]].rename(
                columns={
                    "barrier_latency": f"{label}_barrier_latency",
                    "config": f"{label}_config",
                }
            )
            pieces.append(subset)
        if missing:
            raise SystemExit(f"{run_dir} is missing strategies: {', '.join(missing)}")

        paired = pieces[0]
        for piece in pieces[1:]:
            paired = paired.merge(piece, on="iteration", how="inner", validate="one_to_one")
        if paired.empty:
            raise SystemExit(f"{run_dir} has no paired iterations")

        guard_config = paired["guard_config"].astype(str)
        paired["guard_fallback"] = guard_config.str.contains("fallback-static", na=False)
        paired["guard_enabled"] = guard_config.str.contains("guarded-portfolio-enable", na=False)
        paired["guard_static_fb"] = paired["guard_barrier_latency"]
        paired["guard_speed_fb"] = np.where(
            paired["guard_fallback"],
            paired["speed_barrier_latency"],
            paired["guard_barrier_latency"],
        )
        paired["guard_best_safe_fb"] = np.where(
            paired["guard_fallback"],
            np.minimum(paired["speed_barrier_latency"], paired["static_barrier_latency"]),
            paired["guard_barrier_latency"],
        )
        paired["workers"] = workers
        paired["seed"] = seed
        paired["run"] = run_dir.name
        detail_rows.extend(paired.to_dict(orient="records"))

        baseline = float(paired["static_barrier_latency"].mean())
        fallback_rate = float(paired["guard_fallback"].mean())
        enable_rate = float(paired["guard_enabled"].mean())
        for policy, column in [
            ("static", "static_barrier_latency"),
            ("speed_aware_uncoded", "speed_barrier_latency"),
            ("portfolio", "portfolio_barrier_latency"),
            ("guard_static_fb", "guard_static_fb"),
            ("guard_speed_fb", "guard_speed_fb"),
            ("guard_best_safe_fb", "guard_best_safe_fb"),
        ]:
            value = float(paired[column].mean())
            seed_rows.append(
                {
                    "workers": workers,
                    "seed": seed,
                    "run": run_dir.name,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "mean_barrier_latency": value,
                    "mean_barrier_ms": 1000.0 * value,
                    "gain_vs_static_pct": _percent_gain(baseline, value),
                    "fallback_rate": fallback_rate,
                    "enable_rate": enable_rate,
                    "iterations": len(paired),
                }
            )

    if not seed_rows:
        raise SystemExit(f"No K3s replay runs found under {root}")
    return pd.DataFrame(seed_rows), pd.DataFrame(detail_rows)


def summarize(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (workers, policy, label), subset in seed_summary.groupby(
        ["workers", "policy", "policy_label"], sort=True
    ):
        low, high = _bootstrap_ci(subset["gain_vs_static_pct"])
        rows.append(
            {
                "workers": workers,
                "policy": policy,
                "policy_label": label,
                "n_seeds": int(subset["seed"].nunique()),
                "mean_barrier_ms": float(subset["mean_barrier_ms"].mean()),
                "gain_vs_static_pct": float(subset["gain_vs_static_pct"].mean()),
                "gain_ci_low": low,
                "gain_ci_high": high,
                "fallback_rate_pct": 100.0 * float(subset["fallback_rate"].mean()),
                "enable_rate_pct": 100.0 * float(subset["enable_rate"].mean()),
            }
        )
    order = {
        "static": 0,
        "speed_aware_uncoded": 1,
        "portfolio": 2,
        "guard_static_fb": 3,
        "guard_speed_fb": 4,
        "guard_best_safe_fb": 5,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out["policy"].map(order)
    return out.sort_values(["workers", "_order"]).drop(columns=["_order"])


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    keep = summary[summary["policy"].str.startswith("guard_")].copy()
    pivot = keep.pivot(index="workers", columns="policy")

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{Paired upper-bound fallback diagnostic for the performance-mode guarded portfolio.}",
        r"  \label{tab:portfolio-fallback}",
        r"  \begin{tabular}{@{}rrrrr@{}}",
        r"    \toprule",
        r"    W & Fallback & Static fb & Speed fb & Best-safe replay \\",
        r"      & rounds & gain & gain & gain \\",
        r"    \midrule",
    ]
    for workers in sorted(keep["workers"].unique()):
        fallback = pivot.loc[workers, ("fallback_rate_pct", "guard_static_fb")]
        static_gain = pivot.loc[workers, ("gain_vs_static_pct", "guard_static_fb")]
        speed_gain = pivot.loc[workers, ("gain_vs_static_pct", "guard_speed_fb")]
        best_gain = pivot.loc[workers, ("gain_vs_static_pct", "guard_best_safe_fb")]
        lines.append(
            f"    {workers:d} & {fallback:.1f}\\% & {static_gain:.1f}\\% "
            f"& {speed_gain:.1f}\\% & {best_gain:.1f}\\% \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_seed_latex_table(seed_summary: pd.DataFrame, path: Path) -> None:
    keep = seed_summary[
        seed_summary["policy"].isin(
            ["static", "portfolio", "guard_static_fb", "guard_best_safe_fb"]
        )
    ].copy()
    wide = (
        keep.pivot_table(
            index=["workers", "seed"],
            columns="policy",
            values="mean_barrier_ms",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["workers", "seed"])
    )
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2pt}",
        r"  \caption{Per-seed performance-mode K3s barrier latencies. Latencies are in ms; best-safe replay is an offline paired diagnostic over fallback rounds.}",
        r"  \label{tab:k8s-performance-seeds}",
        r"  \begin{tabular}{@{}rrrrrr@{}}",
        r"    \toprule",
        r"    W & Seed & Static & Portfolio & G-port & Best-safe \\",
        r"    \midrule",
    ]
    for _, row in wide.iterrows():
        lines.append(
            "    "
            f"{int(row['workers']):2d} & {int(row['seed']):2d} & "
            f"{row['static']:.1f} & {row['portfolio']:.1f} & "
            f"{row['guard_static_fb']:.1f} & {row['guard_best_safe_fb']:.1f} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "W": int(row["workers"]),
                "policy": row["policy_label"],
                "barrier_ms": f"{row['mean_barrier_ms']:.2f}",
                "gain_pct": (
                    f"{row['gain_vs_static_pct']:.1f} "
                    f"[{row['gain_ci_low']:.1f},{row['gain_ci_high']:.1f}]"
                ),
                "fallback_pct": f"{row['fallback_rate_pct']:.1f}",
            }
        )
    table = pd.DataFrame(rows).to_markdown(index=False)
    text = (
        "# Performance-mode fallback replay\n\n"
        "Only guarded-portfolio rounds whose saved config contains "
        "`fallback-static` are replayed with alternate fallbacks; enabled rounds "
        "keep the deployed guarded-portfolio latency.  `best-safe` uses the "
        "cheaper observed safe baseline in the paired round, so this is an "
        "upper-bound diagnostic rather than a deployed controller. Gains are "
        "paired by seed against static sparse-flexible placement.\n\n"
        f"{table}\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("majorrev_k8s_diagnostics"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or args.root
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_summary, details = load_replay(args.root)
    summary = summarize(seed_summary)

    seed_summary.to_csv(out_dir / "performance_fallback_seed_summary.csv", index=False)
    details.to_csv(out_dir / "performance_fallback_details.csv", index=False)
    summary.to_csv(out_dir / "performance_fallback_summary.csv", index=False)
    write_latex_table(summary, out_dir / "performance_fallback_table.tex")
    write_per_seed_latex_table(seed_summary, out_dir / "performance_fallback_per_seed_table.tex")
    write_markdown(summary, out_dir / "performance_fallback_report.md")

    print(summary.to_string(index=False))
    print(f"\nWrote performance fallback replay files to {out_dir}")


if __name__ == "__main__":
    main()
