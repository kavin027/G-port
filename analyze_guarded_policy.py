from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the diagnostic guard for first-decode scheduling."
    )
    parser.add_argument("--out", type=Path, default=Path("guarded_policy_diagnostics"))
    parser.add_argument(
        "--mismatch-threshold",
        type=float,
        default=0.75,
        help="Minimum decode-speed mismatch used by the conservative real-data guard.",
    )
    parser.add_argument(
        "--prefix-tolerance",
        type=float,
        default=1e-9,
        help="Allowed completed-row prefix growth before falling back.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    rows.extend(_alignment_rows(args.mismatch_threshold, args.prefix_tolerance))
    rows.extend(_scaling_rows(args.prefix_tolerance))
    rows.extend(_docker_rows(args.prefix_tolerance))
    rows.extend(_embedding_rows(args.prefix_tolerance))
    rows.extend(_network_rows())

    regimes = pd.DataFrame(rows)
    regimes.to_csv(args.out / "guarded_policy_regime_summary.csv", index=False)

    aggregate = _aggregate(regimes)
    aggregate.to_csv(args.out / "guarded_policy_aggregate.csv", index=False)
    ablation = _ablation(regimes, args.mismatch_threshold, args.prefix_tolerance)
    ablation.to_csv(args.out / "guard_ablation_summary.csv", index=False)
    chronological = _chronological_replay(args.prefix_tolerance)
    chronological.to_csv(args.out / "chronological_guard_replay.csv", index=False)

    _plot(regimes, args.out)
    _plot_mechanism_trace(args.out)
    _write_report(
        regimes,
        aggregate,
        ablation,
        chronological,
        args.out / "guarded_policy_report.md",
    )

    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nGuard ablation:")
    print(ablation.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    if not chronological.empty:
        print("\nChronological replay:")
        print(chronological.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nWrote guarded-policy diagnostics to {args.out}")


def _alignment_rows(mismatch_threshold: float, prefix_tolerance: float) -> list[dict[str, object]]:
    path = Path("realdata_alignment_diagnostics") / "alignment_mechanism_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for (dataset, mode), group in df.groupby(["dataset", "alignment_mode"], sort=False):
        static = _single(group, "sparse_flexible_static")
        rank = _single(group, "rank_aware_sparse_flexible")
        mismatch = float(rank["decode_speed_mismatch"])
        prefix_delta = float(rank["completed_rows_delta"])
        enable = mismatch >= mismatch_threshold and prefix_delta <= prefix_tolerance
        rank_gain = 100.0 * float(rank["p95_improvement"])
        rows.append(
            {
                "suite": "real-data alignment",
                "regime": f"{dataset}-{mode}",
                "baseline": "static coded",
                "candidate": "rank-aware coded",
                "guard_action": "enable" if enable else "fallback",
                "guard_policy": "rank-aware coded" if enable else "static coded",
                "decode_speed_mismatch": mismatch,
                "completed_prefix_delta": prefix_delta,
                "always_on_p95_gain_pct": rank_gain,
                "guarded_p95_gain_pct": rank_gain if enable else 0.0,
                "baseline_p95_ms": 1000.0 * float(static["p95_decode_latency"]),
            }
        )
    return rows


def _scaling_rows(prefix_tolerance: float) -> list[dict[str, object]]:
    path = Path("runtime_scaling_diagnostics") / "runtime_scaling_diagnostic_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for (experiment, workers, shards), group in df.groupby(
        ["experiment", "workers", "shards"], sort=False
    ):
        static = _single(group, "sparse_flexible_static")
        rank = _single(group, "rank_aware_sparse_flexible")
        prefix_delta = float(rank["completed_fraction"]) - float(static["completed_fraction"])
        enable = prefix_delta <= prefix_tolerance
        rank_gain = 100.0 * float(rank["p95_decode_improvement"])
        rows.append(
            {
                "suite": "worker scaling",
                "regime": f"{experiment}-w{int(workers)}",
                "baseline": "static coded",
                "candidate": "rank-aware coded",
                "guard_action": "enable" if enable else "fallback",
                "guard_policy": "rank-aware coded" if enable else "static coded",
                "decode_speed_mismatch": np.nan,
                "completed_prefix_delta": prefix_delta,
                "always_on_p95_gain_pct": rank_gain,
                "guarded_p95_gain_pct": rank_gain if enable else 0.0,
                "baseline_p95_ms": np.nan,
            }
        )
    return rows


def _docker_rows(prefix_tolerance: float) -> list[dict[str, object]]:
    root = Path("local_docker_container_sweep")
    frames = []
    for path in sorted(root.glob("w*_run*")):
        summary = path / "network_summary.csv"
        if not summary.exists():
            continue
        frame = pd.read_csv(summary)
        frame["run"] = path.name
        frame["n_workers"] = int(path.name.split("_", 1)[0].removeprefix("w"))
        frames.append(frame)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for workers, group in df.groupby("n_workers", sort=True):
        aggregate = group.groupby("strategy", sort=False).mean(numeric_only=True)
        static = aggregate.loc["sparse_flexible_static"]
        rank = aggregate.loc["rank_aware_sparse_flexible"]
        prefix_delta = float(rank["mean_completed_rows"]) - float(static["mean_completed_rows"])
        enable = prefix_delta <= prefix_tolerance
        rank_gain = _improvement_pct(float(static["p95_decode_latency"]), float(rank["p95_decode_latency"]))
        rows.append(
            {
                "suite": "local Docker TCP",
                "regime": f"docker-w{int(workers)}",
                "baseline": "static coded",
                "candidate": "rank-aware coded",
                "guard_action": "enable" if enable else "fallback",
                "guard_policy": "rank-aware coded" if enable else "static coded",
                "decode_speed_mismatch": np.nan,
                "completed_prefix_delta": prefix_delta,
                "always_on_p95_gain_pct": rank_gain,
                "guarded_p95_gain_pct": rank_gain if enable else 0.0,
                "baseline_p95_ms": 1000.0 * float(static["p95_decode_latency"]),
            }
        )
    return rows


def _network_rows() -> list[dict[str, object]]:
    specs = [
        (
            "network TCP stress",
            "tcp-stress-model",
            Path("network_wan_common_stream_newserver_diagnostics") / "aggregate_vs_speed_aware_uncoded.csv",
        ),
        (
            "network TCP stress",
            "tcp-fresh-direct",
            Path("direct_tcp_fresh_server_diagnostics") / "aggregate_vs_speed_aware_uncoded.csv",
        ),
    ]
    rows: list[dict[str, object]] = []
    for suite, regime, path in specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        speed = _single(df, "speed_aware_uncoded")
        rank = _single(df, "rank_aware_sparse_flexible")
        portfolio = _single(df, "system_portfolio")
        rank_gain = 100.0 * float(
            rank["p95_decode_latency_improvement_vs_speed_aware_uncoded_mean"]
        )
        guarded_gain = 100.0 * float(
            portfolio["p95_decode_latency_improvement_vs_speed_aware_uncoded_mean"]
        )
        rows.append(
            {
                "suite": suite,
                "regime": regime,
                "baseline": "speed-aware uncoded",
                "candidate": "rank-aware coded",
                "guard_action": "portfolio",
                "guard_policy": "system portfolio",
                "decode_speed_mismatch": np.nan,
                "completed_prefix_delta": np.nan,
                "always_on_p95_gain_pct": rank_gain,
                "guarded_p95_gain_pct": guarded_gain,
                "baseline_p95_ms": float(speed["p95_decode_latency_ms_mean"]),
            }
        )
    return rows


def _embedding_rows(prefix_tolerance: float) -> list[dict[str, object]]:
    path = Path("embedding_microbenchmark_diagnostics") / "aggregate_embedding_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    static = _single(df, "sparse_flexible_static")
    rank = _single(df, "rank_aware_sparse_flexible")
    prefix_delta = float(rank["selected_rows_delta_mean"])
    enable = prefix_delta <= prefix_tolerance
    rank_gain = 100.0 * float(rank["p95_latency_gain_mean"])
    return [
        {
            "suite": "embedding update",
            "regime": "zipfian-embedding-w24",
            "baseline": "static coded",
            "candidate": "rank-aware coded",
            "guard_action": "enable" if enable else "fallback",
            "guard_policy": "rank-aware coded" if enable else "static coded",
            "decode_speed_mismatch": np.nan,
            "completed_prefix_delta": prefix_delta,
            "always_on_p95_gain_pct": rank_gain,
            "guarded_p95_gain_pct": rank_gain if enable else 0.0,
            "baseline_p95_ms": 1000.0 * float(static["p95_latency_mean"]),
        }
    ]


def _aggregate(regimes: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        regimes.groupby("suite", sort=False)
        .agg(
            regimes=("regime", "count"),
            enabled=("guard_action", lambda values: int((values != "fallback").sum())),
            always_on_p95_gain_pct=("always_on_p95_gain_pct", "mean"),
            guarded_p95_gain_pct=("guarded_p95_gain_pct", "mean"),
            always_on_negative_regimes=("always_on_p95_gain_pct", lambda values: int((values < 0.0).sum())),
            guarded_negative_regimes=("guarded_p95_gain_pct", lambda values: int((values < 0.0).sum())),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "suite": "overall",
                "regimes": len(regimes),
                "enabled": int((regimes["guard_action"] != "fallback").sum()),
                "always_on_p95_gain_pct": float(regimes["always_on_p95_gain_pct"].mean()),
                "guarded_p95_gain_pct": float(regimes["guarded_p95_gain_pct"].mean()),
                "always_on_negative_regimes": int((regimes["always_on_p95_gain_pct"] < 0.0).sum()),
                "guarded_negative_regimes": int((regimes["guarded_p95_gain_pct"] < 0.0).sum()),
            }
        ]
    )
    return pd.concat([grouped, overall], ignore_index=True)


def _ablation(regimes: pd.DataFrame, mismatch_threshold: float, prefix_tolerance: float) -> pd.DataFrame:
    counter_regimes = regimes[regimes["suite"] != "network TCP stress"].copy()
    policies = [
        ("always-on rank-aware", "always"),
        ("mismatch-only guard", "mismatch"),
        ("prefix-only guard", "prefix"),
        ("full counter guard", "full"),
    ]
    rows: list[dict[str, object]] = []
    for label, policy in policies:
        enabled = []
        gains = []
        for _, row in counter_regimes.iterrows():
            mismatch = row["decode_speed_mismatch"]
            prefix = row["completed_prefix_delta"]
            mismatch_ok = True if pd.isna(mismatch) else float(mismatch) >= mismatch_threshold
            prefix_ok = True if pd.isna(prefix) else float(prefix) <= prefix_tolerance
            if policy == "always":
                enable = True
            elif policy == "mismatch":
                enable = mismatch_ok
            elif policy == "prefix":
                enable = prefix_ok
            elif policy == "full":
                enable = mismatch_ok and prefix_ok
            else:
                raise ValueError(policy)
            enabled.append(enable)
            gains.append(float(row["always_on_p95_gain_pct"]) if enable else 0.0)
        rows.append(
            {
                "policy": label,
                "regimes": len(counter_regimes),
                "enabled": int(sum(enabled)),
                "mean_p95_gain_pct": float(np.mean(gains)),
                "negative_p95_regimes": int(sum(gain < 0.0 for gain in gains)),
            }
        )
    return pd.DataFrame(rows)


def _chronological_replay(prefix_tolerance: float) -> pd.DataFrame:
    root = Path("local_docker_container_sweep")
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("w*_run*")):
        metrics_path = path / "network_metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        if not {"sparse_flexible_static", "rank_aware_sparse_flexible"}.issubset(
            set(metrics["strategy"])
        ):
            continue
        static = metrics[metrics["strategy"] == "sparse_flexible_static"].sort_values("iteration")
        rank = metrics[metrics["strategy"] == "rank_aware_sparse_flexible"].sort_values("iteration")
        paired = static.merge(rank, on="iteration", suffixes=("_static", "_rank"))
        if len(paired) < 2:
            continue
        calibration_count = max(1, int(np.ceil(0.2 * len(paired))))
        calibration = paired.iloc[:calibration_count]
        evaluation = paired.iloc[calibration_count:]
        prefix_delta = float(
            (calibration["completed_rows_rank"] - calibration["completed_rows_static"]).mean()
        )
        enable = prefix_delta <= prefix_tolerance
        static_p95 = float(np.quantile(evaluation["decode_latency_static"], 0.95))
        rank_p95 = float(np.quantile(evaluation["decode_latency_rank"], 0.95))
        always_gain = _improvement_pct(static_p95, rank_p95)
        guarded_gain = always_gain if enable else 0.0
        rows.append(
            {
                "run": path.name,
                "workers": int(path.name.split("_", 1)[0].removeprefix("w")),
                "iterations": len(paired),
                "calibration_iterations": calibration_count,
                "calibration_prefix_delta": prefix_delta,
                "guard_action": "enable" if enable else "fallback",
                "always_on_p95_gain_pct": always_gain,
                "chronological_guard_p95_gain_pct": guarded_gain,
                "eval_static_p95_ms": 1000.0 * static_p95,
                "eval_rank_p95_ms": 1000.0 * rank_p95,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run",
                "workers",
                "iterations",
                "calibration_iterations",
                "calibration_prefix_delta",
                "guard_action",
                "always_on_p95_gain_pct",
                "chronological_guard_p95_gain_pct",
                "eval_static_p95_ms",
                "eval_rank_p95_ms",
            ]
        )
    return pd.DataFrame(rows)


def _plot(regimes: pd.DataFrame, out_dir: Path) -> None:
    aggregate = _aggregate(regimes)
    aggregate = aggregate[aggregate["suite"] != "overall"]
    x = np.arange(len(aggregate))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    ax.bar(
        x - width / 2,
        aggregate["always_on_p95_gain_pct"],
        width=width,
        label="Always-on rank",
        color="#4c72b0",
    )
    ax.bar(
        x + width / 2,
        aggregate["guarded_p95_gain_pct"],
        width=width,
        label="Guarded policy",
        color="#55a868",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, aggregate["suite"], rotation=12, ha="right")
    ax.set_ylabel("Mean p95 gain (%)")
    ax.set_title("Guarded policy across diagnostic regimes")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir / "guarded_policy_p95_by_suite")


def _plot_mechanism_trace(out_dir: Path) -> None:
    path = Path("runtime_realdata_alignment_bjb1") / "rcv1_anti_seed11" / "runtime_metrics.csv"
    if not path.exists():
        return
    metrics = pd.read_csv(path)
    static = metrics[metrics["strategy"] == "sparse_flexible_static"].sort_values("iteration")
    rank = metrics[metrics["strategy"] == "rank_aware_sparse_flexible"].sort_values("iteration")
    if static.empty or rank.empty:
        return

    static_p95 = float(np.quantile(static["decode_latency"], 0.95))
    rank_p95 = float(np.quantile(rank["decode_latency"], 0.95))
    p95_gain = _improvement_pct(static_p95, rank_p95)
    prefix_delta = float(rank["completed_rows"].mean() - static["completed_rows"].mean())

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True, constrained_layout=True)
    axes[0].plot(
        static["iteration"],
        1000.0 * static["decode_latency"],
        marker="o",
        linewidth=1.4,
        markersize=3.5,
        label="Static coded",
        color="#4c72b0",
    )
    axes[0].plot(
        rank["iteration"],
        1000.0 * rank["decode_latency"],
        marker="s",
        linewidth=1.4,
        markersize=3.5,
        label="Rank-aware coded",
        color="#55a868",
    )
    axes[0].set_ylabel("First-decode\nlatency (ms)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_title(f"Representative mechanism trace: p95 gain {p95_gain:.1f}%")

    axes[1].plot(
        static["iteration"],
        static["completed_rows"],
        marker="o",
        linewidth=1.4,
        markersize=3.5,
        label="Static coded",
        color="#4c72b0",
    )
    axes[1].plot(
        rank["iteration"],
        rank["completed_rows"],
        marker="s",
        linewidth=1.4,
        markersize=3.5,
        label="Rank-aware coded",
        color="#55a868",
    )
    axes[1].axhline(static["completed_rows"].mean(), color="#4c72b0", linestyle="--", linewidth=0.9)
    axes[1].axhline(rank["completed_rows"].mean(), color="#55a868", linestyle="--", linewidth=0.9)
    axes[1].text(
        0.01,
        0.04,
        f"Mean prefix delta {prefix_delta:+.2f} rows",
        transform=axes[1].transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8"},
    )
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Completed rows\nat first decode")
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_axisbelow(True)
    _save(fig, out_dir / "mechanism_trace_prefix_latency")


def _write_report(
    regimes: pd.DataFrame,
    aggregate: pd.DataFrame,
    ablation: pd.DataFrame,
    chronological: pd.DataFrame,
    path: Path,
) -> None:
    overall = aggregate[aggregate["suite"] == "overall"].iloc[0]
    lines = [
        "# Guarded Policy Diagnostics",
        "",
        "The guard enables rank-aware coded placement only when runtime diagnostics indicate that "
        "the first-decode completed-row prefix is not growing.  For controlled real-data "
        "alignment, the guard is conservative and also requires strong decode-speed mismatch.  "
        "For network-constrained TCP, the guard selects the existing system portfolio because "
        "transfer and cancellation costs make speed-only placement a boundary case.",
        "",
        "## Overall",
        "",
        (
            f"Across {int(overall['regimes'])} aggregate regimes, always-on rank-aware placement "
            f"has mean p95 gain {overall['always_on_p95_gain_pct']:.1f}%, while the guarded "
            f"policy has mean p95 gain {overall['guarded_p95_gain_pct']:.1f}%."
        ),
        (
            f"Negative p95 regimes drop from {int(overall['always_on_negative_regimes'])} to "
            f"{int(overall['guarded_negative_regimes'])}."
        ),
        "",
        "## Aggregate by suite",
        "",
        aggregate.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Guard ablation",
        "",
        ablation.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Chronological guard replay",
        "",
        (
            "The chronological replay uses the first 20% of iterations in each local Docker "
            "run to estimate completed-prefix growth, locks the guard action, and reports "
            "p95 gain on the remaining iterations."
        ),
        "",
        chronological.to_markdown(index=False, floatfmt=".2f") if not chronological.empty else "No replay data.",
        "",
        "## Regime decisions",
        "",
        regimes.to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _single(group: pd.DataFrame, strategy: str) -> pd.Series:
    rows = group[group["strategy"] == strategy]
    if rows.empty:
        raise KeyError(f"Missing strategy {strategy}")
    return rows.iloc[0]


def _improvement_pct(baseline: float, value: float) -> float:
    return 100.0 * (baseline - value) / max(baseline, 1e-12)


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    for paper_dir in [Path("paper") / "socc26" / "figures", Path("paper") / "socc26_zh" / "figures"]:
        paper_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_dir / stem.with_suffix(".png").name, dpi=220)


if __name__ == "__main__":
    main()
