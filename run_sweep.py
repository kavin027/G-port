from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.coded_learning_exp.experiment import ExperimentConfig, run_experiment
from src.coded_learning_exp.plotting import write_plots


ADAPTIVE_STRATEGIES = {
    "adaptive_sparse_flexible",
    "ucb_sparse_flexible",
    "adaptive_latency_only",
    "window_sparse_flexible",
    "contextual_sparse_flexible",
    "contextual_ucb_sparse_flexible",
    "worker_aware_adaptive_sparse_flexible",
    "worker_aware_ucb_sparse_flexible",
    "rank_aware_adaptive_sparse_flexible",
    "rank_aware_ucb_sparse_flexible",
    "balanced_rank_aware_ucb_sparse_flexible",
}

STATIC_FLEXIBLE_STRATEGIES = {
    "flexible_thin_static",
    "sparse_flexible_static",
    "flexible_robust_static",
    "flexible_dense_static",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sweep over straggler scenarios and sparsity levels."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("stable", "burst", "drift", "phase"),
        default=["stable", "burst", "drift", "phase"],
    )
    parser.add_argument(
        "--densities",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.03],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7])
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--features", type=int, default=500)
    parser.add_argument("--shards", type=int, default=14)
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--rounds", type=int, default=70)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--drift-period", type=int, default=25)
    parser.add_argument("--straggler-fraction", type=float, default=0.25)
    parser.add_argument("--straggler-slowdown", type=float, default=0.22)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--out", type=Path, default=Path("sweep_results"))
    parser.add_argument(
        "--write-individual-plots",
        action="store_true",
        help="Also write per-run latency/loss plots. The combined plots are always written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_summaries: list[pd.DataFrame] = []
    for scenario in args.scenarios:
        for density in args.densities:
            for seed in args.seeds:
                run_name = f"{scenario}_density{density:g}_seed{seed}"
                run_dir = args.out / run_name
                config = ExperimentConfig(
                    n_samples=args.samples,
                    n_features=args.features,
                    density=density,
                    n_shards=args.shards,
                    n_workers=args.workers,
                    rounds=args.rounds,
                    learning_rate=args.lr,
                    l2=args.l2,
                    scenario=scenario,
                    drift_period=args.drift_period,
                    straggler_fraction=args.straggler_fraction,
                    straggler_slowdown=args.straggler_slowdown,
                    burst_probability=args.burst_probability,
                    seed=seed,
                    output_dir=run_dir,
                )
                metrics, summary = run_experiment(config)
                if args.write_individual_plots:
                    write_plots(metrics, summary, run_dir)

                summary.insert(0, "run", run_name)
                summary.insert(1, "scenario", scenario)
                summary.insert(2, "density", density)
                summary.insert(3, "seed", seed)
                all_summaries.append(summary)
                best_row = summary.loc[summary["mean_latency"].idxmin()]
                print(
                    f"{run_name}: best={best_row['strategy']} "
                    f"latency={best_row['mean_latency']:.4f}"
                )

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(args.out / "combined_summary.csv", index=False)

    grouped = _aggregate_summary(combined)
    grouped.to_csv(args.out / "aggregate_by_strategy.csv", index=False)

    idea_report = _build_idea_report(combined)
    idea_report.to_csv(args.out / "idea_report.csv", index=False)

    _write_sweep_plots(grouped, idea_report, args.out)
    print(f"Wrote combined summary to {args.out / 'combined_summary.csv'}")
    print(f"Wrote idea report to {args.out / 'idea_report.csv'}")
    print(idea_report.to_string(index=False))


def _aggregate_summary(combined: pd.DataFrame) -> pd.DataFrame:
    return (
        combined.groupby(["scenario", "density", "strategy"], sort=False)
        .agg(
            mean_latency=("mean_latency", "mean"),
            p95_latency=("p95_latency", "mean"),
            final_loss=("final_loss", "mean"),
            total_wall_clock=("total_wall_clock", "mean"),
            steady_mean_latency=("steady_mean_latency", "mean"),
            steady_p95_latency=("steady_p95_latency", "mean"),
            decode_success_rate=("decode_success_rate", "mean"),
            mean_extra_compute=("mean_extra_compute", "mean"),
            second_layer_rate=("second_layer_rate", "mean"),
            latency_improvement_vs_static_flexible=(
                "latency_improvement_vs_static_flexible",
                "mean",
            ),
            latency_gap_to_oracle_static_flexible=(
                "latency_gap_to_oracle_static_flexible",
                "mean",
            ),
        )
        .reset_index()
    )


def _build_idea_report(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    grouped = combined.groupby(["scenario", "density"], sort=False)
    for (scenario, density), frame in grouped:
        means = frame.groupby("strategy").agg(
            mean_latency=("mean_latency", "mean"),
            p95_latency=("p95_latency", "mean"),
            mean_extra_compute=("mean_extra_compute", "mean"),
            success_rate=("decode_success_rate", "mean"),
            oracle_mean=("oracle_static_flexible_mean_latency", "mean"),
        )

        baseline = _metric(means, "sparse_flexible_static", "mean_latency")
        non_context = _metric(means, "adaptive_sparse_flexible", "mean_latency")
        ucb = _metric(means, "ucb_sparse_flexible", "mean_latency")
        latency_only = _metric(means, "adaptive_latency_only", "mean_latency")
        latency_only_compute = _metric(means, "adaptive_latency_only", "mean_extra_compute")
        compute_aware = _metric(means, "adaptive_sparse_flexible", "mean_latency")
        compute_aware_compute = _metric(means, "adaptive_sparse_flexible", "mean_extra_compute")
        window = _metric(means, "window_sparse_flexible", "mean_latency")
        contextual = _metric(means, "contextual_sparse_flexible", "mean_latency")
        worker_aware_static = _metric(means, "worker_aware_sparse_flexible", "mean_latency")
        worker_aware_adaptive = _metric(
            means, "worker_aware_adaptive_sparse_flexible", "mean_latency"
        )
        worker_aware_ucb = _metric(means, "worker_aware_ucb_sparse_flexible", "mean_latency")
        rank_aware_static = _metric(means, "rank_aware_sparse_flexible", "mean_latency")
        rank_aware_ucb = _metric(means, "rank_aware_ucb_sparse_flexible", "mean_latency")
        balanced_rank_static = _metric(
            means, "balanced_rank_aware_sparse_flexible", "mean_latency"
        )
        contextual_ucb = _metric(means, "contextual_ucb_sparse_flexible", "mean_latency")

        fixed_means = means[means.index.isin(STATIC_FLEXIBLE_STRATEGIES)]
        adaptive_means = means[means.index.isin(ADAPTIVE_STRATEGIES)]
        best_fixed_strategy = str(fixed_means["mean_latency"].idxmin())
        best_fixed_latency = float(fixed_means["mean_latency"].min())
        best_adaptive_strategy = str(adaptive_means["mean_latency"].idxmin())
        best_adaptive_latency = float(adaptive_means["mean_latency"].min())
        oracle = float(means["oracle_mean"].dropna().iloc[0])

        rows.append(
            {
                "scenario": scenario,
                "density": density,
                "baseline_sparse_flexible_latency": baseline,
                "best_fixed_strategy": best_fixed_strategy,
                "best_fixed_latency": best_fixed_latency,
                "best_adaptive_strategy": best_adaptive_strategy,
                "best_adaptive_latency": best_adaptive_latency,
                "best_adaptive_improvement_vs_baseline": _improvement(
                    baseline, best_adaptive_latency
                ),
                "best_adaptive_improvement_vs_best_fixed": _improvement(
                    best_fixed_latency, best_adaptive_latency
                ),
                "contextual_improvement_vs_noncontext": _improvement(
                    non_context, contextual
                ),
                "ucb_improvement_vs_epsilon_greedy": _improvement(non_context, ucb),
                "contextual_ucb_improvement_vs_contextual_ema": _improvement(
                    contextual, contextual_ucb
                ),
                "window_improvement_vs_noncontext": _improvement(non_context, window),
                "worker_aware_static_improvement_vs_static": _improvement(
                    baseline, worker_aware_static
                ),
                "rank_aware_static_improvement_vs_static": _improvement(
                    baseline, rank_aware_static
                ),
                "balanced_rank_static_improvement_vs_rank_static": _improvement(
                    rank_aware_static, balanced_rank_static
                ),
                "worker_aware_adaptive_improvement_vs_non_worker_aware": _improvement(
                    non_context, worker_aware_adaptive
                ),
                "worker_aware_ucb_improvement_vs_worker_aware_epsilon": _improvement(
                    worker_aware_adaptive, worker_aware_ucb
                ),
                "rank_aware_ucb_improvement_vs_non_worker_aware": _improvement(
                    non_context, rank_aware_ucb
                ),
                "compute_aware_latency_improvement_vs_latency_only": _improvement(
                    latency_only, compute_aware
                ),
                "compute_aware_extra_compute_reduction_vs_latency_only": _improvement(
                    latency_only_compute, compute_aware_compute
                ),
                "oracle_static_flexible_latency": oracle,
                "best_adaptive_gap_to_oracle": (best_adaptive_latency - oracle) / oracle,
                "best_adaptive_decode_success_rate": float(
                    adaptive_means.loc[best_adaptive_strategy, "success_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_sweep_plots(
    grouped: pd.DataFrame, idea_report: pd.DataFrame, output_dir: Path
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_frame = grouped.copy()
    plot_frame["case"] = (
        plot_frame["scenario"] + "\n" + plot_frame["density"].map(lambda x: f"d={x:g}")
    )
    selected = plot_frame[
        plot_frame["strategy"].isin(
            [
                "sparse_flexible_static",
                "worker_aware_sparse_flexible",
                "rank_aware_sparse_flexible",
                "adaptive_sparse_flexible",
                "ucb_sparse_flexible",
                "worker_aware_adaptive_sparse_flexible",
                "worker_aware_ucb_sparse_flexible",
                "rank_aware_ucb_sparse_flexible",
                "balanced_rank_aware_ucb_sparse_flexible",
                "window_sparse_flexible",
                "contextual_sparse_flexible",
                "contextual_ucb_sparse_flexible",
            ]
        )
    ]
    for strategy, frame in selected.groupby("strategy", sort=False):
        ax.plot(frame["case"], frame["mean_latency"], marker="o", label=strategy)
    ax.set_ylabel("Mean iteration latency")
    ax.set_title("Adaptive variants across scenarios and sparsity levels")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "sweep_strategy_latency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    report = idea_report.copy()
    report["case"] = (
        report["scenario"] + "\n" + report["density"].map(lambda x: f"d={x:g}")
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(
        report["case"],
        100.0 * report["best_adaptive_improvement_vs_baseline"],
        marker="o",
        label="best adaptive vs sparse_flexible_static",
    )
    ax.plot(
        report["case"],
        100.0 * report["contextual_improvement_vs_noncontext"],
        marker="s",
        label="contextual vs non-context adaptive",
    )
    ax.plot(
        report["case"],
        100.0 * report["window_improvement_vs_noncontext"],
        marker="^",
        label="window vs non-context adaptive",
    )
    ax.plot(
        report["case"],
        100.0 * report["worker_aware_adaptive_improvement_vs_non_worker_aware"],
        marker="D",
        label="worker-aware adaptive vs non-worker-aware",
    )
    ax.plot(
        report["case"],
        100.0 * report["ucb_improvement_vs_epsilon_greedy"],
        marker="x",
        label="UCB vs epsilon adaptive",
    )
    ax.plot(
        report["case"],
        100.0 * report["rank_aware_ucb_improvement_vs_non_worker_aware"],
        marker="P",
        label="rank-aware UCB vs non-worker-aware",
    )
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Which ideas help?")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "idea_improvements.png", dpi=180)
    plt.close(fig)


def _metric(frame: pd.DataFrame, strategy: str, column: str) -> float:
    return float(frame.loc[strategy, column])


def _improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


if __name__ == "__main__":
    main()
