from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.coded_learning_exp.multiprocess_runtime import (
    MultiprocessExperimentConfig,
    run_multiprocess_experiment,
)


DEFAULT_STRATEGIES = (
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run worker-count scaling experiments for the multi-process runtime."
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 31, 43])
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--features", type=int, default=2500)
    parser.add_argument("--density", type=float, default=0.004)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument(
        "--scale-shards-with-workers",
        action="store_true",
        help="Use n_shards = n_workers for each scaling point.",
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=8)
    parser.add_argument("--straggler-fraction", type=float, default=0.45)
    parser.add_argument("--straggler-slowdown", type=float, default=0.08)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--out", type=Path, default=Path("runtime_worker_scaling"))
    parser.add_argument("--sleep-scale", type=float, default=0.06)
    parser.add_argument("--cost-scale", type=float, default=0.004)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.004)
    parser.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], default=None)
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base_config = MultiprocessExperimentConfig(
        n_samples=args.samples,
        n_features=args.features,
        density=args.density,
        n_shards=args.shards,
        rounds=args.rounds,
        learning_rate=args.learning_rate,
        l2=args.l2,
        scenario=args.scenario,
        drift_period=args.drift_period,
        straggler_fraction=args.straggler_fraction,
        straggler_slowdown=args.straggler_slowdown,
        burst_probability=args.burst_probability,
        output_dir=args.out,
        strategy_names=tuple(args.strategies),
        sleep_scale=args.sleep_scale,
        cost_scale=args.cost_scale,
        cancel_poll_seconds=args.cancel_poll_seconds,
        start_method=args.start_method,
    )

    all_summaries: list[pd.DataFrame] = []
    for n_workers in args.workers:
        n_shards = n_workers if args.scale_shards_with_workers else args.shards
        for seed in args.seeds:
            run_dir = args.out / f"workers_{n_workers}_shards_{n_shards}" / f"seed_{seed}"
            print(
                f"Running workers={n_workers}, shards={n_shards}, "
                f"seed={seed} -> {run_dir}"
            )
            config = replace(
                base_config,
                n_workers=n_workers,
                n_shards=n_shards,
                seed=seed,
                output_dir=run_dir,
            )
            _, summary = run_multiprocess_experiment(config)
            summary.insert(0, "seed", seed)
            summary.insert(0, "shards", n_shards)
            summary.insert(0, "workers", n_workers)
            all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(args.out / "combined_worker_scaling.csv", index=False)
    aggregate = aggregate_worker_scaling(combined)
    aggregate.to_csv(args.out / "aggregate_worker_scaling.csv", index=False)
    comparison = comparison_vs_baseline(combined)
    comparison.to_csv(args.out / "comparison_vs_sparse_flexible.csv", index=False)

    print("\nAggregate worker scaling summary")
    print(aggregate.to_string(index=False))
    print("\nComparison vs sparse-flexible")
    print(comparison.to_string(index=False))


def aggregate_worker_scaling(combined: pd.DataFrame) -> pd.DataFrame:
    return (
        combined.groupby(["workers", "shards", "strategy"], sort=False)
        .agg(
            mean_decode_latency=("mean_decode_latency", "mean"),
            p95_decode_latency=("p95_decode_latency", "mean"),
            mean_barrier_latency=("mean_barrier_latency", "mean"),
            decode_improvement=("decode_latency_improvement_vs_sparse_flexible", "mean"),
            p95_improvement=("p95_decode_latency_improvement_vs_sparse_flexible", "mean"),
            barrier_improvement=("barrier_latency_improvement_vs_sparse_flexible", "mean"),
            scheduler_seconds=("mean_scheduler_seconds", "mean"),
            extra_compute=("mean_extra_compute", "mean"),
            selected_rows=("mean_selected_rows", "mean"),
            completed_rows=("mean_completed_rows", "mean"),
        )
        .reset_index()
    )


def comparison_vs_baseline(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (workers, shards), worker_df in combined.groupby(["workers", "shards"], sort=False):
        baseline = worker_df[worker_df["strategy"] == "sparse_flexible_static"]
        if baseline.empty:
            continue
        baseline_by_seed = baseline.set_index("seed")
        for _, candidate in worker_df.iterrows():
            strategy = str(candidate["strategy"])
            seed = int(candidate["seed"])
            base = baseline_by_seed.loc[seed]
            rows.append(
                {
                    "workers": int(workers),
                    "shards": int(shards),
                    "seed": seed,
                    "strategy": strategy,
                    "mean_decode_improvement": (
                        float(base["mean_decode_latency"]) - float(candidate["mean_decode_latency"])
                    )
                    / float(base["mean_decode_latency"]),
                    "p95_decode_improvement": (
                        float(base["p95_decode_latency"]) - float(candidate["p95_decode_latency"])
                    )
                    / float(base["p95_decode_latency"]),
                    "barrier_improvement": (
                        float(base["mean_barrier_latency"]) - float(candidate["mean_barrier_latency"])
                    )
                    / float(base["mean_barrier_latency"]),
                    "scheduler_overhead_ms": float(candidate["mean_scheduler_seconds"]) * 1000.0,
                    "extra_compute": float(candidate["mean_extra_compute"]),
                    "selected_rows": float(candidate["mean_selected_rows"]),
                }
            )

    comparisons = pd.DataFrame.from_records(rows)
    return (
        comparisons.groupby(["workers", "shards", "strategy"], sort=False)
        .agg(
            mean_decode_improvement=("mean_decode_improvement", "mean"),
            p95_decode_improvement=("p95_decode_improvement", "mean"),
            barrier_improvement=("barrier_improvement", "mean"),
            scheduler_overhead_ms=("scheduler_overhead_ms", "mean"),
            extra_compute=("extra_compute", "mean"),
            selected_rows=("selected_rows", "mean"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    main()
