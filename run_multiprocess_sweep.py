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
    DEFAULT_RUNTIME_STRATEGIES,
    MultiprocessExperimentConfig,
    run_multiprocess_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed sweep for the multi-process runtime."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 31, 43])
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--features", type=int, default=2500)
    parser.add_argument("--density", type=float, default=0.004)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=8)
    parser.add_argument("--straggler-fraction", type=float, default=0.45)
    parser.add_argument("--straggler-slowdown", type=float, default=0.08)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--out", type=Path, default=Path("runtime_sweep_highhetero"))
    parser.add_argument("--sleep-scale", type=float, default=0.06)
    parser.add_argument("--cost-scale", type=float, default=0.004)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.004)
    parser.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], default=None)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "sparse_flexible_static",
            "worker_aware_sparse_flexible",
            "rank_aware_sparse_flexible",
            "deadline_aware_sparse_flexible",
        ],
        help="Runtime strategies to compare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base_config = MultiprocessExperimentConfig(
        n_samples=args.samples,
        n_features=args.features,
        density=args.density,
        n_shards=args.shards,
        n_workers=args.workers,
        rounds=args.rounds,
        learning_rate=args.learning_rate,
        l2=args.l2,
        scenario=args.scenario,
        drift_period=args.drift_period,
        straggler_fraction=args.straggler_fraction,
        straggler_slowdown=args.straggler_slowdown,
        burst_probability=args.burst_probability,
        output_dir=args.out,
        strategy_names=tuple(args.strategies or DEFAULT_RUNTIME_STRATEGIES),
        sleep_scale=args.sleep_scale,
        cost_scale=args.cost_scale,
        cancel_poll_seconds=args.cancel_poll_seconds,
        start_method=args.start_method,
    )

    summaries: list[pd.DataFrame] = []
    for seed in args.seeds:
        run_dir = args.out / f"seed_{seed}"
        print(f"Running seed={seed} -> {run_dir}")
        config = replace(base_config, seed=seed, output_dir=run_dir)
        _, summary = run_multiprocess_experiment(config)
        summary.insert(0, "seed", seed)
        summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.out / "combined_runtime_summary.csv", index=False)
    aggregate = _aggregate(combined)
    aggregate.to_csv(args.out / "aggregate_runtime_summary.csv", index=False)
    print("\nAggregate runtime summary")
    print(aggregate.to_string(index=False))


def _aggregate(combined: pd.DataFrame) -> pd.DataFrame:
    return (
        combined.groupby("strategy", sort=False)
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
        )
        .reset_index()
    )


if __name__ == "__main__":
    main()
