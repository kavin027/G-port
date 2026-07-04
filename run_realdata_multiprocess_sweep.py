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
    run_multiprocess_problem,
)
from src.coded_learning_exp.realdata import make_libsvm_ridge_problem


DEFAULT_STRATEGIES = (
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed real-data multi-process runtime sweep."
    )
    parser.add_argument("--dataset", default="a9a")
    parser.add_argument("--url", default=None)
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data") / "libsvm")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-normalize-rows", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 31, 43])
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=8)
    parser.add_argument("--straggler-fraction", type=float, default=0.45)
    parser.add_argument("--straggler-slowdown", type=float, default=0.08)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--out", type=Path, default=Path("runtime_realdata_sweep"))
    parser.add_argument("--sleep-scale", type=float, default=0.06)
    parser.add_argument("--cost-scale", type=float, default=0.004)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.004)
    parser.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], default=None)
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []

    for seed in args.seeds:
        run_dir = args.out / f"{args.dataset}_seed_{seed}"
        print(f"Running dataset={args.dataset}, seed={seed} -> {run_dir}")
        problem = make_libsvm_ridge_problem(
            dataset=args.dataset,
            n_shards=args.shards,
            l2=args.l2,
            seed=seed,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            normalize_rows=not args.no_normalize_rows,
            append_bias=not args.no_bias,
            url=args.url,
            n_features=args.n_features,
        )
        config = MultiprocessExperimentConfig(
            n_samples=problem.n_samples,
            n_features=problem.n_features,
            density=problem.x.nnz / max(1, problem.x.shape[0] * problem.x.shape[1]),
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
            seed=seed,
            output_dir=run_dir,
            strategy_names=tuple(args.strategies),
            sleep_scale=args.sleep_scale,
            cost_scale=args.cost_scale,
            cancel_poll_seconds=args.cancel_poll_seconds,
            start_method=args.start_method,
        )
        _, summary = run_multiprocess_problem(config, problem, dataset_name=args.dataset)
        summary.insert(0, "seed", seed)
        summary.insert(0, "dataset", args.dataset)
        summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.out / "combined_realdata_summary.csv", index=False)
    aggregate = aggregate_realdata_summary(combined)
    aggregate.to_csv(args.out / "aggregate_realdata_summary.csv", index=False)
    print("\nAggregate real-data runtime summary")
    print(aggregate.to_string(index=False))


def aggregate_realdata_summary(combined: pd.DataFrame) -> pd.DataFrame:
    return (
        combined.groupby(["dataset", "strategy"], sort=False)
        .agg(
            mean_decode_latency=("mean_decode_latency", "mean"),
            p95_decode_latency=("p95_decode_latency", "mean"),
            mean_barrier_latency=("mean_barrier_latency", "mean"),
            decode_improvement=("decode_latency_improvement_vs_sparse_flexible", "mean"),
            p95_improvement=("p95_decode_latency_improvement_vs_sparse_flexible", "mean"),
            barrier_improvement=("barrier_latency_improvement_vs_sparse_flexible", "mean"),
            final_loss=("final_loss", "mean"),
            scheduler_seconds=("mean_scheduler_seconds", "mean"),
            extra_compute=("mean_extra_compute", "mean"),
            selected_rows=("mean_selected_rows", "mean"),
            completed_rows=("mean_completed_rows", "mean"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    main()
