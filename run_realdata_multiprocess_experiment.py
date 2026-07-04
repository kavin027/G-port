from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path

from src.coded_learning_exp.multiprocess_runtime import (
    DEFAULT_RUNTIME_STRATEGIES,
    MultiprocessExperimentConfig,
    run_multiprocess_problem,
)
from src.coded_learning_exp.realdata import make_libsvm_ridge_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-process runtime on a real LIBSVM/SVMLight dataset."
    )
    parser.add_argument("--dataset", default="a9a", help="Built-in dataset name or custom label.")
    parser.add_argument("--url", default=None, help="Optional LIBSVM/SVMLight file URL.")
    parser.add_argument("--n-features", type=int, default=None, help="Optional feature count for custom URL.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data") / "libsvm")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-normalize-rows", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--quick", action="store_true")
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("runtime_realdata_results"))
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
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.max_samples = 3000 if args.max_samples is None else args.max_samples
        args.shards = 8
        args.workers = 8
        args.rounds = 5
        args.sleep_scale = 0.02
        args.cost_scale = 0.002

    problem = make_libsvm_ridge_problem(
        dataset=args.dataset,
        n_shards=args.shards,
        l2=args.l2,
        seed=args.seed,
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
        seed=args.seed,
        output_dir=args.out,
        strategy_names=tuple(args.strategies or DEFAULT_RUNTIME_STRATEGIES),
        sleep_scale=args.sleep_scale,
        cost_scale=args.cost_scale,
        cancel_poll_seconds=args.cancel_poll_seconds,
        start_method=args.start_method,
    )
    _, summary = run_multiprocess_problem(config, problem, dataset_name=args.dataset)
    print(
        f"Loaded {args.dataset}: samples={problem.n_samples}, "
        f"features={problem.n_features}, nnz={problem.x.nnz}"
    )
    print(summary.to_string(index=False))
    print(f"\nWrote real-data runtime metrics to {config.output_dir}")


if __name__ == "__main__":
    main()
