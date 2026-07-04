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
    run_multiprocess_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real multi-process sparse-flexible coded learning experiment."
    )
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test setup.")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--features", type=int, default=1600)
    parser.add_argument("--density", type=float, default=0.006)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=10)
    parser.add_argument("--straggler-fraction", type=float, default=0.30)
    parser.add_argument("--straggler-slowdown", type=float, default=0.18)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("runtime_results"))
    parser.add_argument("--sleep-scale", type=float, default=0.03)
    parser.add_argument("--cost-scale", type=float, default=0.006)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.004)
    parser.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.samples = 3000
        args.features = 500
        args.density = 0.012
        args.shards = 8
        args.workers = 6
        args.rounds = 6
        args.sleep_scale = 0.01
        args.cost_scale = 0.002
        if args.strategies is None:
            args.strategies = [
                "sparse_flexible_static",
                "worker_aware_sparse_flexible",
                "rank_aware_sparse_flexible",
                "deadline_aware_sparse_flexible",
            ]

    config = MultiprocessExperimentConfig(
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
        seed=args.seed,
        output_dir=args.out,
        strategy_names=tuple(args.strategies or DEFAULT_RUNTIME_STRATEGIES),
        sleep_scale=args.sleep_scale,
        cost_scale=args.cost_scale,
        cancel_poll_seconds=args.cancel_poll_seconds,
        start_method=args.start_method,
    )
    _, summary = run_multiprocess_experiment(config)
    print(summary.to_string(index=False))
    print(f"\nWrote runtime metrics to {config.output_dir}")


if __name__ == "__main__":
    main()
