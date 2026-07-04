from __future__ import annotations

import argparse
from pathlib import Path

from src.coded_learning_exp.experiment import ExperimentConfig, run_experiment
from src.coded_learning_exp.plotting import write_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive sparse-flexible coded learning experiments."
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--features", type=int, default=800)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=140)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--scenario",
        choices=("stable", "burst", "drift", "phase"),
        default="drift",
        help="Worker slowdown pattern.",
    )
    parser.add_argument("--drift-period", type=int, default=35)
    parser.add_argument("--straggler-fraction", type=float, default=0.25)
    parser.add_argument("--straggler-slowdown", type=float, default=0.22)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Optional subset of strategy names to run.",
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small configuration for a fast smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.samples = 2500
        args.features = 400
        args.rounds = 50
        args.workers = 20
        args.shards = 12

    config = ExperimentConfig(
        n_samples=args.samples,
        n_features=args.features,
        density=args.density,
        n_shards=args.shards,
        n_workers=args.workers,
        rounds=args.rounds,
        learning_rate=args.lr,
        l2=args.l2,
        scenario=args.scenario,
        drift_period=args.drift_period,
        straggler_fraction=args.straggler_fraction,
        straggler_slowdown=args.straggler_slowdown,
        burst_probability=args.burst_probability,
        seed=args.seed,
        output_dir=args.out,
        strategy_names=tuple(args.strategies) if args.strategies else None,
    )
    metrics, summary = run_experiment(config)
    write_plots(metrics, summary, args.out)

    print(f"Wrote metrics to {args.out / 'metrics.csv'}")
    print(f"Wrote summary to {args.out / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
