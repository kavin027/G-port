from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path

from src.coded_learning_exp.network_runtime import (
    NetworkExperimentConfig,
    run_network_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a TCP master-worker experiment. Use --use-docker-workers for "
            "container-per-worker execution; otherwise workers run as isolated "
            "Python services on separate ports."
        )
    )
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test setup.")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--features", type=int, default=800)
    parser.add_argument("--density", type=float, default=0.008)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=6)
    parser.add_argument("--straggler-fraction", type=float, default=0.35)
    parser.add_argument("--straggler-slowdown", type=float, default=0.12)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("network_runtime_results"))
    parser.add_argument("--sleep-scale", type=float, default=0.025)
    parser.add_argument("--cost-scale", type=float, default=0.005)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.003)
    parser.add_argument("--network-rtt-ms", type=float, default=0.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=0.0)
    parser.add_argument(
        "--common-jitter-across-strategies",
        action="store_true",
        help="Use the same worker/iteration jitter draw across strategies for paired comparisons.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=19000)
    parser.add_argument("--alignment-mode", choices=["none", "aligned", "anti"], default="none")
    parser.add_argument(
        "--use-docker-workers",
        action="store_true",
        help="Launch each TCP worker as a separate Docker container.",
    )
    parser.add_argument(
        "--docker-image",
        default="coded-learning-network-worker:local",
        help="Docker image used for worker containers.",
    )
    parser.add_argument(
        "--docker-internal-port",
        type=int,
        default=19000,
        help="Container-internal TCP port exposed by each worker.",
    )
    parser.add_argument(
        "--docker-container-prefix",
        default=None,
        help="Optional container name prefix; stale containers with this prefix/id are removed per worker.",
    )
    parser.add_argument(
        "--portfolio-fallback",
        choices=["static", "speed", "best_safe"],
        default="static",
        help=(
            "Fallback used by guarded_system_portfolio when the guard does not "
            "enable the candidate arm. The default preserves prior K3s traces."
        ),
    )
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
        args.samples = 1600
        args.features = 240
        args.density = 0.014
        args.shards = 6
        args.workers = 6
        args.rounds = 4
        args.sleep_scale = 0.01
        args.cost_scale = 0.002

    config = NetworkExperimentConfig(
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
        strategy_names=tuple(args.strategies),
        sleep_scale=args.sleep_scale,
        cost_scale=args.cost_scale,
        cancel_poll_seconds=args.cancel_poll_seconds,
        network_rtt_seconds=args.network_rtt_ms / 1000.0,
        network_bandwidth_mbps=args.network_bandwidth_mbps,
        common_jitter_across_strategies=args.common_jitter_across_strategies,
        host=args.host,
        base_port=args.base_port,
        alignment_mode=args.alignment_mode,
        use_docker_workers=args.use_docker_workers,
        docker_image=args.docker_image,
        docker_internal_port=args.docker_internal_port,
        docker_container_prefix=args.docker_container_prefix,
        portfolio_fallback=args.portfolio_fallback,
    )
    _, summary = run_network_experiment(config)
    print(summary.to_string(index=False))
    print(f"\nWrote network runtime metrics to {config.output_dir}")


if __name__ == "__main__":
    main()
