from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from src.coded_learning_exp.network_runtime import NetworkExperimentConfig, run_network_experiment


BASELINE_TO_STRATEGY = {
    "original_sfcl": ["original_sfcl_static"],
    "rltune_style": ["rltune_style_selector"],
    "sailor_style": ["sailor_style_heterogeneity_aware"],
    "straggler_whatif": [],
}

DEFAULT_EXTERNAL_BASELINES = [
    "original_sfcl",
    "rltune_style",
    "sailor_style",
    "straggler_whatif",
]

DEFAULT_COMPARISON_STRATEGIES = [
    "guarded_system_portfolio",
    "online_counter_guard_deadline_aware_sparse_flexible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run external-style baselines in the same TCP/K3s worker-service runtime. "
            "The baselines are adaptations for fair same-runtime comparison, not full "
            "ports of the external systems."
        )
    )
    parser.add_argument(
        "--external-baselines",
        nargs="+",
        default=DEFAULT_EXTERNAL_BASELINES,
        choices=sorted(BASELINE_TO_STRATEGY),
    )
    parser.add_argument(
        "--external-baseline-mode",
        choices=["tcp", "k3s", "network_stress"],
        default="network_stress",
    )
    parser.add_argument(
        "--external-baseline-seeds",
        default="7,11,17,23,31,37,43,53",
        help="Comma-separated seed list.",
    )
    parser.add_argument("--artifact-out", type=Path, default=Path("results/external_baselines"))
    parser.add_argument("--workers", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--features", type=int, default=240)
    parser.add_argument("--density", type=float, default=0.014)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=4)
    parser.add_argument("--straggler-fraction", type=float, default=0.45)
    parser.add_argument("--straggler-slowdown", type=float, default=0.08)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--sleep-scale", type=float, default=0.01)
    parser.add_argument("--cost-scale", type=float, default=0.002)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.003)
    parser.add_argument("--network-rtt-ms", type=float, default=None)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=19000)
    parser.add_argument("--alignment-mode", choices=["none", "aligned", "anti"], default="none")
    parser.add_argument("--portfolio-fallback", choices=["static", "speed", "best_safe"], default="static")
    parser.add_argument(
        "--include-arms",
        action="store_true",
        help="Also run speed-aware uncoded, speculative replication, and rank-aware coded arms.",
    )
    parser.add_argument("--use-docker-workers", action="store_true")
    parser.add_argument("--docker-image", default="coded-learning-network-worker:local")
    parser.add_argument("--docker-internal-port", type=int, default=19000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.external_baseline_seeds)
    strategies = strategy_list(args.external_baselines, args.include_arms)
    args.artifact_out.mkdir(parents=True, exist_ok=True)
    plan = {
        "mode": args.external_baseline_mode,
        "workers": args.workers,
        "seeds": seeds,
        "strategies": strategies,
        "external_baselines": args.external_baselines,
        "note": "External systems are adapted as same-runtime baselines for fair accounting.",
    }
    (args.artifact_out / "external_baseline_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )

    if args.external_baseline_mode == "k3s":
        run_k3s_wrapper(args, seeds, strategies)
    else:
        run_tcp_matrix(args, seeds, strategies)

    if not args.dry_run:
        subprocess.run(
            [
                sys.executable,
                "analyze_external_baselines.py",
                "--root",
                str(args.artifact_out),
                "--out",
                str(args.artifact_out),
            ],
            check=True,
        )


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def strategy_list(external_baselines: list[str], include_arms: bool) -> list[str]:
    strategies: list[str] = []
    if include_arms:
        strategies.extend(
            [
                "speed_aware_uncoded",
                "speculative_replication",
                "rank_aware_sparse_flexible",
            ]
        )
    for baseline in external_baselines:
        strategies.extend(BASELINE_TO_STRATEGY[baseline])
    strategies.extend(DEFAULT_COMPARISON_STRATEGIES)
    return unique(strategies)


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def run_tcp_matrix(args: argparse.Namespace, seeds: list[int], strategies: list[str]) -> None:
    network_rtt_ms = args.network_rtt_ms
    bandwidth_mbps = args.network_bandwidth_mbps
    if network_rtt_ms is None:
        network_rtt_ms = 3.0 if args.external_baseline_mode == "network_stress" else 0.0
    if bandwidth_mbps is None:
        bandwidth_mbps = 250.0 if args.external_baseline_mode == "network_stress" else 0.0

    run_index = 0
    for workers in args.workers:
        for seed in seeds:
            run_dir = args.artifact_out / f"{args.external_baseline_mode}_w{workers}_seed{seed}"
            summary_path = run_dir / "network_summary.csv"
            if args.skip_existing and summary_path.exists():
                print(f"Skipping existing run {run_dir}")
                continue
            metadata = {
                "external_mode": args.external_baseline_mode,
                "n_workers": workers,
                "seed": seed,
                "strategies": strategies,
                "network_rtt_ms": network_rtt_ms,
                "network_bandwidth_mbps": bandwidth_mbps,
            }
            if args.dry_run:
                print(json.dumps({"run_dir": str(run_dir), **metadata}, indent=2))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "external_baseline_run.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            config = NetworkExperimentConfig(
                n_samples=args.samples,
                n_features=args.features,
                density=args.density,
                n_shards=args.shards,
                n_workers=workers,
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
                strategy_names=tuple(strategies),
                sleep_scale=args.sleep_scale,
                cost_scale=args.cost_scale,
                cancel_poll_seconds=args.cancel_poll_seconds,
                network_rtt_seconds=float(network_rtt_ms) / 1000.0,
                network_bandwidth_mbps=float(bandwidth_mbps),
                common_jitter_across_strategies=True,
                host=args.host,
                base_port=args.base_port + run_index * 100,
                alignment_mode=args.alignment_mode,
                use_docker_workers=args.use_docker_workers,
                docker_image=args.docker_image,
                docker_internal_port=args.docker_internal_port,
                docker_container_prefix=f"external-baseline-{workers}-{seed}",
                portfolio_fallback=args.portfolio_fallback,
            )
            print(
                f"Running {args.external_baseline_mode} workers={workers} seed={seed} "
                f"strategies={','.join(strategies)}"
            )
            run_network_experiment(config)
            run_index += 1


def run_k3s_wrapper(args: argparse.Namespace, seeds: list[int], strategies: list[str]) -> None:
    command = [
        sys.executable,
        "run_majorrev_k8s_extended.py",
        "--workers",
        *[str(worker) for worker in args.workers],
        "--seeds",
        *[str(seed) for seed in seeds],
        "--samples",
        str(args.samples),
        "--features",
        str(args.features),
        "--density",
        str(args.density),
        "--shards",
        str(args.shards),
        "--rounds",
        str(args.rounds),
        "--scenario",
        args.scenario,
        "--drift-period",
        str(args.drift_period),
        "--straggler-fraction",
        str(args.straggler_fraction),
        "--straggler-slowdown",
        str(args.straggler_slowdown),
        "--burst-probability",
        str(args.burst_probability),
        "--sleep-scale",
        str(args.sleep_scale),
        "--cost-scale",
        str(args.cost_scale),
        "--cancel-poll-seconds",
        str(args.cancel_poll_seconds),
        "--out-root",
        str(args.artifact_out),
        "--portfolio-fallback",
        args.portfolio_fallback,
        "--strategies",
        *strategies,
    ]
    if args.network_rtt_ms is not None:
        command.extend(["--network-rtt-ms", str(args.network_rtt_ms)])
    if args.network_bandwidth_mbps is not None:
        command.extend(["--network-bandwidth-mbps", str(args.network_bandwidth_mbps)])
    if args.skip_existing:
        command.append("--skip-existing")
    if args.dry_run:
        command.append("--dry-run")
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
