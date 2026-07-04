from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .network_runtime import (
    NetworkExperimentConfig,
    load_problem,
    recv_message,
    run_network_problem,
    send_message,
    worker_endpoint,
)


class DirectEndpointWorkerPool:
    """External worker pool for containers already attached to the master network."""

    def __init__(self, config: NetworkExperimentConfig) -> None:
        self.config = config

    def __enter__(self) -> DirectEndpointWorkerPool:
        self._wait_until_ready()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_workers()

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.config.startup_timeout_seconds
        pending = set(range(self.config.n_workers))
        while pending and time.time() < deadline:
            for worker_id in list(pending):
                host, port = worker_endpoint(self.config, worker_id)
                try:
                    with socket.create_connection((host, port), timeout=0.5) as sock:
                        send_message(sock, {"type": "ping"})
                        recv_message(sock)
                    pending.remove(worker_id)
                except OSError:
                    pass
            if pending:
                time.sleep(0.1)
        if pending:
            raise TimeoutError(f"Timed out waiting for workers: {sorted(pending)}")

    def cancel_round(self, round_id: int) -> float:
        start = time.perf_counter()
        threads = []
        for worker_id in range(self.config.n_workers):
            thread = threading.Thread(
                target=self._cancel_one,
                args=(worker_id, round_id),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout=2.0)
        return time.perf_counter() - start

    def _cancel_one(self, worker_id: int, round_id: int) -> None:
        host, port = worker_endpoint(self.config, worker_id)
        try:
            with socket.create_connection((host, port), timeout=1.0) as sock:
                send_message(sock, {"type": "cancel", "round_id": int(round_id)})
                recv_message(sock)
        except Exception:
            pass

    def stop_workers(self) -> None:
        for worker_id in range(self.config.n_workers):
            host, port = worker_endpoint(self.config, worker_id)
            try:
                with socket.create_connection((host, port), timeout=1.0) as sock:
                    send_message(sock, {"type": "stop"})
                    recv_message(sock)
            except Exception:
                pass


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _split_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TCP master inside a Docker network against prestarted worker containers."
    )
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("/out"))
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--worker-hosts", required=True)
    parser.add_argument("--worker-ports", default="")
    parser.add_argument("--worker-port", type=int, default=19000)
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--features", type=int, default=240)
    parser.add_argument("--density", type=float, default=0.014)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=4)
    parser.add_argument("--straggler-fraction", type=float, default=0.35)
    parser.add_argument("--straggler-slowdown", type=float, default=0.12)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--sleep-scale", type=float, default=0.01)
    parser.add_argument("--cost-scale", type=float, default=0.002)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.003)
    parser.add_argument("--network-rtt-ms", type=float, default=0.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--common-jitter-across-strategies", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alignment-mode", choices=["none", "aligned", "anti"], default="none")
    parser.add_argument("--startup-timeout-seconds", type=float, default=45.0)
    parser.add_argument(
        "--portfolio-fallback",
        choices=["static", "speed", "best_safe"],
        default="static",
        help="Fallback used by guarded_system_portfolio when the guard fails.",
    )
    parser.add_argument(
        "--worker-failure-recovery",
        choices=["none", "reissue"],
        default="none",
        help="Prototype recovery for closed worker connections before first decode.",
    )
    parser.add_argument(
        "--dataset-name",
        default="direct_docker_bridge",
        help="Dataset/runtime label written to network_metrics.csv.",
    )
    parser.add_argument(
        "--summary-name",
        default="direct_endpoint_summary.csv",
        help="Additional summary filename for direct endpoint runs.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "sparse_flexible_static",
            "rank_aware_sparse_flexible",
            "deadline_aware_sparse_flexible",
            "guarded_system_portfolio",
        ],
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parse_args()
    worker_hosts = _split_csv(args.worker_hosts)
    if len(worker_hosts) != args.workers:
        raise ValueError("--worker-hosts must contain exactly --workers hosts.")
    worker_ports = _split_int_csv(args.worker_ports) if args.worker_ports else (args.worker_port,) * args.workers
    if len(worker_ports) != args.workers:
        raise ValueError("--worker-ports must contain exactly --workers ports.")

    problem = load_problem(args.problem_dir)
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
        host=worker_hosts[0],
        base_port=worker_ports[0],
        worker_hosts=worker_hosts,
        worker_ports=worker_ports,
        startup_timeout_seconds=args.startup_timeout_seconds,
        alignment_mode=args.alignment_mode,
        portfolio_fallback=args.portfolio_fallback,
        worker_failure_recovery=args.worker_failure_recovery,
    )
    pool = DirectEndpointWorkerPool(config)
    _, summary = run_network_problem(
        config,
        problem,
        dataset_name=args.dataset_name,
        external_worker_pool=pool,
    )
    summary.to_csv(args.out / args.summary_name, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
