from __future__ import annotations

import argparse
import os
import posixpath
import select
import shlex
import socket
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import paramiko

from src.coded_learning_exp.data import make_sparse_ridge_problem
from src.coded_learning_exp.multiprocess_runtime import (
    _apply_worker_alignment,
    _make_strategy_specs,
    _make_worker_states,
    summarize_runtime_metrics,
)
from src.coded_learning_exp.network_runtime import (
    NetworkExperimentConfig,
    recv_message,
    save_problem,
    send_message,
    _run_network_round,
)


@dataclass(frozen=True)
class RemoteServerConfig:
    host: str
    port: int
    user: str
    password: str
    repo_dir: str
    remote_out: str
    remote_base_port: int


class _ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _pipe(src: Any, dst: Any) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        _shutdown_write(dst)


def _shutdown_write(stream: Any) -> None:
    try:
        if isinstance(stream, socket.socket):
            stream.shutdown(socket.SHUT_WR)
        elif hasattr(stream, "shutdown_write"):
            stream.shutdown_write()
        elif hasattr(stream, "shutdown"):
            stream.shutdown(1)
    except Exception:
        pass


def _make_forward_handler(
    transport: paramiko.Transport,
    remote_host: str,
    remote_port: int,
) -> type[socketserver.BaseRequestHandler]:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                channel = transport.open_channel(
                    "direct-tcpip",
                    (remote_host, remote_port),
                    self.request.getsockname(),
                )
            except Exception:
                return
            if channel is None:
                return
            try:
                if os.name == "nt":
                    left = threading.Thread(target=_pipe, args=(self.request, channel), daemon=True)
                    right = threading.Thread(target=_pipe, args=(channel, self.request), daemon=True)
                    left.start()
                    right.start()
                    left.join()
                    right.join()
                else:
                    while True:
                        readable, _, _ = select.select([self.request, channel], [], [], 30.0)
                        if not readable:
                            continue
                        if self.request in readable:
                            data = self.request.recv(65536)
                            if not data:
                                break
                            channel.sendall(data)
                        if channel in readable:
                            data = channel.recv(65536)
                            if not data:
                                break
                            self.request.sendall(data)
            finally:
                try:
                    channel.close()
                except Exception:
                    pass
                try:
                    self.request.close()
                except Exception:
                    pass

    return Handler


class LocalForward:
    def __init__(
        self,
        *,
        transport: paramiko.Transport,
        local_host: str,
        local_port: int,
        remote_host: str,
        remote_port: int,
    ) -> None:
        handler = _make_forward_handler(transport, remote_host, remote_port)
        self.server = _ForwardServer((local_host, local_port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


class ExternalWorkerPool:
    def __init__(self, config: NetworkExperimentConfig) -> None:
        self.config = config

    def cancel_round(self, round_id: int) -> float:
        start = time.perf_counter()
        threads = []
        for worker_id in range(self.config.n_workers):
            thread = threading.Thread(
                target=self._cancel_one,
                args=(self.config.base_port + worker_id, round_id),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout=4.0)
        return time.perf_counter() - start

    def stop_workers(self) -> None:
        for worker_id in range(self.config.n_workers):
            try:
                with socket.create_connection(
                    (self.config.host, self.config.base_port + worker_id),
                    timeout=2.0,
                ) as sock:
                    send_message(sock, {"type": "stop"})
                    recv_message(sock)
            except Exception:
                pass

    def _cancel_one(self, port: int, round_id: int) -> None:
        try:
            with socket.create_connection((self.config.host, port), timeout=2.0) as sock:
                send_message(sock, {"type": "cancel", "round_id": int(round_id)})
                recv_message(sock)
        except Exception:
            pass


class RemoteWorkerHost:
    def __init__(
        self,
        *,
        server: RemoteServerConfig,
        n_workers: int,
        local_base_port: int,
        local_host: str = "127.0.0.1",
    ) -> None:
        self.server = server
        self.n_workers = n_workers
        self.local_base_port = local_base_port
        self.local_host = local_host
        self.client: paramiko.SSHClient | None = None
        self.sftp: paramiko.SFTPClient | None = None
        self.forwards: list[LocalForward] = []
        self.worker_pids: list[int] = []

    @property
    def remote_problem_dir(self) -> str:
        return posixpath.join(self.server.repo_dir, self.server.remote_out, "problem")

    @property
    def remote_ready_dir(self) -> str:
        return posixpath.join(self.server.repo_dir, self.server.remote_out, "worker_ready")

    @property
    def remote_logs_dir(self) -> str:
        return posixpath.join(self.server.repo_dir, self.server.remote_out, "worker_logs")

    def connect(self) -> None:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.server.host,
            port=self.server.port,
            username=self.server.user,
            password=self.server.password,
            timeout=20,
            banner_timeout=20,
            auth_timeout=20,
        )
        self.client = client
        self.sftp = client.open_sftp()

    def close(self) -> None:
        for forward in self.forwards:
            forward.close()
        self.forwards.clear()
        if self.sftp is not None:
            self.sftp.close()
            self.sftp = None
        if self.client is not None:
            self.client.close()
            self.client = None

    def prepare_problem(self, local_problem_dir: Path) -> None:
        self._exec(f"mkdir -p {shlex.quote(self.remote_problem_dir)}")
        if self.sftp is None:
            raise RuntimeError("SFTP client is not connected.")
        for name in ["x.npz", "meta.npz"]:
            self.sftp.put(
                str(local_problem_dir / name),
                posixpath.join(self.remote_problem_dir, name),
            )

    def start_workers(self) -> None:
        self._exec(
            "mkdir -p "
            + shlex.quote(self.remote_ready_dir)
            + " "
            + shlex.quote(self.remote_logs_dir)
            + " && rm -f "
            + shlex.quote(self.remote_ready_dir)
            + "/*.ready"
        )
        self._kill_stale_workers()
        for worker_id in range(self.n_workers):
            port = self.server.remote_base_port + worker_id
            ready = posixpath.join(self.remote_ready_dir, f"worker_{worker_id}.ready")
            log = posixpath.join(self.remote_logs_dir, f"worker_{worker_id}.log")
            pidfile = posixpath.join(self.remote_logs_dir, f"worker_{worker_id}.pid")
            cmd = (
                f"cd {shlex.quote(self.server.repo_dir)} && "
                "("
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "NUMEXPR_NUM_THREADS=1 "
                f".venv/bin/python -m src.coded_learning_exp.network_runtime worker "
                f"--worker-id {worker_id} --host 127.0.0.1 --port {port} "
                f"--problem-dir {shlex.quote(self.remote_problem_dir)} "
                f"--ready-file {shlex.quote(ready)} "
                f"< /dev/null > {shlex.quote(log)} 2>&1 & "
                f"echo $! > {shlex.quote(pidfile)}"
                ") >/dev/null 2>&1 &"
            )
            out = self._exec(cmd, get_pty=False)
            del out
        self._wait_remote_ready()
        self._start_forwards()
        self._wait_local_ready()

    def stop_workers(self, pool: ExternalWorkerPool) -> None:
        pool.stop_workers()
        if self.worker_pids:
            pids = " ".join(str(pid) for pid in self.worker_pids)
            self._exec(f"kill {pids} >/dev/null 2>&1 || true")
        self._kill_stale_workers()

    def _kill_stale_workers(self) -> None:
        for worker_id in range(self.n_workers):
            port = self.server.remote_base_port + worker_id
            pattern = f"network_runtime worker.*--port {port}"
            try:
                self._exec(f"pkill -f {shlex.quote(pattern)} >/dev/null 2>&1 || true", get_pty=False)
            except Exception:
                pass

    def _start_forwards(self) -> None:
        if self.client is None or self.client.get_transport() is None:
            raise RuntimeError("SSH client is not connected.")
        transport = self.client.get_transport()
        assert transport is not None
        for worker_id in range(self.n_workers):
            forward = LocalForward(
                transport=transport,
                local_host=self.local_host,
                local_port=self.local_base_port + worker_id,
                remote_host="127.0.0.1",
                remote_port=self.server.remote_base_port + worker_id,
            )
            forward.start()
            self.forwards.append(forward)

    def _wait_remote_ready(self) -> None:
        if self.sftp is None:
            raise RuntimeError("SFTP client is not connected.")
        deadline = time.time() + 60.0
        pending = set(range(self.n_workers))
        while pending and time.time() < deadline:
            for worker_id in list(pending):
                ready = posixpath.join(self.remote_ready_dir, f"worker_{worker_id}.ready")
                try:
                    self.sftp.stat(ready)
                    pending.remove(worker_id)
                except FileNotFoundError:
                    pass
            if pending:
                time.sleep(0.2)
        if pending:
            raise TimeoutError(f"Timed out waiting for remote workers: {sorted(pending)}")
        for worker_id in range(self.n_workers):
            pidfile = posixpath.join(self.remote_logs_dir, f"worker_{worker_id}.pid")
            try:
                with self.sftp.open(pidfile, "r") as handle:
                    self.worker_pids.append(int(handle.read().decode().strip()))
            except Exception:
                pass

    def _wait_local_ready(self) -> None:
        deadline = time.time() + 60.0
        pending = set(range(self.n_workers))
        while pending and time.time() < deadline:
            for worker_id in list(pending):
                try:
                    with socket.create_connection(
                        (self.local_host, self.local_base_port + worker_id),
                        timeout=0.5,
                    ) as sock:
                        send_message(sock, {"type": "ping"})
                        recv_message(sock)
                    pending.remove(worker_id)
                except Exception:
                    pass
            if pending:
                time.sleep(0.1)
        if pending:
            raise TimeoutError(f"Timed out waiting for local SSH forwards: {sorted(pending)}")

    def _exec(self, command: str, *, get_pty: bool = True, timeout: float = 30.0) -> str:
        if self.client is None:
            raise RuntimeError("SSH client is not connected.")
        stdin, stdout, stderr = self.client.exec_command(command, get_pty=get_pty, timeout=timeout)
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        if rc != 0:
            raise RuntimeError(f"Remote command failed ({rc}): {command}\n{out}\n{err}")
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the TCP master locally while workers run on a remote host behind "
            "SSH local forwards. This gives a small real cross-host TCP path "
            "when direct worker ports are unavailable."
        )
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--features", type=int, default=800)
    parser.add_argument("--density", type=float, default=0.008)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=4)
    parser.add_argument("--straggler-fraction", type=float, default=0.45)
    parser.add_argument("--straggler-slowdown", type=float, default=0.08)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("tunneled_remote_network_results"))
    parser.add_argument("--sleep-scale", type=float, default=0.03)
    parser.add_argument("--cost-scale", type=float, default=0.006)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.003)
    parser.add_argument("--network-rtt-ms", type=float, default=0.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--local-base-port", type=int, default=25000)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-ssh-port", type=int, required=True)
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--remote-password", default=os.environ.get("REMOTE_PASSWORD", ""))
    parser.add_argument("--remote-repo", default="/root/coded_distributed_computing_socc_runtime")
    parser.add_argument("--remote-out", default="tunneled_remote_network_results")
    parser.add_argument("--remote-base-port", type=int, default=26000)
    parser.add_argument("--alignment-mode", choices=["none", "aligned", "anti"], default="none")
    parser.add_argument("--common-jitter-across-strategies", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "speed_aware_uncoded",
            "speculative_replication",
            "sparse_flexible_static",
            "rank_aware_sparse_flexible",
            "system_portfolio",
            "learned_system_portfolio",
        ],
    )
    return parser.parse_args()


def run_with_external_workers(
    *,
    config: NetworkExperimentConfig,
    worker_pool: ExternalWorkerPool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    problem = make_sparse_ridge_problem(
        n_samples=config.n_samples,
        n_features=config.n_features,
        density=config.density,
        n_shards=config.n_shards,
        l2=config.l2,
        seed=config.seed,
    )
    local_problem_dir = config.output_dir / "problem"
    save_problem(problem, local_problem_dir)
    return _run_external_network_problem(config, problem, worker_pool, dataset_name="synthetic")


def _run_external_network_problem(
    config: NetworkExperimentConfig,
    problem: Any,
    worker_pool: ExternalWorkerPool,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_specs = _make_strategy_specs(problem, config)
    worker_states = _apply_worker_alignment(problem, config, _make_worker_states(config))
    unknown = set(config.strategy_names) - set(strategy_specs)
    if unknown:
        raise ValueError(f"Unknown network strategies: {', '.join(sorted(unknown))}")

    records: list[dict[str, Any]] = []
    for strategy_index, strategy_name in enumerate(config.strategy_names):
        weights = np.zeros(problem.n_features, dtype=float)
        decode_wall_clock = 0.0
        barrier_wall_clock = 0.0
        spec = strategy_specs[strategy_name]
        for iteration, worker_state in enumerate(worker_states):
            round_id = strategy_index * 1_000_000 + iteration
            jitter_seed = config.seed + iteration * 997
            if not config.common_jitter_across_strategies:
                jitter_seed += strategy_index * 100_003
            schedule_start = time.perf_counter()
            rows, assignments, second_flags, config_label = spec(worker_state)
            scheduler_seconds = time.perf_counter() - schedule_start
            result = _run_network_round(
                config=config,
                worker_pool=worker_pool,
                problem=problem,
                strategy=strategy_name,
                round_id=round_id,
                rows=rows,
                assignments=assignments,
                second_layer_flags=second_flags,
                weights=weights,
                worker_state=worker_state,
                jitter_seed=jitter_seed,
            )
            weights = weights - config.learning_rate * result["gradient"]
            decode_wall_clock += float(result["decode_latency"])
            barrier_wall_clock += float(result["barrier_latency"])
            records.append(
                {
                    "iteration": iteration,
                    "strategy": strategy_name,
                    "config": config_label,
                    "dataset": dataset_name,
                    "scenario": config.scenario,
                    "alignment_mode": config.alignment_mode,
                    "density": problem.x.nnz / np.prod(problem.x.shape),
                    "n_workers": config.n_workers,
                    "n_shards": config.n_shards,
                    "n_samples": problem.n_samples,
                    "n_features": problem.n_features,
                    "decode_latency": result["decode_latency"],
                    "barrier_latency": result["barrier_latency"],
                    "decode_wall_clock": decode_wall_clock,
                    "barrier_wall_clock": barrier_wall_clock,
                    "loss": problem.loss(weights),
                    "decode_success": result["decode_success"],
                    "decode_residual": result["decode_residual"],
                    "decode_cpu_seconds": result["decode_cpu_seconds"],
                    "scheduler_seconds": scheduler_seconds,
                    "dispatch_seconds": result["dispatch_seconds"],
                    "cancel_seconds": result["cancel_seconds"],
                    "network_response_bytes": result["network_response_bytes"],
                    "network_response_sleep_seconds": result["network_response_sleep_seconds"],
                    "worker_compute_cpu_seconds": result["worker_compute_cpu_seconds"],
                    "selected_rows": result["selected_rows"],
                    "completed_rows": result["completed_rows"],
                    "cancelled_rows": result["cancelled_rows"],
                    "rows_after_decode": result["rows_after_decode"],
                    "extra_compute": result["extra_compute"],
                    "second_layer_used": result["second_layer_used"],
                    "slow_workers": int(worker_state.slow_mask.sum()),
                    "mean_worker_speed": float(worker_state.speeds.mean()),
                }
            )

    metrics = pd.DataFrame.from_records(records)
    summary = summarize_runtime_metrics(metrics)
    extra = metrics.groupby("strategy", sort=False).agg(
        mean_dispatch_seconds=("dispatch_seconds", "mean"),
        mean_cancel_seconds=("cancel_seconds", "mean"),
        mean_network_response_mb=("network_response_bytes", lambda x: x.mean() / 1_000_000.0),
        mean_network_response_sleep_seconds=("network_response_sleep_seconds", "mean"),
    ).reset_index()
    summary = summary.merge(extra, on="strategy", how="left")
    metrics.to_csv(config.output_dir / "network_metrics.csv", index=False)
    summary.to_csv(config.output_dir / "network_summary.csv", index=False)
    return metrics, summary


def main() -> None:
    args = parse_args()
    if not args.remote_password:
        raise SystemExit("Set --remote-password or REMOTE_PASSWORD.")

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
        host=args.local_host,
        base_port=args.local_base_port,
        alignment_mode=args.alignment_mode,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    problem = make_sparse_ridge_problem(
        n_samples=config.n_samples,
        n_features=config.n_features,
        density=config.density,
        n_shards=config.n_shards,
        l2=config.l2,
        seed=config.seed,
    )
    local_problem_dir = config.output_dir / "problem"
    save_problem(problem, local_problem_dir)

    remote = RemoteWorkerHost(
        server=RemoteServerConfig(
            host=args.remote_host,
            port=args.remote_ssh_port,
            user=args.remote_user,
            password=args.remote_password,
            repo_dir=args.remote_repo,
            remote_out=args.remote_out,
            remote_base_port=args.remote_base_port,
        ),
        n_workers=config.n_workers,
        local_base_port=args.local_base_port,
        local_host=args.local_host,
    )
    worker_pool = ExternalWorkerPool(config)
    try:
        remote.connect()
        remote.prepare_problem(local_problem_dir)
        remote.start_workers()
        metrics, summary = _run_external_network_problem(
            config=config,
            problem=problem,
            worker_pool=worker_pool,
            dataset_name="synthetic-tunneled-remote",
        )
        del metrics
        print(summary.to_string(index=False))
        print(f"\nWrote tunneled remote metrics to {config.output_dir}")
    finally:
        try:
            remote.stop_workers(worker_pool)
        finally:
            remote.close()


if __name__ == "__main__":
    main()
