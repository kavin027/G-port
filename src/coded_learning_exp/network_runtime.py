from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .coding import DecodeResult, decode_coefficients
from .data import SparseRidgeProblem, make_sparse_ridge_problem
from .multiprocess_runtime import (
    DEFAULT_RUNTIME_STRATEGIES,
    _apply_worker_alignment,
    _make_strategy_specs,
    _make_worker_states,
    summarize_runtime_metrics,
)
from .workers import WorkerState


@dataclass(frozen=True)
class NetworkExperimentConfig:
    n_samples: int = 6000
    n_features: int = 800
    density: float = 0.008
    n_shards: int = 8
    n_workers: int = 8
    rounds: int = 12
    learning_rate: float = 0.25
    l2: float = 1e-3
    scenario: str = "phase"
    drift_period: int = 6
    straggler_fraction: float = 0.35
    straggler_slowdown: float = 0.12
    burst_probability: float = 0.45
    seed: int = 17
    output_dir: Path = Path("network_runtime_results")
    strategy_names: tuple[str, ...] = (
        "uncoded_sync",
        "replication",
        "speculative_replication",
        "sparse_flexible_static",
        "worker_aware_sparse_flexible",
        "rank_aware_sparse_flexible",
        "deadline_aware_sparse_flexible",
        "guarded_system_portfolio",
    )
    sleep_scale: float = 0.025
    cost_scale: float = 0.005
    cancel_poll_seconds: float = 0.003
    network_rtt_seconds: float = 0.0
    network_bandwidth_mbps: float = 0.0
    common_jitter_across_strategies: bool = False
    host: str = "127.0.0.1"
    base_port: int = 19000
    worker_hosts: tuple[str, ...] | None = None
    worker_ports: tuple[int, ...] | None = None
    startup_timeout_seconds: float = 30.0
    alignment_mode: str = "none"
    use_docker_workers: bool = False
    docker_image: str = "coded-learning-network-worker:local"
    docker_internal_port: int = 19000
    docker_container_prefix: str | None = None
    portfolio_fallback: str = "static"
    worker_failure_recovery: str = "none"


@dataclass
class WorkerProcess:
    worker_id: int
    port: int
    log_file: Any
    process: subprocess.Popen | None = None
    container_name: str | None = None


class ProtocolError(RuntimeError):
    pass


def worker_endpoint(config: NetworkExperimentConfig, worker_id: int) -> tuple[str, int]:
    if config.worker_hosts is not None:
        if len(config.worker_hosts) != config.n_workers:
            raise ValueError("worker_hosts length must match n_workers.")
        host = config.worker_hosts[worker_id]
    else:
        host = config.host
    if config.worker_ports is not None:
        if len(config.worker_ports) != config.n_workers:
            raise ValueError("worker_ports length must match n_workers.")
        port = int(config.worker_ports[worker_id])
    else:
        port = config.base_port + worker_id
    return host, port


def save_problem(problem: SparseRidgeProblem, problem_dir: Path) -> None:
    problem_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(problem_dir / "x.npz", problem.x)
    boundaries = np.asarray(
        [problem.shard_slices[0].start]
        + [shard_slice.stop for shard_slice in problem.shard_slices],
        dtype=np.int64,
    )
    np.savez(problem_dir / "meta.npz", y=problem.y, boundaries=boundaries, l2=problem.l2)


def load_problem(problem_dir: Path) -> SparseRidgeProblem:
    x = sparse.load_npz(problem_dir / "x.npz").tocsr()
    meta = np.load(problem_dir / "meta.npz")
    y = np.asarray(meta["y"], dtype=float)
    boundaries = np.asarray(meta["boundaries"], dtype=np.int64)
    shard_slices = [
        slice(int(boundaries[i]), int(boundaries[i + 1]))
        for i in range(boundaries.size - 1)
    ]
    return SparseRidgeProblem(x=x, y=y, shard_slices=shard_slices, l2=float(meta["l2"]))


def send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, 8)
    if not header:
        raise EOFError("Socket closed while reading message length.")
    (size,) = struct.unpack("!Q", header)
    if size <= 0:
        raise ProtocolError(f"Invalid message size: {size}")
    data = _recv_exact(sock, size)
    if len(data) != size:
        raise EOFError("Socket closed while reading message payload.")
    payload = pickle.loads(data)
    if not isinstance(payload, dict):
        raise ProtocolError("Expected a dictionary payload.")
    return payload


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def worker_server_main(
    *,
    worker_id: int,
    host: str,
    port: int,
    problem_dir: Path,
    ready_file: Path | None = None,
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    problem = load_problem(problem_dir)
    shard_costs = problem.shard_costs()
    cancelled_rounds: set[int] = set()
    cancel_lock = threading.Lock()
    stop_event = threading.Event()
    stress_target = os.environ.get("CODED_STRESS_WORKER_ID", "").strip().lower()
    cancel_ack_delay_seconds = max(0.0, _float_env("CODED_CANCEL_ACK_DELAY_MS", 0.0) / 1000.0)
    close_on_task = _bool_env("CODED_CLOSE_ON_TASK")
    exit_on_task = _bool_env("CODED_EXIT_ON_TASK")

    def stress_applies() -> bool:
        return stress_target in {"", "all", str(worker_id)}

    def is_cancelled(round_id: int) -> bool:
        with cancel_lock:
            return int(round_id) in cancelled_rounds

    def mark_cancelled(round_id: int) -> None:
        with cancel_lock:
            cancelled_rounds.add(int(round_id))

    def handle_connection(conn: socket.socket) -> None:
        with conn:
            try:
                message = recv_message(conn)
                kind = message.get("type")
                if kind == "ping":
                    send_message(conn, {"type": "ack", "worker_id": worker_id})
                    return
                if kind == "cancel":
                    mark_cancelled(int(message["round_id"]))
                    if cancel_ack_delay_seconds > 0.0 and stress_applies():
                        time.sleep(cancel_ack_delay_seconds)
                    send_message(conn, {"type": "ack", "round_id": int(message["round_id"])})
                    return
                if kind == "stop":
                    stop_event.set()
                    send_message(conn, {"type": "ack"})
                    return
                if kind != "task":
                    raise ProtocolError(f"Unknown worker message type: {kind}")
                if stress_applies() and close_on_task:
                    return
                if stress_applies() and exit_on_task:
                    os._exit(97)
                _execute_network_task(
                    conn=conn,
                    worker_id=worker_id,
                    problem=problem,
                    shard_costs=shard_costs,
                    task=message,
                    is_cancelled=is_cancelled,
                )
            except Exception as exc:  # pragma: no cover - reported over the socket when possible.
                traceback.print_exc()
                sys.stderr.flush()
                try:
                    send_message(conn, {"type": "error", "worker_id": worker_id, "error": repr(exc)})
                except Exception:
                    pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        server.settimeout(0.5)
        if ready_file is not None:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.write_text("ready", encoding="utf-8")
        while not stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            thread = threading.Thread(target=handle_connection, args=(conn,), daemon=True)
            thread.start()


def _execute_network_task(
    *,
    conn: socket.socket,
    worker_id: int,
    problem: SparseRidgeProblem,
    shard_costs: np.ndarray,
    task: dict[str, Any],
    is_cancelled,
) -> None:
    rows = np.asarray(task["rows"], dtype=float)
    row_ids = np.asarray(task["row_ids"], dtype=int)
    second_layer_flags = np.asarray(task["second_layer_flags"], dtype=bool)
    weights = np.asarray(task["weights"], dtype=float)
    speed = float(task["speed"])
    delay = float(task["delay"])
    sleep_scale = float(task["sleep_scale"])
    cost_scale = float(task["cost_scale"])
    cancel_poll_seconds = float(task["cancel_poll_seconds"])
    network_rtt_seconds = float(task.get("network_rtt_seconds", 0.0))
    network_bandwidth_mbps = float(task.get("network_bandwidth_mbps", 0.0))
    round_id = int(task["round_id"])
    strategy = str(task["strategy"])
    rng = np.random.default_rng(int(task["jitter_seed"]) + worker_id * 7919)
    completed = 0
    task_start = time.perf_counter()

    for local_idx, row in enumerate(rows):
        if is_cancelled(round_id):
            break
        row_cost = _row_cost(row, shard_costs)
        synthetic_seconds = _synthetic_delay_seconds(
            row_cost=row_cost,
            speed=speed,
            delay=delay,
            sleep_scale=sleep_scale,
            cost_scale=cost_scale,
            rng=rng,
        )
        if _interruptible_sleep(synthetic_seconds, round_id, cancel_poll_seconds, is_cancelled):
            break
        compute_start = time.perf_counter()
        gradient = _compute_encoded_gradient(problem, row, weights)
        compute_seconds = time.perf_counter() - compute_start
        payload_bytes = int(gradient.nbytes + row.nbytes + 512)
        network_seconds = _network_transfer_seconds(
            payload_bytes=payload_bytes,
            rtt_seconds=network_rtt_seconds,
            bandwidth_mbps=network_bandwidth_mbps,
        )
        if _interruptible_sleep(network_seconds, round_id, cancel_poll_seconds, is_cancelled):
            break
        completed += 1
        send_message(
            conn,
            {
                "type": "row",
                "round_id": round_id,
                "strategy": strategy,
                "worker_id": worker_id,
                "row_id": int(row_ids[local_idx]),
                "row": row,
                "gradient": gradient,
                "row_cost": row_cost,
                "network_payload_bytes": payload_bytes,
                "network_sleep_seconds": network_seconds,
                "compute_cpu_seconds": compute_seconds,
                "elapsed_seconds": time.perf_counter() - task_start,
                "second_layer": bool(second_layer_flags[local_idx]),
            },
        )

    send_message(
        conn,
        {
            "type": "done",
            "round_id": round_id,
            "strategy": strategy,
            "worker_id": worker_id,
            "completed_rows": completed,
        },
    )


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _row_cost(row: np.ndarray, shard_costs: np.ndarray) -> float:
    support = np.flatnonzero(np.abs(row) > 0.0)
    if support.size == 0:
        return 0.0
    return float(shard_costs[support].sum())


def _synthetic_delay_seconds(
    *,
    row_cost: float,
    speed: float,
    delay: float,
    sleep_scale: float,
    cost_scale: float,
    rng: np.random.Generator,
) -> float:
    jitter = float(rng.lognormal(mean=0.0, sigma=0.08))
    return max(0.0, sleep_scale * delay + cost_scale * row_cost * jitter / max(speed, 1e-8))


def _network_transfer_seconds(
    *,
    payload_bytes: int,
    rtt_seconds: float,
    bandwidth_mbps: float,
) -> float:
    seconds = max(0.0, float(rtt_seconds))
    if bandwidth_mbps > 0.0 and payload_bytes > 0:
        seconds += (8.0 * float(payload_bytes)) / (float(bandwidth_mbps) * 1_000_000.0)
    return seconds


def _interruptible_sleep(seconds: float, round_id: int, poll_seconds: float, is_cancelled) -> bool:
    deadline = time.perf_counter() + seconds
    while True:
        if is_cancelled(round_id):
            return True
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        time.sleep(min(max(poll_seconds, 1e-4), remaining))


def _compute_encoded_gradient(problem: SparseRidgeProblem, row: np.ndarray, weights: np.ndarray) -> np.ndarray:
    gradient = np.zeros(problem.n_features, dtype=float)
    scale = 1.0 / problem.n_samples
    support = np.flatnonzero(np.abs(row) > 0.0)
    for shard_id in support:
        coeff = float(row[shard_id])
        shard_slice = problem.shard_slices[int(shard_id)]
        x_shard = problem.x[shard_slice]
        residual = x_shard @ weights - problem.y[shard_slice]
        gradient += coeff * np.asarray(x_shard.T @ residual).ravel() * scale
    return gradient


class NetworkWorkerPool:
    def __init__(self, config: NetworkExperimentConfig, problem_dir: Path) -> None:
        self.config = config
        self.problem_dir = problem_dir
        self.workers: list[WorkerProcess] = []

    def __enter__(self) -> NetworkWorkerPool:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        logs_dir = self.config.output_dir / "worker_logs"
        ready_dir = self.config.output_dir / "worker_ready"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ready_dir.mkdir(parents=True, exist_ok=True)
        if self.config.use_docker_workers:
            self._start_docker_workers(logs_dir, ready_dir)
            self._wait_until_ready()
            return

        self._start_local_workers(logs_dir, ready_dir)
        self._wait_until_ready()

    def _start_local_workers(self, logs_dir: Path, ready_dir: Path) -> None:
        for worker_id in range(self.config.n_workers):
            port = self.config.base_port + worker_id
            ready_file = ready_dir / f"worker_{worker_id}.ready"
            if ready_file.exists():
                ready_file.unlink()
            log_file = open(logs_dir / f"worker_{worker_id}.log", "w", encoding="utf-8")
            cmd = [
                sys.executable,
                "-m",
                "src.coded_learning_exp.network_runtime",
                "worker",
                "--worker-id",
                str(worker_id),
                "--host",
                self.config.host,
                "--port",
                str(port),
                "--problem-dir",
                str(self.problem_dir),
                "--ready-file",
                str(ready_file),
            ]
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            self.workers.append(
                WorkerProcess(worker_id=worker_id, port=port, log_file=log_file, process=process)
            )

    def _start_docker_workers(self, logs_dir: Path, ready_dir: Path) -> None:
        prefix = self.config.docker_container_prefix or self._default_container_prefix()
        problem_source = str(self.problem_dir.resolve())
        ready_source = str(ready_dir.resolve())
        for worker_id in range(self.config.n_workers):
            port = self.config.base_port + worker_id
            ready_file = ready_dir / f"worker_{worker_id}.ready"
            if ready_file.exists():
                ready_file.unlink()
            log_file = open(logs_dir / f"worker_{worker_id}.log", "w", encoding="utf-8")
            container_name = f"{prefix}-w{worker_id}"
            self._remove_stale_container(container_name)
            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--cpus",
                "1",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-e",
                "MKL_NUM_THREADS=1",
                "-e",
                "NUMEXPR_NUM_THREADS=1",
                "-p",
                f"{self.config.host}:{port}:{self.config.docker_internal_port}",
                "--mount",
                f"type=bind,source={problem_source},target=/problem,readonly",
                "--mount",
                f"type=bind,source={ready_source},target=/ready",
                self.config.docker_image,
                "--worker-id",
                str(worker_id),
                "--host",
                "0.0.0.0",
                "--port",
                str(self.config.docker_internal_port),
                "--problem-dir",
                "/problem",
                "--ready-file",
                f"/ready/worker_{worker_id}.ready",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log_file.write(result.stdout)
                log_file.write(result.stderr)
                log_file.close()
                raise RuntimeError(
                    f"Failed to start Docker worker {worker_id}: {result.stderr.strip()}"
                )
            log_file.write(result.stdout)
            log_file.flush()
            self.workers.append(
                WorkerProcess(
                    worker_id=worker_id,
                    port=port,
                    log_file=log_file,
                    container_name=container_name,
                )
            )

    def _default_container_prefix(self) -> str:
        digest = hashlib.sha1(str(self.config.output_dir.resolve()).encode("utf-8")).hexdigest()[:10]
        return f"coded-net-{digest}"

    def _remove_stale_container(self, container_name: str) -> None:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.config.startup_timeout_seconds
        for worker in self.workers:
            while time.time() < deadline:
                if worker.process is not None and worker.process.poll() is not None:
                    raise RuntimeError(f"Worker {worker.worker_id} exited during startup.")
                if worker.container_name is not None and not self._container_is_running(worker.container_name):
                    logs = self._docker_logs(worker.container_name)
                    raise RuntimeError(
                        f"Docker worker {worker.worker_id} exited during startup.\n{logs}"
                    )
                try:
                    host, port = worker_endpoint(self.config, worker.worker_id)
                    with socket.create_connection((host, port), timeout=0.25) as sock:
                        send_message(sock, {"type": "ping"})
                        try:
                            recv_message(sock)
                        except Exception:
                            pass
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                raise TimeoutError(f"Timed out waiting for worker {worker.worker_id}.")

    def _container_is_running(self, container_name: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"

    def _docker_logs(self, container_name: str) -> str:
        result = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True)
        return (result.stdout + result.stderr).strip()

    def close(self) -> None:
        for worker in self.workers:
            try:
                host, port = worker_endpoint(self.config, worker.worker_id)
                with socket.create_connection((host, port), timeout=1.0) as sock:
                    send_message(sock, {"type": "stop"})
                    recv_message(sock)
            except Exception:
                pass
        for worker in self.workers:
            if worker.container_name is not None:
                logs = self._docker_logs(worker.container_name)
                if logs:
                    worker.log_file.write("\n[docker logs]\n")
                    worker.log_file.write(logs)
                    worker.log_file.write("\n")
                subprocess.run(
                    ["docker", "rm", "-f", worker.container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            elif worker.process is not None:
                try:
                    worker.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    worker.process.terminate()
                    try:
                        worker.process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        worker.process.kill()
            worker.log_file.close()

    def cancel_round(self, round_id: int) -> float:
        start = time.perf_counter()
        threads = []
        for worker in self.workers:
            host, port = worker_endpoint(self.config, worker.worker_id)
            thread = threading.Thread(target=self._cancel_one, args=(host, port, round_id), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout=2.0)
        return time.perf_counter() - start

    def _cancel_one(self, host: str, port: int, round_id: int) -> None:
        try:
            with socket.create_connection((host, port), timeout=1.0) as sock:
                send_message(sock, {"type": "cancel", "round_id": int(round_id)})
                recv_message(sock)
        except Exception:
            pass


def run_network_experiment(config: NetworkExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    problem = make_sparse_ridge_problem(
        n_samples=config.n_samples,
        n_features=config.n_features,
        density=config.density,
        n_shards=config.n_shards,
        l2=config.l2,
        seed=config.seed,
    )
    return run_network_problem(config, problem, dataset_name="synthetic")


def run_network_problem(
    config: NetworkExperimentConfig,
    problem: SparseRidgeProblem,
    dataset_name: str = "external",
    external_worker_pool: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if problem.n_shards != config.n_shards:
        raise ValueError(f"Problem has {problem.n_shards} shards, config has {config.n_shards}.")
    problem_dir = config.output_dir / "problem"
    save_problem(problem, problem_dir)
    strategy_specs = _make_strategy_specs(problem, config)
    worker_states = _apply_worker_alignment(problem, config, _make_worker_states(config))
    unknown = set(config.strategy_names) - set(strategy_specs)
    if unknown:
        raise ValueError(f"Unknown network strategies: {', '.join(sorted(unknown))}")

    records: list[dict[str, Any]] = []
    pool_context = external_worker_pool if external_worker_pool is not None else NetworkWorkerPool(config, problem_dir)
    with pool_context as worker_pool:
        for strategy_index, strategy_name in enumerate(config.strategy_names):
            weights = np.zeros(problem.n_features, dtype=float)
            decode_wall_clock = 0.0
            barrier_wall_clock = 0.0
            spec = strategy_specs[strategy_name]
            for iteration, worker_state in enumerate(worker_states):
                network_round_id = strategy_index * 1_000_000 + iteration
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
                    round_id=network_round_id,
                    rows=rows,
                    assignments=assignments,
                    second_layer_flags=second_flags,
                    weights=weights,
                    worker_state=worker_state,
                    jitter_seed=jitter_seed,
                )
                if hasattr(spec, "update"):
                    spec.update(
                        worker_state=worker_state,
                        result=result,
                        scheduler_seconds=scheduler_seconds,
                        config_label=config_label,
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
                        "worker_errors": result["worker_errors"],
                        "worker_recoveries": result["worker_recoveries"],
                        "reissued_rows": result["reissued_rows"],
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
        mean_worker_errors=("worker_errors", "mean"),
        mean_worker_recoveries=("worker_recoveries", "mean"),
        mean_reissued_rows=("reissued_rows", "mean"),
    ).reset_index()
    summary = summary.merge(extra, on="strategy", how="left")
    metrics.to_csv(config.output_dir / "network_metrics.csv", index=False)
    summary.to_csv(config.output_dir / "network_summary.csv", index=False)
    return metrics, summary


def _run_network_round(
    *,
    config: NetworkExperimentConfig,
    worker_pool: NetworkWorkerPool,
    problem: SparseRidgeProblem,
    strategy: str,
    round_id: int,
    rows: np.ndarray,
    assignments: np.ndarray,
    second_layer_flags: np.ndarray,
    weights: np.ndarray,
    worker_state: WorkerState,
    jitter_seed: int,
) -> dict[str, Any]:
    result_queue: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
    sent_events: list[threading.Event] = []
    dispatch_times = np.zeros(config.n_workers, dtype=float)
    expected_rows = int(rows.shape[0])
    start = time.perf_counter()

    def send_worker_task(
        *,
        logical_worker_id: int,
        target_worker_id: int,
        row_ids: np.ndarray,
        is_reissue: bool = False,
    ) -> None:
        host, port = worker_endpoint(config, target_worker_id)
        payload = {
            "type": "task",
            "round_id": round_id,
            "strategy": strategy,
            "rows": rows[row_ids].copy(),
            "row_ids": row_ids,
            "second_layer_flags": second_layer_flags[row_ids].copy(),
            "weights": weights.copy(),
            "speed": float(worker_state.speeds[target_worker_id]),
            "delay": float(worker_state.delays[target_worker_id]),
            "sleep_scale": config.sleep_scale,
            "cost_scale": config.cost_scale,
            "cancel_poll_seconds": config.cancel_poll_seconds,
            "network_rtt_seconds": config.network_rtt_seconds,
            "network_bandwidth_mbps": config.network_bandwidth_mbps,
            "jitter_seed": jitter_seed,
        }
        event = sent_events[logical_worker_id] if not is_reissue else None
        try:
            with socket.create_connection((host, port), timeout=10.0) as sock:
                send_start = time.perf_counter()
                request_bytes = (
                    int(payload["rows"].nbytes)
                    + int(payload["row_ids"].nbytes)
                    + int(payload["second_layer_flags"].nbytes)
                    + int(payload["weights"].nbytes)
                    + 1024
                )
                request_sleep = _network_transfer_seconds(
                    payload_bytes=request_bytes,
                    rtt_seconds=config.network_rtt_seconds,
                    bandwidth_mbps=config.network_bandwidth_mbps,
                )
                if request_sleep > 0.0:
                    time.sleep(request_sleep)
                send_message(sock, payload)
                if not is_reissue:
                    dispatch_times[logical_worker_id] = time.perf_counter() - send_start
                if event is not None:
                    event.set()
                while True:
                    message = recv_message(sock)
                    message["_reissue"] = bool(is_reissue)
                    message["logical_worker_id"] = int(logical_worker_id)
                    message["target_worker_id"] = int(target_worker_id)
                    result_queue.put((logical_worker_id, message))
                    if message.get("type") in {"done", "error"}:
                        break
        except Exception as exc:
            if event is not None:
                event.set()
            result_queue.put(
                (
                    logical_worker_id,
                    {
                        "type": "error",
                        "worker_id": target_worker_id,
                        "logical_worker_id": int(logical_worker_id),
                        "target_worker_id": int(target_worker_id),
                        "error": repr(exc),
                        "error_type": type(exc).__name__,
                        "assigned_rows": int(row_ids.size),
                        "row_ids": [int(value) for value in row_ids],
                        "_reissue": bool(is_reissue),
                    },
                )
            )

    def worker_client(worker_id: int) -> None:
        row_ids = np.flatnonzero(assignments == worker_id).astype(int)
        send_worker_task(
            logical_worker_id=worker_id,
            target_worker_id=worker_id,
            row_ids=row_ids,
            is_reissue=False,
        )

    threads: list[threading.Thread] = []
    for worker_id in range(config.n_workers):
        sent_events.append(threading.Event())
    for worker_id in range(config.n_workers):
        thread = threading.Thread(target=worker_client, args=(worker_id,), daemon=True)
        thread.start()
        threads.append(thread)
    for event in sent_events:
        event.wait(timeout=10.0)
    dispatch_seconds = float(dispatch_times.max(initial=0.0))

    selected_rows: list[np.ndarray] = []
    selected_gradients: list[np.ndarray] = []
    selected_costs: list[float] = []
    selected_second_flags: list[bool] = []
    completed_rows = 0
    done_workers: set[int] = set()
    failed_workers: set[int] = set()
    completed_row_ids: set[int] = set()
    rows_after_decode = 0
    worker_errors = 0
    worker_recoveries = 0
    reissued_rows = 0
    reissue_pending = 0
    compute_cpu_seconds = 0.0
    network_response_bytes = 0
    network_response_sleep_seconds = 0.0
    decode_cpu_seconds = 0.0
    decode_residual = float("inf")
    decode_success = False
    decoded_gradient: np.ndarray | None = None
    decode_latency = 0.0
    cancel_seconds = 0.0

    recovery_enabled = config.worker_failure_recovery == "reissue"

    while len(done_workers) < config.n_workers or (
        recovery_enabled and not decode_success and reissue_pending > 0
    ):
        worker_id, message = result_queue.get(timeout=120.0)
        is_reissue = bool(message.get("_reissue", False))
        logical_worker_id = int(message.get("logical_worker_id", worker_id))
        if message.get("type") == "error":
            worker_errors += 1
            if is_reissue:
                reissue_pending = max(0, reissue_pending - 1)
                continue
            done_workers.add(logical_worker_id)
            failed_workers.add(logical_worker_id)
            error_type = message.get("error_type")
            if error_type != "EOFError":
                raise RuntimeError(f"Worker {worker_id} failed: {message.get('error')}")
            if recovery_enabled and not decode_success:
                failed_row_ids = np.asarray(message.get("row_ids", []), dtype=int)
                if failed_row_ids.size:
                    missing = np.asarray(
                        [row_id for row_id in failed_row_ids if int(row_id) not in completed_row_ids],
                        dtype=int,
                    )
                else:
                    missing = failed_row_ids
                live_workers = [
                    candidate
                    for candidate in range(config.n_workers)
                    if candidate not in failed_workers
                ]
                if missing.size and live_workers:
                    target_worker = max(
                        live_workers,
                        key=lambda candidate: float(worker_state.speeds[candidate])
                        / (1.0 + float(worker_state.delays[candidate])),
                    )
                    reissue_pending += 1
                    worker_recoveries += 1
                    reissued_rows += int(missing.size)
                    thread = threading.Thread(
                        target=send_worker_task,
                        kwargs={
                            "logical_worker_id": logical_worker_id,
                            "target_worker_id": int(target_worker),
                            "row_ids": missing,
                            "is_reissue": True,
                        },
                        daemon=True,
                    )
                    thread.start()
                    threads.append(thread)
            continue
        if int(message.get("round_id", round_id)) != round_id or message.get("strategy", strategy) != strategy:
            continue
        if message["type"] == "done":
            if is_reissue:
                reissue_pending = max(0, reissue_pending - 1)
            else:
                done_workers.add(logical_worker_id)
            continue

        row_id = int(message.get("row_id", -1))
        if row_id in completed_row_ids:
            continue
        if row_id >= 0:
            completed_row_ids.add(row_id)
        completed_rows += 1
        compute_cpu_seconds += float(message["compute_cpu_seconds"])
        network_response_bytes += int(message.get("network_payload_bytes", 0))
        network_response_sleep_seconds += float(message.get("network_sleep_seconds", 0.0))
        if decode_success:
            rows_after_decode += 1
            continue

        selected_rows.append(np.asarray(message["row"], dtype=float))
        selected_gradients.append(np.asarray(message["gradient"], dtype=float))
        selected_costs.append(float(message["row_cost"]))
        selected_second_flags.append(bool(message["second_layer"]))

        decode = decode_coefficients(np.vstack(selected_rows))
        decode_cpu_seconds += decode.cpu_seconds
        decode_residual = decode.residual
        if decode.success:
            decode_success = True
            decode_latency = time.perf_counter() - start
            decoded_gradient = decode.coefficients @ np.vstack(selected_gradients)
            decoded_gradient = decoded_gradient + problem.l2 * weights
            cancel_seconds = worker_pool.cancel_round(round_id)

    for thread in threads:
        thread.join(timeout=2.0)

    barrier_latency = time.perf_counter() - start
    if not decode_success:
        decoded_gradient = problem.full_gradient(weights)
        decode_latency = barrier_latency
        decode = DecodeResult(False, np.empty(0), decode_residual, 0.0)
    else:
        decode = DecodeResult(True, np.empty(0), decode_residual, decode_cpu_seconds)

    selected_cost = float(np.sum(selected_costs)) if selected_costs else 0.0
    return {
        "gradient": decoded_gradient,
        "decode_latency": float(decode_latency),
        "barrier_latency": float(barrier_latency),
        "decode_success": bool(decode_success),
        "decode_residual": float(decode.residual),
        "decode_cpu_seconds": float(decode_cpu_seconds),
        "selected_rows": int(len(selected_rows)),
        "completed_rows": int(completed_rows),
        "cancelled_rows": int(max(0, expected_rows - completed_rows)),
        "rows_after_decode": int(rows_after_decode),
        "worker_errors": int(worker_errors),
        "worker_recoveries": int(worker_recoveries),
        "reissued_rows": int(reissued_rows),
        "extra_compute": selected_cost / max(float(problem.n_shards), 1.0),
        "second_layer_used": bool(any(selected_second_flags)),
        "worker_compute_cpu_seconds": float(compute_cpu_seconds),
        "network_response_bytes": int(network_response_bytes),
        "network_response_sleep_seconds": float(network_response_sleep_seconds),
        "dispatch_seconds": dispatch_seconds,
        "cancel_seconds": float(cancel_seconds),
    }


def _parse_worker_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Network coded-learning worker.")
    parser.add_argument("worker", nargs="?")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "worker":
        args = _parse_worker_args(argv)
        worker_server_main(
            worker_id=args.worker_id,
            host=args.host,
            port=args.port,
            problem_dir=args.problem_dir,
            ready_file=args.ready_file,
        )
        return
    raise SystemExit("Use `python -m src.coded_learning_exp.network_runtime worker ...`.")


if __name__ == "__main__":
    main()
