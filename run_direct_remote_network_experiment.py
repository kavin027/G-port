from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import socket
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import paramiko

from src.coded_learning_exp.data import make_sparse_ridge_problem
from src.coded_learning_exp.network_runtime import NetworkExperimentConfig, recv_message, save_problem, send_message
from run_tunneled_remote_network_experiment import ExternalWorkerPool, _run_external_network_problem


@dataclass(frozen=True)
class DirectRemoteServerConfig:
    ssh_host: str
    ssh_port: int
    user: str
    password: str
    repo_dir: str
    remote_out: str
    remote_base_port: int
    worker_host: str


class DirectRemoteWorkerHost:
    def __init__(self, *, server: DirectRemoteServerConfig, n_workers: int) -> None:
        self.server = server
        self.n_workers = n_workers
        self.client: paramiko.SSHClient | None = None
        self.sftp: paramiko.SFTPClient | None = None
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
            self.server.ssh_host,
            port=self.server.ssh_port,
            username=self.server.user,
            password=self.server.password,
            timeout=20,
            banner_timeout=20,
            auth_timeout=20,
        )
        self.client = client
        self.sftp = client.open_sftp()

    def close(self) -> None:
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
            self.sftp.put(str(local_problem_dir / name), posixpath.join(self.remote_problem_dir, name))

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
                "PY=$(command -v python3 || command -v python || "
                "([ -x /root/miniconda3/bin/python ] && echo /root/miniconda3/bin/python) || true); "
                'test -n "$PY"; '
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "NUMEXPR_NUM_THREADS=1 "
                f'"$PY" -m src.coded_learning_exp.network_runtime worker '
                f"--worker-id {worker_id} --host 0.0.0.0 --port {port} "
                f"--problem-dir {shlex.quote(self.remote_problem_dir)} "
                f"--ready-file {shlex.quote(ready)} "
                f"< /dev/null > {shlex.quote(log)} 2>&1 & "
                f"echo $! > {shlex.quote(pidfile)}"
                ") >/dev/null 2>&1 &"
            )
            try:
                self._exec(cmd, get_pty=False)
            except RuntimeError:
                # Some restricted SSH frontends do not return an exit-status for
                # backgrounded commands. The ready-file and direct-ping checks
                # below are the real success criteria.
                pass
        self._wait_remote_ready()
        self._wait_direct_ready()

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

    def _wait_direct_ready(self) -> None:
        deadline = time.time() + 60.0
        pending = set(range(self.n_workers))
        while pending and time.time() < deadline:
            for worker_id in list(pending):
                try:
                    with socket.create_connection(
                        (self.server.worker_host, self.server.remote_base_port + worker_id),
                        timeout=0.75,
                    ) as sock:
                        send_message(sock, {"type": "ping"})
                        recv_message(sock)
                    pending.remove(worker_id)
                except Exception:
                    pass
            if pending:
                time.sleep(0.1)
        if pending:
            raise TimeoutError(f"Timed out waiting for direct worker ports: {sorted(pending)}")

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
            "Run the TCP master locally while workers run on a remote host with "
            "directly reachable TCP ports. SSH is used only to start workers and "
            "copy the problem files, not to forward experiment traffic."
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
    parser.add_argument("--out", type=Path, default=Path("direct_remote_network_results"))
    parser.add_argument("--sleep-scale", type=float, default=0.03)
    parser.add_argument("--cost-scale", type=float, default=0.006)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.003)
    parser.add_argument("--network-rtt-ms", type=float, default=0.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--worker-host", required=True, help="Public or routable host used by the master.")
    parser.add_argument("--remote-ssh-host", required=True)
    parser.add_argument("--remote-ssh-port", type=int, required=True)
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--remote-password", default=os.environ.get("REMOTE_PASSWORD", ""))
    parser.add_argument("--remote-repo", default="/root/coded_distributed_computing_socc_runtime")
    parser.add_argument("--remote-out", default="direct_remote_network_results")
    parser.add_argument("--remote-base-port", type=int, default=38000)
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
        ],
    )
    return parser.parse_args()


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
        host=args.worker_host,
        base_port=args.remote_base_port,
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

    remote = DirectRemoteWorkerHost(
        server=DirectRemoteServerConfig(
            ssh_host=args.remote_ssh_host,
            ssh_port=args.remote_ssh_port,
            user=args.remote_user,
            password=args.remote_password,
            repo_dir=args.remote_repo,
            remote_out=args.remote_out,
            remote_base_port=args.remote_base_port,
            worker_host=args.worker_host,
        ),
        n_workers=config.n_workers,
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
            dataset_name="synthetic-direct-remote",
        )
        del metrics
        print(summary.to_string(index=False))
        print(f"\nWrote direct remote metrics to {config.output_dir}")
    finally:
        try:
            remote.stop_workers(worker_pool)
        finally:
            remote.close()


if __name__ == "__main__":
    main()
