from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from src.coded_learning_exp.data import make_sparse_ridge_problem
from src.coded_learning_exp.network_runtime import save_problem


IMAGE = "coded-learning-network-worker:local"
INTERNAL_PORT = 19000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a direct Docker-bridge TCP experiment: master and worker "
            "containers share a Docker network and communicate by container DNS, "
            "with no host port publishing or SSH forwarding on the data path."
        )
    )
    parser.add_argument("--out", type=Path, default=Path("direct_docker_multinode_smoke"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--features", type=int, default=240)
    parser.add_argument("--density", type=float, default=0.014)
    parser.add_argument("--shards", type=int, default=8)
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
    parser.add_argument("--network-rtt-ms", type=float, default=4.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alignment-mode", choices=["none", "aligned", "anti"], default="none")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--network-name", default=None)
    parser.add_argument("--container-prefix", default=None)
    parser.add_argument(
        "--portfolio-fallback",
        choices=["static", "speed", "best_safe"],
        default="static",
        help="Fallback used by guarded_system_portfolio when the guard fails.",
    )
    parser.add_argument(
        "--worker-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable injected into every worker container. "
        "Can be repeated for worker-service stress tests.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--keep-containers", action="store_true")
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


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=capture)
    if check and result.returncode != 0:
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stdout}\n{stderr}")
    return result


def docker_available() -> None:
    run(["docker", "info"], capture=True)


def build_image(image: str) -> None:
    run(["docker", "build", "-f", "docker/Dockerfile.network-worker", "-t", image, "."])


def remove_container(name: str) -> None:
    run(["docker", "rm", "-f", name], check=False, capture=True)


def remove_network(name: str) -> None:
    run(["docker", "network", "rm", name], check=False, capture=True)


def container_running(name: str) -> bool:
    result = run(["docker", "inspect", "-f", "{{.State.Running}}", name], check=False, capture=True)
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def docker_logs(name: str) -> str:
    result = run(["docker", "logs", name], check=False, capture=True)
    return (result.stdout or "") + (result.stderr or "")


def parse_worker_env(values: list[str]) -> list[tuple[str, str]]:
    env: list[tuple[str, str]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"--worker-env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--worker-env has an empty key: {item!r}")
        env.append((key, value))
    return env


def docker_env_args(env: list[tuple[str, str]]) -> list[str]:
    args: list[str] = []
    for key, value in env:
        args.extend(["-e", f"{key}={value}"])
    return args


def write_worker_logs(prefix: str, workers: int, logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    for worker_id in range(workers):
        name = f"{prefix}-w{worker_id}"
        (logs_dir / f"worker_{worker_id}.log").write_text(docker_logs(name), encoding="utf-8")


def wait_ready(out: Path, prefix: str, workers: int, timeout: float = 60.0) -> None:
    ready_dir = out / "worker_ready"
    deadline = time.time() + timeout
    pending = set(range(workers))
    while pending and time.time() < deadline:
        for worker_id in list(pending):
            if not container_running(f"{prefix}-w{worker_id}"):
                raise RuntimeError(f"Worker container {prefix}-w{worker_id} exited early:\n{docker_logs(f'{prefix}-w{worker_id}')}")
            if (ready_dir / f"worker_{worker_id}.ready").exists():
                pending.remove(worker_id)
        if pending:
            time.sleep(0.2)
    if pending:
        raise TimeoutError(f"Timed out waiting for workers: {sorted(pending)}")


def mount_arg(source: Path, target: str, readonly: bool = False) -> str:
    suffix = ",readonly" if readonly else ""
    return f"type=bind,source={source.resolve()},target={target}{suffix}"


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    args = parse_args()
    worker_env = parse_worker_env(args.worker_env)
    docker_available()
    if not args.skip_build:
        build_image(args.image)

    out = args.out.resolve()
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    ready_dir = out / "worker_ready"
    problem_dir = out / "problem"
    ready_dir.mkdir(parents=True, exist_ok=True)
    for ready_file in ready_dir.glob("*.ready"):
        ready_file.unlink()

    problem = make_sparse_ridge_problem(
        n_samples=args.samples,
        n_features=args.features,
        density=args.density,
        n_shards=args.shards,
        l2=args.l2,
        seed=args.seed,
    )
    save_problem(problem, problem_dir)

    digest = hashlib.sha1(str(out).encode("utf-8")).hexdigest()[:10]
    prefix = args.container_prefix or f"coded-direct-{digest}"
    network = args.network_name or f"{prefix}-net"
    manifest = {
        "mode": "direct_docker_bridge",
        "data_path": "master container -> Docker bridge DNS -> worker containers",
        "host_port_publishing": False,
        "ssh_forwarding": False,
        "workers": args.workers,
        "internal_port": INTERNAL_PORT,
        "network": network,
        "image": args.image,
        "strategies": args.strategies,
        "portfolio_fallback": args.portfolio_fallback,
        "network_rtt_ms": args.network_rtt_ms,
        "network_bandwidth_mbps": args.network_bandwidth_mbps,
        "worker_env": {key: value for key, value in worker_env},
    }
    (out / "direct_docker_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    worker_names = [f"{prefix}-w{worker_id}" for worker_id in range(args.workers)]
    master_name = f"{prefix}-master"
    try:
        remove_container(master_name)
        for name in worker_names:
            remove_container(name)
        remove_network(network)
        run(["docker", "network", "create", network])
        for worker_id, name in enumerate(worker_names):
            run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    name,
                    "--network",
                    network,
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
                    *docker_env_args(worker_env),
                    "--mount",
                    mount_arg(problem_dir, "/problem", readonly=True),
                    "--mount",
                    mount_arg(ready_dir, "/ready"),
                    args.image,
                    "--worker-id",
                    str(worker_id),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(INTERNAL_PORT),
                    "--problem-dir",
                    "/problem",
                    "--ready-file",
                    f"/ready/worker_{worker_id}.ready",
                ]
            )
        wait_ready(out, prefix, args.workers)
        master_cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            master_name,
            "--network",
            network,
            "--cpus",
            "2",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-e",
            "MKL_NUM_THREADS=1",
            "-e",
            "NUMEXPR_NUM_THREADS=1",
            "--mount",
            mount_arg(out, "/out"),
            "--entrypoint",
            "python",
            args.image,
            "-m",
            "src.coded_learning_exp.direct_docker_master",
            "--problem-dir",
            "/out/problem",
            "--out",
            "/out",
            "--workers",
            str(args.workers),
            "--worker-hosts",
            ",".join(worker_names),
            "--worker-port",
            str(INTERNAL_PORT),
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
            "--learning-rate",
            str(args.learning_rate),
            "--l2",
            str(args.l2),
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
            "--network-rtt-ms",
            str(args.network_rtt_ms),
            "--network-bandwidth-mbps",
            str(args.network_bandwidth_mbps),
            "--seed",
            str(args.seed),
            "--alignment-mode",
            args.alignment_mode,
            "--portfolio-fallback",
            args.portfolio_fallback,
            "--common-jitter-across-strategies",
            "--strategies",
            *args.strategies,
        ]
        result = run(master_cmd, capture=True)
        (out / "master_stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (out / "master_stderr.log").write_text(result.stderr or "", encoding="utf-8")
        print(result.stdout)
        print(f"Wrote direct Docker network results to {out}")
    finally:
        write_worker_logs(prefix, args.workers, out / "worker_logs")
        if not args.keep_containers:
            remove_container(master_name)
            for name in worker_names:
                remove_container(name)
            remove_network(network)


if __name__ == "__main__":
    main()
