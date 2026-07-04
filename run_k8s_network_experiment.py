from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a direct multi-node Kubernetes TCP worker-service experiment. "
            "The source tree and problem directory must already exist at the "
            "same host paths on the scheduled nodes."
        )
    )
    parser.add_argument("--namespace", default="coded-learning-exp")
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--source-host-path", default="/root/coded_distributed_computing")
    parser.add_argument("--problem-host-path", default="/root/coded_k8s_problem")
    parser.add_argument("--out-host-path", default="/root/coded_k8s_results")
    parser.add_argument(
        "--deps-host-path",
        default="/root/coded_k8s_deps",
        help=(
            "Per-node hostPath used as a shared pip --target directory. "
            "This avoids reinstalling scipy/pandas into every pod overlay."
        ),
    )
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
    parser.add_argument("--network-rtt-ms", type=float, default=0.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alignment-mode", choices=["none", "aligned", "anti"], default="none")
    parser.add_argument("--worker-port", type=int, default=19000)
    parser.add_argument("--startup-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--portfolio-fallback",
        choices=["static", "speed", "best_safe"],
        default="static",
        help=(
            "Fallback used by guarded_system_portfolio when the guard fails. "
            "Use best_safe to deploy the recommended performance-mode rule."
        ),
    )
    parser.add_argument(
        "--worker-failure-recovery",
        choices=["none", "reissue"],
        default="none",
        help="Prototype master-side recovery for closed worker connections.",
    )
    parser.add_argument("--master-node", default="")
    parser.add_argument("--worker-nodes", nargs="*", default=[])
    parser.add_argument(
        "--worker-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable injected into every worker pod. "
        "Can be repeated for worker-service stress runs.",
    )
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-timeout", default="900s")
    parser.add_argument(
        "--pip-index-url",
        default="https://pypi.tuna.tsinghua.edu.cn/simple",
        help="Python package index used inside Kubernetes pods.",
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
    args = parse_args()
    out_path = Path(args.out_host_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.clean:
        _run(["kubectl", "delete", "namespace", args.namespace, "--ignore-not-found=true"], check=False)
        _wait_namespace_deleted(args.namespace)

    if args.master_node:
        _run(["kubectl", "label", "node", args.master_node, "coded-role=master", "--overwrite"])
    for node in args.worker_nodes:
        _run(["kubectl", "label", "node", node, "coded-role=worker", "--overwrite"])

    manifest = _build_manifest(args)
    worker_manifest, job_manifest = _split_worker_and_job_manifest(manifest)
    manifest_path = out_path / "k8s_multinode_manifest.yaml"
    worker_manifest_path = out_path / "k8s_workers.yaml"
    job_manifest_path = out_path / "k8s_master_job.yaml"
    manifest_path.write_text(manifest, encoding="utf-8")
    worker_manifest_path.write_text(worker_manifest, encoding="utf-8")
    job_manifest_path.write_text(job_manifest, encoding="utf-8")
    (out_path / "k8s_multinode_run_config.json").write_text(
        json.dumps(_run_config(args), indent=2),
        encoding="utf-8",
    )

    _run(["kubectl", "apply", "-f", str(worker_manifest_path)])
    _run(
        [
            "kubectl",
            "-n",
            args.namespace,
            "rollout",
            "status",
            "statefulset/coded-worker",
            f"--timeout={args.wait_timeout}",
        ]
    )
    _run(["kubectl", "apply", "-f", str(job_manifest_path)])
    _run(
        [
            "kubectl",
            "-n",
            args.namespace,
            "wait",
            "--for=condition=complete",
            "job/coded-master",
            f"--timeout={args.wait_timeout}",
        ]
    )
    logs = _run(["kubectl", "-n", args.namespace, "logs", "job/coded-master"], capture=True)
    (out_path / "master_job.log").write_text(logs.stdout, encoding="utf-8")
    pods = _run(["kubectl", "-n", args.namespace, "get", "pods", "-o", "wide"], capture=True)
    (out_path / "k8s_pods_wide.txt").write_text(pods.stdout, encoding="utf-8")
    print(logs.stdout)
    print(f"Wrote Kubernetes run outputs to {out_path}")


def _run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": "direct_kubernetes_multinode",
        "source_host_path": args.source_host_path,
        "problem_host_path": args.problem_host_path,
        "out_host_path": args.out_host_path,
        "deps_host_path": args.deps_host_path,
        "workers": args.workers,
        "worker_port": args.worker_port,
        "image": args.image,
        "namespace": args.namespace,
        "master_node": args.master_node,
        "worker_nodes": args.worker_nodes,
        "strategies": args.strategies,
        "worker_env": dict(_parse_worker_env(args.worker_env)),
        "portfolio_fallback": args.portfolio_fallback,
        "host_network": False,
        "data_path": "master pod -> Kubernetes DNS/headless service -> worker pods",
    }


def _build_manifest(args: argparse.Namespace) -> str:
    worker_hosts = [
        f"coded-worker-{idx}.coded-workers.{args.namespace}.svc.cluster.local"
        for idx in range(args.workers)
    ]
    master_command = _shell_command(
        [
            _pip_install(args.pip_index_url),
            "cd /app",
            _master_command(args, worker_hosts),
        ]
    )
    worker_command = _shell_command(
        [
            _pip_install(args.pip_index_url),
            "cd /app",
            (
                "exec python -m src.coded_learning_exp.network_runtime worker "
                "--worker-id ${HOSTNAME##*-} "
                "--host 0.0.0.0 "
                f"--port {args.worker_port} "
                "--problem-dir /problem "
                "--ready-file /tmp/worker.ready"
            ),
        ]
    )
    master_selector = _node_selector("master") if args.master_node else ""
    worker_selector = _node_selector("worker") if args.worker_nodes else ""
    return textwrap.dedent(
        f"""\
        apiVersion: v1
        kind: Namespace
        metadata:
          name: {args.namespace}
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: coded-workers
          namespace: {args.namespace}
        spec:
          clusterIP: None
          selector:
            app: coded-worker
          ports:
            - name: tcp-worker
              port: {args.worker_port}
              targetPort: {args.worker_port}
        ---
        apiVersion: apps/v1
        kind: StatefulSet
        metadata:
          name: coded-worker
          namespace: {args.namespace}
        spec:
          serviceName: coded-workers
          podManagementPolicy: Parallel
          replicas: {args.workers}
          selector:
            matchLabels:
              app: coded-worker
          template:
            metadata:
              labels:
                app: coded-worker
            spec:
{_indent(worker_selector, 14)}
              terminationGracePeriodSeconds: 5
              topologySpreadConstraints:
                - maxSkew: 1
                  topologyKey: kubernetes.io/hostname
                  whenUnsatisfiable: ScheduleAnyway
                  labelSelector:
                    matchLabels:
                      app: coded-worker
              containers:
                - name: worker
                  image: {args.image}
                  imagePullPolicy: IfNotPresent
                  command: ["/bin/sh", "-lc"]
                  args:
                    - |
{_indent(worker_command, 22)}
                  ports:
                    - containerPort: {args.worker_port}
                      name: tcp-worker
                  readinessProbe:
                    tcpSocket:
                      port: {args.worker_port}
                    periodSeconds: 2
                    failureThreshold: 60
                  env:
                    - name: OMP_NUM_THREADS
                      value: "1"
                    - name: OPENBLAS_NUM_THREADS
                      value: "1"
                    - name: MKL_NUM_THREADS
                      value: "1"
                    - name: NUMEXPR_NUM_THREADS
                      value: "1"
                    - name: PYTHONUNBUFFERED
                      value: "1"
{_indent(_worker_env_yaml(args.worker_env), 20)}
                  volumeMounts:
                    - name: source
                      mountPath: /app
                      readOnly: true
                    - name: problem
                      mountPath: /problem
                      readOnly: true
                    - name: pip-cache
                      mountPath: /root/.cache/pip
                    - name: python-deps
                      mountPath: /deps
              volumes:
                - name: source
                  hostPath:
                    path: {args.source_host_path}
                    type: Directory
                - name: problem
                  hostPath:
                    path: {args.problem_host_path}
                    type: Directory
                - name: pip-cache
                  hostPath:
                    path: /root/.cache/pip
                    type: DirectoryOrCreate
                - name: python-deps
                  hostPath:
                    path: {args.deps_host_path}
                    type: DirectoryOrCreate
        ---
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: coded-master
          namespace: {args.namespace}
        spec:
          backoffLimit: 0
          activeDeadlineSeconds: 1200
          template:
            metadata:
              labels:
                app: coded-master
            spec:
{_indent(master_selector, 14)}
              restartPolicy: Never
              containers:
                - name: master
                  image: {args.image}
                  imagePullPolicy: IfNotPresent
                  command: ["/bin/sh", "-lc"]
                  args:
                    - |
{_indent(master_command, 22)}
                  env:
                    - name: OMP_NUM_THREADS
                      value: "1"
                    - name: OPENBLAS_NUM_THREADS
                      value: "1"
                    - name: MKL_NUM_THREADS
                      value: "1"
                    - name: NUMEXPR_NUM_THREADS
                      value: "1"
                    - name: PYTHONUNBUFFERED
                      value: "1"
                  volumeMounts:
                    - name: source
                      mountPath: /app
                      readOnly: true
                    - name: problem
                      mountPath: /problem
                      readOnly: true
                    - name: out
                      mountPath: /out
                    - name: pip-cache
                      mountPath: /root/.cache/pip
                    - name: python-deps
                      mountPath: /deps
              volumes:
                - name: source
                  hostPath:
                    path: {args.source_host_path}
                    type: Directory
                - name: problem
                  hostPath:
                    path: {args.problem_host_path}
                    type: Directory
                - name: out
                  hostPath:
                    path: {args.out_host_path}
                    type: DirectoryOrCreate
                - name: pip-cache
                  hostPath:
                    path: /root/.cache/pip
                    type: DirectoryOrCreate
                - name: python-deps
                  hostPath:
                    path: {args.deps_host_path}
                    type: DirectoryOrCreate
        """
    )


def _master_command(args: argparse.Namespace, worker_hosts: list[str]) -> str:
    parts = [
        "python -m src.coded_learning_exp.direct_docker_master",
        "--problem-dir /problem",
        "--out /out",
        f"--workers {args.workers}",
        f"--worker-hosts {','.join(worker_hosts)}",
        f"--worker-port {args.worker_port}",
        f"--samples {args.samples}",
        f"--features {args.features}",
        f"--density {args.density}",
        f"--shards {args.shards}",
        f"--rounds {args.rounds}",
        f"--learning-rate {args.learning_rate}",
        f"--l2 {args.l2}",
        f"--scenario {args.scenario}",
        f"--drift-period {args.drift_period}",
        f"--straggler-fraction {args.straggler_fraction}",
        f"--straggler-slowdown {args.straggler_slowdown}",
        f"--burst-probability {args.burst_probability}",
        f"--sleep-scale {args.sleep_scale}",
        f"--cost-scale {args.cost_scale}",
        f"--cancel-poll-seconds {args.cancel_poll_seconds}",
        f"--network-rtt-ms {args.network_rtt_ms}",
        f"--network-bandwidth-mbps {args.network_bandwidth_mbps}",
        f"--seed {args.seed}",
        f"--alignment-mode {args.alignment_mode}",
        f"--startup-timeout-seconds {args.startup_timeout_seconds}",
        f"--portfolio-fallback {args.portfolio_fallback}",
        f"--worker-failure-recovery {args.worker_failure_recovery}",
        "--common-jitter-across-strategies",
        "--dataset-name direct_kubernetes_multinode",
        "--summary-name k8s_summary.csv",
        "--strategies",
        *args.strategies,
    ]
    return " ".join(parts)


def _split_worker_and_job_manifest(manifest: str) -> tuple[str, str]:
    marker = "\n---\napiVersion: batch/v1\nkind: Job\n"
    if marker not in manifest:
        raise ValueError("Could not split Kubernetes manifest into worker and job parts.")
    before, after = manifest.split(marker, 1)
    return before.rstrip() + "\n", "apiVersion: batch/v1\nkind: Job\n" + after


def _parse_worker_env(values: list[str]) -> list[tuple[str, str]]:
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


def _worker_env_yaml(values: list[str]) -> str:
    lines: list[str] = []
    for key, value in _parse_worker_env(values):
        lines.append(f"- name: {key}")
        lines.append(f"  value: {json.dumps(value)}")
    return "\n".join(lines)


def _pip_install(index_url: str) -> str:
    return (
        "if [ ! -f /deps/.coded_deps_ready_v2 ]; then "
        "while ! mkdir /deps/.install.lock 2>/dev/null; do "
        "[ -f /deps/.coded_deps_ready_v2 ] && break; sleep 1; "
        "done; "
        "if [ ! -f /deps/.coded_deps_ready_v2 ]; then "
        "python -m pip install --prefer-binary --target /deps "
        f"-i {index_url} numpy scipy pandas && "
        "touch /deps/.coded_deps_ready_v2; "
        "fi; "
        "rmdir /deps/.install.lock 2>/dev/null || true; "
        "fi; "
        "export PYTHONPATH=/deps:${PYTHONPATH:-}"
    )


def _shell_command(parts: list[str]) -> str:
    return " && \\\n".join(parts)


def _node_selector(role: str) -> str:
    return textwrap.dedent(
        f"""\
        nodeSelector:
          coded-role: {role}
        """
    ).rstrip()


def _indent(value: str, spaces: int) -> str:
    if not value:
        return ""
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in value.splitlines())


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd))
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=capture,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stdout or ''}\n{result.stderr or ''}"
        )
    return result


def _wait_namespace_deleted(namespace: str) -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        result = _run(["kubectl", "get", "namespace", namespace], check=False, capture=True)
        if result.returncode != 0:
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
