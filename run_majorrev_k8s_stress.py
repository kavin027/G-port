"""Run small K3s interference/failure stress cases for the worker-service path.

The script is a thin orchestrator around ``run_majorrev_k8s_extended.py``.  It
keeps the same direct worker-service entrypoint and output layout, adding only
case-level stress: a CPU-hog deployment or worker environment variables that
exercise cancellation and connection-failure hooks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path


CASE_ENV = {
    "baseline": [],
    "cancel_ack_20ms": ["CODED_CANCEL_ACK_DELAY_MS=20"],
    "cancel_ack_50ms": ["CODED_CANCEL_ACK_DELAY_MS=50"],
    "close_connection": ["CODED_STRESS_WORKER_ID=0", "CODED_CLOSE_ON_TASK=1"],
    "exit_on_task": ["CODED_STRESS_WORKER_ID=0", "CODED_EXIT_ON_TASK=1"],
}

DEFAULT_STRATEGIES = [
    "speed_aware_uncoded",
    "speculative_replication",
    "sparse_flexible_static",
    "rank_aware_sparse_flexible",
    "system_portfolio",
    "guarded_system_portfolio",
    "online_counter_guard_deadline_aware_sparse_flexible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", type=int, default=[24])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31, 53])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["baseline", "cpu_hog", "cancel_ack_20ms"],
        choices=["baseline", "cpu_hog", *CASE_ENV.keys()],
    )
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--features", type=int, default=240)
    parser.add_argument("--density", type=float, default=0.014)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--source-host-path", default="/root/coded_distributed_computing")
    parser.add_argument("--problem-host-root", default="/root/coded_k8s_stress_problem")
    parser.add_argument("--out-root", type=Path, default=Path("/root/coded_k8s_stress_results"))
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--master-node", default="")
    parser.add_argument("--worker-nodes", nargs="*", default=[])
    parser.add_argument("--namespace-prefix", default="coded-stress")
    parser.add_argument("--pip-index-url", default="https://pypi.tuna.tsinghua.edu.cn/simple")
    parser.add_argument("--hog-namespace", default="coded-interference")
    parser.add_argument("--hog-replicas", type=int, default=2)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--worker-failure-recovery",
        choices=["none", "reissue"],
        default="none",
        help="Prototype master-side recovery used by the underlying K3s runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "k8s_stress_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    label_nodes(args)
    failures: list[dict[str, object]] = []
    for case in args.cases:
        if case == "cpu_hog":
            apply_cpu_hog(args)
            time.sleep(10)
        try:
            status = run_case(args, case)
        finally:
            if case == "cpu_hog":
                delete_cpu_hog(args)
        if status != 0:
            failures.append({"case": case, "status": status})
            if not args.continue_on_error:
                break

    if failures:
        (args.out_root / "k8s_stress_failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
        if not args.continue_on_error:
            raise SystemExit(1)

    run_one(
        [
            sys.executable,
            "analyze_k8s_stress.py",
            "--root",
            str(args.out_root),
        ],
        args.out_root / "analyze_k8s_stress.log",
    )
    print(f"K3s stress sweep complete. Results root: {args.out_root}")


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "created_unix": time.time(),
        "cases": args.cases,
        "workers": args.workers,
        "seeds": args.seeds,
        "strategies": args.strategies,
        "source_host_path": args.source_host_path,
        "out_root": str(args.out_root),
        "cpu_hog": {
            "namespace": args.hog_namespace,
            "replicas": args.hog_replicas,
            "image": args.image,
        },
    }


def run_case(args: argparse.Namespace, case: str) -> int:
    case_slug = case.replace("_", "-")
    case_root = args.out_root / case
    case_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "run_majorrev_k8s_extended.py",
        "--workers",
        *[str(value) for value in args.workers],
        "--seeds",
        *[str(value) for value in args.seeds],
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
        "--source-host-path",
        args.source_host_path,
        "--problem-host-root",
        f"{args.problem_host_root}_{case}",
        "--out-root",
        str(case_root),
        "--image",
        args.image,
        "--worker-failure-recovery",
        args.worker_failure_recovery,
        "--namespace-prefix",
        f"{args.namespace_prefix}-{case_slug}",
        "--pip-index-url",
        args.pip_index_url,
        "--diagnostics-out",
        str(case_root / "guard_prediction_diagnostics"),
        "--strategies",
        *args.strategies,
    ]
    if args.master_node:
        command.extend(["--master-node", args.master_node])
    if args.worker_nodes:
        command.append("--worker-nodes")
        command.extend(args.worker_nodes)
    if args.skip_existing:
        command.append("--skip-existing")
    if args.continue_on_error:
        command.append("--continue-on-error")
    for worker_env in CASE_ENV.get(case, []):
        command.extend(["--worker-env", worker_env])
    return run_one(command, case_root / "run_case.log")


def label_nodes(args: argparse.Namespace) -> None:
    if args.master_node:
        run_one(
            ["kubectl", "label", "node", args.master_node, "coded-role=master", "--overwrite"],
            args.out_root / "k8s_label_master.log",
        )
    for idx, node in enumerate(args.worker_nodes):
        run_one(
            ["kubectl", "label", "node", node, "coded-role=worker", "--overwrite"],
            args.out_root / f"k8s_label_worker_{idx}.log",
        )


def apply_cpu_hog(args: argparse.Namespace) -> None:
    manifest = cpu_hog_manifest(args)
    path = args.out_root / "cpu_hog.yaml"
    path.write_text(manifest, encoding="utf-8")
    run_one(["kubectl", "apply", "-f", str(path)], args.out_root / "cpu_hog_apply.log")
    run_one(
        [
            "kubectl",
            "-n",
            args.hog_namespace,
            "rollout",
            "status",
            "deployment/coded-cpu-hog",
            "--timeout=180s",
        ],
        args.out_root / "cpu_hog_rollout.log",
    )


def delete_cpu_hog(args: argparse.Namespace) -> None:
    run_one(
        ["kubectl", "delete", "namespace", args.hog_namespace, "--ignore-not-found=true"],
        args.out_root / "cpu_hog_delete.log",
    )


def cpu_hog_manifest(args: argparse.Namespace) -> str:
    selector = ""
    if args.worker_nodes:
        selector = textwrap.dedent(
            """\
            nodeSelector:
              coded-role: worker
            """
        ).rstrip()
    return textwrap.dedent(
        f"""\
        apiVersion: v1
        kind: Namespace
        metadata:
          name: {args.hog_namespace}
        ---
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: coded-cpu-hog
          namespace: {args.hog_namespace}
        spec:
          replicas: {args.hog_replicas}
          selector:
            matchLabels:
              app: coded-cpu-hog
          template:
            metadata:
              labels:
                app: coded-cpu-hog
            spec:
{indent(selector, 14)}
              terminationGracePeriodSeconds: 1
              containers:
                - name: hog
                  image: {args.image}
                  imagePullPolicy: IfNotPresent
                  command: ["/bin/sh", "-lc"]
                  args:
                    - |
                      python - <<'PY'
                      import time
                      while True:
                          end = time.time() + 0.95
                          while time.time() < end:
                              pass
                          time.sleep(0.05)
                      PY
        """
    )


def indent(value: str, spaces: int) -> str:
    if not value:
        return ""
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in value.splitlines())


def run_one(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    log_path.write_text(
        (result.stdout or "") + ("\n[stderr]\n" + (result.stderr or "") if result.stderr else ""),
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return int(result.returncode)


if __name__ == "__main__":
    main()
