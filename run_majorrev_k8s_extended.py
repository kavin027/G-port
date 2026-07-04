from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.coded_learning_exp.data import make_sparse_ridge_problem
from src.coded_learning_exp.network_runtime import save_problem


DEFAULT_STRATEGIES = [
    "speed_aware_uncoded",
    "speculative_replication",
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
    "system_portfolio",
    "guarded_system_portfolio",
    "online_counter_guard_rank_aware_sparse_flexible",
    "online_counter_guard_deadline_aware_sparse_flexible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the major-revision direct-K3s worker-service extension: "
            "more seeds, resource-counter snapshots, and post-run diagnostics."
        )
    )
    parser.add_argument("--workers", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 37, 43, 53])
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
    parser.add_argument("--network-rtt-ms", type=float, default=0.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--problem-seed", type=int, default=17)
    parser.add_argument("--source-host-path", default="/root/coded_distributed_computing")
    parser.add_argument("--problem-host-root", default="/root/coded_k8s_problem")
    parser.add_argument("--out-root", type=Path, default=Path("/root/coded_k8s_results"))
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--worker-port", type=int, default=19000)
    parser.add_argument("--startup-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--wait-timeout", default="1200s")
    parser.add_argument("--master-node", default="")
    parser.add_argument("--worker-nodes", nargs="*", default=[])
    parser.add_argument("--namespace-prefix", default="coded-majorrev")
    parser.add_argument("--pip-index-url", default="https://pypi.tuna.tsinghua.edu.cn/simple")
    parser.add_argument(
        "--portfolio-fallback",
        choices=["static", "speed", "best_safe"],
        default="static",
        help="Fallback mode passed to guarded_system_portfolio K3s runs.",
    )
    parser.add_argument(
        "--worker-failure-recovery",
        choices=["none", "reissue"],
        default="none",
        help="Prototype master-side recovery for closed worker connections.",
    )
    parser.add_argument("--diagnostics-out", type=Path, default=Path("guard_prediction_diagnostics"))
    parser.add_argument("--skip-problem-build", action="store_true")
    parser.add_argument(
        "--prepare-problems-only",
        action="store_true",
        help="Create /root/coded_k8s_problem_w* directories and exit. "
        "Use this before rsyncing hostPath inputs to worker nodes.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra env passed through to worker pods. Repeat for stress tests.",
    )
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "majorrev_k8s_extended_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )

    if not args.skip_problem_build:
        for workers in args.workers:
            prepare_problem(args, workers)
    if args.prepare_problems_only:
        print("Prepared problem directories only; sync them to worker nodes before running K3s.")
        return

    failures: list[dict[str, object]] = []
    for item in plan["runs"]:
        run_dir = Path(item["out_host_path"])
        summary_path = run_dir / "network_summary.csv"
        if args.skip_existing and summary_path.exists():
            print(f"Skipping existing run {run_dir}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        status = run_one(item["command"], run_dir / "k8s_run_command.log")
        collect_status = run_one(item["resource_command"], run_dir / "k8s_resource_collect.log")
        cleanup_status = cleanup_namespace(item["namespace"], run_dir)
        if status != 0 or collect_status != 0 or cleanup_status != 0:
            failures.append(
                {
                    "workers": item["workers"],
                    "seed": item["seed"],
                    "run_status": status,
                    "resource_status": collect_status,
                    "cleanup_status": cleanup_status,
                    "out": str(run_dir),
                }
            )
            if not args.continue_on_error:
                break

    if failures:
        (args.out_root / "majorrev_k8s_extended_failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
        print(f"Encountered {len(failures)} failed run(s). See majorrev_k8s_extended_failures.json.")
        if not args.continue_on_error:
            raise SystemExit(1)

    run_one(
        [sys.executable, "analyze_majorrev_k8s.py", "--root", str(args.out_root)],
        args.out_root / "analyze_majorrev_k8s.log",
    )
    run_one(
        [
            sys.executable,
            "analyze_guard_prediction.py",
            "--root",
            str(args.out_root),
            "--out",
            str(args.diagnostics_out),
        ],
        args.out_root / "analyze_guard_prediction.log",
    )
    run_one(
        [
            sys.executable,
            "analyze_online_tail_predictor.py",
            "--per-round",
            str(args.diagnostics_out / "guard_prediction_per_round.csv"),
            "--out",
            str(args.out_root / "tail_predictor_diagnostics"),
        ],
        args.out_root / "analyze_online_tail_predictor.log",
    )
    print(f"Extended K3s sweep complete. Results root: {args.out_root}")


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    runs = []
    for workers in args.workers:
        problem_host_path = f"{args.problem_host_root}_w{workers}"
        for seed in args.seeds:
            namespace = f"{args.namespace_prefix}-w{workers}-{seed}"
            out_host_path = str(args.out_root / f"majorrev_k8s_w{workers}_seed{seed}")
            command = [
                sys.executable,
                "run_k8s_network_experiment.py",
                "--namespace",
                namespace,
                "--image",
                args.image,
                "--source-host-path",
                args.source_host_path,
                "--problem-host-path",
                problem_host_path,
                "--out-host-path",
                out_host_path,
                "--workers",
                str(workers),
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
                str(seed),
                "--worker-port",
                str(args.worker_port),
                "--startup-timeout-seconds",
                str(args.startup_timeout_seconds),
                "--wait-timeout",
                args.wait_timeout,
                "--pip-index-url",
                args.pip_index_url,
                "--portfolio-fallback",
                args.portfolio_fallback,
                "--worker-failure-recovery",
                args.worker_failure_recovery,
                "--strategies",
                *args.strategies,
            ]
            if args.master_node:
                command.extend(["--master-node", args.master_node])
            if args.worker_nodes:
                command.append("--worker-nodes")
                command.extend(args.worker_nodes)
            for worker_env in args.worker_env:
                command.extend(["--worker-env", worker_env])
            resource_command = [
                sys.executable,
                "collect_k8s_resource_counters.py",
                "--namespace",
                namespace,
                "--out",
                out_host_path,
            ]
            runs.append(
                {
                    "workers": workers,
                    "seed": seed,
                    "namespace": namespace,
                    "problem_host_path": problem_host_path,
                    "out_host_path": out_host_path,
                    "command": command,
                    "resource_command": resource_command,
                }
            )
    return {
        "created_unix": time.time(),
        "workers": args.workers,
        "seeds": args.seeds,
        "source_host_path": args.source_host_path,
        "out_root": str(args.out_root),
        "problem_seed": args.problem_seed,
        "problem_host_root": args.problem_host_root,
        "runs": runs,
    }


def prepare_problem(args: argparse.Namespace, workers: int) -> None:
    problem_dir = Path(f"{args.problem_host_root}_w{workers}")
    problem_dir.mkdir(parents=True, exist_ok=True)
    problem = make_sparse_ridge_problem(
        n_samples=args.samples,
        n_features=args.features,
        density=args.density,
        n_shards=args.shards,
        l2=args.l2,
        seed=args.problem_seed,
    )
    save_problem(problem, problem_dir)
    (problem_dir / "problem_manifest.json").write_text(
        json.dumps(
            {
                "samples": args.samples,
                "features": args.features,
                "density": args.density,
                "shards": args.shards,
                "l2": args.l2,
                "problem_seed": args.problem_seed,
                "workers_label": workers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared problem at {problem_dir}")


def run_one(cmd: list[str], log_path: Path) -> int:
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


def cleanup_namespace(namespace: str, run_dir: Path) -> int:
    return run_one(
        [
            "kubectl",
            "delete",
            "namespace",
            namespace,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=180s",
        ],
        run_dir / "k8s_namespace_cleanup.log",
    )


if __name__ == "__main__":
    main()
