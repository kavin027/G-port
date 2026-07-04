from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_STRATEGIES = [
    "speed_aware_uncoded",
    "speculative_replication",
    "sparse_flexible_static",
    "rank_aware_sparse_flexible",
    "system_portfolio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-seed direct remote TCP validation sweep. Workers run on "
            "a remote host with routable TCP ports; SSH is used only for setup."
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
    parser.add_argument("--sleep-scale", type=float, default=0.03)
    parser.add_argument("--cost-scale", type=float, default=0.006)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.003)
    parser.add_argument("--network-rtt-ms", type=float, default=0.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--alignment-mode", choices=["none", "aligned", "anti"], default="none")
    parser.add_argument("--common-jitter-across-strategies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31, 43])
    parser.add_argument("--output-root", type=Path, default=Path("direct_remote_sweep"))
    parser.add_argument("--diagnostics-out", type=Path, default=Path("direct_remote_diagnostics"))
    parser.add_argument("--baseline-strategy", default="speed_aware_uncoded")
    parser.add_argument("--analyze", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--remote-ssh-host", required=True)
    parser.add_argument("--remote-ssh-port", type=int, required=True)
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--remote-password", default=os.environ.get("REMOTE_PASSWORD", ""))
    parser.add_argument("--remote-repo", default="/root/coded_distributed_computing_socc_runtime")
    parser.add_argument("--remote-out-prefix", default="direct_remote_seed")
    parser.add_argument("--remote-base-port", type=int, default=38000)
    parser.add_argument("--port-stride", type=int, default=100)
    return parser.parse_args()


def _append(cmd: list[str], name: str, value: object) -> None:
    cmd.extend([name, str(value)])


def build_run_command(args: argparse.Namespace, seed: int, index: int, out_dir: Path) -> list[str]:
    remote_base_port = args.remote_base_port + index * args.port_stride
    cmd = [sys.executable, "run_direct_remote_network_experiment.py"]
    for name, value in [
        ("--samples", args.samples),
        ("--features", args.features),
        ("--density", args.density),
        ("--shards", args.shards),
        ("--workers", args.workers),
        ("--rounds", args.rounds),
        ("--learning-rate", args.learning_rate),
        ("--l2", args.l2),
        ("--scenario", args.scenario),
        ("--drift-period", args.drift_period),
        ("--straggler-fraction", args.straggler_fraction),
        ("--straggler-slowdown", args.straggler_slowdown),
        ("--burst-probability", args.burst_probability),
        ("--seed", seed),
        ("--out", out_dir),
        ("--sleep-scale", args.sleep_scale),
        ("--cost-scale", args.cost_scale),
        ("--cancel-poll-seconds", args.cancel_poll_seconds),
        ("--network-rtt-ms", args.network_rtt_ms),
        ("--network-bandwidth-mbps", args.network_bandwidth_mbps),
        ("--worker-host", args.worker_host),
        ("--remote-ssh-host", args.remote_ssh_host),
        ("--remote-ssh-port", args.remote_ssh_port),
        ("--remote-user", args.remote_user),
        ("--remote-repo", args.remote_repo),
        ("--remote-out", f"{args.remote_out_prefix}_{seed}"),
        ("--remote-base-port", remote_base_port),
        ("--alignment-mode", args.alignment_mode),
    ]:
        _append(cmd, name, value)
    if args.common_jitter_across_strategies:
        cmd.append("--common-jitter-across-strategies")
    cmd.append("--strategies")
    cmd.extend(args.strategies)
    return cmd


def main() -> None:
    args = parse_args()
    if not args.remote_password:
        raise SystemExit("Set --remote-password or REMOTE_PASSWORD.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["REMOTE_PASSWORD"] = args.remote_password

    out_dirs: list[Path] = []
    for index, seed in enumerate(args.seeds):
        out_dir = args.output_root / f"seed_{seed}"
        out_dirs.append(out_dir)
        cmd = build_run_command(args, seed, index, out_dir)
        print(
            "\n== Running direct remote seed "
            f"{seed} with remote_base_port={args.remote_base_port + index * args.port_stride} =="
        )
        print(" ".join(str(part) for part in cmd))
        subprocess.run(cmd, check=True, env=env)

    if args.analyze:
        cmd = [
            sys.executable,
            "analyze_network_container_results.py",
            *[str(path) for path in out_dirs],
            "--baseline-strategy",
            args.baseline_strategy,
            "--out",
            str(args.diagnostics_out),
        ]
        print("\n== Analyzing direct remote sweep ==")
        print(" ".join(str(part) for part in cmd))
        subprocess.run(cmd, check=True, env=env)
        print(f"\nWrote diagnostics to {args.diagnostics_out}")


if __name__ == "__main__":
    main()
