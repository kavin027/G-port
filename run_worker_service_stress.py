from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


CASES: dict[str, list[str]] = {
    "baseline": [],
    "cancel_ack_20ms": ["CODED_CANCEL_ACK_DELAY_MS=20"],
    "cancel_ack_50ms": ["CODED_CANCEL_ACK_DELAY_MS=50"],
    "close_connection": ["CODED_STRESS_WORKER_ID=0", "CODED_CLOSE_ON_TASK=1"],
    "exit_on_task": ["CODED_STRESS_WORKER_ID=0", "CODED_EXIT_ON_TASK=1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small Docker worker-service stress suite for cancellation "
            "delay, closed TCP connections, and worker exits."
        )
    )
    parser.add_argument("--out-root", type=Path, default=Path("worker_service_stress_diagnostics"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--features", type=int, default=240)
    parser.add_argument("--density", type=float, default=0.014)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=4)
    parser.add_argument("--straggler-fraction", type=float, default=0.45)
    parser.add_argument("--straggler-slowdown", type=float, default=0.08)
    parser.add_argument("--sleep-scale", type=float, default=0.01)
    parser.add_argument("--cost-scale", type=float, default=0.002)
    parser.add_argument("--network-rtt-ms", type=float, default=4.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=100.0)
    parser.add_argument("--image", default="coded-learning-network-worker:local")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "sparse_flexible_static",
            "rank_aware_sparse_flexible",
            "online_counter_guard_deadline_aware_sparse_flexible",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "cases": {name: CASES[name] for name in args.cases},
        "workers": args.workers,
        "rounds": args.rounds,
        "seed": args.seed,
        "strategies": args.strategies,
    }
    (args.out_root / "worker_service_stress_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    image_built = args.skip_build
    for case in args.cases:
        if case not in CASES:
            raise SystemExit(f"Unknown stress case {case!r}; choices are {', '.join(CASES)}")
        case_dir = args.out_root / case
        cmd = build_case_command(args, case, case_dir, skip_build=image_built)
        status = run_case(cmd, case_dir / "stress_case.log")
        image_built = True
        rows.extend(summarize_case(case, case_dir, status))

    write_csv(args.out_root / "worker_service_stress_summary.csv", rows)
    write_report(args.out_root / "worker_service_stress_report.md", rows)
    print(f"Wrote {args.out_root / 'worker_service_stress_summary.csv'}")
    print(f"Wrote {args.out_root / 'worker_service_stress_report.md'}")


def build_case_command(args: argparse.Namespace, case: str, out: Path, *, skip_build: bool) -> list[str]:
    cmd = [
        sys.executable,
        "run_direct_docker_network_experiment.py",
        "--out",
        str(out),
        "--workers",
        str(args.workers),
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
        "--sleep-scale",
        str(args.sleep_scale),
        "--cost-scale",
        str(args.cost_scale),
        "--network-rtt-ms",
        str(args.network_rtt_ms),
        "--network-bandwidth-mbps",
        str(args.network_bandwidth_mbps),
        "--seed",
        str(args.seed),
        "--image",
        args.image,
        "--clean",
        "--strategies",
        *args.strategies,
    ]
    if skip_build:
        cmd.append("--skip-build")
    for env in CASES[case]:
        cmd.extend(["--worker-env", env])
    return cmd


def run_case(cmd: list[str], log_path: Path) -> int:
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (result.stdout or "") + ("\n[stderr]\n" + (result.stderr or "") if result.stderr else ""),
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return int(result.returncode)


def summarize_case(case: str, case_dir: Path, status: int) -> list[dict[str, object]]:
    summary_path = case_dir / "network_summary.csv"
    metrics_path = case_dir / "network_metrics.csv"
    if not summary_path.exists():
        return [
            {
                "case": case,
                "strategy": "",
                "returncode": status,
                "completed": False,
                "decode_success_rate": "",
                "mean_barrier_ms": "",
                "p95_barrier_ms": "",
                "mean_decode_ms": "",
                "p95_decode_ms": "",
                "mean_cancel_ms": "",
                "mean_dispatch_ms": "",
                "worker_errors_mean": "",
                "worker_errors_sum": "",
                "note": "network_summary.csv missing; run likely failed before summary",
            }
        ]
    worker_errors_sum = worker_error_sum(metrics_path)
    rows = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            rows.append(
                {
                    "case": case,
                    "strategy": item.get("strategy", ""),
                    "returncode": status,
                    "completed": status == 0,
                    "decode_success_rate": item.get("decode_success_rate", ""),
                    "mean_barrier_ms": ms(item.get("mean_barrier_latency")),
                    "p95_barrier_ms": ms(item.get("p95_barrier_latency")),
                    "mean_decode_ms": ms(item.get("mean_decode_latency")),
                    "p95_decode_ms": ms(item.get("p95_decode_latency")),
                    "mean_cancel_ms": ms(item.get("mean_cancel_seconds")),
                    "mean_dispatch_ms": ms(item.get("mean_dispatch_seconds")),
                    "worker_errors_mean": item.get("mean_worker_errors", ""),
                    "worker_errors_sum": worker_errors_sum,
                    "note": "",
                }
            )
    return rows


def worker_error_sum(path: Path) -> str:
    if not path.exists():
        return ""
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values.append(float_or_zero(row.get("worker_errors")))
    return f"{sum(values):.3f}" if values else ""


def ms(value: object) -> str:
    try:
        return f"{1000.0 * float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def float_or_zero(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Worker-Service Failure and Cancellation Stress",
        "",
        "This Docker suite uses the same TCP worker entrypoint as the main "
        "worker-service experiments. It injects stress through worker environment "
        "variables rather than changing scheduler code.",
        "",
        "| Case | Strategy | Completed | Mean barrier (ms) | p95 barrier (ms) | "
        "Cancel (ms) | Worker errors | Note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case", "")),
                    str(row.get("strategy", "")),
                    str(row.get("completed", "")),
                    str(row.get("mean_barrier_ms", "")),
                    str(row.get("p95_barrier_ms", "")),
                    str(row.get("mean_cancel_ms", "")),
                    str(row.get("worker_errors_sum", "")),
                    str(row.get("note", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- `cancel_ack_*` isolates cancellation-path delay while keeping worker processes alive.",
            "- `close_connection` closes one worker connection on task receipt; the master records EOF worker errors and may still decode if the remaining rows suffice.",
            "- `exit_on_task` terminates one worker process. A nonzero return code is an explicit prototype limitation rather than hidden recovery.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
