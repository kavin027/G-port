from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StageResult:
    name: str
    status: str
    seconds: float
    command: list[str]
    log_path: Path
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short SoCC artifact fast path: TCP worker-service smoke, "
            "direct Docker-bridge smoke, and guarded-policy replay."
        )
    )
    parser.add_argument("--out-root", type=Path, default=Path("socc_fast_path_artifact"))
    parser.add_argument("--base-port", type=int, default=32000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-docker-build", action="store_true")
    parser.add_argument("--skip-guard", action="store_true")
    return parser.parse_args()


def run_stage(name: str, command: list[str], log_path: Path) -> StageResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    log = [
        f"$ {' '.join(command)}",
        "",
        "## stdout",
        result.stdout or "",
        "",
        "## stderr",
        result.stderr or "",
    ]
    log_path.write_text("\n".join(log), encoding="utf-8")
    status = "ok" if result.returncode == 0 else f"failed({result.returncode})"
    note = "" if result.returncode == 0 else "See the stage log for stdout/stderr."
    return StageResult(name, status, elapsed, command, log_path, note)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary_csv(rows: list[dict[str, str]], out: Path) -> None:
    if not rows:
        return
    keys = [
        "strategy",
        "mean_decode_latency_ms",
        "p95_decode_latency_ms",
        "mean_barrier_latency_ms",
        "p95_gain_vs_sparse_flexible_pct",
        "decode_success_rate",
        "mean_worker_errors",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_network(path: Path, out: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_rows(path / "network_summary.csv"):
        p95_gain = row.get("p95_decode_latency_improvement_vs_sparse_flexible", "")
        errors = row.get("mean_worker_errors", "0.0")
        rows.append(
            {
                "strategy": row.get("strategy", ""),
                "mean_decode_latency_ms": _ms(row.get("mean_decode_latency", "")),
                "p95_decode_latency_ms": _ms(row.get("p95_decode_latency", "")),
                "mean_barrier_latency_ms": _ms(row.get("mean_barrier_latency", "")),
                "p95_gain_vs_sparse_flexible_pct": _pct(p95_gain),
                "decode_success_rate": _fmt(row.get("decode_success_rate", "")),
                "mean_worker_errors": _fmt(errors),
            }
        )
    write_summary_csv(rows, out)
    return rows


def summarize_guard(path: Path) -> dict[str, str]:
    rows = read_rows(path / "guarded_policy_aggregate.csv")
    overall = next((row for row in rows if row.get("suite") == "overall"), {})
    if not overall:
        return {}
    return {
        "regimes": overall.get("regimes", ""),
        "enabled": overall.get("enabled", ""),
        "always_on_p95_gain_pct": _fmt(overall.get("always_on_p95_gain_pct", "")),
        "guarded_p95_gain_pct": _fmt(overall.get("guarded_p95_gain_pct", "")),
        "always_on_negative_regimes": overall.get("always_on_negative_regimes", ""),
        "guarded_negative_regimes": overall.get("guarded_negative_regimes", ""),
    }


def _ms(value: str) -> str:
    try:
        return f"{1000.0 * float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _pct(value: str) -> str:
    try:
        return f"{100.0 * float(value):.1f}"
    except (TypeError, ValueError):
        return ""


def _fmt(value: str) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return value or ""


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows produced._"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    return lines


def write_report(
    out_root: Path,
    stages: list[StageResult],
    tcp_rows: list[dict[str, str]],
    docker_rows: list[dict[str, str]],
    guard_summary: dict[str, str],
    docker_manifest: dict[str, object],
) -> None:
    report = [
        "# SoCC Artifact Fast Path Report",
        "",
        "This report is a short reviewer-facing sanity path. It verifies that the",
        "TCP worker-service, direct Docker-bridge worker-service, and fixed guard",
        "replay can run from one command. It is not a replacement for the full",
        "paper-scale experiments.",
        "",
        "## Stage Status",
        "",
        "| Stage | Status | Seconds | Log |",
        "| --- | --- | ---: | --- |",
    ]
    for stage in stages:
        report.append(
            f"| {stage.name} | {stage.status} | {stage.seconds:.1f} | `{stage.log_path}` |"
        )
    report.extend(["", "## TCP Worker-Service Smoke", ""])
    report.extend(
        markdown_table(
            tcp_rows,
            [
                "strategy",
                "mean_decode_latency_ms",
                "p95_decode_latency_ms",
                "mean_barrier_latency_ms",
                "p95_gain_vs_sparse_flexible_pct",
                "decode_success_rate",
                "mean_worker_errors",
            ],
        )
    )
    report.extend(["", "## Direct Docker-Bridge Smoke", ""])
    if docker_manifest:
        report.extend(
            [
                f"- Data path: `{docker_manifest.get('data_path', '')}`",
                f"- Host port publishing: `{docker_manifest.get('host_port_publishing', '')}`",
                f"- SSH forwarding: `{docker_manifest.get('ssh_forwarding', '')}`",
                f"- Workers: `{docker_manifest.get('workers', '')}`",
            ]
        )
    report.extend(
        markdown_table(
            docker_rows,
            [
                "strategy",
                "mean_decode_latency_ms",
                "p95_decode_latency_ms",
                "mean_barrier_latency_ms",
                "p95_gain_vs_sparse_flexible_pct",
                "decode_success_rate",
                "mean_worker_errors",
            ],
        )
    )
    report.extend(["", "## Guard Replay", ""])
    if guard_summary:
        report.extend(
            [
                f"- Regimes: `{guard_summary['regimes']}`",
                f"- Enabled by full guard: `{guard_summary['enabled']}`",
                (
                    "- Mean p95 gain, always-on vs guarded: "
                    f"`{guard_summary['always_on_p95_gain_pct']}%` -> "
                    f"`{guard_summary['guarded_p95_gain_pct']}%`"
                ),
                (
                    "- Negative regimes, always-on vs guarded: "
                    f"`{guard_summary['always_on_negative_regimes']}` -> "
                    f"`{guard_summary['guarded_negative_regimes']}`"
                ),
            ]
        )
    else:
        report.append("_Guard replay was skipped or produced no aggregate row._")
    report.extend(
        [
            "",
            "## Output Map",
            "",
            "- `tcp_smoke/`: independent TCP worker-service smoke outputs.",
            "- `tcp_smoke_analysis/`: paired summary for the TCP smoke.",
            "- `direct_docker_bridge_smoke/`: direct container-to-container Docker bridge outputs.",
            "- `direct_docker_bridge_analysis/`: paired summary for the Docker smoke.",
            "- `guarded_policy_diagnostics/`: fixed guard replay outputs.",
            "- `summary_report.md`: this file.",
        ]
    )
    (out_root / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    if out_root.exists() and args.clean:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    logs = out_root / "logs"
    stages: list[StageResult] = []

    tcp_out = out_root / "tcp_smoke"
    tcp_cmd = [
        args.python,
        "run_network_container_experiment.py",
        "--quick",
        "--strategies",
        "speed_aware_uncoded",
        "sparse_flexible_static",
        "rank_aware_sparse_flexible",
        "system_portfolio",
        "--common-jitter-across-strategies",
        "--out",
        str(tcp_out),
        "--base-port",
        str(args.base_port),
    ]
    stages.append(run_stage("tcp-worker-service-smoke", tcp_cmd, logs / "tcp_smoke.log"))

    tcp_analysis = out_root / "tcp_smoke_analysis"
    tcp_analysis_cmd = [
        args.python,
        "analyze_network_container_results.py",
        str(tcp_out),
        "--baseline-strategy",
        "sparse_flexible_static",
        "--out",
        str(tcp_analysis),
    ]
    stages.append(run_stage("tcp-smoke-analysis", tcp_analysis_cmd, logs / "tcp_analysis.log"))

    docker_out = out_root / "direct_docker_bridge_smoke"
    if args.skip_docker:
        stages.append(
            StageResult(
                "direct-docker-bridge-smoke",
                "skipped",
                0.0,
                [],
                logs / "docker_smoke.log",
                "--skip-docker was set.",
            )
        )
    else:
        docker_cmd = [
            args.python,
            "run_direct_docker_network_experiment.py",
            "--out",
            str(docker_out),
            "--workers",
            "8",
            "--shards",
            "8",
            "--samples",
            "1000",
            "--features",
            "180",
            "--density",
            "0.02",
            "--rounds",
            "2",
            "--sleep-scale",
            "0.008",
            "--cost-scale",
            "0.0015",
            "--network-rtt-ms",
            "4",
            "--network-bandwidth-mbps",
            "100",
            "--alignment-mode",
            "anti",
            "--clean",
        ]
        if args.skip_docker_build:
            docker_cmd.append("--skip-build")
        stages.append(run_stage("direct-docker-bridge-smoke", docker_cmd, logs / "docker_smoke.log"))

        docker_analysis = out_root / "direct_docker_bridge_analysis"
        docker_analysis_cmd = [
            args.python,
            "analyze_network_container_results.py",
            str(docker_out),
            "--baseline-strategy",
            "sparse_flexible_static",
            "--out",
            str(docker_analysis),
        ]
        stages.append(
            run_stage("direct-docker-bridge-analysis", docker_analysis_cmd, logs / "docker_analysis.log")
        )

    guard_out = out_root / "guarded_policy_diagnostics"
    if args.skip_guard:
        stages.append(
            StageResult(
                "guard-replay",
                "skipped",
                0.0,
                [],
                logs / "guard_replay.log",
                "--skip-guard was set.",
            )
        )
    else:
        guard_cmd = [
            args.python,
            "analyze_guarded_policy.py",
            "--out",
            str(guard_out),
        ]
        stages.append(run_stage("guard-replay", guard_cmd, logs / "guard_replay.log"))

    tcp_rows = summarize_network(tcp_out, out_root / "tcp_smoke_summary.csv")
    docker_rows = summarize_network(docker_out, out_root / "direct_docker_bridge_summary.csv")
    guard_summary = summarize_guard(guard_out)
    manifest_path = docker_out / "direct_docker_manifest.json"
    docker_manifest = {}
    if manifest_path.exists():
        docker_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    write_report(out_root, stages, tcp_rows, docker_rows, guard_summary, docker_manifest)

    failed = [stage for stage in stages if stage.status.startswith("failed")]
    print(f"Wrote SoCC artifact fast-path report to {out_root / 'summary_report.md'}")
    if failed:
        for stage in failed:
            print(f"FAILED: {stage.name}; see {stage.log_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
