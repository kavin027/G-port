from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect lightweight Kubernetes resource and reliability counters "
            "for a completed coded-learning worker-service run."
        )
    )
    parser.add_argument("--namespace", default="coded-learning-exp")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kubectl", default="kubectl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "k8s_resource_raw"
    raw.mkdir(parents=True, exist_ok=True)

    commands = {
        "pods_wide": [args.kubectl, "-n", args.namespace, "get", "pods", "-o", "wide"],
        "pods_json": [args.kubectl, "-n", args.namespace, "get", "pods", "-o", "json"],
        "describe_pods": [args.kubectl, "-n", args.namespace, "describe", "pods"],
        "events": [
            args.kubectl,
            "-n",
            args.namespace,
            "get",
            "events",
            "--sort-by=.lastTimestamp",
        ],
        "top_pods": [
            args.kubectl,
            "top",
            "pods",
            "-n",
            args.namespace,
            "--containers",
            "--no-headers",
        ],
        "top_nodes": [args.kubectl, "top", "nodes", "--no-headers"],
        "nodes_json": [args.kubectl, "get", "nodes", "-o", "json"],
    }

    results: dict[str, subprocess.CompletedProcess[str]] = {}
    for name, cmd in commands.items():
        results[name] = run_and_save(cmd, raw / f"{name}.txt")

    pod_json = load_json_result(results["pods_json"])
    node_json = load_json_result(results["nodes_json"])
    pod_stats = parse_pods_json(pod_json)
    top_pod_stats = parse_top_pods(results["top_pods"].stdout)
    top_node_stats = parse_top_nodes(results["top_nodes"].stdout)
    stats_summary = collect_node_stats(args.kubectl, args.namespace, node_json, raw)
    network_stats = parse_network_metrics(out / "network_metrics.csv")

    notes = []
    for name in ["top_pods", "top_nodes"]:
        result = results[name]
        if result.returncode != 0:
            notes.append(f"{name} unavailable: {(result.stderr or '').strip()[:160]}")
    if not stats_summary:
        notes.append("node stats summary unavailable or did not include this namespace")

    row = {
        "namespace": args.namespace,
        "timestamp_unix": f"{time.time():.3f}",
        **pod_stats,
        **top_pod_stats,
        **top_node_stats,
        **stats_summary,
        **network_stats,
        "notes": " | ".join(notes),
    }
    write_csv(out / "k8s_resource_counters.csv", row)
    write_report(out / "k8s_resource_report.md", row)
    print(f"Wrote {out / 'k8s_resource_counters.csv'}")
    print(f"Wrote {out / 'k8s_resource_report.md'}")


def run_and_save(cmd: list[str], path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=True)
    path.write_text(result.stdout or "", encoding="utf-8")
    err_path = path.with_suffix(path.suffix + ".stderr")
    err_path.write_text(result.stderr or "", encoding="utf-8")
    return result


def load_json_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if result.returncode != 0 or not (result.stdout or "").strip():
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def parse_pods_json(data: dict[str, Any]) -> dict[str, Any]:
    items = data.get("items") or []
    phases: dict[str, int] = {}
    nodes: set[str] = set()
    restart_count = 0
    worker_pods = 0
    master_pods = 0
    tcp_error_markers = 0
    for pod in items:
        metadata = pod.get("metadata") or {}
        status = pod.get("status") or {}
        spec = pod.get("spec") or {}
        name = str(metadata.get("name", ""))
        phase = str(status.get("phase", "unknown"))
        phases[phase] = phases.get(phase, 0) + 1
        if name.startswith("coded-worker-"):
            worker_pods += 1
        if name.startswith("coded-master-"):
            master_pods += 1
        node_name = spec.get("nodeName")
        if node_name:
            nodes.add(str(node_name))
        for item in status.get("containerStatuses") or []:
            restart_count += int(item.get("restartCount") or 0)
        message = json.dumps(status)
        tcp_error_markers += message.lower().count("connection refused")
    return {
        "pod_count": len(items),
        "worker_pods": worker_pods,
        "master_pods": master_pods,
        "running_pods": phases.get("Running", 0),
        "succeeded_pods": phases.get("Succeeded", 0),
        "failed_pods": phases.get("Failed", 0),
        "restart_count": restart_count,
        "pod_node_count": len(nodes),
        "pod_nodes": ";".join(sorted(nodes)),
        "pod_status_tcp_error_markers": tcp_error_markers,
    }


def parse_top_pods(text: str) -> dict[str, Any]:
    cpu_values = []
    mem_values = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cpu = parse_cpu_mcores(parts[-2])
        mem = parse_mem_mib(parts[-1])
        if cpu is not None:
            cpu_values.append(cpu)
        if mem is not None:
            mem_values.append(mem)
    return {
        "top_pod_cpu_mean_mcores": fmt_stat(cpu_values, "mean"),
        "top_pod_cpu_p95_mcores": fmt_stat(cpu_values, "p95"),
        "top_pod_mem_mean_mib": fmt_stat(mem_values, "mean"),
        "top_pod_mem_p95_mib": fmt_stat(mem_values, "p95"),
    }


def parse_top_nodes(text: str) -> dict[str, Any]:
    cpu_values = []
    mem_values = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cpu = parse_cpu_mcores(parts[1])
        mem = parse_mem_mib(parts[3])
        if cpu is not None:
            cpu_values.append(cpu)
        if mem is not None:
            mem_values.append(mem)
    return {
        "top_node_count": len(cpu_values),
        "top_node_cpu_mean_mcores": fmt_stat(cpu_values, "mean"),
        "top_node_cpu_p95_mcores": fmt_stat(cpu_values, "p95"),
        "top_node_mem_mean_mib": fmt_stat(mem_values, "mean"),
        "top_node_mem_p95_mib": fmt_stat(mem_values, "p95"),
    }


def collect_node_stats(
    kubectl: str,
    namespace: str,
    node_json: dict[str, Any],
    raw: Path,
) -> dict[str, Any]:
    cpu_values = []
    mem_values = []
    for node in node_json.get("items") or []:
        name = ((node.get("metadata") or {}).get("name") or "").strip()
        if not name:
            continue
        result = run_and_save(
            [kubectl, "get", "--raw", f"/api/v1/nodes/{name}/proxy/stats/summary"],
            raw / f"node_stats_{safe_filename(name)}.json",
        )
        if result.returncode != 0:
            continue
        data = load_json_result(result)
        for pod in data.get("pods") or []:
            pod_ref = pod.get("podRef") or {}
            if pod_ref.get("namespace") != namespace:
                continue
            for container in pod.get("containers") or []:
                cpu = ((container.get("cpu") or {}).get("usageNanoCores"))
                mem = ((container.get("memory") or {}).get("workingSetBytes"))
                if cpu is not None:
                    cpu_values.append(float(cpu) / 1_000_000.0)
                if mem is not None:
                    mem_values.append(float(mem) / (1024.0 * 1024.0))
    if not cpu_values and not mem_values:
        return {}
    return {
        "stats_pod_cpu_mean_mcores": fmt_stat(cpu_values, "mean"),
        "stats_pod_cpu_p95_mcores": fmt_stat(cpu_values, "p95"),
        "stats_pod_mem_mean_mib": fmt_stat(mem_values, "mean"),
        "stats_pod_mem_p95_mib": fmt_stat(mem_values, "p95"),
    }


def parse_network_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "network_worker_errors_sum": "",
            "network_worker_errors_mean": "",
            "network_dispatch_ms_mean": "",
            "network_cancel_ms_mean": "",
            "network_response_kb_mean": "",
            "network_decode_success_rate": "",
        }
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        return {}
    worker_errors = [float_or_zero(row.get("worker_errors")) for row in rows]
    dispatch = [1000.0 * float_or_zero(row.get("dispatch_seconds")) for row in rows]
    cancel = [1000.0 * float_or_zero(row.get("cancel_seconds")) for row in rows]
    response = [float_or_zero(row.get("network_response_bytes")) / 1000.0 for row in rows]
    success = [float_or_zero(row.get("decode_success")) for row in rows]
    return {
        "network_worker_errors_sum": f"{sum(worker_errors):.3f}",
        "network_worker_errors_mean": f"{mean(worker_errors):.6f}",
        "network_dispatch_ms_mean": f"{mean(dispatch):.3f}",
        "network_cancel_ms_mean": f"{mean(cancel):.3f}",
        "network_response_kb_mean": f"{mean(response):.3f}",
        "network_decode_success_rate": f"{mean(success):.6f}",
    }


def parse_cpu_mcores(value: str) -> float | None:
    match = re.fullmatch(r"([0-9.]+)(n|u|m)?", value.strip())
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "n":
        return amount / 1_000_000.0
    if unit == "u":
        return amount / 1_000.0
    if unit == "m":
        return amount
    return amount * 1000.0


def parse_mem_mib(value: str) -> float | None:
    match = re.fullmatch(r"([0-9.]+)([KMGTP]i?|)", value.strip())
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    factors = {
        "": 1.0 / (1024.0 * 1024.0),
        "Ki": 1.0 / 1024.0,
        "K": 1.0 / 1000.0,
        "Mi": 1.0,
        "M": 1.0,
        "Gi": 1024.0,
        "G": 1000.0,
        "Ti": 1024.0 * 1024.0,
        "T": 1000.0 * 1000.0,
    }
    return amount * factors.get(unit, 1.0)


def fmt_stat(values: list[float], kind: str) -> str:
    if not values:
        return ""
    sorted_values = sorted(values)
    if kind == "mean":
        return f"{mean(sorted_values):.3f}"
    if kind == "p95":
        idx = min(len(sorted_values) - 1, int(round(0.95 * (len(sorted_values) - 1))))
        return f"{sorted_values[idx]:.3f}"
    raise ValueError(kind)


def float_or_zero(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_report(path: Path, row: dict[str, Any]) -> None:
    lines = [
        "# Kubernetes Resource Counter Snapshot",
        "",
        f"- Namespace: `{row.get('namespace')}`",
        f"- Pods: {row.get('pod_count')} total, {row.get('worker_pods')} workers, "
        f"{row.get('master_pods')} master jobs, restarts={row.get('restart_count')}",
        f"- Pod nodes: {row.get('pod_nodes')}",
        f"- Pod CPU from `kubectl top`: mean={row.get('top_pod_cpu_mean_mcores')} mcores, "
        f"p95={row.get('top_pod_cpu_p95_mcores')} mcores",
        f"- Pod memory from `kubectl top`: mean={row.get('top_pod_mem_mean_mib')} MiB, "
        f"p95={row.get('top_pod_mem_p95_mib')} MiB",
        f"- Node CPU from `kubectl top`: mean={row.get('top_node_cpu_mean_mcores')} mcores, "
        f"p95={row.get('top_node_cpu_p95_mcores')} mcores",
        f"- Node memory from `kubectl top`: mean={row.get('top_node_mem_mean_mib')} MiB, "
        f"p95={row.get('top_node_mem_p95_mib')} MiB",
        f"- API stats pod CPU: mean={row.get('stats_pod_cpu_mean_mcores')} mcores, "
        f"p95={row.get('stats_pod_cpu_p95_mcores')} mcores",
        f"- API stats pod memory: mean={row.get('stats_pod_mem_mean_mib')} MiB, "
        f"p95={row.get('stats_pod_mem_p95_mib')} MiB",
        f"- Network worker errors: sum={row.get('network_worker_errors_sum')}, "
        f"mean={row.get('network_worker_errors_mean')}",
        f"- Dispatch/cancel means: {row.get('network_dispatch_ms_mean')} ms / "
        f"{row.get('network_cancel_ms_mean')} ms",
        f"- Response payload mean: {row.get('network_response_kb_mean')} KB",
        "",
        f"Notes: {row.get('notes') or 'none'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
