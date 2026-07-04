from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


RUN_RE = re.compile(r"w(?P<workers>\d+)_(?P<alignment>[a-z]+)_seed(?P<seed>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize direct Docker-bridge scaling runs.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("direct_docker_scale_diagnostics"))
    parser.add_argument("--baseline", default="sparse_flexible_static")
    return parser.parse_args()


def _metadata(path: Path) -> dict[str, object]:
    manifest = path / "direct_docker_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        data = {}
    match = RUN_RE.search(path.name)
    if match:
        data.setdefault("workers", int(match.group("workers")))
        data.setdefault("alignment", match.group("alignment"))
        data.setdefault("seed", int(match.group("seed")))
    return data


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in args.paths:
        summary = path / "network_summary.csv"
        if not summary.exists():
            continue
        frame = pd.read_csv(summary)
        meta = _metadata(path)
        frame["run"] = path.name
        frame["workers"] = int(meta.get("workers", -1))
        frame["alignment"] = str(meta.get("alignment", "unknown"))
        frame["worker_errors"] = frame.get("mean_worker_errors", 0.0)
        frames.append(frame)
    if not frames:
        raise SystemExit("No network_summary.csv files found.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.out / "direct_docker_scale_combined.csv", index=False)

    rows = []
    for (run, workers, alignment), group in combined.groupby(["run", "workers", "alignment"], sort=False):
        baseline_rows = group[group["strategy"] == args.baseline]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for _, row in group.iterrows():
            rows.append(
                {
                    "run": run,
                    "workers": int(workers),
                    "alignment": alignment,
                    "strategy": row["strategy"],
                    "mean_decode_ms": 1000.0 * float(row["mean_decode_latency"]),
                    "p95_decode_ms": 1000.0 * float(row["p95_decode_latency"]),
                    "mean_barrier_ms": 1000.0 * float(row["mean_barrier_latency"]),
                    "mean_decode_gain_pct": 100.0
                    * (float(baseline["mean_decode_latency"]) - float(row["mean_decode_latency"]))
                    / max(float(baseline["mean_decode_latency"]), 1e-12),
                    "p95_decode_gain_pct": 100.0
                    * (float(baseline["p95_decode_latency"]) - float(row["p95_decode_latency"]))
                    / max(float(baseline["p95_decode_latency"]), 1e-12),
                    "mean_barrier_gain_pct": 100.0
                    * (float(baseline["mean_barrier_latency"]) - float(row["mean_barrier_latency"]))
                    / max(float(baseline["mean_barrier_latency"]), 1e-12),
                    "completed_rows": float(row["mean_completed_rows"]),
                    "selected_rows": float(row["mean_selected_rows"]),
                    "worker_errors": float(row.get("mean_worker_errors", 0.0)),
                }
            )
    paired = pd.DataFrame(rows)
    paired.to_csv(args.out / "direct_docker_scale_paired.csv", index=False)

    metrics = [
        "mean_decode_ms",
        "p95_decode_ms",
        "mean_barrier_ms",
        "mean_decode_gain_pct",
        "p95_decode_gain_pct",
        "mean_barrier_gain_pct",
        "completed_rows",
        "selected_rows",
        "worker_errors",
    ]
    grouped = (
        paired.groupby(["workers", "alignment", "strategy"], sort=True)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = ["_".join(col).rstrip("_") for col in grouped.columns]
    grouped.to_csv(args.out / "direct_docker_scale_by_workers.csv", index=False)

    report = [
        "# Direct Docker-Bridge Scaling Diagnostics",
        "",
        "The master and workers run as containers on one Docker network. Worker ports are not published.",
        "Positive gains are paired against `sparse_flexible_static` within the same run.",
        "",
        "```",
        grouped.to_string(index=False, float_format=lambda value: f"{value:.2f}"),
        "```",
        "",
    ]
    (args.out / "direct_docker_scale_report.md").write_text("\n".join(report), encoding="utf-8")
    print(grouped.to_string(index=False))
    print(f"\nWrote direct Docker scale diagnostics to {args.out}")


if __name__ == "__main__":
    main()
