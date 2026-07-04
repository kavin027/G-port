from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a direct Docker-bridge sweep. Each run starts one master "
            "container and one TCP worker-service container per logical worker "
            "on the same Docker network; worker ports are not published."
        )
    )
    parser.add_argument("--out-root", type=Path, default=Path("direct_docker_scale_sweep"))
    parser.add_argument("--diagnostics-out", type=Path, default=Path("direct_docker_scale_diagnostics"))
    parser.add_argument("--workers", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--alignments", nargs="+", choices=["none", "aligned", "anti"], default=["none", "anti"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--features", type=int, default=220)
    parser.add_argument("--density", type=float, default=0.02)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--sleep-scale", type=float, default=0.010)
    parser.add_argument("--cost-scale", type=float, default=0.002)
    parser.add_argument("--network-rtt-ms", type=float, default=4.0)
    parser.add_argument("--network-bandwidth-mbps", type=float, default=100.0)
    parser.add_argument("--straggler-fraction", type=float, default=0.35)
    parser.add_argument("--straggler-slowdown", type=float, default=0.12)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=4)
    parser.add_argument("--image", default="coded-learning-network-worker:local")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "sparse_flexible_static",
            "rank_aware_sparse_flexible",
            "deadline_aware_sparse_flexible",
        ],
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    result_dirs: list[Path] = []
    build_needed = not args.skip_build

    for workers in args.workers:
        for alignment in args.alignments:
            for seed in args.seeds:
                out = args.out_root / f"w{workers}_{alignment}_seed{seed}"
                cmd = [
                    sys.executable,
                    "run_direct_docker_network_experiment.py",
                    "--out",
                    str(out),
                    "--workers",
                    str(workers),
                    "--shards",
                    str(workers),
                    "--samples",
                    str(args.samples),
                    "--features",
                    str(args.features),
                    "--density",
                    str(args.density),
                    "--rounds",
                    str(args.rounds),
                    "--sleep-scale",
                    str(args.sleep_scale),
                    "--cost-scale",
                    str(args.cost_scale),
                    "--network-rtt-ms",
                    str(args.network_rtt_ms),
                    "--network-bandwidth-mbps",
                    str(args.network_bandwidth_mbps),
                    "--scenario",
                    args.scenario,
                    "--drift-period",
                    str(args.drift_period),
                    "--straggler-fraction",
                    str(args.straggler_fraction),
                    "--straggler-slowdown",
                    str(args.straggler_slowdown),
                    "--seed",
                    str(seed),
                    "--alignment-mode",
                    alignment,
                    "--image",
                    args.image,
                    "--strategies",
                    *args.strategies,
                ]
                if args.no_clean:
                    pass
                else:
                    cmd.append("--clean")
                if not build_needed:
                    cmd.append("--skip-build")
                run(cmd)
                build_needed = False
                result_dirs.append(out)

    if not args.no_analyze and result_dirs:
        run(
            [
                sys.executable,
                "analyze_network_container_results.py",
                *[str(path) for path in result_dirs],
                "--baseline-strategy",
                "sparse_flexible_static",
                "--out",
                str(args.diagnostics_out),
            ]
        )
        manifest = [
            "# Direct Docker-Bridge Sweep",
            "",
            "Each run starts the master and all TCP workers as containers on one Docker network.",
            "Worker ports are not published to the host, and the master reaches workers through Docker DNS.",
            "",
            "## Runs",
            "",
            *[f"- `{path}`" for path in result_dirs],
            "",
            "## Diagnostics",
            "",
            f"- `{args.diagnostics_out / 'network_report.md'}`",
            f"- `{args.diagnostics_out / 'aggregate_vs_sparse_flexible_static.csv'}`",
        ]
        (args.diagnostics_out / "direct_docker_sweep_manifest.md").write_text(
            "\n".join(manifest),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
