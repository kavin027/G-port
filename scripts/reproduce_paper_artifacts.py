from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "tcp_root": ROOT / "results/external_baselines_tcp_plain_full_best_safe",
    "stress_root": ROOT / "results/external_baselines_network_stress_full_best_safe",
    "k3s_main_root": ROOT / "results/server_k3s_20260702/coded_k3s_external_full",
    "k3s_best_safe_root": ROOT / "results/server_k3s_20260702/coded_k3s_best_safe_full",
    "k3s_stress_root": ROOT / "results/server_k3s_20260702/coded_k3s_stress_full",
    "k3s_recovery_root": ROOT / "results/server_k3s_20260702/coded_k3s_recovery_stress",
    "paper_repro_root": ROOT / "results/paper_reproduction",
    "paper_dir": ROOT / "paper/socc26",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-platform dispatcher for paper figure/table reproduction."
    )
    parser.add_argument(
        "items",
        nargs="*",
        default=["all"],
        choices=[
            "all",
            "figure1",
            "figure2",
            "figure3",
            "table1",
            "table2",
            "table3",
            "table4",
            "table5",
            "compile",
        ],
        help="Paper items to rebuild. 'all' rebuilds figures 2-3 and tables 1-5.",
    )
    parser.add_argument("--tcp-root", type=Path, default=DEFAULTS["tcp_root"])
    parser.add_argument("--stress-root", type=Path, default=DEFAULTS["stress_root"])
    parser.add_argument("--k3s-main-root", type=Path, default=DEFAULTS["k3s_main_root"])
    parser.add_argument("--k3s-best-safe-root", type=Path, default=DEFAULTS["k3s_best_safe_root"])
    parser.add_argument("--k3s-stress-root", type=Path, default=DEFAULTS["k3s_stress_root"])
    parser.add_argument("--k3s-recovery-root", type=Path, default=DEFAULTS["k3s_recovery_root"])
    parser.add_argument("--paper-repro-root", type=Path, default=DEFAULTS["paper_repro_root"])
    parser.add_argument("--paper-dir", type=Path, default=DEFAULTS["paper_dir"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = expand_items(args.items)
    for item in items:
        run_item(item, args)


def expand_items(items: list[str]) -> list[str]:
    if "all" not in items:
        return items
    return ["figure2", "figure3", "table1", "table2", "table3", "table4", "table5"]


def run_item(item: str, args: argparse.Namespace) -> None:
    if item in {"figure1", "compile"}:
        run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=args.paper_dir)
        return
    if item == "figure2":
        run(
            [
                sys.executable,
                "tools/export_drawio_figure.py",
                "paper/socc26/figures/figure1_architecture.drawio",
                "paper/socc26/figures/figure1_architecture.svg",
                "paper/socc26/figures/figure1_architecture.pdf",
            ]
        )
        return
    if item in {"figure3", "table3"}:
        guard_out = args.k3s_main_root / "guard_prediction_diagnostics_rebuild"
        if has_file(args.k3s_main_root, "network_metrics.csv"):
            run(
                [
                    sys.executable,
                    "analyze_guard_prediction.py",
                    "--root",
                    str(args.k3s_main_root),
                    "--out",
                    str(guard_out),
                    "--paper-figures-dir",
                    str(args.paper_dir / "figures"),
                ]
            )
        else:
            require_existing(guard_out / "guard_threshold_sensitivity.csv")
            require_existing(args.paper_dir / "figures/guard_threshold_sensitivity.pdf")
            print(f"Using compact guard diagnostics in {guard_out}", flush=True)
        return
    if item in {"table1", "table2"}:
        rebuild_external_tables(args)
        return
    if item == "table4":
        rebuild_or_use_stress(args.k3s_stress_root)
        rebuild_or_use_stress(args.k3s_recovery_root)
        return
    if item == "table5":
        run([sys.executable, "analyze_score_ablation.py"])
        return
    raise ValueError(f"Unsupported item: {item}")


def rebuild_external_tables(args: argparse.Namespace) -> None:
    maybe_rebuild_external(args.tcp_root, args.tcp_root / "tcp_external_summary.csv")
    maybe_rebuild_external(args.stress_root, args.stress_root / "tcp_external_summary.csv")
    k3s_summary = args.k3s_main_root / "external_analysis_rebuild/k3s_external_summary.csv"
    if has_file(args.k3s_main_root, "network_metrics.csv"):
        command = [
            sys.executable,
            "analyze_external_baselines.py",
            "--root",
            str(args.k3s_main_root),
            "--out",
            str(args.k3s_main_root / "external_analysis_rebuild"),
        ]
        if args.k3s_best_safe_root.exists():
            command.extend(["--best-safe-root", str(args.k3s_best_safe_root)])
        run(command)
    else:
        require_existing(k3s_summary)
        print(f"Using compact K3s external summary {k3s_summary}", flush=True)
    run(
        [
            sys.executable,
            "build_paper_artifact_tables.py",
            "--tcp-summary",
            str(args.tcp_root / "tcp_external_summary.csv"),
            "--stress-summary",
            str(args.stress_root / "tcp_external_summary.csv"),
            "--k3s-summary",
            str(k3s_summary),
            "--out",
            str(args.paper_repro_root),
        ]
    )


def maybe_rebuild_external(root: Path, summary: Path) -> None:
    if has_file(root, "network_metrics.csv"):
        run(
            [
                sys.executable,
                "analyze_external_baselines.py",
                "--root",
                str(root),
                "--out",
                str(root),
            ]
        )
    else:
        require_existing(summary)
        print(f"Using compact external summary {summary}", flush=True)


def rebuild_or_use_stress(root: Path) -> None:
    out = root / "analysis_rebuild"
    if has_file(root, "network_summary.csv"):
        run(
            [
                sys.executable,
                "analyze_k8s_stress.py",
                "--root",
                str(root),
                "--out",
                str(out),
            ]
        )
    else:
        require_existing(out / "k8s_stress_table.tex")
        print(f"Using compact K3s stress analysis in {out}", flush=True)


def has_file(root: Path, name: str) -> bool:
    return root.exists() and any(root.glob(f"**/{name}"))


def require_existing(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, check=True)


if __name__ == "__main__":
    main()
