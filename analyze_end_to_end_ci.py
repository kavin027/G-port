from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGY_LABELS = {
    "sparse_flexible_static": "Static",
    "worker_aware_sparse_flexible": "Worker-aware",
    "rank_aware_sparse_flexible": "Decode-aware",
    "deadline_aware_sparse_flexible": "Deadline-aware",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze end-to-end time-to-loss and bootstrap confidence intervals."
    )
    parser.add_argument(
        "--network-dirs",
        nargs="+",
        type=Path,
        default=sorted(Path("network_container_server").glob("w16_seed_*")),
        help="Directories containing network_metrics.csv and network_summary.csv.",
    )
    parser.add_argument(
        "--realdata-roots",
        nargs="*",
        type=Path,
        default=[
            Path("runtime_realdata_a9a_sweep_bjb1"),
            Path("runtime_realdata_w8a_sweep_bjb1"),
            Path("runtime_realdata_rcv1_sweep_bjb1"),
        ],
        help="Real-data sweep roots containing per-seed runtime_summary.csv files.",
    )
    parser.add_argument("--out", type=Path, default=Path("end_to_end_ci_diagnostics"))
    parser.add_argument("--progress", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    network_time = compute_network_time_to_loss(args.network_dirs, progress=args.progress)
    network_time.to_csv(args.out / "tcp_time_to_loss.csv", index=False)

    network_summary = load_seed_summaries(args.network_dirs, summary_name="network_summary.csv")
    network_ci = bootstrap_summary(
        network_summary,
        experiment="tcp_w16",
        metrics=[
            "decode_latency_improvement_vs_sparse_flexible",
            "p95_decode_latency_improvement_vs_sparse_flexible",
            "barrier_latency_improvement_vs_sparse_flexible",
        ],
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )

    time_ci = bootstrap_summary(
        network_time,
        experiment="tcp_w16_time_to_loss",
        metrics=[
            "decode_time_improvement_vs_static",
            "barrier_time_improvement_vs_static",
        ],
        n_boot=args.bootstrap_samples,
        seed=args.seed + 1,
    )

    realdata = load_realdata_seed_summaries(args.realdata_roots)
    realdata_ci = bootstrap_summary(
        realdata,
        experiment_col="experiment",
        metrics=[
            "decode_latency_improvement_vs_sparse_flexible",
            "p95_decode_latency_improvement_vs_sparse_flexible",
            "barrier_latency_improvement_vs_sparse_flexible",
        ],
        n_boot=args.bootstrap_samples,
        seed=args.seed + 2,
    )

    ci_summary = pd.concat([network_ci, time_ci, realdata_ci], ignore_index=True)
    ci_summary.to_csv(args.out / "bootstrap_ci_summary.csv", index=False)
    network_ci.to_csv(args.out / "tcp_bootstrap_ci.csv", index=False)
    time_ci.to_csv(args.out / "tcp_time_to_loss_ci.csv", index=False)
    realdata_ci.to_csv(args.out / "realdata_bootstrap_ci.csv", index=False)

    plot_time_to_loss(network_time, args.out / "tcp_time_to_loss.png", args.out / "tcp_time_to_loss.pdf")
    write_report(args.out, network_time, ci_summary, progress=args.progress)
    print(ci_summary.to_string(index=False))
    print(f"\nWrote end-to-end diagnostics to {args.out}")


def load_seed_summaries(paths: list[Path], summary_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        summary_path = path / summary_name
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        frame["run"] = path.name
        frame["seed"] = extract_seed(path.name)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {summary_name} files found under {paths}.")
    return pd.concat(frames, ignore_index=True)


def load_realdata_seed_summaries(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        if not root.exists():
            continue
        dataset = infer_dataset(root.name)
        for summary_path in sorted(root.glob("*/runtime_summary.csv")):
            frame = pd.read_csv(summary_path)
            frame["run"] = summary_path.parent.name
            frame["seed"] = extract_seed(summary_path.parent.name)
            frame["experiment"] = f"real_{dataset}"
            frame["dataset_name"] = dataset
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def infer_dataset(name: str) -> str:
    for dataset in ("a9a", "w8a", "rcv1"):
        if dataset in name:
            return dataset
    return name


def extract_seed(text: str) -> int:
    parts = text.replace("-", "_").split("_")
    for idx, part in enumerate(parts):
        if part == "seed" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                pass
        if part.startswith("seed"):
            try:
                return int(part.removeprefix("seed"))
            except ValueError:
                pass
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else -1


def compute_network_time_to_loss(paths: list[Path], progress: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in paths:
        metrics_path = path / "network_metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        seed = extract_seed(path.name)
        static = metrics[metrics["strategy"] == "sparse_flexible_static"].sort_values("iteration")
        if static.empty:
            continue
        initial_loss = float(static["loss"].iloc[0])
        final_loss = float(static["loss"].iloc[-1])
        target_loss = initial_loss - progress * (initial_loss - final_loss)
        static_times = _time_for_strategy(static, target_loss)

        for strategy, group in metrics.groupby("strategy", sort=False):
            times = _time_for_strategy(group.sort_values("iteration"), target_loss)
            records.append(
                {
                    "run": path.name,
                    "seed": seed,
                    "strategy": strategy,
                    "label": STRATEGY_LABELS.get(strategy, strategy),
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "target_loss": target_loss,
                    "decode_time_to_loss": times["decode_wall_clock"],
                    "barrier_time_to_loss": times["barrier_wall_clock"],
                    "target_iteration": times["iteration"],
                    "decode_time_improvement_vs_static": (
                        static_times["decode_wall_clock"] - times["decode_wall_clock"]
                    )
                    / static_times["decode_wall_clock"],
                    "barrier_time_improvement_vs_static": (
                        static_times["barrier_wall_clock"] - times["barrier_wall_clock"]
                    )
                    / static_times["barrier_wall_clock"],
                }
            )
    if not records:
        raise FileNotFoundError("No network_metrics.csv files found.")
    return pd.DataFrame.from_records(records)


def _time_for_strategy(group: pd.DataFrame, target_loss: float) -> dict[str, float]:
    reached = group[group["loss"] <= target_loss]
    row = reached.iloc[0] if not reached.empty else group.iloc[-1]
    return {
        "decode_wall_clock": float(row["decode_wall_clock"]),
        "barrier_wall_clock": float(row["barrier_wall_clock"]),
        "iteration": int(row["iteration"]),
    }


def bootstrap_summary(
    frame: pd.DataFrame,
    *,
    metrics: list[str],
    n_boot: int,
    seed: int,
    experiment: str | None = None,
    experiment_col: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    experiments = [experiment] if experiment is not None else sorted(frame[experiment_col].dropna().unique())
    for exp in experiments:
        exp_frame = frame if experiment is not None else frame[frame[experiment_col] == exp]
        for strategy, group in exp_frame.groupby("strategy", sort=False):
            if strategy == "sparse_flexible_static":
                continue
            for metric in metrics:
                if metric not in group.columns:
                    continue
                values = group[metric].dropna().to_numpy(dtype=float)
                if values.size == 0:
                    continue
                means = np.empty(n_boot, dtype=float)
                for idx in range(n_boot):
                    sample = rng.choice(values, size=values.size, replace=True)
                    means[idx] = float(sample.mean())
                records.append(
                    {
                        "experiment": exp,
                        "strategy": strategy,
                        "label": STRATEGY_LABELS.get(strategy, strategy),
                        "metric": metric,
                        "n": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                        "ci_low": float(np.quantile(means, 0.025)),
                        "ci_high": float(np.quantile(means, 0.975)),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_time_to_loss(time_df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    order = [
        "sparse_flexible_static",
        "worker_aware_sparse_flexible",
        "rank_aware_sparse_flexible",
        "deadline_aware_sparse_flexible",
    ]
    plot_df = time_df[time_df["strategy"].isin(order)].copy()
    grouped = plot_df.groupby("strategy", sort=False)["barrier_time_to_loss"]
    means = grouped.mean().reindex(order)
    stds = grouped.std().reindex(order).fillna(0.0)
    labels = [STRATEGY_LABELS.get(strategy, strategy) for strategy in order]
    colors = ["#6b7280", "#b45309", "#2563eb", "#059669"]

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(order))
    ax.bar(x, means.to_numpy() * 1000.0, yerr=stds.to_numpy() * 1000.0, capsize=4, color=colors)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Barrier time to 90% loss progress (ms)")
    ax.set_title("TCP-isolated worker runtime: time to loss")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def write_report(out_dir: Path, time_df: pd.DataFrame, ci_summary: pd.DataFrame, progress: float) -> None:
    time_summary = (
        time_df.groupby(["strategy", "label"], sort=False)
        .agg(
            decode_time_ms=("decode_time_to_loss", lambda x: 1000.0 * x.mean()),
            barrier_time_ms=("barrier_time_to_loss", lambda x: 1000.0 * x.mean()),
            barrier_time_std_ms=("barrier_time_to_loss", lambda x: 1000.0 * x.std(ddof=1)),
            barrier_improvement=("barrier_time_improvement_vs_static", "mean"),
        )
        .reset_index()
    )
    report = [
        "# End-to-end and CI diagnostics",
        "",
        f"Time-to-loss uses the first iteration reaching {progress:.0%} of the observed static loss reduction for each seed.",
        "",
        "## TCP time to loss",
        "",
        "```",
        time_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Bootstrap CI summary",
        "",
        "```",
        ci_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
    ]
    (out_dir / "end_to_end_ci_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
