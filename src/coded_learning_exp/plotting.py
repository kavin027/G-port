from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_plots(metrics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, frame in metrics.groupby("strategy", sort=False):
        ax.plot(frame["iteration"], frame["iteration_time"], label=strategy, linewidth=1.4)
    ax.set_xlabel("Training round")
    ax.set_ylabel("Simulated iteration time")
    ax.set_title("Iteration latency under dynamic stragglers")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "latency_over_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, frame in metrics.groupby("strategy", sort=False):
        ax.plot(frame["wall_clock"], frame["loss"], label=strategy, linewidth=1.5)
    ax.set_xlabel("Simulated wall-clock time")
    ax.set_ylabel("Sparse ridge loss")
    ax.set_yscale("log")
    ax.set_title("End-to-end convergence")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_over_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(summary))
    ax.bar(x, summary["mean_latency"], label="mean", alpha=0.85)
    ax.scatter(x, summary["p95_latency"], label="P95", color="black", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["strategy"], rotation=25, ha="right")
    ax.set_ylabel("Simulated iteration time")
    ax.set_title("Mean and P95 latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "summary_latency.png", dpi=180)
    plt.close(fig)
