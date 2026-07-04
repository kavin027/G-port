from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from src.coded_learning_exp.realdata import (
    download_libsvm_dataset,
    l2_normalize_rows,
    load_svmlight_file,
    make_even_shard_slices,
    _resolve_spec,
)
from src.coded_learning_exp.strategies import default_strategies
from src.coded_learning_exp.workers import WorkerPool, WorkerPoolConfig


DEFAULT_STRATEGIES = (
    "uncoded_sync",
    "sparse_flexible_static",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
)


@dataclass(frozen=True)
class SparseLogisticProblem:
    x: sparse.csr_matrix
    y: np.ndarray
    shard_slices: list[slice]
    l2: float

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def n_features(self) -> int:
        return self.x.shape[1]

    @property
    def n_shards(self) -> int:
        return len(self.shard_slices)

    def shard_costs(self) -> np.ndarray:
        costs = np.asarray([float(self.x[slc].nnz) for slc in self.shard_slices], dtype=float)
        return costs / max(float(costs.mean()), 1e-12)

    def shard_gradients(self, weights: np.ndarray) -> np.ndarray:
        gradients = np.zeros((self.n_shards, self.n_features), dtype=float)
        scale = 1.0 / self.n_samples
        for shard_id, slc in enumerate(self.shard_slices):
            x_shard = self.x[slc]
            y_shard = self.y[slc]
            margin = y_shard * np.asarray(x_shard @ weights).ravel()
            factors = -y_shard * expit(-margin)
            gradients[shard_id] = np.asarray(x_shard.T @ factors).ravel() * scale
        return gradients

    def full_gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.shard_gradients(weights).sum(axis=0) + self.l2 * weights

    def loss(self, weights: np.ndarray) -> float:
        margin = self.y * np.asarray(self.x @ weights).ravel()
        data_loss = float(np.logaddexp(0.0, -margin).mean())
        reg_loss = 0.5 * self.l2 * float(np.dot(weights, weights))
        return data_loss + reg_loss

    def accuracy(self, weights: np.ndarray) -> float:
        scores = np.asarray(self.x @ weights).ravel()
        pred = np.where(scores >= 0.0, 1.0, -1.0)
        return float(np.mean(pred == self.y))

    def auc(self, weights: np.ndarray) -> float:
        scores = np.asarray(self.x @ weights).ravel()
        positive = self.y > 0
        n_pos = int(positive.sum())
        n_neg = int((~positive).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        pos_rank_sum = float(ranks[positive].sum())
        return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass(frozen=True)
class LogisticWorkloadConfig:
    datasets: tuple[str, ...] = ("a9a", "w8a")
    cache_dir: Path = Path("data") / "libsvm"
    max_samples: int = 9000
    n_shards: int = 16
    n_workers: int = 24
    rounds: int = 35
    learning_rate: float = 0.8
    l2: float = 1e-4
    scenario: str = "phase"
    drift_period: int = 10
    straggler_fraction: float = 0.45
    straggler_slowdown: float = 0.08
    burst_probability: float = 0.45
    target_loss_fraction: float = 0.90
    run_ids: tuple[int, ...] = (11, 23, 31)
    output_dir: Path = Path("logistic_workload_diagnostics")
    strategy_names: tuple[str, ...] = DEFAULT_STRATEGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end sparse logistic classification workload."
    )
    parser.add_argument("--out", type=Path, default=Path("logistic_workload_diagnostics"))
    parser.add_argument("--datasets", nargs="+", default=["a9a", "w8a"])
    parser.add_argument("--cache-dir", type=Path, default=Path("data") / "libsvm")
    parser.add_argument("--max-samples", type=int, default=9000)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=10)
    parser.add_argument("--straggler-fraction", type=float, default=0.45)
    parser.add_argument("--straggler-slowdown", type=float, default=0.08)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--target-loss-fraction", type=float, default=0.90)
    parser.add_argument("--run-ids", type=int, nargs="+", default=[11, 23, 31])
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LogisticWorkloadConfig(
        datasets=tuple(args.datasets),
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        n_shards=args.shards,
        n_workers=args.workers,
        rounds=args.rounds,
        learning_rate=args.learning_rate,
        l2=args.l2,
        scenario=args.scenario,
        drift_period=args.drift_period,
        straggler_fraction=args.straggler_fraction,
        straggler_slowdown=args.straggler_slowdown,
        burst_probability=args.burst_probability,
        target_loss_fraction=args.target_loss_fraction,
        run_ids=tuple(args.run_ids),
        output_dir=args.out,
        strategy_names=tuple(args.strategies),
    )
    _, summary, aggregate = run_logistic_workload(config)
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote sparse logistic workload outputs to {config.output_dir}")


def run_logistic_workload(
    config: LogisticWorkloadConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frames = []
    summary_frames = []
    for dataset in config.datasets:
        for run_id in config.run_ids:
            problem = make_libsvm_logistic_problem(
                dataset=dataset,
                n_shards=config.n_shards,
                l2=config.l2,
                seed=run_id,
                cache_dir=config.cache_dir,
                max_samples=config.max_samples,
            )
            metrics, summary = _run_one(config, problem, dataset, run_id)
            run_dir = config.output_dir / f"{dataset}_run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(run_dir / "logistic_metrics.csv", index=False)
            summary.to_csv(run_dir / "logistic_summary.csv", index=False)
            metrics_frames.append(metrics)
            summary_frames.append(summary)

    combined_metrics = pd.concat(metrics_frames, ignore_index=True)
    combined_summary = pd.concat(summary_frames, ignore_index=True)
    aggregate = _aggregate(combined_summary)
    combined_metrics.to_csv(config.output_dir / "combined_logistic_metrics.csv", index=False)
    combined_summary.to_csv(config.output_dir / "combined_logistic_summary.csv", index=False)
    aggregate.to_csv(config.output_dir / "aggregate_logistic_summary.csv", index=False)
    _plot(aggregate, config.output_dir)
    _write_report(config, aggregate, config.output_dir / "logistic_workload_report.md")
    return combined_metrics, combined_summary, aggregate


def make_libsvm_logistic_problem(
    dataset: str,
    n_shards: int,
    l2: float,
    seed: int,
    cache_dir: Path,
    max_samples: int | None = None,
    normalize_rows: bool = True,
    append_bias: bool = True,
) -> SparseLogisticProblem:
    spec = _resolve_spec(dataset, url=None, n_features=None)
    path = download_libsvm_dataset(spec, cache_dir)
    x, raw_y = load_svmlight_file(path, n_features=spec.n_features)
    y = np.where(raw_y > 0, 1.0, -1.0)
    x = x.astype(float).tocsr()
    if normalize_rows:
        x = l2_normalize_rows(x)
    if append_bias:
        bias = sparse.csr_matrix(np.ones((x.shape[0], 1), dtype=float))
        x = sparse.hstack([x, bias], format="csr")
    if max_samples is not None and max_samples < x.shape[0]:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(x.shape[0], size=max_samples, replace=False))
        x = x[indices]
        y = y[indices]
    return SparseLogisticProblem(
        x=x,
        y=y,
        shard_slices=make_even_shard_slices(x.shape[0], n_shards),
        l2=l2,
    )


def _run_one(
    config: LogisticWorkloadConfig,
    problem: SparseLogisticProblem,
    dataset: str,
    run_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    worker_pool = WorkerPool(
        WorkerPoolConfig(
            n_workers=config.n_workers,
            scenario=config.scenario,
            drift_period=config.drift_period,
            straggler_fraction=config.straggler_fraction,
            straggler_slowdown=config.straggler_slowdown,
            burst_probability=config.burst_probability,
        ),
        np.random.default_rng(run_id + 100),
    )
    strategies = _filter_strategies(default_strategies(), config.strategy_names)
    weights = {strategy.name: np.zeros(problem.n_features, dtype=float) for strategy in strategies}
    wall_clock = {strategy.name: 0.0 for strategy in strategies}
    strategy_rngs = {
        strategy.name: np.random.default_rng(run_id + 1000 + idx * 7919)
        for idx, strategy in enumerate(strategies)
    }
    initial_loss = problem.loss(np.zeros(problem.n_features, dtype=float))
    records: list[dict[str, float | str | int | bool]] = []
    for iteration in range(config.rounds):
        worker_state = worker_pool.sample(iteration)
        for strategy in strategies:
            name = strategy.name
            result = strategy.run_round(
                problem=problem,
                weights=weights[name],
                worker_state=worker_state,
                rng=strategy_rngs[name],
                iteration=iteration,
            )
            weights[name] = weights[name] - config.learning_rate * result.gradient
            wall_clock[name] += result.iteration_time
            records.append(
                {
                    "dataset": dataset,
                    "run_id": run_id,
                    "iteration": iteration,
                    "strategy": name,
                    "config": result.config_label,
                    "scenario": config.scenario,
                    "n_workers": config.n_workers,
                    "n_shards": config.n_shards,
                    "iteration_time": result.iteration_time,
                    "wall_clock": wall_clock[name],
                    "loss": problem.loss(weights[name]),
                    "accuracy": problem.accuracy(weights[name]),
                    "auc": problem.auc(weights[name]),
                    "initial_loss": initial_loss,
                    "decode_success": result.decode_success,
                    "decode_residual": result.decode_residual,
                    "decode_cpu_seconds": result.decode_cpu_seconds,
                    "selected_rows": result.selected_rows,
                    "extra_compute": result.extra_compute,
                    "nnz_expansion": result.nnz_expansion,
                    "second_layer_used": result.second_layer_used,
                    "slow_workers": int(worker_state.slow_mask.sum()),
                    "mean_worker_speed": float(worker_state.speeds.mean()),
                }
            )
    metrics = pd.DataFrame.from_records(records)
    return metrics, _summarize(metrics, config.target_loss_fraction)


def _summarize(metrics: pd.DataFrame, target_fraction: float) -> pd.DataFrame:
    static = metrics[metrics["strategy"] == "sparse_flexible_static"]
    static_final_loss = float(static.groupby("run_id")["loss"].last().iloc[0])
    initial_loss = float(metrics["initial_loss"].iloc[0])
    target_loss = initial_loss - target_fraction * (initial_loss - static_final_loss)
    rows = []
    for strategy, group in metrics.groupby("strategy", sort=False):
        reached = group[group["loss"] <= target_loss]
        if reached.empty:
            time_to_target = float(group["wall_clock"].iloc[-1])
            target_reached = False
        else:
            time_to_target = float(reached["wall_clock"].iloc[0])
            target_reached = True
        rows.append(
            {
                "dataset": group["dataset"].iloc[0],
                "run_id": int(group["run_id"].iloc[0]),
                "strategy": strategy,
                "mean_latency": float(group["iteration_time"].mean()),
                "p95_latency": float(group["iteration_time"].quantile(0.95)),
                "total_wall_clock": float(group["wall_clock"].iloc[-1]),
                "time_to_target_loss": time_to_target,
                "target_loss": target_loss,
                "target_reached": target_reached,
                "final_loss": float(group["loss"].iloc[-1]),
                "final_accuracy": float(group["accuracy"].iloc[-1]),
                "final_auc": float(group["auc"].iloc[-1]),
                "decode_success_rate": float(group["decode_success"].mean()),
                "mean_extra_compute": float(group["extra_compute"].mean()),
                "mean_selected_rows": float(group["selected_rows"].mean()),
                "second_layer_rate": float(group["second_layer_used"].mean()),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    static_row = _lookup(summary, "sparse_flexible_static")
    for column in ["mean_latency", "p95_latency", "total_wall_clock", "time_to_target_loss"]:
        gain_column = f"{column}_gain_vs_static"
        summary[gain_column] = (static_row[column] - summary[column]) / max(static_row[column], 1e-12)
    summary["selected_rows_delta_vs_static"] = (
        summary["mean_selected_rows"] - static_row["mean_selected_rows"]
    )
    return summary


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["dataset", "strategy"], sort=False)
        .agg(
            mean_latency_mean=("mean_latency", "mean"),
            mean_latency_std=("mean_latency", "std"),
            p95_latency_mean=("p95_latency", "mean"),
            p95_latency_std=("p95_latency", "std"),
            total_wall_clock_mean=("total_wall_clock", "mean"),
            total_wall_clock_std=("total_wall_clock", "std"),
            time_to_target_loss_mean=("time_to_target_loss", "mean"),
            time_to_target_loss_std=("time_to_target_loss", "std"),
            total_wall_clock_gain_mean=("total_wall_clock_gain_vs_static", "mean"),
            time_to_target_gain_mean=("time_to_target_loss_gain_vs_static", "mean"),
            mean_latency_gain_mean=("mean_latency_gain_vs_static", "mean"),
            p95_latency_gain_mean=("p95_latency_gain_vs_static", "mean"),
            final_loss_mean=("final_loss", "mean"),
            final_accuracy_mean=("final_accuracy", "mean"),
            final_auc_mean=("final_auc", "mean"),
            decode_success_rate_mean=("decode_success_rate", "mean"),
            selected_rows_delta_mean=("selected_rows_delta_vs_static", "mean"),
        )
        .reset_index()
    )


def _plot(aggregate: pd.DataFrame, out_dir: Path) -> None:
    plot_df = aggregate[aggregate["strategy"].isin(["rank_aware_sparse_flexible", "deadline_aware_sparse_flexible"])]
    if plot_df.empty:
        return
    datasets = list(dict.fromkeys(plot_df["dataset"].tolist()))
    strategies = ["rank_aware_sparse_flexible", "deadline_aware_sparse_flexible"]
    labels = ["Rank-aware", "Deadline-aware"]
    x = np.arange(len(datasets))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for idx, strategy in enumerate(strategies):
        offsets = x + (idx - 0.5) * width
        rows = plot_df.set_index(["dataset", "strategy"])
        p95 = [100.0 * rows.loc[(dataset, strategy), "p95_latency_gain_mean"] for dataset in datasets]
        ttl = [100.0 * rows.loc[(dataset, strategy), "total_wall_clock_gain_mean"] for dataset in datasets]
        axes[0].bar(offsets, p95, width=width, label=labels[idx])
        axes[1].bar(offsets, ttl, width=width, label=labels[idx])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("p95 latency gain (%)")
    axes[1].set_ylabel("fixed-update time gain (%)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "logistic_workload_summary.png", dpi=220)
    fig.savefig(out_dir / "logistic_workload_summary.pdf")
    plt.close(fig)


def _write_report(config: LogisticWorkloadConfig, aggregate: pd.DataFrame, out: Path) -> None:
    display = aggregate.copy()
    for column in [
        "mean_latency_mean",
        "p95_latency_mean",
        "total_wall_clock_mean",
        "time_to_target_loss_mean",
    ]:
        display[column] = 1000.0 * display[column]
    for column in [
        "time_to_target_gain_mean",
        "mean_latency_gain_mean",
        "p95_latency_gain_mean",
        "total_wall_clock_gain_mean",
    ]:
        display[column] = 100.0 * display[column]
    report = [
        "# Sparse Logistic Workload Diagnostics",
        "",
        "This experiment runs an end-to-end sparse binary logistic-regression",
        "training loop on real LIBSVM data. Shard gradients are additive, so",
        "successful coded decoding recovers the exact full logistic gradient.",
        "The workload is intended to address ridge-only concerns; it is still a",
        "prototype workload, not a production ML training system.",
        "",
        "## Configuration",
        "",
        f"- Datasets: `{', '.join(config.datasets)}`",
        f"- Workers/shards: `{config.n_workers}/{config.n_shards}`",
        f"- Rounds per run: `{config.rounds}`",
        f"- Target loss reduction fraction: `{config.target_loss_fraction}`",
        "",
        "## Aggregate Results",
        "",
        "Positive gains are relative to `sparse_flexible_static` within each run.",
        "Fixed-update time is the wall-clock time to complete the same number of",
        "exact logistic-gradient updates, reaching the same final loss/accuracy.",
        "",
        "```",
        display[
            [
                "dataset",
                "strategy",
                "p95_latency_mean",
                "p95_latency_gain_mean",
                "total_wall_clock_mean",
                "total_wall_clock_gain_mean",
                "time_to_target_loss_mean",
                "time_to_target_gain_mean",
                "final_accuracy_mean",
                "final_auc_mean",
                "decode_success_rate_mean",
                "selected_rows_delta_mean",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}"),
        "```",
    ]
    out.write_text("\n".join(report) + "\n", encoding="utf-8")


def _filter_strategies(strategies, names: tuple[str, ...]):
    wanted = set(names)
    selected = [strategy for strategy in strategies if strategy.name in wanted]
    found = {strategy.name for strategy in selected}
    missing = wanted - found
    if missing:
        raise ValueError(f"Unknown strategies: {sorted(missing)}")
    return selected


def _lookup(summary: pd.DataFrame, strategy: str) -> pd.Series:
    match = summary[summary["strategy"] == strategy]
    if match.empty:
        raise ValueError(f"Missing strategy {strategy!r} in summary")
    return match.iloc[0]


if __name__ == "__main__":
    main()
