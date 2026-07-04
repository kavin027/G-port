from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import make_sparse_ridge_problem
from .strategies import default_strategies
from .workers import WorkerPool, WorkerPoolConfig


STATIC_FLEXIBLE_STRATEGIES = {
    "flexible_thin_static",
    "sparse_flexible_static",
    "flexible_robust_static",
    "flexible_dense_static",
}


@dataclass(frozen=True)
class ExperimentConfig:
    n_samples: int = 6000
    n_features: int = 800
    density: float = 0.01
    n_shards: int = 16
    n_workers: int = 24
    rounds: int = 140
    learning_rate: float = 0.35
    l2: float = 1e-3
    scenario: str = "drift"
    drift_period: int = 35
    straggler_fraction: float = 0.25
    straggler_slowdown: float = 0.22
    burst_probability: float = 0.45
    seed: int = 13
    output_dir: Path = Path("results")
    strategy_names: tuple[str, ...] | None = None


def run_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    problem = make_sparse_ridge_problem(
        n_samples=config.n_samples,
        n_features=config.n_features,
        density=config.density,
        n_shards=config.n_shards,
        l2=config.l2,
        seed=config.seed,
    )

    worker_rng = np.random.default_rng(config.seed + 100)
    worker_pool = WorkerPool(
        WorkerPoolConfig(
            n_workers=config.n_workers,
            scenario=config.scenario,
            drift_period=config.drift_period,
            straggler_fraction=config.straggler_fraction,
            straggler_slowdown=config.straggler_slowdown,
            burst_probability=config.burst_probability,
        ),
        worker_rng,
    )

    strategies = default_strategies()
    if config.strategy_names is not None:
        strategies = _filter_strategies(strategies, config.strategy_names)
    weights = {
        strategy.name: np.zeros(config.n_features, dtype=float) for strategy in strategies
    }
    wall_clock = {strategy.name: 0.0 for strategy in strategies}
    strategy_rngs = {
        strategy.name: np.random.default_rng(config.seed + 1000 + idx * 7919)
        for idx, strategy in enumerate(strategies)
    }

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
                    "iteration": iteration,
                    "strategy": name,
                    "config": result.config_label,
                    "scenario": config.scenario,
                    "density": config.density,
                    "n_workers": config.n_workers,
                    "n_shards": config.n_shards,
                    "iteration_time": result.iteration_time,
                    "wall_clock": wall_clock[name],
                    "loss": problem.loss(weights[name]),
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
    summary = _summarize(metrics)
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    summary.to_csv(config.output_dir / "summary.csv", index=False)
    return metrics, summary


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby("strategy", sort=False)
    summary = grouped.agg(
        mean_latency=("iteration_time", "mean"),
        p95_latency=("iteration_time", lambda x: x.quantile(0.95)),
        p99_latency=("iteration_time", lambda x: x.quantile(0.99)),
        final_loss=("loss", "last"),
        total_wall_clock=("wall_clock", "last"),
        decode_success_rate=("decode_success", "mean"),
        mean_extra_compute=("extra_compute", "mean"),
        mean_selected_rows=("selected_rows", "mean"),
        second_layer_rate=("second_layer_used", "mean"),
    ).reset_index()

    cutoff = int(metrics["iteration"].max() * 0.30)
    steady = (
        metrics[metrics["iteration"] >= cutoff]
        .groupby("strategy", sort=False)
        .agg(
            steady_mean_latency=("iteration_time", "mean"),
            steady_p95_latency=("iteration_time", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    summary = summary.merge(steady, on="strategy", how="left")

    baseline_latency = _lookup_metric(summary, "sparse_flexible_static", "mean_latency")
    baseline_wall_clock = _lookup_metric(summary, "sparse_flexible_static", "total_wall_clock")
    summary["latency_improvement_vs_static_flexible"] = (
        baseline_latency - summary["mean_latency"]
    ) / baseline_latency
    summary["wall_clock_improvement_vs_static_flexible"] = (
        baseline_wall_clock - summary["total_wall_clock"]
    ) / baseline_wall_clock

    fixed_summary = summary[summary["strategy"].isin(STATIC_FLEXIBLE_STRATEGIES)]
    if not fixed_summary.empty:
        best_fixed_latency = float(fixed_summary["mean_latency"].min())
        summary["latency_improvement_vs_best_fixed_flexible"] = (
            best_fixed_latency - summary["mean_latency"]
        ) / best_fixed_latency

    oracle = oracle_static_flexible(metrics)
    summary["oracle_static_flexible_mean_latency"] = oracle["mean_latency"]
    summary["oracle_static_flexible_p95_latency"] = oracle["p95_latency"]
    summary["latency_gap_to_oracle_static_flexible"] = (
        summary["mean_latency"] - oracle["mean_latency"]
    ) / oracle["mean_latency"]
    return summary


def oracle_static_flexible(metrics: pd.DataFrame) -> dict[str, float]:
    fixed_metrics = metrics[metrics["strategy"].isin(STATIC_FLEXIBLE_STRATEGIES)]
    if fixed_metrics.empty:
        return {"mean_latency": float("nan"), "p95_latency": float("nan")}
    oracle_latencies = fixed_metrics.groupby("iteration")["iteration_time"].min()
    return {
        "mean_latency": float(oracle_latencies.mean()),
        "p95_latency": float(oracle_latencies.quantile(0.95)),
    }


def _lookup_metric(summary: pd.DataFrame, strategy: str, column: str) -> float:
    matches = summary.loc[summary["strategy"] == strategy, column]
    if matches.empty:
        return float("nan")
    return float(matches.iloc[0])


def _filter_strategies(strategies: list, strategy_names: tuple[str, ...]) -> list:
    wanted = set(strategy_names)
    available = {strategy.name for strategy in strategies}
    missing = sorted(wanted - available)
    if missing:
        raise ValueError(
            "Unknown strategy name(s): "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(sorted(available))
        )
    return [strategy for strategy in strategies if strategy.name in wanted]
