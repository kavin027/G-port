from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.coded_learning_exp.data import make_sparse_embedding_problem
from src.coded_learning_exp.strategies import default_strategies
from src.coded_learning_exp.workers import WorkerPool, WorkerPoolConfig


DEFAULT_STRATEGIES = (
    "uncoded_sync",
    "sparse_flexible_static",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
)


@dataclass(frozen=True)
class EmbeddingMicrobenchmarkConfig:
    n_interactions: int = 6000
    n_users: int = 384
    n_items: int = 1024
    embedding_dim: int = 8
    n_shards: int = 16
    n_workers: int = 24
    rounds: int = 45
    learning_rate: float = 0.35
    l2: float = 1e-4
    scenario: str = "phase"
    drift_period: int = 15
    straggler_fraction: float = 0.42
    straggler_slowdown: float = 0.10
    burst_probability: float = 0.45
    zipf_exponent: float = 1.15
    shard_cost_skew: float = 1.75
    run_ids: tuple[int, ...] = (17, 23, 31)
    output_dir: Path = Path("embedding_microbenchmark_diagnostics")
    strategy_names: tuple[str, ...] = DEFAULT_STRATEGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sparse embedding-style additive-update microbenchmark."
    )
    parser.add_argument("--out", type=Path, default=Path("embedding_microbenchmark_diagnostics"))
    parser.add_argument("--interactions", type=int, default=6000)
    parser.add_argument("--users", type=int, default=384)
    parser.add_argument("--items", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=45)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--scenario", default="phase")
    parser.add_argument("--drift-period", type=int, default=15)
    parser.add_argument("--straggler-fraction", type=float, default=0.42)
    parser.add_argument("--straggler-slowdown", type=float, default=0.10)
    parser.add_argument("--burst-probability", type=float, default=0.45)
    parser.add_argument("--zipf-exponent", type=float, default=1.15)
    parser.add_argument("--shard-cost-skew", type=float, default=1.75)
    parser.add_argument("--run-ids", type=int, nargs="+", default=[17, 23, 31])
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EmbeddingMicrobenchmarkConfig(
        n_interactions=args.interactions,
        n_users=args.users,
        n_items=args.items,
        embedding_dim=args.embedding_dim,
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
        zipf_exponent=args.zipf_exponent,
        shard_cost_skew=args.shard_cost_skew,
        run_ids=tuple(args.run_ids),
        output_dir=args.out,
        strategy_names=tuple(args.strategies),
    )
    metrics, summary, aggregate = run_embedding_microbenchmark(config)
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote sparse embedding microbenchmark outputs to {config.output_dir}")


def run_embedding_microbenchmark(
    config: EmbeddingMicrobenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics = []
    run_summaries = []
    for run_id in config.run_ids:
        metrics, summary = _run_one(config, run_id)
        run_dir = config.output_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(run_dir / "embedding_metrics.csv", index=False)
        summary.to_csv(run_dir / "embedding_summary.csv", index=False)
        run_metrics.append(metrics)
        run_summaries.append(summary)

    combined_metrics = pd.concat(run_metrics, ignore_index=True)
    combined_summary = pd.concat(run_summaries, ignore_index=True)
    aggregate = _aggregate(combined_summary)
    combined_metrics.to_csv(config.output_dir / "combined_embedding_metrics.csv", index=False)
    combined_summary.to_csv(config.output_dir / "combined_embedding_summary.csv", index=False)
    aggregate.to_csv(config.output_dir / "aggregate_embedding_summary.csv", index=False)
    _write_report(config, aggregate, config.output_dir / "embedding_microbenchmark_report.md")
    return combined_metrics, combined_summary, aggregate


def _run_one(
    config: EmbeddingMicrobenchmarkConfig, run_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    problem = make_sparse_embedding_problem(
        n_interactions=config.n_interactions,
        n_users=config.n_users,
        n_items=config.n_items,
        embedding_dim=config.embedding_dim,
        n_shards=config.n_shards,
        l2=config.l2,
        seed=run_id,
        zipf_exponent=config.zipf_exponent,
        shard_cost_skew=config.shard_cost_skew,
    )
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
    init_rng = np.random.default_rng(run_id + 5000)
    initial_weights = init_rng.normal(0.0, 0.02, size=problem.n_features)
    weights = {strategy.name: initial_weights.copy() for strategy in strategies}
    wall_clock = {strategy.name: 0.0 for strategy in strategies}
    strategy_rngs = {
        strategy.name: np.random.default_rng(run_id + 1000 + idx * 7919)
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
    return metrics, _summarize(metrics, run_id)


def _summarize(metrics: pd.DataFrame, run_id: int) -> pd.DataFrame:
    summary = (
        metrics.groupby("strategy", sort=False)
        .agg(
            mean_latency=("iteration_time", "mean"),
            p95_latency=("iteration_time", lambda x: x.quantile(0.95)),
            final_loss=("loss", "last"),
            total_wall_clock=("wall_clock", "last"),
            decode_success_rate=("decode_success", "mean"),
            mean_extra_compute=("extra_compute", "mean"),
            mean_selected_rows=("selected_rows", "mean"),
            second_layer_rate=("second_layer_used", "mean"),
        )
        .reset_index()
    )
    static = _lookup(summary, "sparse_flexible_static")
    summary["mean_latency_gain_vs_static"] = (
        static["mean_latency"] - summary["mean_latency"]
    ) / static["mean_latency"]
    summary["p95_latency_gain_vs_static"] = (
        static["p95_latency"] - summary["p95_latency"]
    ) / static["p95_latency"]
    summary["selected_rows_delta_vs_static"] = (
        summary["mean_selected_rows"] - static["mean_selected_rows"]
    )
    summary.insert(0, "run_id", run_id)
    return summary


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("strategy", sort=False)
        .agg(
            mean_latency_mean=("mean_latency", "mean"),
            mean_latency_std=("mean_latency", "std"),
            p95_latency_mean=("p95_latency", "mean"),
            p95_latency_std=("p95_latency", "std"),
            mean_latency_gain_mean=("mean_latency_gain_vs_static", "mean"),
            p95_latency_gain_mean=("p95_latency_gain_vs_static", "mean"),
            selected_rows_delta_mean=("selected_rows_delta_vs_static", "mean"),
            decode_success_rate_mean=("decode_success_rate", "mean"),
            final_loss_mean=("final_loss", "mean"),
        )
        .reset_index()
    )


def _write_report(
    config: EmbeddingMicrobenchmarkConfig, aggregate: pd.DataFrame, path: Path
) -> None:
    lines = [
        "# Sparse Embedding Microbenchmark",
        "",
        "This experiment is a recommendation-style additive-update check.  "
        "Each interaction touches one user embedding and one item embedding; "
        "shard gradients are additive, so the same sparse-flexible decoding "
        "path can recover the full update.",
        "",
        "## Configuration",
        "",
        f"- Interactions: {config.n_interactions}",
        f"- Users/items: {config.n_users}/{config.n_items}",
        f"- Embedding dimension: {config.embedding_dim}",
        f"- Workers/shards: {config.n_workers}/{config.n_shards}",
        f"- Rounds: {config.rounds}",
        f"- Worker scenario: {config.scenario}",
        "",
        "## Aggregate Results",
        "",
        aggregate.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Positive gains are relative to static sparse-flexible placement.  "
        "This is not an end-to-end recommender claim; it checks that the "
        "mismatch-prefix-guard mechanism extends beyond ridge regression when "
        "the workload exposes additive sparse updates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lookup(summary: pd.DataFrame, strategy: str) -> pd.Series:
    matches = summary[summary["strategy"] == strategy]
    if matches.empty:
        raise ValueError(f"Missing strategy {strategy!r} in summary.")
    return matches.iloc[0]


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


if __name__ == "__main__":
    main()
