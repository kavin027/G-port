from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.coded_learning_exp.multiprocess_runtime import (
    FLEXIBLE_CONFIG,
    MultiprocessExperimentConfig,
    _make_worker_states,
    run_multiprocess_problem,
)
from src.coded_learning_exp.realdata import make_libsvm_ridge_problem
from src.coded_learning_exp.strategies import (
    _decode_pair_priorities,
    _make_flexible_code,
    _row_cost,
    _stable_seed,
)


DEFAULT_STRATEGIES = (
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity sweep for real sparse datasets. It varies worker "
            "heterogeneity and records a decode-speed mismatch diagnostic."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["w8a", "rcv1"])
    parser.add_argument("--cache-dir", type=Path, default=Path("data") / "libsvm")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 31])
    parser.add_argument("--straggler-fractions", type=float, nargs="+", default=[0.25, 0.45])
    parser.add_argument("--straggler-slowdowns", type=float, nargs="+", default=[0.04, 0.08, 0.16])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=8)
    parser.add_argument("--sleep-scale", type=float, default=0.06)
    parser.add_argument("--cost-scale", type=float, default=0.004)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.004)
    parser.add_argument("--out", type=Path, default=Path("runtime_realdata_sensitivity"))
    parser.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], default=None)
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    diagnostics: list[dict[str, float | int | str]] = []

    for dataset in args.datasets:
        problems = {
            seed: make_libsvm_ridge_problem(
                dataset=dataset,
                n_shards=args.shards,
                l2=args.l2,
                seed=seed,
                cache_dir=args.cache_dir,
            )
            for seed in args.seeds
        }
        for fraction in args.straggler_fractions:
            for slowdown in args.straggler_slowdowns:
                for seed, problem in problems.items():
                    label = (
                        f"{dataset}_f{fraction:.2f}_s{slowdown:.2f}_seed{seed}"
                        .replace(".", "p")
                    )
                    run_dir = args.out / label
                    print(
                        "Running "
                        f"dataset={dataset} seed={seed} "
                        f"fraction={fraction} slowdown={slowdown} -> {run_dir}",
                        flush=True,
                    )
                    config = MultiprocessExperimentConfig(
                        n_samples=problem.n_samples,
                        n_features=problem.n_features,
                        density=problem.x.nnz
                        / max(1, problem.x.shape[0] * problem.x.shape[1]),
                        n_shards=args.shards,
                        n_workers=args.workers,
                        rounds=args.rounds,
                        learning_rate=args.learning_rate,
                        l2=args.l2,
                        scenario=args.scenario,
                        drift_period=args.drift_period,
                        straggler_fraction=fraction,
                        straggler_slowdown=slowdown,
                        seed=seed,
                        output_dir=run_dir,
                        strategy_names=tuple(args.strategies),
                        sleep_scale=args.sleep_scale,
                        cost_scale=args.cost_scale,
                        cancel_poll_seconds=args.cancel_poll_seconds,
                        start_method=args.start_method,
                    )
                    diag = compute_assignment_diagnostics(problem, config)
                    diagnostics.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "straggler_fraction": fraction,
                            "straggler_slowdown": slowdown,
                            **diag,
                        }
                    )
                    _, summary = run_multiprocess_problem(config, problem, dataset_name=dataset)
                    summary.insert(0, "decode_speed_mismatch", diag["decode_speed_mismatch"])
                    summary.insert(0, "cost_speed_mismatch", diag["cost_speed_mismatch"])
                    summary.insert(0, "decode_priority_cv", diag["decode_priority_cv"])
                    summary.insert(0, "cost_priority_cv", diag["cost_priority_cv"])
                    summary.insert(0, "worker_capacity_cv", diag["worker_capacity_cv"])
                    summary.insert(0, "cost_decode_corr", diag["cost_decode_corr"])
                    summary.insert(0, "straggler_slowdown", slowdown)
                    summary.insert(0, "straggler_fraction", fraction)
                    summary.insert(0, "seed", seed)
                    summary.insert(0, "dataset", dataset)
                    summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.out / "combined_sensitivity_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out / "assignment_mismatch_diagnostics.csv", index=False)
    aggregate = aggregate_sensitivity(combined)
    aggregate.to_csv(args.out / "aggregate_sensitivity_summary.csv", index=False)
    print("\nAggregate sensitivity summary")
    print(aggregate.to_string(index=False))


def compute_assignment_diagnostics(
    problem,
    config: MultiprocessExperimentConfig,
) -> dict[str, float]:
    first, second = _make_flexible_code(
        "random",
        FLEXIBLE_CONFIG,
        config.n_workers,
        config.n_shards,
        np.random.default_rng(
            _stable_seed(FLEXIBLE_CONFIG.label, config.n_workers, config.n_shards)
        ),
    )
    shard_costs = problem.shard_costs()
    density = problem.x.nnz / max(1, problem.x.shape[0] * problem.x.shape[1])
    pair_costs = np.zeros(config.n_workers, dtype=float)
    for task_id in range(config.n_workers):
        first_cost, _ = _row_cost(first[task_id], shard_costs, density)
        second_cost, _ = _row_cost(second[task_id], shard_costs, density)
        pair_costs[task_id] = first_cost + second_cost

    decode_priorities = _decode_pair_priorities(first, second, pair_costs)
    states = _make_worker_states(config)
    decode_mismatch = []
    cost_mismatch = []
    capacity_cv = []
    static_decode_corr = []
    static_cost_corr = []

    for state in states:
        capacity = state.speeds / (1.0 + state.delays)
        decode_mismatch.append(normalized_assignment_regret(decode_priorities, capacity))
        cost_mismatch.append(normalized_assignment_regret(pair_costs, capacity))
        capacity_cv.append(coefficient_of_variation(capacity))
        static_decode_corr.append(pearson_corr(decode_priorities, capacity))
        static_cost_corr.append(pearson_corr(pair_costs, capacity))

    return {
        "decode_speed_mismatch": float(np.mean(decode_mismatch)),
        "cost_speed_mismatch": float(np.mean(cost_mismatch)),
        "decode_priority_cv": coefficient_of_variation(decode_priorities),
        "cost_priority_cv": coefficient_of_variation(pair_costs),
        "worker_capacity_cv": float(np.mean(capacity_cv)),
        "cost_decode_corr": pearson_corr(pair_costs, decode_priorities),
        "static_decode_speed_corr": float(np.mean(static_decode_corr)),
        "static_cost_speed_corr": float(np.mean(static_cost_corr)),
    }


def normalized_assignment_regret(priorities: np.ndarray, capacity: np.ndarray) -> float:
    priorities = np.asarray(priorities, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    static = float(np.dot(priorities, capacity))
    best = float(np.dot(np.sort(priorities), np.sort(capacity)))
    worst = float(np.dot(np.sort(priorities), np.sort(capacity)[::-1]))
    denom = max(abs(best - worst), 1e-12)
    return float((best - static) / denom)


def coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(np.abs(values)))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(values) / mean)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def aggregate_sensitivity(combined: pd.DataFrame) -> pd.DataFrame:
    return (
        combined.groupby(
            [
                "dataset",
                "straggler_fraction",
                "straggler_slowdown",
                "strategy",
            ],
            sort=False,
        )
        .agg(
            mean_decode_latency=("mean_decode_latency", "mean"),
            p95_decode_latency=("p95_decode_latency", "mean"),
            mean_barrier_latency=("mean_barrier_latency", "mean"),
            decode_improvement=("decode_latency_improvement_vs_sparse_flexible", "mean"),
            decode_improvement_std=("decode_latency_improvement_vs_sparse_flexible", sample_std),
            p95_improvement=("p95_decode_latency_improvement_vs_sparse_flexible", "mean"),
            p95_improvement_std=("p95_decode_latency_improvement_vs_sparse_flexible", sample_std),
            barrier_improvement=("barrier_latency_improvement_vs_sparse_flexible", "mean"),
            selected_rows=("mean_selected_rows", "mean"),
            completed_rows=("mean_completed_rows", "mean"),
            extra_compute=("mean_extra_compute", "mean"),
            scheduler_seconds=("mean_scheduler_seconds", "mean"),
            decode_speed_mismatch=("decode_speed_mismatch", "mean"),
            cost_speed_mismatch=("cost_speed_mismatch", "mean"),
            worker_capacity_cv=("worker_capacity_cv", "mean"),
            decode_priority_cv=("decode_priority_cv", "mean"),
            cost_priority_cv=("cost_priority_cv", "mean"),
            cost_decode_corr=("cost_decode_corr", "mean"),
        )
        .reset_index()
    )


def sample_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


if __name__ == "__main__":
    main()
