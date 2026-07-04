from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.coded_learning_exp.multiprocess_runtime import (
    FLEXIBLE_CONFIG,
    MultiprocessExperimentConfig,
    _alignment_priorities,
    _apply_worker_alignment,
    _make_worker_states,
    run_multiprocess_problem,
)
from src.coded_learning_exp.realdata import make_libsvm_ridge_problem


DEFAULT_STRATEGIES = (
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled alignment sweep for real sparse datasets."
    )
    parser.add_argument("--datasets", nargs="+", default=["w8a", "rcv1"])
    parser.add_argument("--cache-dir", type=Path, default=Path("data") / "libsvm")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 31])
    parser.add_argument("--alignment-modes", nargs="+", default=["aligned", "none", "anti"])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--scenario", choices=["stable", "burst", "drift", "phase"], default="phase")
    parser.add_argument("--drift-period", type=int, default=8)
    parser.add_argument("--straggler-fraction", type=float, default=0.45)
    parser.add_argument("--straggler-slowdown", type=float, default=0.08)
    parser.add_argument("--sleep-scale", type=float, default=0.06)
    parser.add_argument("--cost-scale", type=float, default=0.004)
    parser.add_argument("--cancel-poll-seconds", type=float, default=0.004)
    parser.add_argument("--out", type=Path, default=Path("runtime_realdata_alignment"))
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
        for alignment_mode in args.alignment_modes:
            for seed, problem in problems.items():
                label = f"{dataset}_{alignment_mode}_seed{seed}"
                run_dir = args.out / label
                print(
                    f"Running dataset={dataset} seed={seed} "
                    f"alignment={alignment_mode} -> {run_dir}",
                    flush=True,
                )
                config = MultiprocessExperimentConfig(
                    n_samples=problem.n_samples,
                    n_features=problem.n_features,
                    density=problem.x.nnz / max(1, problem.x.shape[0] * problem.x.shape[1]),
                    n_shards=args.shards,
                    n_workers=args.workers,
                    rounds=args.rounds,
                    learning_rate=args.learning_rate,
                    l2=args.l2,
                    scenario=args.scenario,
                    drift_period=args.drift_period,
                    straggler_fraction=args.straggler_fraction,
                    straggler_slowdown=args.straggler_slowdown,
                    seed=seed,
                    output_dir=run_dir,
                    strategy_names=tuple(args.strategies),
                    sleep_scale=args.sleep_scale,
                    cost_scale=args.cost_scale,
                    cancel_poll_seconds=args.cancel_poll_seconds,
                    start_method=args.start_method,
                    alignment_mode=alignment_mode,
                )
                diag = compute_alignment_diagnostics(problem, config)
                diagnostics.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "alignment_mode": alignment_mode,
                        **diag,
                    }
                )
                _, summary = run_multiprocess_problem(config, problem, dataset_name=dataset)
                summary.insert(0, "decode_speed_mismatch", diag["decode_speed_mismatch"])
                summary.insert(0, "static_decode_speed_corr", diag["static_decode_speed_corr"])
                summary.insert(0, "worker_capacity_cv", diag["worker_capacity_cv"])
                summary.insert(0, "alignment_mode", alignment_mode)
                summary.insert(0, "seed", seed)
                summary.insert(0, "dataset", dataset)
                summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.out / "combined_alignment_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out / "alignment_diagnostics.csv", index=False)
    aggregate = aggregate_alignment(combined)
    aggregate.to_csv(args.out / "aggregate_alignment_summary.csv", index=False)
    print("\nAggregate controlled-alignment summary")
    print(aggregate.to_string(index=False))


def compute_alignment_diagnostics(problem, config: MultiprocessExperimentConfig) -> dict[str, float]:
    priorities = _alignment_priorities(problem, config)
    states = _apply_worker_alignment(problem, config, _make_worker_states(config))
    regrets = []
    corrs = []
    cvs = []
    for state in states:
        capacity = state.speeds / (1.0 + state.delays)
        regrets.append(normalized_assignment_regret(priorities, capacity))
        corrs.append(pearson_corr(priorities, capacity))
        cvs.append(coefficient_of_variation(capacity))
    return {
        "decode_speed_mismatch": float(np.mean(regrets)),
        "static_decode_speed_corr": float(np.mean(corrs)),
        "worker_capacity_cv": float(np.mean(cvs)),
        "decode_priority_cv": coefficient_of_variation(priorities),
    }


def normalized_assignment_regret(priorities: np.ndarray, capacity: np.ndarray) -> float:
    static = float(np.dot(priorities, capacity))
    best = float(np.dot(np.sort(priorities), np.sort(capacity)))
    worst = float(np.dot(np.sort(priorities), np.sort(capacity)[::-1]))
    return float((best - static) / max(abs(best - worst), 1e-12))


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(np.abs(values)))
    return 0.0 if mean <= 1e-12 else float(np.std(values) / mean)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def aggregate_alignment(combined: pd.DataFrame) -> pd.DataFrame:
    return (
        combined.groupby(["dataset", "alignment_mode", "strategy"], sort=False)
        .agg(
            mean_decode_latency=("mean_decode_latency", "mean"),
            p95_decode_latency=("p95_decode_latency", "mean"),
            mean_barrier_latency=("mean_barrier_latency", "mean"),
            decode_improvement=("decode_latency_improvement_vs_sparse_flexible", "mean"),
            decode_improvement_std=("decode_latency_improvement_vs_sparse_flexible", sample_std),
            p95_improvement=("p95_decode_latency_improvement_vs_sparse_flexible", "mean"),
            p95_improvement_std=("p95_decode_latency_improvement_vs_sparse_flexible", sample_std),
            selected_rows=("mean_selected_rows", "mean"),
            completed_rows=("mean_completed_rows", "mean"),
            extra_compute=("mean_extra_compute", "mean"),
            scheduler_seconds=("mean_scheduler_seconds", "mean"),
            decode_speed_mismatch=("decode_speed_mismatch", "mean"),
            static_decode_speed_corr=("static_decode_speed_corr", "mean"),
            worker_capacity_cv=("worker_capacity_cv", "mean"),
        )
        .reset_index()
    )


def sample_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


if __name__ == "__main__":
    main()
