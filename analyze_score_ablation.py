"""Small score-ablation diagnostic for first-decode scheduling.

This script is intentionally a small offline diagnostic, not a new systems
benchmark.  It compares row-pair priority scores using the same fixed flexible
code and worker states, then predicts first-decode time with the same row-span
check used by the online guard.  For 8 workers it also enumerates minimal
decodable subsets to obtain an oracle-style row criticality signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.coded_learning_exp.coding import decode_coefficients, make_flexible_rows
from src.coded_learning_exp.data import make_sparse_ridge_problem
from src.coded_learning_exp.workers import WorkerPool, WorkerPoolConfig, WorkerState


@dataclass(frozen=True)
class ScoreConfig:
    workers: int = 8
    shards: int = 8
    degree_first: int = 2
    degree_second: int = 3
    code_seeds: tuple[int, ...] = (17, 23, 31, 43, 59)
    rounds_per_seed: int = 12
    sampled_minset_sizes: tuple[int, ...] = (128, 512)
    sleep_scale: float = 0.010
    cost_scale: float = 0.002


def _row_costs(rows: np.ndarray, shard_costs: np.ndarray) -> np.ndarray:
    costs = np.zeros(rows.shape[0], dtype=float)
    for row_id, row in enumerate(rows):
        support = np.flatnonzero(np.abs(row) > 0.0)
        costs[row_id] = float(shard_costs[support].sum()) if support.size else 0.0
    return costs


def _pair_costs(first: np.ndarray, second: np.ndarray, shard_costs: np.ndarray) -> np.ndarray:
    return _row_costs(first, shard_costs) + _row_costs(second, shard_costs)


def _pair_rho(rows: np.ndarray, workers: int) -> np.ndarray:
    decode = decode_coefficients(rows)
    if not decode.success:
        return np.ones(workers, dtype=float)
    coeff = np.abs(decode.coefficients)
    return coeff[:workers] + coeff[workers : 2 * workers]


def _minimal_subset_pair_frequency(rows: np.ndarray, workers: int) -> np.ndarray:
    """Count row participation in inclusion-minimal decodable subsets."""
    n_rows = rows.shape[0]
    if n_rows > 20:
        raise ValueError("Minimal-subset enumeration is intended only for small codes.")

    decodable = np.zeros(1 << n_rows, dtype=bool)
    for size in range(1, n_rows + 1):
        for subset in combinations(range(n_rows), size):
            mask = 0
            for row_id in subset:
                mask |= 1 << row_id
            decodable[mask] = decode_coefficients(rows[list(subset)]).success

    row_counts = np.zeros(n_rows, dtype=float)
    minimal_count = 0
    for mask in np.flatnonzero(decodable):
        minimal = True
        for row_id in range(n_rows):
            if mask & (1 << row_id) and decodable[mask & ~(1 << row_id)]:
                minimal = False
                break
        if not minimal:
            continue
        minimal_count += 1
        for row_id in range(n_rows):
            if mask & (1 << row_id):
                row_counts[row_id] += 1.0

    if minimal_count == 0:
        return np.ones(workers, dtype=float)
    row_freq = row_counts / float(minimal_count)
    return row_freq[:workers] + row_freq[workers : 2 * workers]


def _sampled_minset_pair_frequency(
    rows: np.ndarray,
    workers: int,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate minimal-subset frequency by random arrival orders."""
    n_rows = rows.shape[0]
    row_counts = np.zeros(n_rows, dtype=float)
    successful = 0
    for _ in range(samples):
        order = rng.permutation(n_rows)
        prefix: list[int] = []
        for row_id in order:
            prefix.append(int(row_id))
            if decode_coefficients(rows[prefix]).success:
                break
        if not prefix or not decode_coefficients(rows[prefix]).success:
            continue

        minimal = list(prefix)
        changed = True
        while changed:
            changed = False
            for row_id in list(minimal):
                candidate = [item for item in minimal if item != row_id]
                if candidate and decode_coefficients(rows[candidate]).success:
                    minimal = candidate
                    changed = True
        successful += 1
        for row_id in minimal:
            row_counts[int(row_id)] += 1.0

    if successful == 0:
        return np.ones(workers, dtype=float)
    row_freq = row_counts / float(successful)
    return row_freq[:workers] + row_freq[workers : 2 * workers]


def _assign_by_priority(priority: np.ndarray, state: WorkerState) -> np.ndarray:
    task_order = np.argsort(-priority)
    worker_order = np.argsort(-state.speeds)
    assignments = np.empty(priority.size, dtype=int)
    assignments[task_order] = worker_order
    return assignments


def _predict_first_decode(
    rows: np.ndarray,
    pair_assignments: np.ndarray,
    state: WorkerState,
    row_costs: np.ndarray,
    cfg: ScoreConfig,
) -> tuple[float, int]:
    assignments = np.concatenate([pair_assignments, pair_assignments])
    worker_available = np.zeros(state.speeds.size, dtype=float)
    event_times = np.zeros(rows.shape[0], dtype=float)
    for row_id, worker_id in enumerate(assignments):
        duration = (
            cfg.sleep_scale * float(state.delays[int(worker_id)])
            + cfg.cost_scale
            * float(row_costs[row_id])
            / max(float(state.speeds[int(worker_id)]), 1e-12)
        )
        worker_available[int(worker_id)] += max(0.0, duration)
        event_times[row_id] = worker_available[int(worker_id)]

    selected: list[int] = []
    target_residual = np.ones(rows.shape[1], dtype=float)
    target_scale = np.sqrt(max(rows.shape[1], 1))
    basis: list[np.ndarray] = []
    tol = 1e-7
    for row_id in np.argsort(event_times):
        selected.append(int(row_id))
        vector = np.asarray(rows[row_id], dtype=float).copy()
        for q in basis:
            vector -= float(vector @ q) * q
        norm = float(np.linalg.norm(vector))
        if norm <= tol:
            continue
        q = vector / norm
        basis.append(q)
        target_residual -= float(target_residual @ q) * q
        if float(np.linalg.norm(target_residual)) / target_scale <= tol:
            return float(event_times[row_id]), len(selected)
    return float(event_times.max(initial=0.0)), len(selected)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run(cfg: ScoreConfig, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_out: list[dict[str, float | int | str]] = []
    corr_out: list[dict[str, float | int]] = []
    timing_out: list[dict[str, float | int | str]] = []
    out.mkdir(parents=True, exist_ok=True)

    for seed in cfg.code_seeds:
        rng = np.random.default_rng(seed)
        first, second = make_flexible_rows(
            cfg.workers,
            cfg.shards,
            cfg.degree_first,
            cfg.degree_second,
            rng,
        )
        rows = np.vstack([first, second])
        problem = make_sparse_ridge_problem(
            n_samples=1600,
            n_features=240,
            density=0.014,
            n_shards=cfg.shards,
            l2=1e-3,
            seed=seed + 1000,
        )
        shard_costs = problem.shard_costs()
        row_costs = _row_costs(rows, shard_costs)
        pair_costs = _pair_costs(first, second, shard_costs)
        start = time.perf_counter()
        rho = _pair_rho(rows, cfg.workers)
        timing_out.append(
            {
                "seed": seed,
                "policy": "rho-family",
                "samples": 0,
                "build_ms": 1000.0 * (time.perf_counter() - start),
            }
        )
        start = time.perf_counter()
        oracle = _minimal_subset_pair_frequency(rows, cfg.workers)
        timing_out.append(
            {
                "seed": seed,
                "policy": "oracle-minset",
                "samples": 0,
                "build_ms": 1000.0 * (time.perf_counter() - start),
            }
        )
        sampled = {}
        for sample_size in cfg.sampled_minset_sizes:
            start = time.perf_counter()
            sampled[sample_size] = _sampled_minset_pair_frequency(
                rows,
                cfg.workers,
                samples=sample_size,
                rng=np.random.default_rng(seed + 3000 + sample_size),
            )
            timing_out.append(
                {
                    "seed": seed,
                    "policy": f"sampled-minset-{sample_size}",
                    "samples": sample_size,
                    "build_ms": 1000.0 * (time.perf_counter() - start),
                }
            )

        corr_out.append(
            {
                "seed": seed,
                "rho_oracle_corr": _corr(rho, oracle),
                "rho_cost_oracle_corr": _corr(rho * np.maximum(pair_costs, 1e-12), oracle),
                "pair_cost_oracle_corr": _corr(pair_costs, oracle),
                **{
                    f"sampled_minset_{sample_size}_oracle_corr": _corr(values, oracle)
                    for sample_size, values in sampled.items()
                },
            }
        )

        priority_by_policy = {
            "static": None,
            "random": None,
            "cost-only": pair_costs,
            "rho": rho,
            "rho*C": rho * np.maximum(pair_costs, 1e-12),
            "rho/C": rho / np.maximum(pair_costs, 1e-12),
            **{
                f"sampled-minset-{sample_size}": values
                for sample_size, values in sampled.items()
            },
            "oracle-minset": oracle,
        }
        pool = WorkerPool(
            WorkerPoolConfig(
                n_workers=cfg.workers,
                scenario="phase",
                drift_period=4,
                straggler_fraction=0.45,
                straggler_slowdown=0.08,
            ),
            np.random.default_rng(seed + 2000),
        )
        for round_id in range(cfg.rounds_per_seed):
            state = pool.sample(round_id)
            random_assignment = np.random.default_rng(seed * 100 + round_id).permutation(
                cfg.workers
            )
            static_assignment = np.arange(cfg.workers, dtype=int)
            static_time, static_prefix = _predict_first_decode(
                rows, static_assignment, state, row_costs, cfg
            )
            for policy, priority in priority_by_policy.items():
                if policy == "static":
                    assignment = static_assignment
                elif policy == "random":
                    assignment = random_assignment
                else:
                    assignment = _assign_by_priority(np.asarray(priority), state)
                pred_time, prefix = _predict_first_decode(rows, assignment, state, row_costs, cfg)
                rows_out.append(
                    {
                        "seed": seed,
                        "round": round_id,
                        "policy": policy,
                        "predicted_ms": 1000.0 * pred_time,
                        "prefix_rows": prefix,
                        "gain_vs_static_pct": 100.0
                        * (static_time - pred_time)
                        / max(static_time, 1e-12),
                        "prefix_delta_vs_static": prefix - static_prefix,
                    }
                )

    details = pd.DataFrame(rows_out)
    summary = (
        details.groupby("policy", as_index=False)
        .agg(
            mean_predicted_ms=("predicted_ms", "mean"),
            p95_predicted_ms=("predicted_ms", lambda s: float(s.quantile(0.95))),
            mean_prefix_rows=("prefix_rows", "mean"),
            mean_prefix_delta_vs_static=("prefix_delta_vs_static", "mean"),
        )
    )
    static_row = summary.loc[summary["policy"] == "static"].iloc[0]
    static_mean = float(static_row["mean_predicted_ms"])
    static_p95 = float(static_row["p95_predicted_ms"])
    summary["mean_gain_vs_static_pct"] = (
        100.0 * (static_mean - summary["mean_predicted_ms"]) / max(static_mean, 1e-12)
    )
    summary["p95_gain_vs_static_pct"] = (
        100.0 * (static_p95 - summary["p95_predicted_ms"]) / max(static_p95, 1e-12)
    )
    summary = summary.sort_values("mean_predicted_ms")
    correlations = pd.DataFrame(corr_out)
    timings = pd.DataFrame(timing_out)
    timing_summary = (
        timings.groupby(["policy", "samples"], as_index=False)
        .agg(
            mean_build_ms=("build_ms", "mean"),
            p95_build_ms=("build_ms", lambda s: float(s.quantile(0.95))),
            max_build_ms=("build_ms", "max"),
        )
        .sort_values("mean_build_ms")
    )
    details.to_csv(out / "score_ablation_runs.csv", index=False)
    summary.to_csv(out / "score_ablation_summary.csv", index=False)
    correlations.to_csv(out / "score_ablation_oracle_correlations.csv", index=False)
    timings.to_csv(out / "score_ablation_score_build_times.csv", index=False)
    timing_summary.to_csv(out / "score_ablation_score_build_summary.csv", index=False)

    lines = [
        "# Score Ablation Diagnostic",
        "",
        "Offline 8-worker/8-shard diagnostic.  The oracle column enumerates "
        "inclusion-minimal decodable subsets and is used only as a small-code "
        "reference, not as an online scheduler.",
        "",
        "## Summary",
        "",
        "| Policy | Mean pred. ms | p95 pred. ms | Gain vs static | Prefix delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['policy']} | {row['mean_predicted_ms']:.2f} | "
            f"{row['p95_predicted_ms']:.2f} | {row['mean_gain_vs_static_pct']:.1f}% | "
            f"{row['mean_prefix_delta_vs_static']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Oracle Correlation",
            "",
            f"- mean corr(rho, oracle-minset frequency): "
            f"{correlations['rho_oracle_corr'].mean():.2f}",
            f"- mean corr(rho*C, oracle-minset frequency): "
            f"{correlations['rho_cost_oracle_corr'].mean():.2f}",
            f"- mean corr(cost, oracle-minset frequency): "
            f"{correlations['pair_cost_oracle_corr'].mean():.2f}",
        ]
    )
    for sample_size in cfg.sampled_minset_sizes:
        column = f"sampled_minset_{sample_size}_oracle_corr"
        if column in correlations:
            lines.append(
                f"- mean corr(sampled-minset-{sample_size}, oracle-minset frequency): "
                f"{correlations[column].mean():.2f}"
            )
    lines.extend(
        [
            "",
            "## Score Build Time",
            "",
            "| Policy | Mean build ms | p95 build ms | Max build ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in timing_summary.iterrows():
        lines.append(
            f"| {row['policy']} | {row['mean_build_ms']:.2f} | "
            f"{row['p95_build_ms']:.2f} | {row['max_build_ms']:.2f} |"
        )
    (out / "score_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, correlations, timing_summary


def main() -> None:
    out = Path("score_ablation_diagnostics")
    summary, _, timing_summary = run(ScoreConfig(), out)
    print(summary.to_string(index=False))
    print()
    print(timing_summary.to_string(index=False))
    print(f"Wrote {out / 'score_ablation_report.md'}")


if __name__ == "__main__":
    main()
