from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .coding import DecodeResult, decode_coefficients
from .data import SparseRidgeProblem, make_sparse_ridge_problem
from .strategies import (
    DEFAULT_FLEXIBLE_CONFIGS,
    _deadline_assignment,
    _decode_pair_priorities,
    _make_flexible_code,
    _row_cost as _strategy_row_cost,
    _stable_seed,
)
from .workers import WorkerPool, WorkerPoolConfig, WorkerState


FLEXIBLE_CONFIG = DEFAULT_FLEXIBLE_CONFIGS[2]
DEFAULT_RUNTIME_STRATEGIES = (
    "uncoded_sync",
    "replication",
    "speculative_replication",
    "sparse_flexible_static",
    "worker_aware_sparse_flexible",
    "rank_aware_sparse_flexible",
    "deadline_aware_sparse_flexible",
    "guarded_system_portfolio",
)


@dataclass(frozen=True)
class MultiprocessExperimentConfig:
    n_samples: int = 12000
    n_features: int = 1600
    density: float = 0.006
    n_shards: int = 16
    n_workers: int = 12
    rounds: int = 30
    learning_rate: float = 0.25
    l2: float = 1e-3
    scenario: str = "phase"
    drift_period: int = 10
    straggler_fraction: float = 0.30
    straggler_slowdown: float = 0.18
    burst_probability: float = 0.45
    seed: int = 17
    output_dir: Path = Path("runtime_results")
    strategy_names: tuple[str, ...] = DEFAULT_RUNTIME_STRATEGIES
    sleep_scale: float = 0.03
    cost_scale: float = 0.006
    cancel_poll_seconds: float = 0.004
    start_method: str | None = None
    alignment_mode: str = "none"
    portfolio_fallback: str = "static"


@dataclass
class WorkerTask:
    round_id: int
    strategy: str
    rows: np.ndarray
    row_ids: np.ndarray
    second_layer_flags: np.ndarray
    weights: np.ndarray
    speed: float
    delay: float
    sleep_scale: float
    cost_scale: float
    cancel_poll_seconds: float
    jitter_seed: int


@dataclass
class RuntimeRowResult:
    kind: str
    round_id: int
    strategy: str
    worker_id: int
    row_id: int
    row: np.ndarray
    gradient: np.ndarray
    row_cost: float
    compute_cpu_seconds: float
    elapsed_seconds: float
    second_layer: bool


@dataclass
class RuntimeDoneResult:
    kind: str
    round_id: int
    strategy: str
    worker_id: int
    completed_rows: int


_WORKER_ID: int | None = None
_WORKER_PROBLEM: SparseRidgeProblem | None = None
_WORKER_SHARD_COSTS: np.ndarray | None = None


def _init_worker(worker_id: int, problem: SparseRidgeProblem) -> None:
    global _WORKER_ID, _WORKER_PROBLEM, _WORKER_SHARD_COSTS
    _WORKER_ID = worker_id
    _WORKER_PROBLEM = problem
    _WORKER_SHARD_COSTS = problem.shard_costs()


def _worker_loop(
    worker_id: int,
    problem: SparseRidgeProblem,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    cancel_round: mp.Value,
) -> None:
    _init_worker(worker_id, problem)
    while True:
        task = task_queue.get()
        if task is None:
            break
        completed = _execute_task(task, cancel_round, result_queue)
        result_queue.put(
            RuntimeDoneResult(
                kind="done",
                round_id=task.round_id,
                strategy=task.strategy,
                worker_id=worker_id,
                completed_rows=completed,
            )
        )


def _execute_task(
    task: WorkerTask,
    cancel_round: mp.Value,
    result_queue: mp.Queue,
) -> int:
    worker_id = _require_worker_id()
    completed = 0
    task_start = time.perf_counter()
    rng = np.random.default_rng(task.jitter_seed + worker_id * 7919)

    for local_idx, row in enumerate(task.rows):
        if _is_cancelled(cancel_round, task.round_id):
            break

        row_cost = _row_cost(row)
        synthetic_seconds = _synthetic_delay_seconds(
            row_cost=row_cost,
            speed=task.speed,
            delay=task.delay,
            sleep_scale=task.sleep_scale,
            cost_scale=task.cost_scale,
            rng=rng,
        )
        if _interruptible_sleep(
            synthetic_seconds,
            cancel_round,
            task.round_id,
            task.cancel_poll_seconds,
        ):
            break

        compute_start = time.perf_counter()
        gradient = _compute_encoded_gradient(row, task.weights)
        compute_seconds = time.perf_counter() - compute_start
        completed += 1

        result_queue.put(
            RuntimeRowResult(
                kind="row",
                round_id=task.round_id,
                strategy=task.strategy,
                worker_id=worker_id,
                row_id=int(task.row_ids[local_idx]),
                row=row,
                gradient=gradient,
                row_cost=row_cost,
                compute_cpu_seconds=compute_seconds,
                elapsed_seconds=time.perf_counter() - task_start,
                second_layer=bool(task.second_layer_flags[local_idx]),
            )
        )

    return completed


def _require_worker_id() -> int:
    if _WORKER_ID is None:
        raise RuntimeError("Worker process is not initialized.")
    return _WORKER_ID


def _require_problem() -> SparseRidgeProblem:
    if _WORKER_PROBLEM is None:
        raise RuntimeError("Worker process has no problem instance.")
    return _WORKER_PROBLEM


def _row_cost(row: np.ndarray) -> float:
    if _WORKER_SHARD_COSTS is None:
        raise RuntimeError("Worker shard costs are not initialized.")
    support = np.flatnonzero(np.abs(row) > 0.0)
    if support.size == 0:
        return 0.0
    return float(_WORKER_SHARD_COSTS[support].sum())


def _synthetic_delay_seconds(
    row_cost: float,
    speed: float,
    delay: float,
    sleep_scale: float,
    cost_scale: float,
    rng: np.random.Generator,
) -> float:
    jitter = float(rng.lognormal(mean=0.0, sigma=0.08))
    speed = max(float(speed), 1e-8)
    return max(0.0, sleep_scale * float(delay) + cost_scale * row_cost * jitter / speed)


def _interruptible_sleep(
    seconds: float,
    cancel_round: mp.Value,
    round_id: int,
    poll_seconds: float,
) -> bool:
    deadline = time.perf_counter() + seconds
    while True:
        if _is_cancelled(cancel_round, round_id):
            return True
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        time.sleep(min(max(poll_seconds, 1e-4), remaining))


def _is_cancelled(cancel_round: mp.Value, round_id: int) -> bool:
    return int(cancel_round.value) == int(round_id)


def _compute_encoded_gradient(row: np.ndarray, weights: np.ndarray) -> np.ndarray:
    problem = _require_problem()
    gradient = np.zeros(problem.n_features, dtype=float)
    scale = 1.0 / problem.n_samples
    support = np.flatnonzero(np.abs(row) > 0.0)
    for shard_id in support:
        coeff = float(row[shard_id])
        shard_slice = problem.shard_slices[int(shard_id)]
        x_shard = problem.x[shard_slice]
        residual = x_shard @ weights - problem.y[shard_slice]
        shard_gradient = np.asarray(x_shard.T @ residual).ravel() * scale
        gradient += coeff * shard_gradient
    return gradient


class StreamingWorkerRuntime:
    def __init__(
        self,
        problem: SparseRidgeProblem,
        n_workers: int,
        start_method: str | None = None,
    ) -> None:
        self.problem = problem
        self.n_workers = n_workers
        self.ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self.result_queue = self.ctx.Queue()
        self.cancel_round = self.ctx.Value("i", -1)
        self.task_queues = [self.ctx.Queue() for _ in range(n_workers)]
        self.processes = [
            self.ctx.Process(
                target=_worker_loop,
                args=(worker_id, problem, self.task_queues[worker_id], self.result_queue, self.cancel_round),
            )
            for worker_id in range(n_workers)
        ]

    def __enter__(self) -> StreamingWorkerRuntime:
        for process in self.processes:
            process.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for task_queue in self.task_queues:
            task_queue.put(None)
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    def run_round(
        self,
        *,
        strategy: str,
        round_id: int,
        rows: np.ndarray,
        assignments: np.ndarray,
        second_layer_flags: np.ndarray,
        weights: np.ndarray,
        worker_state: WorkerState,
        sleep_scale: float,
        cost_scale: float,
        cancel_poll_seconds: float,
        jitter_seed: int,
    ) -> dict[str, object]:
        self.cancel_round.value = -1
        start = time.perf_counter()
        expected_rows = int(rows.shape[0])
        for worker_id in range(self.n_workers):
            row_ids = np.flatnonzero(assignments == worker_id).astype(int)
            task = WorkerTask(
                round_id=round_id,
                strategy=strategy,
                rows=rows[row_ids].copy(),
                row_ids=row_ids,
                second_layer_flags=second_layer_flags[row_ids].copy(),
                weights=weights.copy(),
                speed=float(worker_state.speeds[worker_id]),
                delay=float(worker_state.delays[worker_id]),
                sleep_scale=sleep_scale,
                cost_scale=cost_scale,
                cancel_poll_seconds=cancel_poll_seconds,
                jitter_seed=jitter_seed,
            )
            self.task_queues[worker_id].put(task)

        selected_rows: list[np.ndarray] = []
        selected_gradients: list[np.ndarray] = []
        selected_costs: list[float] = []
        selected_second_flags: list[bool] = []
        completed_rows = 0
        done_workers = 0
        rows_after_decode = 0
        compute_cpu_seconds = 0.0
        decode_cpu_seconds = 0.0
        decode_residual = float("inf")
        decode_success = False
        decoded_gradient: np.ndarray | None = None
        decode_latency = 0.0

        while done_workers < self.n_workers:
            try:
                message = self.result_queue.get(timeout=60.0)
            except queue.Empty as exc:
                raise TimeoutError("Timed out while waiting for worker results.") from exc

            if message.round_id != round_id or message.strategy != strategy:
                continue

            now = time.perf_counter()
            if message.kind == "done":
                done_workers += 1
                continue

            completed_rows += 1
            compute_cpu_seconds += float(message.compute_cpu_seconds)
            if decode_success:
                rows_after_decode += 1
                continue

            selected_rows.append(message.row)
            selected_gradients.append(message.gradient)
            selected_costs.append(float(message.row_cost))
            selected_second_flags.append(bool(message.second_layer))

            decode = decode_coefficients(np.vstack(selected_rows))
            decode_cpu_seconds += decode.cpu_seconds
            decode_residual = decode.residual
            if decode.success:
                decode_success = True
                decode_latency = now - start
                decoded_gradient = decode.coefficients @ np.vstack(selected_gradients)
                decoded_gradient = decoded_gradient + self.problem.l2 * weights
                self.cancel_round.value = round_id

        barrier_latency = time.perf_counter() - start
        if not decode_success:
            fallback = self.problem.full_gradient(weights)
            decoded_gradient = fallback
            decode_latency = barrier_latency
            decode = DecodeResult(False, np.empty(0), decode_residual, 0.0)
        else:
            decode = DecodeResult(True, np.empty(0), decode_residual, decode_cpu_seconds)

        selected_cost = float(np.sum(selected_costs)) if selected_costs else 0.0
        return {
            "gradient": decoded_gradient,
            "decode_latency": float(decode_latency),
            "barrier_latency": float(barrier_latency),
            "decode_success": bool(decode_success),
            "decode_residual": float(decode.residual),
            "decode_cpu_seconds": float(decode_cpu_seconds),
            "selected_rows": int(len(selected_rows)),
            "completed_rows": int(completed_rows),
            "cancelled_rows": int(max(0, expected_rows - completed_rows)),
            "rows_after_decode": int(rows_after_decode),
            "extra_compute": selected_cost / max(float(self.problem.n_shards), 1.0),
            "second_layer_used": bool(any(selected_second_flags)),
            "worker_compute_cpu_seconds": float(compute_cpu_seconds),
        }


def run_multiprocess_experiment(
    config: MultiprocessExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    problem = make_sparse_ridge_problem(
        n_samples=config.n_samples,
        n_features=config.n_features,
        density=config.density,
        n_shards=config.n_shards,
        l2=config.l2,
        seed=config.seed,
    )
    return run_multiprocess_problem(config, problem, dataset_name="synthetic")


def run_multiprocess_problem(
    config: MultiprocessExperimentConfig,
    problem: SparseRidgeProblem,
    dataset_name: str = "external",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if problem.n_shards != config.n_shards:
        raise ValueError(
            f"Problem has {problem.n_shards} shards, but config uses {config.n_shards}."
        )
    strategy_specs = _make_strategy_specs(problem, config)
    worker_states = _apply_worker_alignment(problem, config, _make_worker_states(config))
    unknown = set(config.strategy_names) - set(strategy_specs)
    if unknown:
        raise ValueError(f"Unknown runtime strategies: {', '.join(sorted(unknown))}")

    records: list[dict[str, object]] = []
    for strategy_index, strategy_name in enumerate(config.strategy_names):
        weights = np.zeros(problem.n_features, dtype=float)
        decode_wall_clock = 0.0
        barrier_wall_clock = 0.0
        spec = strategy_specs[strategy_name]
        with StreamingWorkerRuntime(
            problem=problem,
            n_workers=config.n_workers,
            start_method=config.start_method,
        ) as runtime:
            for iteration, worker_state in enumerate(worker_states):
                schedule_start = time.perf_counter()
                rows, assignments, second_flags, config_label = spec(worker_state)
                scheduler_seconds = time.perf_counter() - schedule_start
                result = runtime.run_round(
                    strategy=strategy_name,
                    round_id=iteration,
                    rows=rows,
                    assignments=assignments,
                    second_layer_flags=second_flags,
                    weights=weights,
                    worker_state=worker_state,
                    sleep_scale=config.sleep_scale,
                    cost_scale=config.cost_scale,
                    cancel_poll_seconds=config.cancel_poll_seconds,
                    jitter_seed=config.seed + strategy_index * 100_003 + iteration * 997,
                )
                if hasattr(spec, "update"):
                    spec.update(
                        worker_state=worker_state,
                        result=result,
                        scheduler_seconds=scheduler_seconds,
                        config_label=config_label,
                    )
                weights = weights - config.learning_rate * result["gradient"]
                decode_wall_clock += float(result["decode_latency"])
                barrier_wall_clock += float(result["barrier_latency"])
                records.append(
                    {
                        "iteration": iteration,
                        "strategy": strategy_name,
                        "config": config_label,
                        "dataset": dataset_name,
                        "scenario": config.scenario,
                        "alignment_mode": config.alignment_mode,
                        "density": problem.x.nnz / np.prod(problem.x.shape),
                        "n_workers": config.n_workers,
                        "n_shards": config.n_shards,
                        "n_samples": problem.n_samples,
                        "n_features": problem.n_features,
                        "decode_latency": result["decode_latency"],
                        "barrier_latency": result["barrier_latency"],
                        "decode_wall_clock": decode_wall_clock,
                        "barrier_wall_clock": barrier_wall_clock,
                        "loss": problem.loss(weights),
                        "decode_success": result["decode_success"],
                        "decode_residual": result["decode_residual"],
                        "decode_cpu_seconds": result["decode_cpu_seconds"],
                        "scheduler_seconds": scheduler_seconds,
                        "worker_compute_cpu_seconds": result["worker_compute_cpu_seconds"],
                        "selected_rows": result["selected_rows"],
                        "completed_rows": result["completed_rows"],
                        "cancelled_rows": result["cancelled_rows"],
                        "rows_after_decode": result["rows_after_decode"],
                        "extra_compute": result["extra_compute"],
                        "second_layer_used": result["second_layer_used"],
                        "slow_workers": int(worker_state.slow_mask.sum()),
                        "mean_worker_speed": float(worker_state.speeds.mean()),
                    }
                )

    metrics = pd.DataFrame.from_records(records)
    summary = summarize_runtime_metrics(metrics)
    metrics.to_csv(config.output_dir / "runtime_metrics.csv", index=False)
    summary.to_csv(config.output_dir / "runtime_summary.csv", index=False)
    return metrics, summary


def _make_worker_states(config: MultiprocessExperimentConfig) -> list[WorkerState]:
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
    return [worker_pool.sample(iteration) for iteration in range(config.rounds)]


def _apply_worker_alignment(
    problem: SparseRidgeProblem,
    config: MultiprocessExperimentConfig,
    worker_states: list[WorkerState],
) -> list[WorkerState]:
    mode = config.alignment_mode.lower()
    if mode in {"none", "random"}:
        return worker_states
    if mode not in {"aligned", "anti", "anti_aligned", "misaligned"}:
        raise ValueError(f"Unknown alignment mode: {config.alignment_mode}")

    priorities = _alignment_priorities(problem, config)
    priority_order = np.argsort(-priorities)
    aligned_states: list[WorkerState] = []
    for state in worker_states:
        capacity = state.speeds / (1.0 + state.delays)
        worker_order = np.argsort(-capacity)
        if mode in {"anti", "anti_aligned", "misaligned"}:
            worker_order = worker_order[::-1]

        speeds = np.empty_like(state.speeds)
        delays = np.empty_like(state.delays)
        slow_mask = np.empty_like(state.slow_mask)
        for target_worker, source_worker in zip(priority_order, worker_order):
            speeds[target_worker] = state.speeds[source_worker]
            delays[target_worker] = state.delays[source_worker]
            slow_mask[target_worker] = state.slow_mask[source_worker]
        aligned_states.append(
            WorkerState(
                speeds=speeds,
                delays=delays,
                slow_mask=slow_mask,
                scenario=f"{state.scenario}:{mode}",
            )
        )
    return aligned_states


def _alignment_priorities(problem: SparseRidgeProblem, config: MultiprocessExperimentConfig) -> np.ndarray:
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
    density = problem.x.nnz / np.prod(problem.x.shape)
    pair_costs = np.zeros(config.n_workers, dtype=float)
    for task_id in range(config.n_workers):
        first_cost, _ = _strategy_row_cost(first[task_id], shard_costs, density)
        second_cost, _ = _strategy_row_cost(second[task_id], shard_costs, density)
        pair_costs[task_id] = first_cost + second_cost
    return _decode_pair_priorities(first, second, pair_costs)


def _predict_first_decode_latency(
    rows: np.ndarray,
    assignments: np.ndarray,
    worker_state: WorkerState,
    shard_costs: np.ndarray,
    config: MultiprocessExperimentConfig,
) -> tuple[float, int]:
    worker_available = np.zeros(worker_state.speeds.size, dtype=float)
    event_times = np.zeros(rows.shape[0], dtype=float)
    for row_id, row in enumerate(rows):
        worker_id = int(assignments[row_id])
        support = np.flatnonzero(np.abs(row) > 0.0)
        row_cost = float(shard_costs[support].sum()) if support.size else 0.0
        duration = (
            config.sleep_scale * float(worker_state.delays[worker_id])
            + config.cost_scale
            * row_cost
            / max(float(worker_state.speeds[worker_id]), 1e-12)
        )
        worker_available[worker_id] += max(0.0, duration)
        event_times[row_id] = worker_available[worker_id]

    selected: list[int] = []
    target_residual = np.ones(rows.shape[1], dtype=float)
    basis: list[np.ndarray] = []
    residual_tol = 1e-7
    target_scale = np.sqrt(max(rows.shape[1], 1))
    for event_id in np.argsort(event_times):
        selected.append(int(event_id))
        vector = np.asarray(rows[event_id], dtype=float).copy()
        for q in basis:
            vector -= float(vector @ q) * q
        norm = float(np.linalg.norm(vector))
        if norm > residual_tol:
            q = vector / norm
            basis.append(q)
            target_residual -= float(target_residual @ q) * q
            if float(np.linalg.norm(target_residual)) / target_scale <= residual_tol:
                return float(event_times[event_id]), len(selected)
    return float(event_times.max(initial=0.0)), len(selected)


def _make_strategy_specs(problem: SparseRidgeProblem, config: MultiprocessExperimentConfig):
    n_workers = config.n_workers
    n_shards = config.n_shards
    shard_costs = problem.shard_costs()
    density = problem.x.nnz / np.prod(problem.x.shape)
    flexible_rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    flexible_profiles: dict[str, tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def get_flexible_rows(config_name: str):
        config_by_name = {config.label: config for config in DEFAULT_FLEXIBLE_CONFIGS}
        config = config_by_name[config_name]
        if config.label not in flexible_rows:
            flexible_rows[config.label] = _make_flexible_code(
                "random",
                config,
                n_workers,
                n_shards,
                np.random.default_rng(_stable_seed(config.label, n_workers, n_shards)),
            )
        return config, flexible_rows[config.label]

    def get_flexible_profile(config_name: str):
        if config_name not in flexible_profiles:
            config, (first, second) = get_flexible_rows(config_name)
            pair_costs = np.zeros(n_workers, dtype=float)
            for task_id in range(n_workers):
                first_cost, _ = _strategy_row_cost(first[task_id], shard_costs, density)
                second_cost, _ = _strategy_row_cost(second[task_id], shard_costs, density)
                pair_costs[task_id] = first_cost + second_cost
            priorities = _decode_pair_priorities(first, second, pair_costs)
            flexible_profiles[config_name] = (config, first, second, pair_costs, priorities)
        return flexible_profiles[config_name]

    def assign_profile(
        mode: str,
        pair_costs: np.ndarray,
        priorities: np.ndarray,
        worker_state: WorkerState,
    ) -> np.ndarray:
        if mode == "identity":
            return np.arange(n_workers, dtype=int)
        if mode == "deadline":
            return _deadline_assignment(pair_costs, priorities, worker_state)
        if mode == "cost":
            task_priorities = pair_costs
        elif mode == "leverage":
            task_priorities = priorities
        else:
            raise ValueError(f"Unknown assignment mode: {mode}")
        task_order = np.argsort(-task_priorities)
        worker_order = np.argsort(-worker_state.speeds)
        assignments = np.empty(n_workers, dtype=int)
        assignments[task_order] = worker_order
        return assignments

    def uncoded(worker_state: WorkerState):
        rows = np.eye(n_shards, dtype=float)
        assignments = np.arange(n_shards, dtype=int) % n_workers
        flags = np.zeros(n_shards, dtype=bool)
        return rows, assignments, flags, "identity"

    def speed_aware_uncoded(worker_state: WorkerState):
        rows = np.eye(n_shards, dtype=float)
        assignments = np.empty(n_shards, dtype=int)
        capacity = worker_state.speeds / (1.0 + worker_state.delays)
        shard_order = np.argsort(-shard_costs)
        worker_order = np.argsort(-capacity)
        for shard_id, worker_id in zip(shard_order, np.resize(worker_order, n_shards)):
            assignments[int(shard_id)] = int(worker_id)
        flags = np.zeros(n_shards, dtype=bool)
        return rows, assignments, flags, "speed-aware-identity"

    def replication(worker_state: WorkerState):
        n_rows = max(n_workers, n_shards)
        rows = np.zeros((n_rows, n_shards), dtype=float)
        for row_id in range(n_rows):
            rows[row_id, row_id % n_shards] = 1.0
        assignments = np.arange(n_rows, dtype=int) % n_workers
        flags = np.zeros(n_rows, dtype=bool)
        return rows, assignments, flags, "replicas"

    def speculative_replication(worker_state: WorkerState):
        capacity = worker_state.speeds / (1.0 + worker_state.delays)
        worker_order = np.argsort(-capacity)
        primary_workers = np.resize(worker_order, n_shards)

        rows: list[np.ndarray] = []
        assignments: list[int] = []
        nominal = np.zeros(n_shards, dtype=float)
        for shard_id in range(n_shards):
            worker_id = int(primary_workers[shard_id])
            row = np.zeros(n_shards, dtype=float)
            row[shard_id] = 1.0
            rows.append(row)
            assignments.append(worker_id)
            nominal[shard_id] = (
                config.sleep_scale * float(worker_state.delays[worker_id])
                + config.cost_scale
                * float(shard_costs[shard_id])
                / max(float(worker_state.speeds[worker_id]), 1e-12)
            )

        used_workers = set(int(worker_id) for worker_id in primary_workers[: min(n_shards, n_workers)])
        spare_workers = [int(worker_id) for worker_id in worker_order if int(worker_id) not in used_workers]
        slow_shards = np.argsort(-nominal)
        for duplicate_id, worker_id in enumerate(spare_workers):
            shard_id = int(slow_shards[duplicate_id % n_shards])
            row = np.zeros(n_shards, dtype=float)
            row[shard_id] = 1.0
            rows.append(row)
            assignments.append(worker_id)

        flags = np.zeros(len(rows), dtype=bool)
        return np.vstack(rows), np.asarray(assignments, dtype=int), flags, "speculative-replicas"

    def flexible(mode: str, config_name: str = FLEXIBLE_CONFIG.label):
        def build(worker_state: WorkerState):
            config, first, second, pair_costs, priorities = get_flexible_profile(config_name)
            rows = np.vstack([first, second])
            pair_assignments = assign_profile(mode, pair_costs, priorities, worker_state)
            assignments = np.concatenate([pair_assignments, pair_assignments])
            flags = np.concatenate([np.zeros(n_workers, dtype=bool), np.ones(n_workers, dtype=bool)])
            return rows, assignments, flags, config.label

        return build

    def hybrid_decode_replication(worker_state: WorkerState):
        candidates = [
            ("spec", speculative_replication(worker_state)),
            ("decode", flexible("leverage")(worker_state)),
            ("deadline", flexible("deadline")(worker_state)),
        ]
        best_name = ""
        best_payload = None
        best_score = (float("inf"), 10**9)
        for name, payload in candidates:
            rows, assignments, flags, config_label = payload
            predicted_time, selected_rows = _predict_first_decode_latency(
                rows,
                assignments,
                worker_state,
                shard_costs,
                config,
            )
            score = (predicted_time, selected_rows)
            if score < best_score:
                best_name = name
                best_payload = (rows, assignments, flags, config_label)
                best_score = score
        if best_payload is None:
            raise RuntimeError("Hybrid scheduler found no candidate assignment.")
        rows, assignments, flags, config_label = best_payload
        return rows, assignments, flags, f"hybrid-{best_name}:{config_label}"

    rank_flexible = flexible("leverage")

    def fast_hybrid_decode_replication(worker_state: WorkerState):
        candidates = [
            ("spec", speculative_replication(worker_state)),
            ("decode", rank_flexible(worker_state)),
        ]
        best_name = ""
        best_payload = None
        best_score = (float("inf"), 10**9)
        for name, payload in candidates:
            rows, assignments, flags, config_label = payload
            predicted_time, selected_rows = _predict_first_decode_latency(
                rows,
                assignments,
                worker_state,
                shard_costs,
                config,
            )
            score = (predicted_time, selected_rows)
            if score < best_score:
                best_name = name
                best_payload = (rows, assignments, flags, config_label)
                best_score = score
        if best_payload is None:
            raise RuntimeError("Fast hybrid scheduler found no candidate assignment.")
        rows, assignments, flags, config_label = best_payload
        return rows, assignments, flags, f"fast-hybrid-{best_name}:{config_label}"

    def system_portfolio(worker_state: WorkerState):
        candidates = [
            ("uncoded", speed_aware_uncoded(worker_state)),
            ("spec", speculative_replication(worker_state)),
            ("decode", rank_flexible(worker_state)),
        ]
        scored = []
        for name, payload in candidates:
            rows, assignments, flags, config_label = payload
            predicted_time, selected_rows = _predict_first_decode_latency(
                rows,
                assignments,
                worker_state,
                shard_costs,
                config,
            )
            scored.append((predicted_time, selected_rows, name, payload))

        best_time = min(item[0] for item in scored)
        robust_window = max(0.01, 0.15 * best_time)
        near_best = [item for item in scored if item[0] <= best_time + robust_window]
        _, _, best_name, best_payload = min(near_best, key=lambda item: (item[1], item[0]))
        rows, assignments, flags, config_label = best_payload
        return rows, assignments, flags, f"portfolio-{best_name}:{config_label}"

    def _predict_all_done_latency(
        rows: np.ndarray,
        assignments: np.ndarray,
        worker_state: WorkerState,
    ) -> tuple[float, np.ndarray]:
        worker_available = np.zeros(n_workers, dtype=float)
        worker_payload = np.zeros(n_workers, dtype=float)
        for row_id, row in enumerate(rows):
            worker_id = int(assignments[row_id])
            support = np.flatnonzero(np.abs(row) > 0.0)
            row_cost, _ = _strategy_row_cost(row, shard_costs, density)
            duration = (
                config.sleep_scale * float(worker_state.delays[worker_id])
                + config.cost_scale
                * row_cost
                / max(float(worker_state.speeds[worker_id]), 1e-12)
            )
            worker_available[worker_id] += max(duration, 0.0)
            worker_payload[worker_id] += float(len(support))
        return float(worker_available.max(initial=0.0)), worker_payload

    def _payload_compute_rows(payload) -> int:
        rows, _, _, _ = payload
        return int(rows.shape[0])

    class RLTuneStyleSelector:
        """Runtime portfolio selector inspired by cluster-level learning schedulers.

        The selector intentionally uses only pre-round features and previous-round
        counters.  It is not a port of RLTune; it adapts the idea of a learned
        portfolio over heterogeneous scheduling arms to this coded worker-service
        runtime.
        """

        def __init__(self) -> None:
            self.candidates = [
                ("speed", speed_aware_uncoded),
                ("spec", speculative_replication),
                ("original", flexible("identity")),
                ("rank", rank_flexible),
            ]
            self.ema_dispatch_seconds = 0.0
            self.ema_cancel_seconds = 0.0
            self.ema_scheduler_seconds = 0.0
            self.arm_barrier_ema: dict[str, float] = {}
            self.arm_overrun_ema: dict[str, float] = {}
            self.last_arm = "original"

        def __call__(self, worker_state: WorkerState):
            static_payload = flexible("identity")(worker_state)
            static_time, static_prefix = self._first_decode(static_payload, worker_state)
            speed_mean = max(float(worker_state.speeds.mean()), 1e-12)
            speed_cv = float(worker_state.speeds.std() / speed_mean)
            slow_fraction = float(worker_state.slow_mask.mean())

            scored = []
            for name, builder in self.candidates:
                payload = builder(worker_state)
                first_time, prefix_rows = self._first_decode(payload, worker_state)
                all_done_time, worker_payload = _predict_all_done_latency(
                    payload[0], payload[1], worker_state
                )
                load_mean = max(float(worker_payload.mean()), 1e-12)
                load_cv = float(worker_payload.std() / load_mean)
                extra_rows = max(0, _payload_compute_rows(payload) - n_shards)
                prefix_delta = float(prefix_rows - static_prefix)
                predicted_gain = (
                    (static_time - first_time) / max(static_time, 1e-12)
                    if np.isfinite(static_time)
                    else 0.0
                )
                learned_barrier = self.arm_barrier_ema.get(name)
                tail_budget = 0.15 * max(0.0, all_done_time - first_time)
                predicted_barrier = (
                    first_time
                    + tail_budget
                    + self.ema_dispatch_seconds
                    + self.ema_cancel_seconds
                    + self.ema_scheduler_seconds
                )
                if learned_barrier is not None:
                    predicted_barrier = 0.70 * predicted_barrier + 0.30 * learned_barrier
                overrun_penalty = self.arm_overrun_ema.get(name, 0.0)
                exploration_bias = -0.004 * predicted_gain if speed_cv >= 0.20 else 0.0
                score = (
                    predicted_barrier
                    + 0.0015 * max(prefix_delta, 0.0)
                    + 0.0005 * extra_rows
                    + 0.010 * load_cv
                    + 0.005 * slow_fraction
                    + overrun_penalty
                    + exploration_bias
                )
                scored.append(
                    (
                        score,
                        predicted_barrier,
                        prefix_rows,
                        name,
                        payload,
                    )
                )

            _, predicted_barrier, prefix_rows, best_name, best_payload = min(
                scored, key=lambda item: (item[0], item[2])
            )
            self.last_arm = best_name
            rows, assignments, flags, config_label = best_payload
            return rows, assignments, flags, (
                f"rltune-style-{best_name}:{config_label}:"
                f"pred_barrier={predicted_barrier:.4f}:pred_prefix={prefix_rows}"
            )

        def update(
            self,
            *,
            worker_state: WorkerState,
            result: dict[str, object],
            scheduler_seconds: float,
            config_label: str,
        ) -> None:
            del worker_state, config_label
            alpha = 0.30
            dispatch_seconds = float(result.get("dispatch_seconds", 0.0))
            cancel_seconds = float(result.get("cancel_seconds", 0.0))
            self.ema_dispatch_seconds = (
                (1.0 - alpha) * self.ema_dispatch_seconds + alpha * dispatch_seconds
            )
            self.ema_cancel_seconds = (
                (1.0 - alpha) * self.ema_cancel_seconds + alpha * cancel_seconds
            )
            self.ema_scheduler_seconds = (
                (1.0 - alpha) * self.ema_scheduler_seconds + alpha * float(scheduler_seconds)
            )
            barrier = float(result["barrier_latency"])
            prior = self.arm_barrier_ema.get(self.last_arm, barrier)
            self.arm_barrier_ema[self.last_arm] = (1.0 - alpha) * prior + alpha * barrier
            rows_after_decode = float(result.get("rows_after_decode", 0.0))
            prior_overrun = self.arm_overrun_ema.get(self.last_arm, 0.0)
            self.arm_overrun_ema[self.last_arm] = (
                (1.0 - alpha) * prior_overrun + alpha * 0.001 * rows_after_decode
            )

        def _first_decode(self, payload, worker_state: WorkerState) -> tuple[float, int]:
            rows, assignments, _, _ = payload
            return _predict_first_decode_latency(rows, assignments, worker_state, shard_costs, config)

    class SailorStyleHeterogeneityAware:
        """Heterogeneity-aware scheduler that deliberately avoids coded features.

        This baseline adapts Sailor's dynamic heterogeneity motivation to the
        worker-service runtime by modeling speed, delay, availability, payload
        size, and historical throughput.  It does not use row importance,
        row-span decodability, first-decode prefix prediction, or rows-after-
        decode counters.
        """

        def __init__(self) -> None:
            self.candidates = [
                ("speed", speed_aware_uncoded),
                ("spec", speculative_replication),
                ("original", flexible("identity")),
                ("cost", flexible("cost")),
            ]
            self.throughput_ema: dict[str, float] = {}
            self.last_arm = "original"

        def __call__(self, worker_state: WorkerState):
            capacity = worker_state.speeds / np.maximum(1.0 + worker_state.delays, 1e-12)
            capacity_mean = max(float(capacity.mean()), 1e-12)
            capacity_cv = float(capacity.std() / capacity_mean)
            scored = []
            for name, builder in self.candidates:
                payload = builder(worker_state)
                rows, assignments, _, _ = payload
                all_done_time, worker_payload = _predict_all_done_latency(
                    rows, assignments, worker_state
                )
                load_mean = max(float(worker_payload.mean()), 1e-12)
                load_cv = float(worker_payload.std() / load_mean)
                payload_rows = int(rows.shape[0])
                hist_throughput = self.throughput_ema.get(name, 0.0)
                throughput_bonus = 0.0
                if hist_throughput > 0.0:
                    throughput_bonus = min(0.20 * all_done_time, 0.10 * payload_rows / hist_throughput)
                score = (
                    all_done_time * (1.0 + 0.10 * load_cv + 0.05 * capacity_cv)
                    + 0.0005 * max(0, payload_rows - n_shards)
                    - throughput_bonus
                )
                scored.append((score, all_done_time, name, payload))
            _, predicted_barrier, best_name, best_payload = min(
                scored, key=lambda item: (item[0], item[1])
            )
            self.last_arm = best_name
            rows, assignments, flags, config_label = best_payload
            return rows, assignments, flags, (
                f"sailor-style-{best_name}:{config_label}:"
                f"pred_all_done={predicted_barrier:.4f}"
            )

        def update(
            self,
            *,
            worker_state: WorkerState,
            result: dict[str, object],
            scheduler_seconds: float,
            config_label: str,
        ) -> None:
            del worker_state, scheduler_seconds, config_label
            barrier = max(float(result["barrier_latency"]), 1e-12)
            throughput = float(result.get("completed_rows", 0.0)) / barrier
            prior = self.throughput_ema.get(self.last_arm, throughput)
            self.throughput_ema[self.last_arm] = 0.70 * prior + 0.30 * throughput

    rltune_style_selector = RLTuneStyleSelector()
    sailor_style_scheduler = SailorStyleHeterogeneityAware()

    class GuardedPortfolioSpec:
        """Performance-mode guard across uncoded, speculative, static, and coded arms."""

        def __init__(self) -> None:
            self.static_builder = flexible("identity")
            self.candidates = [
                ("uncoded", speed_aware_uncoded),
                ("spec", speculative_replication),
                ("rank", rank_flexible),
            ]
            self.ema_prefix_delta = 0.0
            self.ema_rows_after_decode = 0.0
            self.last_static_pred_rows = 0
            self.last_action = "fallback"
            self.min_speed_cv = 0.20
            self.min_predicted_gain = 0.01
            self.prefix_tolerance_rows = 0.0
            self.rows_after_decode_tolerance = 1.0
            self.fallback_mode = getattr(config, "portfolio_fallback", "static")
            if self.fallback_mode not in {"static", "speed", "best_safe"}:
                raise ValueError(
                    "portfolio_fallback must be one of: static, speed, best_safe"
                )

        def __call__(self, worker_state: WorkerState):
            static_payload = self.static_builder(worker_state)
            static_time, static_rows = self._predict(static_payload, worker_state)
            scored = [(static_time, static_rows, "static", static_payload)]
            for name, builder in self.candidates:
                payload = builder(worker_state)
                pred_time, pred_rows = self._predict(payload, worker_state)
                scored.append((pred_time, pred_rows, name, payload))

            best_time = min(item[0] for item in scored)
            robust_window = max(0.01, 0.15 * best_time)
            near_best = [item for item in scored if item[0] <= best_time + robust_window]
            candidate_time, candidate_rows, candidate_name, candidate_payload = min(
                near_best, key=lambda item: (item[1], item[0])
            )
            fallback_time, fallback_rows, fallback_name, fallback_payload = self._fallback_payload(
                scored
            )
            predicted_gain = (
                (fallback_time - candidate_time) / max(fallback_time, 1e-12)
                if np.isfinite(fallback_time)
                else 0.0
            )
            predicted_prefix_delta = float(candidate_rows - fallback_rows)
            speed_mean = max(float(worker_state.speeds.mean()), 1e-12)
            speed_cv = float(worker_state.speeds.std() / speed_mean)
            history_ok = (
                self.ema_prefix_delta <= self.prefix_tolerance_rows
                and self.ema_rows_after_decode <= self.rows_after_decode_tolerance
            )
            enable = (
                candidate_name != "static"
                and speed_cv >= self.min_speed_cv
                and predicted_gain >= self.min_predicted_gain
                and predicted_prefix_delta <= self.prefix_tolerance_rows
                and history_ok
            )
            self.last_static_pred_rows = int(fallback_rows)
            self.last_action = "enable" if enable else "fallback"
            rows, assignments, flags, config_label = (
                candidate_payload if enable else fallback_payload
            )
            active_name = candidate_name if enable else f"{self.fallback_mode}-{fallback_name}"
            return rows, assignments, flags, (
                f"guarded-portfolio-{self.last_action}-{active_name}:{config_label}:"
                f"pred_gain={predicted_gain:.3f}:"
                f"pred_prefix_delta={predicted_prefix_delta:.1f}"
            )

        def update(
            self,
            *,
            worker_state: WorkerState,
            result: dict[str, object],
            scheduler_seconds: float,
            config_label: str,
        ) -> None:
            del worker_state, scheduler_seconds, config_label
            if self.last_action != "enable":
                self.ema_rows_after_decode *= 0.80
                self.ema_prefix_delta *= 0.80
                return
            observed_prefix_delta = float(result["selected_rows"]) - float(
                self.last_static_pred_rows
            )
            rows_after_decode = float(result.get("rows_after_decode", 0.0))
            self.ema_prefix_delta = 0.70 * self.ema_prefix_delta + 0.30 * observed_prefix_delta
            self.ema_rows_after_decode = (
                0.70 * self.ema_rows_after_decode + 0.30 * rows_after_decode
            )

        def _predict(self, payload, worker_state: WorkerState) -> tuple[float, int]:
            rows, assignments, _, _ = payload
            return _predict_first_decode_latency(rows, assignments, worker_state, shard_costs, config)

        def _fallback_payload(self, scored):
            static_entry = next(item for item in scored if item[2] == "static")
            if self.fallback_mode == "static":
                return static_entry
            speed_entry = next(item for item in scored if item[2] == "uncoded")
            if self.fallback_mode == "speed":
                return speed_entry
            return min((static_entry, speed_entry), key=lambda item: (item[0], item[1]))

    guarded_portfolio = GuardedPortfolioSpec()

    class LearnedPortfolioSpec:
        def __init__(self) -> None:
            self.candidates = [
                ("uncoded", speed_aware_uncoded),
                ("spec", speculative_replication),
                ("decode", rank_flexible),
            ]
            self.counts = np.zeros((2, len(self.candidates)), dtype=float)
            self.reward_sums = np.zeros_like(self.counts)
            self.total_counts = np.zeros(2, dtype=float)
            self.last_context = 0
            self.last_arm = 0
            self.confidence = 0.025

        def __call__(self, worker_state: WorkerState):
            context = self._context(worker_state)
            arm = self._choose_arm(context)
            self.last_context = context
            self.last_arm = arm
            name, builder = self.candidates[arm]
            rows, assignments, flags, config_label = builder(worker_state)
            return rows, assignments, flags, f"learned-{name}:{config_label}"

        def update(
            self,
            *,
            worker_state: WorkerState,
            result: dict[str, object],
            scheduler_seconds: float,
            config_label: str,
        ) -> None:
            del worker_state, config_label
            latency = float(scheduler_seconds) + float(result["barrier_latency"])
            reward = -latency
            context = self.last_context
            arm = self.last_arm
            self.counts[context, arm] += 1.0
            self.reward_sums[context, arm] += reward
            self.total_counts[context] += 1.0

        def _choose_arm(self, context: int) -> int:
            counts = self.counts[context]
            untried = np.flatnonzero(counts == 0)
            if untried.size:
                return int(untried[0])
            means = self.reward_sums[context] / np.maximum(counts, 1.0)
            bonus = self.confidence * np.sqrt(
                np.log(self.total_counts[context] + 1.0) / np.maximum(counts, 1.0)
            )
            return int(np.argmax(means + bonus))

        def _context(self, worker_state: WorkerState) -> int:
            speed_mean = max(float(worker_state.speeds.mean()), 1e-12)
            speed_cv = float(worker_state.speeds.std() / speed_mean)
            slow_fraction = float(worker_state.slow_mask.mean())
            return int(slow_fraction >= 0.55 or speed_cv >= 0.75)

    learned_portfolio = LearnedPortfolioSpec()

    class OnlineCounterGuardSpec:
        """Online counter guard that enables a coded scheduler only in favorable rounds."""

        def __init__(self, name: str, candidate_builder) -> None:
            self.name = name
            self.candidate_builder = candidate_builder
            self.static_builder = flexible("identity")
            self.ema_prefix_delta = 0.0
            self.ema_rows_after_decode = 0.0
            self.last_static_pred_rows = 0
            self.last_action = "fallback"
            self.min_speed_cv = 0.20
            self.min_predicted_gain = 0.01
            self.prefix_tolerance_rows = 0.0
            self.rows_after_decode_tolerance = 1.0

        def __call__(self, worker_state: WorkerState):
            static_payload = self.static_builder(worker_state)
            candidate_payload = self.candidate_builder(worker_state)
            static_time, static_rows = self._predict(static_payload, worker_state)
            candidate_time, candidate_rows = self._predict(candidate_payload, worker_state)
            speed_mean = max(float(worker_state.speeds.mean()), 1e-12)
            speed_cv = float(worker_state.speeds.std() / speed_mean)
            predicted_gain = (
                (static_time - candidate_time) / max(static_time, 1e-12)
                if np.isfinite(static_time)
                else 0.0
            )
            predicted_prefix_delta = float(candidate_rows - static_rows)
            history_ok = (
                self.ema_prefix_delta <= self.prefix_tolerance_rows
                and self.ema_rows_after_decode <= self.rows_after_decode_tolerance
            )
            enable = (
                speed_cv >= self.min_speed_cv
                and predicted_gain >= self.min_predicted_gain
                and predicted_prefix_delta <= self.prefix_tolerance_rows
                and history_ok
            )
            self.last_static_pred_rows = int(static_rows)
            self.last_action = "enable" if enable else "fallback"
            rows, assignments, flags, config_label = candidate_payload if enable else static_payload
            return rows, assignments, flags, (
                f"online-guard-{self.last_action}-{self.name}:{config_label}:"
                f"pred_gain={predicted_gain:.3f}:pred_prefix_delta={predicted_prefix_delta:.1f}"
            )

        def update(
            self,
            *,
            worker_state: WorkerState,
            result: dict[str, object],
            scheduler_seconds: float,
            config_label: str,
        ) -> None:
            del worker_state, scheduler_seconds, config_label
            if self.last_action != "enable":
                self.ema_rows_after_decode *= 0.80
                self.ema_prefix_delta *= 0.80
                return
            observed_prefix_delta = float(result["selected_rows"]) - float(self.last_static_pred_rows)
            rows_after_decode = float(result.get("rows_after_decode", 0.0))
            self.ema_prefix_delta = 0.70 * self.ema_prefix_delta + 0.30 * observed_prefix_delta
            self.ema_rows_after_decode = (
                0.70 * self.ema_rows_after_decode + 0.30 * rows_after_decode
            )

        def _predict(self, payload, worker_state: WorkerState) -> tuple[float, int]:
            rows, assignments, _, _ = payload
            return _predict_first_decode_latency(rows, assignments, worker_state, shard_costs, config)

    online_rank_guard = OnlineCounterGuardSpec("rank", rank_flexible)
    online_deadline_guard = OnlineCounterGuardSpec("deadline", flexible("deadline"))

    return {
        "uncoded_sync": uncoded,
        "speed_aware_uncoded": speed_aware_uncoded,
        "replication": replication,
        "speculative_replication": speculative_replication,
        "hybrid_decode_replication": hybrid_decode_replication,
        "fast_hybrid_decode_replication": fast_hybrid_decode_replication,
        "system_portfolio": system_portfolio,
        "guarded_system_portfolio": guarded_portfolio,
        "learned_system_portfolio": learned_portfolio,
        "original_sfcl_static": flexible("identity"),
        "rltune_style_selector": rltune_style_selector,
        "sailor_style_heterogeneity_aware": sailor_style_scheduler,
        "straggler_whatif_diagnostic": flexible("identity"),
        "online_counter_guard_rank_aware_sparse_flexible": online_rank_guard,
        "online_counter_guard_deadline_aware_sparse_flexible": online_deadline_guard,
        "thin_sparse_flexible_static": flexible("identity", "thin_d1_d2"),
        "thin_rank_aware_sparse_flexible": flexible("leverage", "thin_d1_d2"),
        "thin_deadline_aware_sparse_flexible": flexible("deadline", "thin_d1_d2"),
        "light_sparse_flexible_static": flexible("identity", "light_d2_d2"),
        "light_rank_aware_sparse_flexible": flexible("leverage", "light_d2_d2"),
        "light_deadline_aware_sparse_flexible": flexible("deadline", "light_d2_d2"),
        "sparse_flexible_static": flexible("identity"),
        "worker_aware_sparse_flexible": flexible("cost"),
        "rank_aware_sparse_flexible": flexible("leverage"),
        "deadline_aware_sparse_flexible": flexible("deadline"),
        "robust_sparse_flexible_static": flexible("identity", "robust_d3_d4"),
        "robust_rank_aware_sparse_flexible": flexible("leverage", "robust_d3_d4"),
        "robust_deadline_aware_sparse_flexible": flexible("deadline", "robust_d3_d4"),
    }


def summarize_runtime_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby("strategy", sort=False)
    summary = grouped.agg(
        mean_decode_latency=("decode_latency", "mean"),
        p95_decode_latency=("decode_latency", lambda x: x.quantile(0.95)),
        mean_barrier_latency=("barrier_latency", "mean"),
        p95_barrier_latency=("barrier_latency", lambda x: x.quantile(0.95)),
        final_loss=("loss", "last"),
        total_decode_wall_clock=("decode_wall_clock", "last"),
        total_barrier_wall_clock=("barrier_wall_clock", "last"),
        decode_success_rate=("decode_success", "mean"),
        mean_scheduler_seconds=("scheduler_seconds", "mean"),
        mean_decode_cpu_seconds=("decode_cpu_seconds", "mean"),
        mean_worker_compute_cpu_seconds=("worker_compute_cpu_seconds", "mean"),
        mean_extra_compute=("extra_compute", "mean"),
        mean_selected_rows=("selected_rows", "mean"),
        mean_completed_rows=("completed_rows", "mean"),
        mean_cancelled_rows=("cancelled_rows", "mean"),
        second_layer_rate=("second_layer_used", "mean"),
    ).reset_index()

    baseline_decode = _lookup(summary, "sparse_flexible_static", "mean_decode_latency")
    baseline_p95 = _lookup(summary, "sparse_flexible_static", "p95_decode_latency")
    baseline_barrier = _lookup(summary, "sparse_flexible_static", "mean_barrier_latency")
    if np.isnan(baseline_decode):
        baseline_decode = _lookup(summary, "original_sfcl_static", "mean_decode_latency")
    if np.isnan(baseline_p95):
        baseline_p95 = _lookup(summary, "original_sfcl_static", "p95_decode_latency")
    if np.isnan(baseline_barrier):
        baseline_barrier = _lookup(summary, "original_sfcl_static", "mean_barrier_latency")
    summary["decode_latency_improvement_vs_sparse_flexible"] = (
        baseline_decode - summary["mean_decode_latency"]
    ) / baseline_decode
    summary["p95_decode_latency_improvement_vs_sparse_flexible"] = (
        baseline_p95 - summary["p95_decode_latency"]
    ) / baseline_p95
    summary["barrier_latency_improvement_vs_sparse_flexible"] = (
        baseline_barrier - summary["mean_barrier_latency"]
    ) / baseline_barrier
    return summary


def _lookup(summary: pd.DataFrame, strategy: str, column: str) -> float:
    values = summary.loc[summary["strategy"] == strategy, column]
    if values.empty:
        return float("nan")
    return float(values.iloc[0])
