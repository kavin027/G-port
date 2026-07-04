from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .coding import (
    DecodeResult,
    aggregate_encoded_gradients,
    decode_coefficients,
    make_decode_balanced_flexible_rows,
    make_decodable_sparse_rows,
    make_flexible_rows,
)
from .data import SparseRidgeProblem
from .workers import WorkerState


@dataclass(frozen=True)
class EncodingConfig:
    label: str
    degree_first: int
    degree_second: int


DEFAULT_FLEXIBLE_CONFIGS = [
    EncodingConfig(label="thin_d1_d2", degree_first=1, degree_second=2),
    EncodingConfig(label="light_d2_d2", degree_first=2, degree_second=2),
    EncodingConfig(label="balanced_d2_d3", degree_first=2, degree_second=3),
    EncodingConfig(label="robust_d3_d4", degree_first=3, degree_second=4),
    EncodingConfig(label="dense_d4_d5", degree_first=4, degree_second=5),
]


@dataclass
class RoundResult:
    strategy: str
    gradient: np.ndarray
    iteration_time: float
    decode_success: bool
    decode_residual: float
    decode_cpu_seconds: float
    selected_rows: int
    extra_compute: float
    nnz_expansion: float
    second_layer_used: bool
    config_label: str


class Strategy:
    name: str

    def run_round(
        self,
        problem: SparseRidgeProblem,
        weights: np.ndarray,
        worker_state: WorkerState,
        rng: np.random.Generator,
        iteration: int,
    ) -> RoundResult:
        raise NotImplementedError


def _row_cost(
    row: np.ndarray,
    shard_costs: np.ndarray,
    data_density: float,
    combine_penalty: float = 0.10,
) -> tuple[float, float]:
    support = np.flatnonzero(np.abs(row) > 0.0)
    if support.size == 0:
        return 0.0, 1.0
    base_cost = float(shard_costs[support].sum())
    density_scale = min(2.5, max(0.2, data_density / 0.01))
    expansion = 1.0 + combine_penalty * max(0, support.size - 1) * density_scale
    return base_cost * expansion, expansion


def _event_duration(
    row: np.ndarray,
    worker_id: int,
    shard_costs: np.ndarray,
    data_density: float,
    worker_state: WorkerState,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    cost, expansion = _row_cost(row, shard_costs, data_density)
    jitter = rng.lognormal(mean=0.0, sigma=0.08)
    duration = worker_state.delays[worker_id] + 0.035 * cost * jitter / worker_state.speeds[worker_id]
    return duration, cost, expansion


def _stable_seed(label: str, n_workers: int, n_shards: int) -> int:
    payload = f"{label}:{n_workers}:{n_shards}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32)


def _reward(result: RoundResult, extra_compute_weight: float) -> float:
    reward = -result.iteration_time - extra_compute_weight * result.extra_compute
    if not result.decode_success:
        reward -= 1.0
    return reward


def _context_id(worker_state: WorkerState) -> int:
    slow_fraction = float(worker_state.slow_mask.mean())
    speed_cv = float(worker_state.speeds.std() / max(worker_state.speeds.mean(), 1e-12))
    if slow_fraction < 0.08 and speed_cv < 0.45:
        return 0
    if slow_fraction < 0.20:
        return 1
    if speed_cv < 0.75:
        return 2
    return 3


def _decode_from_events(
    rows: np.ndarray,
    encoded_gradients: np.ndarray,
    event_times: np.ndarray,
    event_costs: np.ndarray,
    event_expansions: np.ndarray,
    full_gradient: np.ndarray,
    l2_gradient: np.ndarray,
    second_layer_flags: np.ndarray,
    n_shards: int,
) -> tuple[np.ndarray, dict[str, float | bool | str | int]]:
    order = np.argsort(event_times)
    selected: list[int] = []
    final_decode = DecodeResult(False, np.empty(0), np.inf, 0.0)
    selected_time = float(event_times[order[-1]]) if order.size else 0.0

    for event_id in order:
        selected.append(int(event_id))
        candidate_rows = rows[selected]
        final_decode = decode_coefficients(candidate_rows)
        if final_decode.success:
            selected_time = float(event_times[event_id])
            break

    selected_ids = np.asarray(selected, dtype=int)
    if final_decode.success:
        gradient = aggregate_encoded_gradients(
            rows[selected_ids], encoded_gradients[selected_ids], final_decode
        ) + l2_gradient
    else:
        gradient = full_gradient

    selected_cost = float(event_costs[selected_ids].sum()) if selected_ids.size else 0.0
    denominator = max(float(n_shards), 1.0)
    decode_overhead = 2.0e-6 * selected_ids.size * n_shards * n_shards
    selected_time += decode_overhead
    mean_expansion = float(event_expansions[selected_ids].mean()) if selected_ids.size else 1.0

    metrics = {
        "iteration_time": selected_time,
        "decode_success": final_decode.success,
        "decode_residual": final_decode.residual,
        "decode_cpu_seconds": final_decode.cpu_seconds,
        "selected_rows": int(selected_ids.size),
        "extra_compute": selected_cost / denominator,
        "nnz_expansion": mean_expansion,
        "second_layer_used": bool(second_layer_flags[selected_ids].any()) if selected_ids.size else False,
    }
    return gradient, metrics


class UncodedSyncStrategy(Strategy):
    name = "uncoded_sync"

    def run_round(
        self,
        problem: SparseRidgeProblem,
        weights: np.ndarray,
        worker_state: WorkerState,
        rng: np.random.Generator,
        iteration: int,
    ) -> RoundResult:
        shard_gradients = problem.shard_gradients(weights)
        rows = np.eye(problem.n_shards, dtype=float)
        encoded = shard_gradients.copy()
        shard_costs = problem.shard_costs()
        event_times = np.zeros(problem.n_shards, dtype=float)
        event_costs = np.zeros(problem.n_shards, dtype=float)
        event_expansions = np.ones(problem.n_shards, dtype=float)

        for shard_id in range(problem.n_shards):
            worker_id = shard_id % worker_state.speeds.size
            event_times[shard_id], event_costs[shard_id], event_expansions[shard_id] = _event_duration(
                rows[shard_id], worker_id, shard_costs, problem.x.nnz / np.prod(problem.x.shape), worker_state, rng
            )

        gradient, metrics = _decode_from_events(
            rows,
            encoded,
            event_times,
            event_costs,
            event_expansions,
            problem.full_gradient(weights),
            problem.l2 * weights,
            np.zeros(problem.n_shards, dtype=bool),
            problem.n_shards,
        )
        return RoundResult(self.name, gradient, config_label="identity", **metrics)


class ReplicationStrategy(Strategy):
    name = "replication"

    def __init__(self) -> None:
        self.cached_rows: dict[tuple[int, int], np.ndarray] = {}

    def run_round(
        self,
        problem: SparseRidgeProblem,
        weights: np.ndarray,
        worker_state: WorkerState,
        rng: np.random.Generator,
        iteration: int,
    ) -> RoundResult:
        cache_key = (worker_state.speeds.size, problem.n_shards)
        if cache_key not in self.cached_rows:
            n_rows = max(worker_state.speeds.size, problem.n_shards)
            rows = np.zeros((n_rows, problem.n_shards), dtype=float)
            for row_id in range(n_rows):
                rows[row_id, row_id % problem.n_shards] = 1.0
            self.cached_rows[cache_key] = rows
        rows = self.cached_rows[cache_key]

        shard_gradients = problem.shard_gradients(weights)
        encoded = rows @ shard_gradients
        shard_costs = problem.shard_costs()
        event_times = np.zeros(rows.shape[0], dtype=float)
        event_costs = np.zeros(rows.shape[0], dtype=float)
        event_expansions = np.ones(rows.shape[0], dtype=float)
        density = problem.x.nnz / np.prod(problem.x.shape)
        for row_id in range(rows.shape[0]):
            worker_id = row_id % worker_state.speeds.size
            event_times[row_id], event_costs[row_id], event_expansions[row_id] = _event_duration(
                rows[row_id], worker_id, shard_costs, density, worker_state, rng
            )

        gradient, metrics = _decode_from_events(
            rows,
            encoded,
            event_times,
            event_costs,
            event_expansions,
            problem.full_gradient(weights),
            problem.l2 * weights,
            np.zeros(rows.shape[0], dtype=bool),
            problem.n_shards,
        )
        return RoundResult(self.name, gradient, config_label="replicas", **metrics)


class StaticSparseCodeStrategy(Strategy):
    name = "static_sparse_code"

    def __init__(self, degree: int = 3) -> None:
        self.degree = degree
        self.rows: np.ndarray | None = None

    def run_round(
        self,
        problem: SparseRidgeProblem,
        weights: np.ndarray,
        worker_state: WorkerState,
        rng: np.random.Generator,
        iteration: int,
    ) -> RoundResult:
        if self.rows is None:
            self.rows = make_decodable_sparse_rows(
                worker_state.speeds.size, problem.n_shards, self.degree, rng
            )
        rows = self.rows
        shard_gradients = problem.shard_gradients(weights)
        encoded = rows @ shard_gradients
        shard_costs = problem.shard_costs()
        density = problem.x.nnz / np.prod(problem.x.shape)
        event_times = np.zeros(rows.shape[0], dtype=float)
        event_costs = np.zeros(rows.shape[0], dtype=float)
        event_expansions = np.ones(rows.shape[0], dtype=float)
        for row_id in range(rows.shape[0]):
            event_times[row_id], event_costs[row_id], event_expansions[row_id] = _event_duration(
                rows[row_id], row_id, shard_costs, density, worker_state, rng
            )

        gradient, metrics = _decode_from_events(
            rows,
            encoded,
            event_times,
            event_costs,
            event_expansions,
            problem.full_gradient(weights),
            problem.l2 * weights,
            np.zeros(rows.shape[0], dtype=bool),
            problem.n_shards,
        )
        return RoundResult(
            self.name,
            gradient,
            config_label=f"degree={self.degree}",
            **metrics,
        )


class FlexibleCodedStrategy(Strategy):
    def __init__(
        self,
        name: str,
        config: EncodingConfig,
        speed_aware: bool = False,
        assignment_mode: str | None = None,
        code_design: str = "random",
    ) -> None:
        self.name = name
        self.config = config
        self.assignment_mode = assignment_mode or ("cost" if speed_aware else "identity")
        self.code_design = code_design
        self.cached_rows: tuple[np.ndarray, np.ndarray] | None = None

    def _get_rows(
        self, n_workers: int, n_shards: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.cached_rows is None:
            row_rng = np.random.default_rng(
                _stable_seed(f"{self.code_design}:{self.config.label}", n_workers, n_shards)
            )
            self.cached_rows = _make_flexible_code(
                self.code_design, self.config, n_workers, n_shards, row_rng
            )
        return self.cached_rows

    def run_round(
        self,
        problem: SparseRidgeProblem,
        weights: np.ndarray,
        worker_state: WorkerState,
        rng: np.random.Generator,
        iteration: int,
    ) -> RoundResult:
        first, second = self._get_rows(worker_state.speeds.size, problem.n_shards, rng)
        return _run_flexible_round(
            self.name,
            self.config,
            first,
            second,
            problem,
            weights,
            worker_state,
            rng,
            self.assignment_mode,
        )


class AdaptiveFlexibleStrategy(Strategy):
    name = "adaptive_sparse_flexible"

    def __init__(
        self,
        configs: list[EncodingConfig],
        name: str = "adaptive_sparse_flexible",
        epsilon: float = 0.04,
        step: float = 0.22,
        extra_compute_weight: float = 0.08,
        speed_aware: bool = False,
        assignment_mode: str | None = None,
        code_design: str = "random",
    ):
        self.name = name
        self.configs = configs
        self.epsilon = epsilon
        self.step = step
        self.extra_compute_weight = extra_compute_weight
        self.assignment_mode = assignment_mode or ("cost" if speed_aware else "identity")
        self.code_design = code_design
        self.values = np.zeros(len(configs), dtype=float)
        self.counts = np.zeros(len(configs), dtype=float)
        self.cached_rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def run_round(
        self,
        problem: SparseRidgeProblem,
        weights: np.ndarray,
        worker_state: WorkerState,
        rng: np.random.Generator,
        iteration: int,
    ) -> RoundResult:
        config_id = self._choose_config(rng)
        config = self.configs[config_id]
        if config.label not in self.cached_rows:
            row_rng = np.random.default_rng(
                _stable_seed(
                    f"{self.code_design}:{config.label}",
                    worker_state.speeds.size,
                    problem.n_shards,
                )
            )
            self.cached_rows[config.label] = _make_flexible_code(
                self.code_design,
                config,
                worker_state.speeds.size,
                problem.n_shards,
                row_rng,
            )
        first, second = self.cached_rows[config.label]
        result = _run_flexible_round(
            self.name,
            config,
            first,
            second,
            problem,
            weights,
            worker_state,
            rng,
            self.assignment_mode,
        )
        self._update(config_id, _reward(result, self.extra_compute_weight))
        return result

    def _choose_config(self, rng: np.random.Generator) -> int:
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(untried[0])
        if rng.random() < self.epsilon:
            return int(rng.integers(len(self.configs)))
        return int(np.argmax(self.values))

    def _update(self, config_id: int, reward: float) -> None:
        self.counts[config_id] += 1.0
        self.values[config_id] = (1.0 - self.step) * self.values[config_id] + self.step * reward


class WindowAdaptiveFlexibleStrategy(AdaptiveFlexibleStrategy):
    def __init__(
        self,
        configs: list[EncodingConfig],
        name: str = "window_sparse_flexible",
        epsilon: float = 0.04,
        window: int = 18,
        extra_compute_weight: float = 0.08,
        speed_aware: bool = False,
        assignment_mode: str | None = None,
        code_design: str = "random",
    ):
        super().__init__(
            configs=configs,
            name=name,
            epsilon=epsilon,
            step=1.0,
            extra_compute_weight=extra_compute_weight,
            speed_aware=speed_aware,
            assignment_mode=assignment_mode,
            code_design=code_design,
        )
        self.history = [deque(maxlen=window) for _ in configs]

    def _choose_config(self, rng: np.random.Generator) -> int:
        untried = [idx for idx, history in enumerate(self.history) if not history]
        if untried:
            return untried[0]
        if rng.random() < self.epsilon:
            return int(rng.integers(len(self.configs)))
        means = np.asarray([np.mean(history) for history in self.history], dtype=float)
        return int(np.argmax(means))

    def _update(self, config_id: int, reward: float) -> None:
        self.counts[config_id] += 1.0
        self.history[config_id].append(reward)
        self.values[config_id] = float(np.mean(self.history[config_id]))


class ContextualAdaptiveFlexibleStrategy(AdaptiveFlexibleStrategy):
    def __init__(
        self,
        configs: list[EncodingConfig],
        name: str = "contextual_sparse_flexible",
        epsilon: float = 0.04,
        step: float = 0.28,
        extra_compute_weight: float = 0.08,
        n_contexts: int = 4,
        speed_aware: bool = False,
        assignment_mode: str | None = None,
        code_design: str = "random",
    ):
        super().__init__(
            configs=configs,
            name=name,
            epsilon=epsilon,
            step=step,
            extra_compute_weight=extra_compute_weight,
            speed_aware=speed_aware,
            assignment_mode=assignment_mode,
            code_design=code_design,
        )
        self.n_contexts = n_contexts
        self.context_values = np.zeros((n_contexts, len(configs)), dtype=float)
        self.context_counts = np.zeros((n_contexts, len(configs)), dtype=float)
        self.current_context = 0

    def run_round(
        self,
        problem: SparseRidgeProblem,
        weights: np.ndarray,
        worker_state: WorkerState,
        rng: np.random.Generator,
        iteration: int,
    ) -> RoundResult:
        self.current_context = _context_id(worker_state)
        return super().run_round(problem, weights, worker_state, rng, iteration)

    def _choose_config(self, rng: np.random.Generator) -> int:
        counts = self.context_counts[self.current_context]
        untried = np.flatnonzero(counts == 0)
        if untried.size:
            return int(untried[0])
        if rng.random() < self.epsilon:
            return int(rng.integers(len(self.configs)))
        return int(np.argmax(self.context_values[self.current_context]))

    def _update(self, config_id: int, reward: float) -> None:
        context = self.current_context
        self.context_counts[context, config_id] += 1.0
        old_value = self.context_values[context, config_id]
        self.context_values[context, config_id] = (1.0 - self.step) * old_value + self.step * reward
        self.counts[config_id] = self.context_counts[:, config_id].sum()
        context_visits = self.context_counts[:, config_id]
        if context_visits.sum() > 0:
            self.values[config_id] = float(
                np.average(self.context_values[:, config_id], weights=context_visits)
            )


class UCBAdaptiveFlexibleStrategy(AdaptiveFlexibleStrategy):
    def __init__(
        self,
        configs: list[EncodingConfig],
        name: str = "ucb_sparse_flexible",
        confidence: float = 0.35,
        extra_compute_weight: float = 0.08,
        speed_aware: bool = False,
        assignment_mode: str | None = None,
        code_design: str = "random",
    ):
        super().__init__(
            configs=configs,
            name=name,
            epsilon=0.0,
            step=1.0,
            extra_compute_weight=extra_compute_weight,
            speed_aware=speed_aware,
            assignment_mode=assignment_mode,
            code_design=code_design,
        )
        self.confidence = confidence
        self.reward_sums = np.zeros(len(configs), dtype=float)
        self.total_pulls = 0.0

    def _choose_config(self, rng: np.random.Generator) -> int:
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(untried[0])
        means = self.reward_sums / np.maximum(self.counts, 1.0)
        bonus = self.confidence * np.sqrt(np.log(self.total_pulls + 1.0) / self.counts)
        return int(np.argmax(means + bonus))

    def _update(self, config_id: int, reward: float) -> None:
        self.total_pulls += 1.0
        self.counts[config_id] += 1.0
        self.reward_sums[config_id] += reward
        self.values[config_id] = self.reward_sums[config_id] / self.counts[config_id]


class ContextualUCBAdaptiveFlexibleStrategy(ContextualAdaptiveFlexibleStrategy):
    def __init__(
        self,
        configs: list[EncodingConfig],
        name: str = "contextual_ucb_sparse_flexible",
        confidence: float = 0.35,
        extra_compute_weight: float = 0.08,
        n_contexts: int = 4,
        speed_aware: bool = False,
        assignment_mode: str | None = None,
        code_design: str = "random",
    ):
        super().__init__(
            configs=configs,
            name=name,
            epsilon=0.0,
            step=1.0,
            extra_compute_weight=extra_compute_weight,
            n_contexts=n_contexts,
            speed_aware=speed_aware,
            assignment_mode=assignment_mode,
            code_design=code_design,
        )
        self.confidence = confidence
        self.context_reward_sums = np.zeros((n_contexts, len(configs)), dtype=float)
        self.context_total_pulls = np.zeros(n_contexts, dtype=float)

    def _choose_config(self, rng: np.random.Generator) -> int:
        context = self.current_context
        counts = self.context_counts[context]
        untried = np.flatnonzero(counts == 0)
        if untried.size:
            return int(untried[0])
        means = self.context_reward_sums[context] / np.maximum(counts, 1.0)
        bonus = self.confidence * np.sqrt(
            np.log(self.context_total_pulls[context] + 1.0) / counts
        )
        return int(np.argmax(means + bonus))

    def _update(self, config_id: int, reward: float) -> None:
        context = self.current_context
        self.context_total_pulls[context] += 1.0
        self.context_counts[context, config_id] += 1.0
        self.context_reward_sums[context, config_id] += reward
        self.context_values[context, config_id] = (
            self.context_reward_sums[context, config_id]
            / self.context_counts[context, config_id]
        )
        self.counts[config_id] = self.context_counts[:, config_id].sum()
        self.values[config_id] = (
            self.context_reward_sums[:, config_id].sum() / max(self.counts[config_id], 1.0)
        )


def _run_flexible_round(
    name: str,
    config: EncodingConfig,
    first: np.ndarray,
    second: np.ndarray,
    problem: SparseRidgeProblem,
    weights: np.ndarray,
    worker_state: WorkerState,
    rng: np.random.Generator,
    assignment_mode: str = "identity",
) -> RoundResult:
    shard_gradients = problem.shard_gradients(weights)
    rows = np.vstack([first, second])
    encoded = rows @ shard_gradients
    shard_costs = problem.shard_costs()
    density = problem.x.nnz / np.prod(problem.x.shape)
    n_workers = worker_state.speeds.size

    first_times = np.zeros(n_workers, dtype=float)
    first_costs = np.zeros(n_workers, dtype=float)
    first_expansions = np.ones(n_workers, dtype=float)
    second_times = np.zeros(n_workers, dtype=float)
    second_costs = np.zeros(n_workers, dtype=float)
    second_expansions = np.ones(n_workers, dtype=float)
    assignments = _assign_tasks_to_workers(
        first, second, shard_costs, density, worker_state, assignment_mode
    )

    for worker_id in range(n_workers):
        assigned_worker = int(assignments[worker_id])
        first_duration, first_costs[worker_id], first_expansions[worker_id] = _event_duration(
            first[worker_id], assigned_worker, shard_costs, density, worker_state, rng
        )
        second_duration, second_costs[worker_id], second_expansions[worker_id] = _event_duration(
            second[worker_id], assigned_worker, shard_costs, density, worker_state, rng
        )
        first_times[worker_id] = first_duration
        second_times[worker_id] = first_duration + second_duration

    event_times = np.concatenate([first_times, second_times])
    event_costs = np.concatenate([first_costs, second_costs])
    event_expansions = np.concatenate([first_expansions, second_expansions])
    second_layer_flags = np.concatenate(
        [np.zeros(n_workers, dtype=bool), np.ones(n_workers, dtype=bool)]
    )

    gradient, metrics = _decode_from_events(
        rows,
        encoded,
        event_times,
        event_costs,
        event_expansions,
        problem.full_gradient(weights),
        problem.l2 * weights,
        second_layer_flags,
        problem.n_shards,
    )
    return RoundResult(name, gradient, config_label=config.label, **metrics)


def _make_flexible_code(
    code_design: str,
    config: EncodingConfig,
    n_workers: int,
    n_shards: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if code_design == "random":
        return make_flexible_rows(
            n_workers,
            n_shards,
            config.degree_first,
            config.degree_second,
            rng,
        )
    if code_design == "balanced":
        return make_decode_balanced_flexible_rows(
            n_workers,
            n_shards,
            config.degree_first,
            config.degree_second,
            rng,
        )
    raise ValueError(f"Unknown code design: {code_design}")


def _assign_tasks_to_workers(
    first: np.ndarray,
    second: np.ndarray,
    shard_costs: np.ndarray,
    data_density: float,
    worker_state: WorkerState,
    assignment_mode: str,
) -> np.ndarray:
    n_workers = worker_state.speeds.size
    if assignment_mode == "identity":
        return np.arange(n_workers, dtype=int)

    pair_costs = np.zeros(n_workers, dtype=float)
    for task_id in range(n_workers):
        first_cost, _ = _row_cost(first[task_id], shard_costs, data_density)
        second_cost, _ = _row_cost(second[task_id], shard_costs, data_density)
        pair_costs[task_id] = first_cost + second_cost

    if assignment_mode == "cost":
        priorities = pair_costs
    elif assignment_mode in {"leverage", "deadline"}:
        priorities = _decode_pair_priorities(first, second, pair_costs)
        if assignment_mode == "deadline":
            return _deadline_assignment(pair_costs, priorities, worker_state)
    else:
        raise ValueError(f"Unknown assignment mode: {assignment_mode}")

    task_order = np.argsort(-priorities)
    worker_order = np.argsort(-worker_state.speeds)
    assignments = np.empty(n_workers, dtype=int)
    assignments[task_order] = worker_order
    return assignments


def _decode_pair_priorities(
    first: np.ndarray, second: np.ndarray, pair_costs: np.ndarray
) -> np.ndarray:
    n_workers = first.shape[0]
    rows = np.vstack([first, second])
    decode = decode_coefficients(rows)
    if not decode.success:
        return pair_costs
    pair_mass = np.abs(decode.coefficients[:n_workers]) + np.abs(
        decode.coefficients[n_workers:]
    )
    return pair_mass * np.maximum(pair_costs, 1e-12)


def _deadline_assignment(
    pair_costs: np.ndarray,
    priorities: np.ndarray,
    worker_state: WorkerState,
) -> np.ndarray:
    n_workers = worker_state.speeds.size
    nominal = (
        worker_state.delays[None, :]
        + 0.035 * pair_costs[:, None] / np.maximum(worker_state.speeds[None, :], 1e-12)
    )
    target_time = float(np.median(nominal)) + 1e-12
    completion_score = np.exp(-nominal / target_time)
    benefit = priorities[:, None] * completion_score
    task_ids, worker_ids = linear_sum_assignment(-benefit)
    assignments = np.empty(n_workers, dtype=int)
    assignments[task_ids] = worker_ids
    return assignments


def default_strategies() -> list[Strategy]:
    return [
        UncodedSyncStrategy(),
        ReplicationStrategy(),
        StaticSparseCodeStrategy(degree=3),
        FlexibleCodedStrategy(
            "flexible_thin_static",
            DEFAULT_FLEXIBLE_CONFIGS[0],
        ),
        FlexibleCodedStrategy(
            "sparse_flexible_static",
            DEFAULT_FLEXIBLE_CONFIGS[2],
        ),
        FlexibleCodedStrategy(
            "worker_aware_sparse_flexible",
            DEFAULT_FLEXIBLE_CONFIGS[2],
            speed_aware=True,
        ),
        FlexibleCodedStrategy(
            "rank_aware_sparse_flexible",
            DEFAULT_FLEXIBLE_CONFIGS[2],
            assignment_mode="leverage",
        ),
        FlexibleCodedStrategy(
            "deadline_aware_sparse_flexible",
            DEFAULT_FLEXIBLE_CONFIGS[2],
            assignment_mode="deadline",
        ),
        FlexibleCodedStrategy(
            "balanced_sparse_flexible",
            DEFAULT_FLEXIBLE_CONFIGS[2],
            code_design="balanced",
        ),
        FlexibleCodedStrategy(
            "balanced_rank_aware_sparse_flexible",
            DEFAULT_FLEXIBLE_CONFIGS[2],
            assignment_mode="leverage",
            code_design="balanced",
        ),
        FlexibleCodedStrategy(
            "balanced_deadline_aware_sparse_flexible",
            DEFAULT_FLEXIBLE_CONFIGS[2],
            assignment_mode="deadline",
            code_design="balanced",
        ),
        FlexibleCodedStrategy(
            "flexible_robust_static",
            DEFAULT_FLEXIBLE_CONFIGS[3],
        ),
        FlexibleCodedStrategy(
            "flexible_dense_static",
            DEFAULT_FLEXIBLE_CONFIGS[4],
        ),
        AdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
        ),
        UCBAdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
        ),
        AdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
            name="worker_aware_adaptive_sparse_flexible",
            speed_aware=True,
        ),
        AdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
            name="rank_aware_adaptive_sparse_flexible",
            assignment_mode="leverage",
        ),
        UCBAdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
            name="worker_aware_ucb_sparse_flexible",
            speed_aware=True,
        ),
        UCBAdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
            name="rank_aware_ucb_sparse_flexible",
            assignment_mode="leverage",
        ),
        UCBAdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
            name="balanced_rank_aware_ucb_sparse_flexible",
            assignment_mode="leverage",
            code_design="balanced",
        ),
        AdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
            name="adaptive_latency_only",
            extra_compute_weight=0.0,
        ),
        WindowAdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
        ),
        ContextualAdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
        ),
        ContextualUCBAdaptiveFlexibleStrategy(
            configs=DEFAULT_FLEXIBLE_CONFIGS,
        ),
    ]
