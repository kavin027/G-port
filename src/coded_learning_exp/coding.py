from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class DecodeResult:
    success: bool
    coefficients: np.ndarray
    residual: float
    cpu_seconds: float


def can_decode(rows: np.ndarray, target: np.ndarray | None = None, tol: float = 1e-7) -> bool:
    return decode_coefficients(rows, target=target, tol=tol).success


def decode_coefficients(
    rows: np.ndarray, target: np.ndarray | None = None, tol: float = 1e-7
) -> DecodeResult:
    """Find coefficients c such that rows.T @ c equals the all-shard sum."""
    start = perf_counter()
    if rows.size == 0:
        return DecodeResult(False, np.empty(0), np.inf, perf_counter() - start)

    n_shards = rows.shape[1]
    rhs = np.ones(n_shards, dtype=float) if target is None else target
    coeffs, *_ = np.linalg.lstsq(rows.T, rhs, rcond=None)
    residual = float(np.linalg.norm(rows.T @ coeffs - rhs) / np.sqrt(n_shards))
    return DecodeResult(residual <= tol, coeffs, residual, perf_counter() - start)


def make_sparse_rows(
    n_rows: int,
    n_shards: int,
    degree: int,
    rng: np.random.Generator,
) -> np.ndarray:
    degree = max(1, min(degree, n_shards))
    rows = np.zeros((n_rows, n_shards), dtype=float)
    scale = 1.0 / np.sqrt(degree)
    for row_id in range(n_rows):
        support = rng.choice(n_shards, size=degree, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=degree)
        rows[row_id, support] = signs * scale
    return rows


def make_decodable_sparse_rows(
    n_rows: int,
    n_shards: int,
    degree: int,
    rng: np.random.Generator,
    max_attempts: int = 500,
) -> np.ndarray:
    for _ in range(max_attempts):
        rows = make_sparse_rows(n_rows, n_shards, degree, rng)
        if can_decode(rows):
            return rows
    raise RuntimeError(
        f"Could not create a decodable sparse matrix: rows={n_rows}, "
        f"shards={n_shards}, degree={degree}"
    )


def make_flexible_rows(
    n_workers: int,
    n_shards: int,
    degree_first: int,
    degree_second: int,
    rng: np.random.Generator,
    max_attempts: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two sparse coding layers whose combined equations can decode."""
    for _ in range(max_attempts):
        first = make_sparse_rows(n_workers, n_shards, degree_first, rng)
        second = make_sparse_rows(n_workers, n_shards, degree_second, rng)
        if can_decode(np.vstack([first, second])):
            return first, second
    raise RuntimeError(
        "Could not create decodable flexible rows: "
        f"workers={n_workers}, shards={n_shards}, "
        f"degrees=({degree_first}, {degree_second})"
    )


def make_decode_balanced_flexible_rows(
    n_workers: int,
    n_shards: int,
    degree_first: int,
    degree_second: int,
    rng: np.random.Generator,
    candidates: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample sparse flexible codes and keep the one with balanced decode mass."""
    best_score = np.inf
    best_rows: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(candidates):
        first, second = make_flexible_rows(
            n_workers,
            n_shards,
            degree_first,
            degree_second,
            rng,
            max_attempts=80,
        )
        rows = np.vstack([first, second])
        score = decode_balance_score(rows, n_workers)
        if score < best_score:
            best_score = score
            best_rows = (first, second)
    if best_rows is None:
        raise RuntimeError("Could not build a decode-balanced flexible code.")
    return best_rows


def decode_balance_score(rows: np.ndarray, n_workers: int) -> float:
    decode = decode_coefficients(rows)
    if not decode.success:
        return np.inf
    coeff_abs = np.abs(decode.coefficients)
    mean_coeff = float(coeff_abs.mean()) + 1e-12
    pair_importance = coeff_abs[:n_workers] + coeff_abs[n_workers : 2 * n_workers]
    mean_pair = float(pair_importance.mean()) + 1e-12

    coeff_concentration = float(coeff_abs.max()) / mean_coeff
    coeff_cv = float(coeff_abs.std()) / mean_coeff
    pair_concentration = float(pair_importance.max()) / mean_pair
    pair_cv = float(pair_importance.std()) / mean_pair
    return coeff_concentration + 0.5 * coeff_cv + 0.5 * pair_concentration + 0.25 * pair_cv


def aggregate_encoded_gradients(
    rows: np.ndarray,
    encoded_gradients: np.ndarray,
    decode: DecodeResult,
) -> np.ndarray:
    if not decode.success:
        raise ValueError("Cannot aggregate gradients without a successful decoder.")
    return decode.coefficients @ encoded_gradients
