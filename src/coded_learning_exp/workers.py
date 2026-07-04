from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkerState:
    speeds: np.ndarray
    delays: np.ndarray
    slow_mask: np.ndarray
    scenario: str


@dataclass
class WorkerPoolConfig:
    n_workers: int
    scenario: str = "drift"
    drift_period: int = 35
    straggler_fraction: float = 0.25
    straggler_slowdown: float = 0.22
    burst_probability: float = 0.45
    base_sigma: float = 0.35
    straggler_delay_low: float = 0.10
    straggler_delay_high: float = 0.55


class WorkerPool:
    def __init__(self, config: WorkerPoolConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.base_speeds = self._draw_base_speeds()
        self.stable_slow = self._draw_slow_mask()
        self.drift_slow = self._draw_slow_mask()

    def sample(self, iteration: int) -> WorkerState:
        cfg = self.config
        if cfg.scenario in {"drift", "phase"} and iteration % cfg.drift_period == 0 and iteration > 0:
            self.base_speeds = self._draw_base_speeds()
            self.drift_slow = self._draw_slow_mask()

        speeds = self.base_speeds.copy()
        delays = np.zeros(cfg.n_workers, dtype=float)
        slow_mask = np.zeros(cfg.n_workers, dtype=bool)
        phase_slowdown = cfg.straggler_slowdown

        if cfg.scenario == "stable":
            slow_mask = self.stable_slow.copy()
        elif cfg.scenario == "burst":
            if self.rng.random() < cfg.burst_probability:
                slow_mask = self._draw_slow_mask()
        elif cfg.scenario == "drift":
            slow_mask = self.drift_slow.copy()
        elif cfg.scenario == "phase":
            phase_id = (iteration // cfg.drift_period) % 3
            if phase_id == 0:
                slow_mask = self._draw_slow_mask_with_fraction(max(0.05, cfg.straggler_fraction / 2.0))
                phase_slowdown = min(0.75, max(0.35, cfg.straggler_slowdown * 2.0))
            elif phase_id == 1:
                slow_mask = self._draw_slow_mask_with_fraction(cfg.straggler_fraction)
                phase_slowdown = cfg.straggler_slowdown
            else:
                slow_mask = self._draw_slow_mask_with_fraction(min(0.50, cfg.straggler_fraction * 1.7))
                phase_slowdown = max(0.08, cfg.straggler_slowdown * 0.75)
        else:
            raise ValueError(f"Unknown scenario: {cfg.scenario}")

        speeds[slow_mask] *= phase_slowdown
        delays[slow_mask] = self.rng.uniform(
            cfg.straggler_delay_low, cfg.straggler_delay_high, size=int(slow_mask.sum())
        )
        return WorkerState(
            speeds=speeds,
            delays=delays,
            slow_mask=slow_mask,
            scenario=cfg.scenario,
        )

    def _draw_base_speeds(self) -> np.ndarray:
        return self.rng.lognormal(mean=0.0, sigma=self.config.base_sigma, size=self.config.n_workers)

    def _draw_slow_mask(self) -> np.ndarray:
        return self._draw_slow_mask_with_fraction(self.config.straggler_fraction)

    def _draw_slow_mask_with_fraction(self, fraction: float) -> np.ndarray:
        n_slow = max(1, int(round(self.config.n_workers * fraction)))
        mask = np.zeros(self.config.n_workers, dtype=bool)
        mask[self.rng.choice(self.config.n_workers, size=n_slow, replace=False)] = True
        return mask
