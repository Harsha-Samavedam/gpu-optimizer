from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .domain import KernelKind, ScheduleConfig, Workload


@dataclass(frozen=True)
class Measurement:
    latency_us: float
    valid: bool = True
    detail: str = ""
    samples_us: tuple[float, ...] = ()
    device: str = ""
    timing_method: str = "simulated"
    batch_size: int = 1

    @property
    def usable(self) -> bool:
        return self.valid and math.isfinite(self.latency_us) and self.latency_us > 0


class Benchmark(ABC):
    @abstractmethod
    def evaluate(self, workload: Workload, config: ScheduleConfig) -> Measurement:
        """Compile/execute a config and return a robust latency measurement."""


class SimulatedBenchmark(Benchmark):
    """Deterministic, noisy stand-in for a real GPU timing loop.

    This is intentionally not a GPU performance model. It gives policy code an
    expensive/noisy reward surface that rewards reasonable tile and pipeline
    choices, while making unit tests independent of CUDA hardware.
    """

    def __init__(self, seed: int = 0, noise_fraction: float = 0.015) -> None:
        self._seed = seed
        self._noise_fraction = noise_fraction

    def evaluate(self, workload: Workload, config: ScheduleConfig) -> Measurement:
        if not config.is_legal():
            return Measurement(
                float("inf"), valid=False, detail="illegal configuration"
            )

        volume = 1
        for dimension in workload.shape:
            volume *= dimension
        if workload.kernel is not KernelKind.MATMUL:
            volume *= 8

        # Prefer tiles near a problem-dependent target; punish underutilization
        # and excessive staging. This creates a nontrivial, reproducible surface.
        target_m = 64 if workload.shape[0] >= 512 else 32
        target_n = 64 if workload.shape[1] >= 512 else 32
        tile_penalty = 1 + abs(config.block_m - target_m) / target_m
        tile_penalty += abs(config.block_n - target_n) / target_n
        k_penalty = 1 + abs(config.block_k - 32) / 96
        warp_target = 4 if volume > 32_000_000 else 2
        warp_penalty = 1 + 0.15 * abs(config.num_warps - warp_target)
        stage_penalty = 1 + 0.06 * abs(config.num_stages - 3)
        base_us = max(3.0, volume / 100_000.0)
        raw = base_us * tile_penalty * k_penalty * warp_penalty * stage_penalty

        signature = hash((self._seed, workload, config)) & 0xFFFFFFFF
        noise = random.Random(signature).uniform(
            -self._noise_fraction, self._noise_fraction
        )
        return Measurement(latency_us=raw * (1 + noise))
