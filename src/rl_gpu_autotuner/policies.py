from __future__ import annotations

from dataclasses import dataclass
import random

from .benchmark import Benchmark, Measurement
from .domain import ScheduleConfig, Workload


@dataclass(frozen=True)
class SearchResult:
    config: ScheduleConfig
    measurement: Measurement
    evaluations: int


def random_search(
    benchmark: Benchmark, workload: Workload, candidates: list[ScheduleConfig], budget: int, seed: int = 0
) -> SearchResult:
    if budget < 1 or not candidates:
        raise ValueError("budget and candidate set must be non-empty")
    rng = random.Random(seed)
    selected = rng.sample(candidates, k=min(budget, len(candidates)))
    scored = [(candidate, benchmark.evaluate(workload, candidate)) for candidate in selected]
    best_config, best_measurement = min(scored, key=lambda item: item[1].latency_us)
    return SearchResult(best_config, best_measurement, len(scored))


def exhaustive_search(
    benchmark: Benchmark, workload: Workload, candidates: list[ScheduleConfig]
) -> SearchResult:
    if not candidates:
        raise ValueError("candidate set must be non-empty")
    scored = [(candidate, benchmark.evaluate(workload, candidate)) for candidate in candidates]
    best_config, best_measurement = min(scored, key=lambda item: item[1].latency_us)
    return SearchResult(best_config, best_measurement, len(scored))
