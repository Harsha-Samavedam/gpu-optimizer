from __future__ import annotations

import random
from dataclasses import dataclass

from .benchmark import Benchmark, Measurement
from .domain import ScheduleConfig, Workload


@dataclass(frozen=True)
class Trial:
    config: ScheduleConfig
    measurement: Measurement


@dataclass(frozen=True)
class SearchResult:
    config: ScheduleConfig
    measurement: Measurement
    evaluations: int
    history: tuple[Trial, ...] = ()


def evaluate_candidates(
    benchmark: Benchmark, workload: Workload, candidates: list[ScheduleConfig]
) -> SearchResult:
    history = tuple(
        Trial(candidate, benchmark.evaluate(workload, candidate))
        for candidate in candidates
    )
    valid = [trial for trial in history if trial.measurement.usable]
    if not valid:
        raise RuntimeError("search produced no valid finite measurements")
    best = min(valid, key=lambda trial: trial.measurement.latency_us)
    return SearchResult(best.config, best.measurement, len(history), history)


def random_search(
    benchmark: Benchmark,
    workload: Workload,
    candidates: list[ScheduleConfig],
    budget: int,
    seed: int = 0,
) -> SearchResult:
    if budget < 1 or not candidates:
        raise ValueError("budget and candidate set must be non-empty")
    rng = random.Random(seed)
    selected = rng.sample(candidates, k=min(budget, len(candidates)))
    return evaluate_candidates(benchmark, workload, selected)


def exhaustive_search(
    benchmark: Benchmark, workload: Workload, candidates: list[ScheduleConfig]
) -> SearchResult:
    if not candidates:
        raise ValueError("candidate set must be non-empty")
    return evaluate_candidates(benchmark, workload, candidates)
