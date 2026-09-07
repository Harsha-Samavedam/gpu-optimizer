from __future__ import annotations

from .benchmark import Benchmark
from .domain import ScheduleConfig, Workload


class TuningEnvironment:
    """Minimal RL-compatible environment: each action spends one measurement."""

    def __init__(self, benchmark: Benchmark, workload: Workload, candidates: list[ScheduleConfig], budget: int):
        self.benchmark, self.workload = benchmark, workload
        self.candidates, self.budget = candidates, budget
        self.reset()

    def reset(self) -> dict[str, int]:
        self.evaluations = 0
        self.best_latency = float("inf")
        return self.observation()

    def observation(self) -> dict[str, int]:
        return {"evaluations_remaining": self.budget - self.evaluations, "candidate_count": len(self.candidates)}

    def step(self, action: int) -> tuple[dict[str, int], float, bool, dict[str, float]]:
        if not 0 <= action < len(self.candidates):
            raise IndexError("action must index a candidate configuration")
        if self.evaluations >= self.budget:
            raise RuntimeError("measurement budget exhausted; call reset")
        measurement = self.benchmark.evaluate(self.workload, self.candidates[action])
        previous_best = self.best_latency
        self.best_latency = min(self.best_latency, measurement.latency_us)
        self.evaluations += 1
        # Positive reward only for an improvement over the incumbent.
        reward = 0.0 if previous_best == float("inf") else max(0.0, previous_best - self.best_latency)
        done = self.evaluations >= self.budget
        return self.observation(), reward, done, {"latency_us": measurement.latency_us, "best_latency_us": self.best_latency}
