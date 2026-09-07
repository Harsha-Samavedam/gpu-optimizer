from __future__ import annotations

import math

from .benchmark import Benchmark
from .domain import ScheduleConfig, Workload
from .policies import Trial


class TuningEnvironment:
    def __init__(
        self,
        benchmark: Benchmark,
        workload: Workload,
        candidates: list[ScheduleConfig],
        budget: int,
        baseline_latency_us: float | None = None,
        device_features: dict[str, object] | None = None,
    ):
        if budget < 1 or not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("positive budget and non-empty unique candidates required")
        if baseline_latency_us is not None and (
            not math.isfinite(baseline_latency_us) or baseline_latency_us <= 0
        ):
            raise ValueError("baseline must be positive and finite")
        self.benchmark, self.workload = benchmark, workload
        self.candidates, self.budget = tuple(candidates), budget
        self.baseline_latency_us = baseline_latency_us
        self.device_features = dict(device_features or {})
        self.reset()

    def reset(self) -> dict[str, object]:
        self.evaluations = 0
        self.best_latency = (
            self.baseline_latency_us
            if self.baseline_latency_us is not None
            else float("inf")
        )
        self.best_config: ScheduleConfig | None = None
        self.history: list[Trial] = []
        self.tried: set[int] = set()
        return self.observation()

    def observation(self) -> dict[str, object]:
        done = self.evaluations >= self.budget or len(self.tried) == len(
            self.candidates
        )
        return {
            "kernel": self.workload.kernel.value,
            "shape": self.workload.shape,
            "dtype": self.workload.dtype,
            "device_features": dict(self.device_features),
            "evaluations_remaining": self.budget - self.evaluations,
            "candidate_count": len(self.candidates),
            "candidates": self.candidates,
            "baseline_latency_us": self.baseline_latency_us,
            "best_latency_us": self.best_latency
            if math.isfinite(self.best_latency)
            else None,
            "best_config": self.best_config,
            "history": tuple(self.history),
            "action_mask": tuple(
                not done and i not in self.tried for i in range(len(self.candidates))
            ),
        }

    def step(
        self, action: int
    ) -> tuple[dict[str, object], float, bool, dict[str, object]]:
        if self.evaluations >= self.budget or len(self.tried) == len(self.candidates):
            raise RuntimeError("measurement budget or candidates exhausted; call reset")
        if not 0 <= action < len(self.candidates):
            raise IndexError("action must index a candidate configuration")
        if action in self.tried:
            raise ValueError("configuration already measured in this episode")
        measurement = self.benchmark.evaluate(self.workload, self.candidates[action])
        previous_best = self.best_latency
        reward = -1.0
        if measurement.usable:
            if measurement.latency_us < self.best_latency:
                self.best_latency = measurement.latency_us
                self.best_config = self.candidates[action]
            reward = (
                0.0
                if self.baseline_latency_us is None
                else math.log(previous_best / self.best_latency)
            )
        self.history.append(Trial(self.candidates[action], measurement))
        self.tried.add(action)
        self.evaluations += 1
        done = self.evaluations >= self.budget or len(self.tried) == len(
            self.candidates
        )
        if (
            done
            and self.baseline_latency_us is None
            and math.isfinite(self.best_latency)
        ):
            reward += 1.0 / (1.0 + self.best_latency)
        return (
            self.observation(),
            reward,
            done,
            {
                "latency_us": measurement.latency_us,
                "best_latency_us": self.best_latency,
                "valid": measurement.usable,
                "detail": measurement.detail,
            },
        )
