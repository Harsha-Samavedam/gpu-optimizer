from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .benchmark import Measurement
from .domain import KernelKind, ScheduleConfig, Workload

FEATURE_VERSION = 1


def context_features(workload: Workload, config: ScheduleConfig) -> np.ndarray:
    if (
        workload.kernel != KernelKind.MATMUL
        or workload.dtype != "fp16"
        or not config.is_legal()
    ):
        raise ValueError("bandit supports legal FP16 matmul configurations")
    m, n, k = workload.shape
    bm, bn, bk = config.block_m, config.block_n, config.block_k
    tiles_m, tiles_n, steps = math.ceil(m / bm), math.ceil(n / bn), math.ceil(k / bk)
    context = np.array(
        [
            math.log2(m) / 12,
            math.log2(n) / 12,
            math.log2(k) / 12,
            math.log2(m / n) / 8,
            math.log2(k / n) / 8,
            math.log2(m * k * n) / 36,
        ]
    )
    action = np.array(
        [
            math.log2(bm) / 8,
            math.log2(bn) / 8,
            math.log2(bk) / 8,
            math.log2(config.num_warps) / 3,
            config.num_stages / 5,
            math.log2(config.group_size_m) / 3,
        ]
    )
    derived = np.array(
        [
            math.log2(tiles_m * tiles_n) / 20,
            math.log2(steps) / 12,
            m * n / (tiles_m * tiles_n * bm * bn),
            k / (steps * bk),
            (bm + bn) * bk * 2 * config.num_stages / 65536,
            bm * bn / (config.threads * 256),
            bm * bn / ((bm + bn) * 128),
            math.log2(bm / bn) / 4,
            min(tiles_m * tiles_n / 128, 1),
            float(m % bm == 0),
            float(n % bn == 0),
            float(k % bk == 0),
        ]
    )
    return np.concatenate(
        ([1.0], context, action, action**2, derived, np.outer(context, action).ravel())
    )


def latency_reward(measurement: Measurement, baseline_latency_us: float) -> float:
    if not math.isfinite(baseline_latency_us) or baseline_latency_us <= 0:
        raise ValueError("baseline must be positive and finite")
    if not measurement.usable:
        return -6.0
    return float(
        np.clip(math.log(baseline_latency_us) - math.log(measurement.latency_us), -6, 6)
    )


class LinearUCB:
    def __init__(self, dimension: int, alpha: float = 0.15, ridge: float = 0.1):
        if (
            dimension < 1
            or not math.isfinite(alpha)
            or alpha < 0
            or not math.isfinite(ridge)
            or ridge <= 0
        ):
            raise ValueError(
                "positive dimension/ridge and nonnegative finite alpha required"
            )
        self.alpha = alpha
        self.ridge = ridge
        self.inverse = np.eye(dimension) / ridge
        self.response = np.zeros(dimension)
        self.observations = 0

    @property
    def weights(self) -> np.ndarray:
        return self.inverse @ self.response

    def copy(self) -> LinearUCB:
        copied = LinearUCB(len(self.response), self.alpha, self.ridge)
        copied.inverse = self.inverse.copy()
        copied.response = self.response.copy()
        copied.observations = self.observations
        return copied

    def update(self, features: np.ndarray, reward: float) -> None:
        x = np.asarray(features, dtype=np.float64)
        if (
            x.shape != self.response.shape
            or not np.isfinite(x).all()
            or not math.isfinite(reward)
        ):
            raise ValueError("invalid feature vector or reward")
        projected = self.inverse @ x
        self.inverse -= np.outer(projected, projected) / (1 + x @ projected)
        self.inverse = (self.inverse + self.inverse.T) / 2
        self.response += reward * x
        self.observations += 1

    def scores(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.response) or not np.isfinite(x).all():
            raise ValueError("invalid candidate feature matrix")
        variance = np.einsum("ij,jk,ik->i", x, self.inverse, x, optimize=True)
        return x @ self.weights + self.alpha * np.sqrt(np.maximum(variance, 0))

    def select(
        self, features: np.ndarray, available: np.ndarray, rng: np.random.Generator
    ) -> int:
        scores = self.scores(features)
        mask = np.asarray(available, dtype=bool)
        if mask.shape != scores.shape or not mask.any():
            raise ValueError("no available candidate or invalid mask")
        scores[~mask] = -np.inf
        ties = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0, atol=1e-12))
        return int(rng.choice(ties))

    def save(self, path: Path, metadata: dict) -> None:
        from .experiments import write_json

        write_json(
            path,
            {
                "version": FEATURE_VERSION,
                "alpha": self.alpha,
                "ridge": self.ridge,
                "inverse": self.inverse.tolist(),
                "response": self.response.tolist(),
                "observations": self.observations,
                "metadata": metadata,
            },
        )

    @classmethod
    def load(cls, path: Path) -> tuple[LinearUCB, dict]:
        data = json.loads(path.read_text())
        if data["version"] != FEATURE_VERSION:
            raise ValueError("unsupported feature version")
        response = np.asarray(data["response"], dtype=np.float64)
        inverse = np.asarray(data["inverse"], dtype=np.float64)
        if response.ndim != 1 or inverse.shape != (response.size, response.size):
            raise ValueError("invalid model dimensions")
        if (
            not np.isfinite(response).all()
            or not np.isfinite(inverse).all()
            or not np.allclose(inverse, inverse.T)
        ):
            raise ValueError("invalid model coefficients")
        np.linalg.cholesky(inverse)
        model = cls(response.size, data["alpha"], data["ridge"])
        model.inverse, model.response = inverse, response
        if not isinstance(data["observations"], int) or data["observations"] < 0:
            raise ValueError("invalid observation count")
        model.observations = data["observations"]
        return model, data["metadata"]
