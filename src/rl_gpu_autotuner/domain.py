from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KernelKind(str, Enum):
    MATMUL = "matmul"
    SOFTMAX = "softmax"
    REDUCTION = "reduction"


@dataclass(frozen=True)
class Workload:
    """A kernel problem instance. Matmul uses (M, N, K); others use (rows, cols)."""

    kernel: KernelKind
    shape: tuple[int, ...]
    dtype: str = "fp16"

    def __post_init__(self) -> None:
        expected_dims = 3 if self.kernel is KernelKind.MATMUL else 2
        if len(self.shape) != expected_dims or any(d <= 0 for d in self.shape):
            raise ValueError(
                f"{self.kernel.value} requires {expected_dims} positive dimensions"
            )
        if not self.dtype:
            raise ValueError("dtype must be non-empty")


@dataclass(frozen=True)
class ScheduleConfig:
    """The small, interpretable action space for our first tuner."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    group_size_m: int = 8

    @property
    def threads(self) -> int:
        return self.num_warps * 32

    def is_legal(self) -> bool:
        # Conservative constraints suitable for a first Triton-oriented space.
        return (
            all(
                size >= 16 and size & (size - 1) == 0
                for size in (self.block_m, self.block_n, self.block_k)
            )
            and self.group_size_m > 0
            and self.num_warps in {1, 2, 4, 8}
            and self.num_stages in {1, 2, 3, 4, 5}
            and self.threads <= 256
            and self.block_m * self.block_n <= 65_536
        )
