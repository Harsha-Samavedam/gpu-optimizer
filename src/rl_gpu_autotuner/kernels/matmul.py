"""FP16 Triton matrix multiplication used by the hardware benchmark backend."""

from __future__ import annotations

import triton
import triton.language as tl

from ..domain import ScheduleConfig


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute one output tile of C = A @ B using FP32 accumulation."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_block * BLOCK_K + offsets_k
        a = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < K) & (offsets_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N))


def matmul_fp16(a, b, c, config: ScheduleConfig, group_size_m: int = 8) -> None:
    """Launch the parameterized FP16 matmul kernel.

    Inputs must be contiguous rank-2 CUDA tensors with shapes ``[M, K]`` and
    ``[K, N]``. Allocation and timing are intentionally owned by the benchmark
    backend so they are not included in kernel latency.
    """
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("matmul_fp16 requires rank-2 tensors")
    M, K = a.shape
    b_k, N = b.shape
    if b_k != K or tuple(c.shape) != (M, N):
        raise ValueError("incompatible matmul tensor shapes")

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
