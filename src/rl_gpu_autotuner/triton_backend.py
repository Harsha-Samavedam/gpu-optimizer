"""CUDA/Triton benchmark backend for the FP16 matmul milestone.

Importing this module is safe on a CPU-only machine. Torch and Triton are
loaded only when a real benchmark or preflight is requested.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from .benchmark import Benchmark, Measurement
from .domain import KernelKind, ScheduleConfig, Workload


class TritonUnavailableError(RuntimeError):
    """Raised when the optional CUDA/Triton runtime is not usable."""


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    device: str
    compute_capability: str
    total_memory_bytes: int
    torch_version: str
    cuda_version: str | None
    triton_version: str | None


def _load_runtime() -> tuple[Any, Any]:
    torch = _load_torch_cuda()
    try:
        import triton
    except ImportError as exc:
        raise TritonUnavailableError(
            "Triton is required for the GPU benchmark backend"
        ) from exc
    return torch, triton


def _load_torch_cuda() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TritonUnavailableError(
            "PyTorch is required for the GPU benchmark backend"
        ) from exc
    if not torch.cuda.is_available():
        raise TritonUnavailableError("No CUDA GPU is available to PyTorch")
    return torch


def triton_preflight(device: str | None = None) -> dict[str, object]:
    """Return machine-readable CUDA/Triton readiness information.

    This intentionally returns an unavailable result rather than raising, so it
    can be run as a setup check after moving the project to a GPU machine.
    """
    try:
        torch, triton = _load_runtime()
        selected = torch.device(device or "cuda")
        properties = torch.cuda.get_device_properties(selected)
        info = DeviceInfo(
            name=properties.name,
            device=str(selected),
            compute_capability=f"{properties.major}.{properties.minor}",
            total_memory_bytes=properties.total_memory,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            triton_version=getattr(triton, "__version__", None),
        )
        return {"available": True, "device": asdict(info)}
    except TritonUnavailableError as exc:
        return {"available": False, "error": str(exc)}


def time_cuda_callable(
    fn: Callable[[], object],
    torch: Any,
    device: str,
    warmup: int = 10,
    repetitions: int = 25,
    timing_method: str = "cuda_graph",
    graph_batch_size: int = 512,
) -> tuple[float, ...]:
    """Return per-call microseconds for hot-buffer, steady-state execution.

    CUDA graphs amortize host dispatch over a captured batch. Events bracket
    replay on the selected device/stream; each sample is a batch mean, not an
    individual launch. No allocation, compilation, or reference work is timed.
    The legacy event mode is retained only for diagnosing launch-gap bias.
    """
    if warmup < 1 or repetitions < 3 or graph_batch_size < 1:
        raise ValueError(
            "positive warmup/batch size and at least 3 repetitions required"
        )
    if timing_method not in {"cuda_graph", "events"}:
        raise ValueError("unknown timing method")
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn()
            stream.synchronize()
            batch_size = 1
            replay = fn
            if timing_method == "cuda_graph":
                batch_size = graph_batch_size
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(batch_size):
                        fn()
                replay = graph.replay
                # Instantiate and warm graph replay before recording any samples.
                for _ in range(3):
                    replay()
                stream.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(repetitions):
                start.record(stream)
                replay()
                end.record(stream)
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0 / batch_size)
            return tuple(samples)


class TritonBenchmark(Benchmark):
    """Measure a parameterized FP16 Triton matmul on a CUDA GPU.

    Compilation and correctness validation precede timing. The default measures
    hot-buffer CUDA graph batches, amortizing host dispatch. Legacy single-launch
    events are available for diagnostics and can include host-induced idle gaps.
    """

    def __init__(
        self,
        device: str | None = None,
        warmup: int = 10,
        repetitions: int = 25,
        seed: int = 0,
        validate: bool = True,
        timing_method: str = "cuda_graph",
        graph_batch_size: int = 512,
    ) -> None:
        if warmup < 1 or repetitions < 3:
            raise ValueError("warmup must be at least 1 and repetitions at least 3")
        if timing_method not in {"cuda_graph", "events"} or graph_batch_size < 1:
            raise ValueError("invalid timing method or graph batch size")
        self.timing_method = timing_method
        self.graph_batch_size = graph_batch_size
        self.device = device or "cuda"
        self.warmup = warmup
        self.repetitions = repetitions
        self.seed = seed
        self.validate = validate
        self._tensor_cache: dict[Workload, tuple[Any, Any, Any]] = {}

    def device_info(self) -> DeviceInfo:
        torch, triton = _load_runtime()
        selected = torch.device(self.device)
        properties = torch.cuda.get_device_properties(selected)
        return DeviceInfo(
            name=properties.name,
            device=str(selected),
            compute_capability=f"{properties.major}.{properties.minor}",
            total_memory_bytes=properties.total_memory,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            triton_version=getattr(triton, "__version__", None),
        )

    def _inputs(self, workload: Workload, torch: Any) -> tuple[Any, Any, Any]:
        if workload in self._tensor_cache:
            return self._tensor_cache[workload]
        if workload.kernel is not KernelKind.MATMUL:
            raise ValueError("TritonBenchmark currently supports only matmul")
        if workload.dtype != "fp16":
            raise ValueError("the first hardware milestone supports dtype='fp16' only")
        m, n, k = workload.shape
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + m + n + k)
        a = torch.randn(
            (m, k), device=self.device, dtype=torch.float16, generator=generator
        )
        b = torch.randn(
            (k, n), device=self.device, dtype=torch.float16, generator=generator
        )
        reference = torch.matmul(a, b)
        self._tensor_cache[workload] = (a, b, reference)
        return a, b, reference

    def _failure(self, detail: str, device: str = "") -> Measurement:
        return Measurement(
            float("inf"),
            valid=False,
            detail=detail[:500],
            device=device,
            timing_method=self.timing_method,
            batch_size=self.graph_batch_size
            if self.timing_method == "cuda_graph"
            else 1,
        )

    def evaluate(self, workload: Workload, config: ScheduleConfig) -> Measurement:
        if not config.is_legal():
            return self._failure("illegal schedule configuration")

        torch, _ = _load_runtime()
        with torch.cuda.device(self.device):
            return self._evaluate_on_device(workload, config, torch)

    def _evaluate_on_device(
        self, workload: Workload, config: ScheduleConfig, torch: Any
    ) -> Measurement:
        device_name = self.device_info().name
        from triton.compiler.errors import CompilationError
        from triton.runtime.errors import OutOfResources

        try:
            from .kernels.matmul import matmul_fp16

            a, b, reference = self._inputs(workload, torch)
            output = torch.empty(
                (workload.shape[0], workload.shape[1]),
                device=self.device,
                dtype=torch.float16,
            )

            # This first launch forces JIT compilation and is deliberately not timed.
            matmul_fp16(a, b, output, config)
            torch.cuda.synchronize(self.device)

            if self.validate:
                torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

            samples_us = time_cuda_callable(
                lambda: matmul_fp16(a, b, output, config),
                torch,
                self.device,
                self.warmup,
                self.repetitions,
                self.timing_method,
                self.graph_batch_size,
            )
            return Measurement(
                latency_us=float(median(samples_us)),
                samples_us=samples_us,
                device=device_name,
                detail=f"median of {self.repetitions} samples; hot buffers; {self.warmup} warm-ups",
                timing_method=self.timing_method,
                batch_size=self.graph_batch_size
                if self.timing_method == "cuda_graph"
                else 1,
            )
        except (
            AssertionError,
            RuntimeError,
            ValueError,
            OutOfResources,
            CompilationError,
            subprocess.CalledProcessError,
        ) as exc:
            # A device-side fault can poison the CUDA context; continuing would
            # incorrectly turn every subsequent candidate into a failed trial.
            if any(
                message in str(exc).lower()
                for message in ("illegal memory access", "device-side assert")
            ):
                raise
            return self._failure(
                f"{type(exc).__name__}: kernel validation or execution failed: {exc}",
                device_name,
            )


def benchmark_torch_matmul(
    workload: Workload,
    device: str | None = None,
    warmup: int = 10,
    repetitions: int = 25,
    seed: int = 0,
    timing_method: str = "cuda_graph",
    graph_batch_size: int = 512,
) -> Measurement:
    """Measure PyTorch matmul (normally cuBLAS) as a library baseline.

    This is intentionally outside the ``Benchmark`` interface: PyTorch matmul
    has no Triton schedule configuration. Its median latency is reported next
    to fixed, searched, and learned Triton schedules.
    """
    if workload.kernel is not KernelKind.MATMUL or workload.dtype != "fp16":
        raise ValueError("the PyTorch baseline currently supports FP16 matmul only")
    if warmup < 1 or repetitions < 3:
        raise ValueError("warmup must be at least 1 and repetitions at least 3")

    torch = _load_torch_cuda()
    selected = device or "cuda"
    m, n, k = workload.shape
    generator = torch.Generator(device=selected)
    generator.manual_seed(seed + m + n + k)
    a = torch.randn((m, k), device=selected, dtype=torch.float16, generator=generator)
    b = torch.randn((k, n), device=selected, dtype=torch.float16, generator=generator)
    output = torch.empty((m, n), device=selected, dtype=torch.float16)

    samples_us = time_cuda_callable(
        lambda: torch.matmul(a, b, out=output),
        torch,
        selected,
        warmup,
        repetitions,
        timing_method,
        graph_batch_size,
    )
    properties = torch.cuda.get_device_properties(selected)
    return Measurement(
        latency_us=float(median(samples_us)),
        samples_us=samples_us,
        device=properties.name,
        detail=f"PyTorch matmul; median of {repetitions} samples; hot buffers; {warmup} warm-ups",
        timing_method=timing_method,
        batch_size=graph_batch_size if timing_method == "cuda_graph" else 1,
    )
