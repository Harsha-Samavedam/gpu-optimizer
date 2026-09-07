"""CUDA/Triton benchmark backend for the FP16 matmul milestone.

Importing this module is safe on a CPU-only machine. Torch and Triton are
loaded only when a real benchmark or preflight is requested.
"""

from __future__ import annotations

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
        raise TritonUnavailableError("Triton is required for the GPU benchmark backend") from exc
    return torch, triton


def _load_torch_cuda() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TritonUnavailableError("PyTorch is required for the GPU benchmark backend") from exc
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


class TritonBenchmark(Benchmark):
    """Measure a parameterized FP16 Triton matmul on a CUDA GPU.

    Compilation happens before warm-up and timing. The timing loop uses CUDA
    events, so Python dispatch, tensor creation, reference computation, and
    compilation are not included in the reported kernel latency.
    """

    def __init__(
        self,
        device: str | None = None,
        warmup: int = 10,
        repetitions: int = 25,
        seed: int = 0,
        validate: bool = True,
    ) -> None:
        if warmup < 1 or repetitions < 3:
            raise ValueError("warmup must be at least 1 and repetitions at least 3")
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
        a = torch.randn((m, k), device=self.device, dtype=torch.float16, generator=generator)
        b = torch.randn((k, n), device=self.device, dtype=torch.float16, generator=generator)
        reference = torch.matmul(a, b)
        self._tensor_cache[workload] = (a, b, reference)
        return a, b, reference

    @staticmethod
    def _failure(detail: str, device: str = "") -> Measurement:
        return Measurement(float("inf"), valid=False, detail=detail[:500], device=device)

    def evaluate(self, workload: Workload, config: ScheduleConfig) -> Measurement:
        if not config.is_legal():
            return self._failure("illegal schedule configuration")

        torch, _ = _load_runtime()
        device_name = self.device_info().name
        try:
            from .kernels.matmul import matmul_fp16

            a, b, reference = self._inputs(workload, torch)
            output = torch.empty((workload.shape[0], workload.shape[1]), device=self.device, dtype=torch.float16)

            # This first launch forces JIT compilation and is deliberately not timed.
            matmul_fp16(a, b, output, config)
            torch.cuda.synchronize(self.device)

            if self.validate:
                torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

            for _ in range(self.warmup):
                matmul_fp16(a, b, output, config)
            torch.cuda.synchronize(self.device)

            samples_us: list[float] = []
            for _ in range(self.repetitions):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                matmul_fp16(a, b, output, config)
                end.record()
                end.synchronize()
                samples_us.append(start.elapsed_time(end) * 1_000.0)

            return Measurement(
                latency_us=float(median(samples_us)),
                samples_us=tuple(samples_us),
                device=device_name,
                detail=f"median of {self.repetitions} CUDA-event timings after {self.warmup} warm-ups",
            )
        except (AssertionError, RuntimeError, ValueError) as exc:
            return self._failure(f"kernel failed validation or execution: {exc}", device_name)


def benchmark_torch_matmul(
    workload: Workload,
    device: str | None = None,
    warmup: int = 10,
    repetitions: int = 25,
    seed: int = 0,
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

    for _ in range(warmup):
        torch.matmul(a, b, out=output)
    torch.cuda.synchronize(selected)

    samples_us: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.matmul(a, b, out=output)
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1_000.0)

    properties = torch.cuda.get_device_properties(selected)
    return Measurement(
        latency_us=float(median(samples_us)),
        samples_us=tuple(samples_us),
        device=properties.name,
        detail=f"PyTorch matmul median of {repetitions} CUDA-event timings after {warmup} warm-ups",
    )
