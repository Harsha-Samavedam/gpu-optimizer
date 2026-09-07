from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .domain import ScheduleConfig
from .benchmark import SimulatedBenchmark
from .domain import KernelKind, Workload
from .policies import exhaustive_search, random_search
from .search_space import configs_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GPU autotuning search policies.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--kernel", choices=[k.value for k in KernelKind], default="matmul")
    compare.add_argument("--shape", default="1024,1024,1024")
    compare.add_argument("--dtype", default="fp16")
    compare.add_argument("--backend", choices=("simulated", "triton"), default="simulated")
    compare.add_argument("--budget", type=int, default=16)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--warmup", type=int, default=10)
    compare.add_argument("--repetitions", type=int, default=25)
    compare.add_argument("--oracle", action="store_true", help="also exhaustively evaluate every candidate")
    compare.add_argument("--results-json", type=Path, default=None, help="write complete comparison results to this file")
    preflight = subparsers.add_parser("preflight", help="check CUDA/Triton availability on this machine")
    preflight.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        from .triton_backend import triton_preflight

        print(json.dumps(triton_preflight(args.device), indent=2, sort_keys=True))
        return

    kernel = KernelKind(args.kernel)
    shape = tuple(int(part) for part in args.shape.split(","))
    workload = Workload(kernel, shape, args.dtype)
    candidates = configs_for(workload)
    if args.backend == "simulated":
        benchmark = SimulatedBenchmark(seed=args.seed)
        torch_baseline = None
    else:
        from .triton_backend import TritonBenchmark, benchmark_torch_matmul

        benchmark = TritonBenchmark(warmup=args.warmup, repetitions=args.repetitions, seed=args.seed)
        torch_baseline = benchmark_torch_matmul(
            workload,
            warmup=args.warmup,
            repetitions=args.repetitions,
            seed=args.seed,
        )

    fixed_config = ScheduleConfig(64, 64, 32, 4, 3)
    fixed_measurement = benchmark.evaluate(workload, fixed_config)
    random_result = random_search(benchmark, workload, candidates, args.budget, args.seed)
    print(f"workload: {kernel.value} {shape} {args.dtype}; candidates: {len(candidates)}")
    if torch_baseline is not None:
        print(f"PyTorch/cuBLAS: {torch_baseline.latency_us:.3f} us")
    print(f"fixed: {fixed_measurement.latency_us:.3f} us | {fixed_config}")
    print(f"random ({random_result.evaluations} evals): {random_result.measurement.latency_us:.3f} us | {random_result.config}")
    oracle_result = None
    if args.oracle:
        oracle_result = exhaustive_search(benchmark, workload, candidates)
        print(f"oracle ({oracle_result.evaluations} evals): {oracle_result.measurement.latency_us:.3f} us | {oracle_result.config}")
    if args.results_json is not None:
        results = {
            "workload": asdict(workload),
            "backend": args.backend,
            "budget": args.budget,
            "seed": args.seed,
            "candidate_count": len(candidates),
            "pytorch_cublas": asdict(torch_baseline) if torch_baseline is not None else None,
            "fixed": {"config": asdict(fixed_config), "measurement": asdict(fixed_measurement)},
            "random": {"config": asdict(random_result.config), "measurement": asdict(random_result.measurement)},
            "oracle": (
                {"config": asdict(oracle_result.config), "measurement": asdict(oracle_result.measurement)}
                if oracle_result is not None
                else None
            ),
        }
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"wrote results: {args.results_json}")


if __name__ == "__main__":
    main()
