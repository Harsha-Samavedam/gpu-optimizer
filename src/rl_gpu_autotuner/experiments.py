"""Reproducible, budget-matched GPU searches and independent confirmation.

Run with ``python -m rl_gpu_autotuner.experiments --output-dir results/suite``.
The study measures hot-buffer graph throughput; it is not end-to-end latency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

from .domain import KernelKind, ScheduleConfig, Workload
from .policies import SearchResult, Trial
from .search_space import configs_for
from .triton_backend import TritonBenchmark, benchmark_torch_matmul, triton_preflight

FIXED = ScheduleConfig(64, 64, 32, 4, 3)
OLD_WINNER = ScheduleConfig(64, 128, 32, 4, 3)


def json_safe(value):
    """Preserve failed trials in standards-compliant JSON."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def curated_configs() -> list[ScheduleConfig]:
    """A predeclared 16-trial heuristic covering reuse, grouping, and pipeline depth.

    Larger tile candidates follow the families in Triton's matmul tutorial.
    This is a hand-written baseline, not a trained policy or an exhaustive oracle.
    """
    return [
        FIXED,
        OLD_WINNER,
        ScheduleConfig(128, 128, 32, 4, 3),
        ScheduleConfig(128, 128, 64, 4, 3),
        ScheduleConfig(128, 256, 32, 8, 3),
        ScheduleConfig(128, 256, 64, 8, 3),
        ScheduleConfig(256, 128, 32, 8, 3),
        ScheduleConfig(256, 128, 64, 8, 3),
        ScheduleConfig(64, 128, 64, 4, 4),
        ScheduleConfig(128, 64, 64, 4, 4),
        ScheduleConfig(64, 64, 128, 4, 3),
        ScheduleConfig(32, 64, 64, 4, 3),
        ScheduleConfig(64, 128, 32, 4, 3, 1),
        ScheduleConfig(64, 128, 32, 4, 3, 4),
        ScheduleConfig(128, 128, 64, 4, 3, 1),
        ScheduleConfig(128, 128, 64, 4, 4, 4),
    ]


def candidate_universe(workload: Workload) -> list[ScheduleConfig]:
    return list(dict.fromkeys(configs_for(workload) + curated_configs()))


def telemetry() -> dict[str, object]:
    fields = "name,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return {
            "utc": datetime.now(timezone.utc).isoformat(),
            "csv": result.stdout.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}


def summarize_case(case: dict) -> dict:
    rounds = case["confirmation"]
    summaries = {}
    for method in rounds[0]["measurements"]:
        records = [row["measurements"][method] for row in rounds]
        valid = [
            record
            for record in records
            if record["valid"] and record["latency_us"] is not None
        ]
        ratios = []
        for row in rounds:
            reference = row["measurements"]["pytorch"]
            measured = row["measurements"][method]
            if (
                reference["valid"]
                and measured["valid"]
                and measured["latency_us"] is not None
            ):
                ratios.append(reference["latency_us"] / measured["latency_us"])
        values = [record["latency_us"] for record in valid]
        summaries[method] = {
            "all_valid": len(valid) == len(records),
            "median_us": median(values) if values else None,
            "round_min_us": min(values) if values else None,
            "round_max_us": max(values) if values else None,
            "paired_speedup_vs_pytorch": median(ratios) if ratios else None,
        }
    return summaries


def run_case(
    workload: Workload,
    seed: int,
    directory: Path,
    *,
    budget: int = 16,
    rounds: int = 5,
    repetitions: int = 15,
    batch_size: int = 512,
) -> dict:
    if not 1 <= budget <= 16 or rounds < 3:
        raise ValueError("budget must be 1..16 and confirmation rounds at least 3")
    directory.mkdir(parents=True, exist_ok=False)
    universe = candidate_universe(workload)
    rng = random.Random(seed)
    selected = {
        "random": rng.sample(universe, budget),
        "curated": curated_configs()[:budget],
    }
    benchmark = TritonBenchmark(
        seed=seed, repetitions=repetitions, graph_batch_size=batch_size
    )
    history: dict[str, list[Trial]] = {name: [] for name in selected}
    start_time = perf_counter()
    # Interleave both searches to reduce order/temperature confounding. No sharing
    # of measurements: each policy pays for its own trials, even on overlap.
    jobs = [
        (method, config) for method, configs in selected.items() for config in configs
    ]
    rng.shuffle(jobs)
    with (directory / "trials.jsonl").open("w") as log:
        for index, (method, config) in enumerate(jobs):
            trial_start = perf_counter()
            measurement = benchmark.evaluate(workload, config)
            history[method].append(Trial(config, measurement))
            log.write(
                json.dumps(
                    json_safe(
                        {
                            "method": method,
                            "trial": history[method][-1],
                            "elapsed_s": perf_counter() - trial_start,
                        }
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
            log.flush()
            if (index + 1) % 8 == 0:
                print(
                    f"{workload.shape} seed={seed}: {index + 1}/{len(jobs)} search trials",
                    flush=True,
                )
    winners = {}
    for method, trials in history.items():
        valid = [trial for trial in trials if trial.measurement.usable]
        if not valid:
            raise RuntimeError(
                f"{method} has no valid candidate; inspect {directory / 'trials.jsonl'}"
            )
        best = min(valid, key=lambda trial: trial.measurement.latency_us)
        winners[method] = SearchResult(
            best.config, best.measurement, len(trials), tuple(trials)
        )
    case = {
        "workload": asdict(workload),
        "seed": seed,
        "candidate_count": len(universe),
        "budget_per_search": budget,
        "search_wall_s": perf_counter() - start_time,
        "search": json_safe(winners),
        "confirmation": [],
        "timing": {
            "method": "cuda_graph",
            "batch_size": batch_size,
            "samples": repetitions,
            "cache_policy": "hot repeated buffers",
            "warmup_launches": 10,
        },
        "confirmation_budget": {"rounds": rounds, "measurements_per_round": 4},
        "selection_rule": "lowest search median; frozen before confirmation",
    }
    write_json(directory / "result.json", case)
    # New inputs and new measurements; winners stay fixed throughout all rounds.
    for round_index in range(rounds):
        confirmation_seed = seed + 10_000 + round_index
        confirmation = TritonBenchmark(
            seed=confirmation_seed, repetitions=repetitions, graph_batch_size=batch_size
        )
        methods = ["pytorch", "fixed", "random", "curated"]
        rng.shuffle(methods)
        row = {
            "round": round_index,
            "input_seed": confirmation_seed,
            "order": methods,
            "telemetry_before": telemetry(),
            "measurements": {},
        }
        for method in methods:
            if method == "pytorch":
                measurement = benchmark_torch_matmul(
                    workload,
                    seed=confirmation_seed,
                    repetitions=repetitions,
                    graph_batch_size=batch_size,
                )
            else:
                config = FIXED if method == "fixed" else winners[method].config
                measurement = confirmation.evaluate(workload, config)
            row["measurements"][method] = json_safe(measurement)
        row["telemetry_after"] = telemetry()
        case["confirmation"].append(row)
        write_json(directory / "result.json", case)
    case["summary"] = summarize_case(case)
    case["total_wall_s"] = perf_counter() - start_time
    write_json(directory / "result.json", case)
    print(
        f"{workload.shape} seed={seed}: confirmed " + str(case["summary"]), flush=True
    )
    return case


def run_diagnostic(directory: Path) -> None:
    """Compare the old timer against three graph batch sizes, in shuffled order."""
    workload = Workload(KernelKind.MATMUL, (1024, 1024, 1024))
    records = []
    jobs = [
        (round_index, method, batch)
        for round_index in range(3)
        for method, batch in [
            ("events", 1),
            ("cuda_graph", 256),
            ("cuda_graph", 512),
            ("cuda_graph", 1024),
        ]
    ]
    random.Random(42).shuffle(jobs)
    for round_index, method, batch in jobs:
        bench = TritonBenchmark(
            seed=round_index,
            timing_method=method,
            graph_batch_size=batch,
            repetitions=15,
        )
        measurements = {
            "pytorch": benchmark_torch_matmul(
                workload,
                seed=round_index,
                timing_method=method,
                graph_batch_size=batch,
                repetitions=15,
            ),
            "fixed": bench.evaluate(workload, FIXED),
            "old_winner": bench.evaluate(workload, OLD_WINNER),
        }
        records.append(
            {
                "round": round_index,
                "timing_method": method,
                "batch_size": batch,
                "measurements": measurements,
                "telemetry_after": telemetry(),
            }
        )
        write_json(directory / "timing_diagnostic.json", records)
        print(
            f"timer {method} batch={batch}: "
            + str({k: round(v.latency_us, 3) for k, v in measurements.items()}),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shapes", default="512,512,512;1024,1024,1024;512,2048,1024;1000,769,513"
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(
            "output directory already exists; choose a new directory to preserve earlier results"
        )
    if (
        not 1 <= args.budget <= 16
        or args.rounds < 3
        or args.repetitions < 3
        or args.batch_size < 1
    ):
        parser.error("budget 1..16, rounds/repetitions >=3, batch-size >=1 required")
    shapes = [
        Workload(KernelKind.MATMUL, tuple(map(int, text.split(","))))
        for text in args.shapes.split(";")
    ]
    seeds = list(map(int, args.seeds.split(",")))
    if len(set(shapes)) != len(shapes) or len(set(seeds)) != len(seeds):
        parser.error("shapes and seeds must be unique")
    preflight = triton_preflight()
    if not preflight["available"]:
        parser.error(str(preflight))
    import torch

    args.output_dir.mkdir(parents=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "preflight": preflight,
            "telemetry": telemetry(),
            "shapes": shapes,
            "seeds": seeds,
            "budget": args.budget,
            "rounds": args.rounds,
            "repetitions": args.repetitions,
            "batch_size": args.batch_size,
            "curated_configs": curated_configs(),
            "torch_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "scope": "single GPU, hot buffers, FP16 inputs/outputs, correctness rtol=atol=0.01",
        },
    )
    if args.diagnostic_only:
        run_diagnostic(args.output_dir)
        return
    cases = []
    for workload in shapes:
        for seed in seeds:
            name = "x".join(map(str, workload.shape)) + f"_seed{seed}"
            cases.append(
                run_case(
                    workload,
                    seed,
                    args.output_dir / name,
                    budget=args.budget,
                    rounds=args.rounds,
                    repetitions=args.repetitions,
                    batch_size=args.batch_size,
                )
            )
    write_json(
        args.output_dir / "summary.json",
        [
            {
                key: case[key]
                for key in (
                    "workload",
                    "seed",
                    "summary",
                    "search_wall_s",
                    "total_wall_s",
                )
            }
            for case in cases
        ],
    )


if __name__ == "__main__":
    main()
