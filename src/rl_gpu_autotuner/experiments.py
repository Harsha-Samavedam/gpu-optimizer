"""Run repeated GPU autotuning experiments."""

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

import numpy as np

from .bandit import LinearUCB, context_features, latency_reward
from .domain import KernelKind, ScheduleConfig, Workload
from .policies import SearchResult, Trial
from .search_space import configs_for
from .triton_backend import TritonBenchmark, benchmark_torch_matmul, triton_preflight

FIXED = ScheduleConfig(64, 64, 32, 4, 3)
OLD_WINNER = ScheduleConfig(64, 128, 32, 4, 3)


def json_safe(value):
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


def usable_record(record: dict) -> bool:
    latency = record.get("latency_us")
    return bool(
        record.get("valid")
        and isinstance(latency, (int, float))
        and math.isfinite(latency)
        and latency > 0
    )


def summarize_case(case: dict) -> dict:
    rounds = case["confirmation"]
    summaries = {}
    for method in rounds[0]["measurements"]:
        records = [row["measurements"][method] for row in rounds]
        valid = [record for record in records if usable_record(record)]
        ratios = []
        for row in rounds:
            reference = row["measurements"]["pytorch"]
            measured = row["measurements"][method]
            if usable_record(reference) and usable_record(measured):
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
    bandit_model: LinearUCB | None = None,
) -> dict:
    if not 1 <= budget <= 16 or rounds < 3:
        raise ValueError("budget must be 1..16 and confirmation rounds at least 3")
    start_time = perf_counter()
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
    model = bandit_model.copy() if bandit_model is not None else None
    baseline = benchmark.evaluate(workload, FIXED) if model is not None else None
    if baseline is not None and not baseline.usable:
        raise RuntimeError(f"bandit reward baseline failed: {baseline.detail}")
    features = (
        np.array([context_features(workload, config) for config in universe])
        if model
        else None
    )
    available = np.ones(len(universe), dtype=bool)
    bandit_rng = np.random.default_rng(seed)
    methods = list(selected) + (["bandit"] if model else [])
    history: dict[str, list[Trial]] = {name: [] for name in methods}
    decision_seconds = {name: 0.0 for name in methods}
    search_started = perf_counter()
    with (directory / "trials.jsonl").open("w") as log:
        for step in range(budget):
            order = methods.copy()
            rng.shuffle(order)
            for method in order:
                decision_start = perf_counter()
                if method == "bandit":
                    action = model.select(features, available, bandit_rng)
                    available[action] = False
                    config = universe[action]
                else:
                    config = selected[method][step]
                decision_seconds[method] += perf_counter() - decision_start
                trial_start = perf_counter()
                measurement = benchmark.evaluate(workload, config)
                history[method].append(Trial(config, measurement))
                trial_elapsed = perf_counter() - trial_start
                if method == "bandit":
                    update_start = perf_counter()
                    model.update(
                        features[action],
                        latency_reward(measurement, baseline.latency_us),
                    )
                    decision_seconds[method] += perf_counter() - update_start
                log.write(
                    json.dumps(
                        json_safe(
                            {
                                "method": method,
                                "step": step + 1,
                                "trial": history[method][-1],
                                "elapsed_s": trial_elapsed,
                            }
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                log.flush()
            if (step + 1) % 4 == 0:
                print(
                    f"{workload.shape} seed={seed}: {step + 1}/{budget} trials per search",
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
        "search_wall_s": perf_counter() - search_started,
        "setup_wall_s": search_started - start_time,
        "search": json_safe(winners),
        "confirmation": [],
        "timing": {
            "method": "cuda_graph",
            "batch_size": batch_size,
            "samples": repetitions,
            "cache_policy": "hot repeated buffers",
            "correctness_reference": "FP32 matmul, TF32 disabled, rounded to FP16; rtol=atol=0.01",
            "warmup_launches": 10,
        },
        "confirmation_budget": {
            "rounds": rounds,
            "measurements_per_round": len(methods) + 2 + int(model is not None),
        },
        "selection_rule": "lowest search median; frozen before confirmation",
        "common_reference_evaluations": 1 if model else 0,
        "first_choice": asdict(history["bandit"][0].config) if model else None,
        "common_reference": json_safe(baseline),
        "decision_seconds": decision_seconds,
        "bandit_updates": model.observations - bandit_model.observations
        if model
        else 0,
        "budget_curve": {
            name: [
                min(
                    (
                        t.measurement.latency_us
                        for t in trials[:i]
                        if t.measurement.usable
                    ),
                    default=None,
                )
                for i in range(1, len(trials) + 1)
            ]
            for name, trials in history.items()
        },
    }
    write_json(directory / "result.json", case)
    for round_index in range(rounds):
        confirmation_seed = 10_000 + seed * rounds + round_index
        confirmation = TritonBenchmark(
            seed=confirmation_seed, repetitions=repetitions, graph_batch_size=batch_size
        )
        methods = ["pytorch", "fixed", *winners] + (["bandit_first"] if model else [])
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
                config = (
                    FIXED
                    if method == "fixed"
                    else history["bandit"][0].config
                    if method == "bandit_first"
                    else winners[method].config
                )
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
    parser.add_argument("--bandit-model", type=Path)
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

    bandit_model, bandit_metadata = None, None
    if args.bandit_model:
        from .train_bandit import device_key

        bandit_model, bandit_metadata = LinearUCB.load(args.bandit_model)
        trained_shapes = {tuple(shape) for shape in bandit_metadata["training_shapes"]}
        if trained_shapes & {workload.shape for workload in shapes}:
            parser.error("test shapes overlap with bandit training shapes")
        if bandit_metadata["device_key"] != device_key(preflight):
            parser.error("bandit GPU/runtime differs from current preflight")
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
            "bandit_metadata": bandit_metadata,
            "bandit_model": str(args.bandit_model) if args.bandit_model else None,
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
                    bandit_model=bandit_model,
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
