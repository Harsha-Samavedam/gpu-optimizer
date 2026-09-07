from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .bandit import FEATURE_VERSION, LinearUCB, context_features, latency_reward
from .domain import KernelKind, ScheduleConfig, Workload
from .experiments import (
    FIXED,
    candidate_universe,
    curated_configs,
    json_safe,
    telemetry,
    write_json,
)
from .triton_backend import TritonBenchmark, triton_preflight

TRAIN_SHAPES = [
    (256, 256, 256),
    (384, 384, 384),
    (768, 768, 768),
    (1536, 1536, 512),
    (256, 1024, 512),
    (384, 1536, 768),
    (768, 384, 1024),
    (1536, 512, 768),
    (767, 513, 385),
    (1151, 779, 651),
]
TEST_SHAPES = [(512, 512, 512), (1024, 1024, 1024), (512, 2048, 1024), (1000, 769, 513)]


def validate_split(training_shapes, test_shapes) -> None:
    training = [tuple(shape) for shape in training_shapes]
    tests = [tuple(shape) for shape in test_shapes]
    if (
        not training
        or not tests
        or len(set(training)) != len(training)
        or len(set(tests)) != len(tests)
    ):
        raise ValueError("nonempty unique shape lists required")
    if set(training) & set(tests):
        raise ValueError("training and test shapes overlap")


def device_key(preflight: dict) -> dict:
    device = preflight["device"]
    return {
        key: device[key]
        for key in (
            "name",
            "compute_capability",
            "torch_version",
            "triton_version",
            "cuda_version",
        )
    }


def collect(directory: Path, repetitions: int = 7, batch_size: int = 256) -> dict:
    validate_split(TRAIN_SHAPES, TEST_SHAPES)
    if repetitions < 3 or batch_size < 1:
        raise ValueError("invalid timing settings")
    preflight = triton_preflight()
    if not preflight["available"]:
        raise RuntimeError(str(preflight))
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    data = {
        "training_shapes": TRAIN_SHAPES,
        "test_shapes": TEST_SHAPES,
        "feature_version": FEATURE_VERSION,
        "correctness_reference": "FP32 matmul, TF32 disabled, rounded to FP16; rtol=atol=0.01",
        "device_key": device_key(preflight),
        "timing": {
            "method": "cuda_graph",
            "repetitions": repetitions,
            "batch_size": batch_size,
        },
        "sampling": "16 curated plus 16 uniform non-curated candidates per shape, shuffled",
        "cases": [],
    }
    write_json(directory / "dataset.json", data)
    with (directory / "trials.jsonl").open("w") as log:
        for index, shape in enumerate(TRAIN_SHAPES):
            workload = Workload(KernelKind.MATMUL, shape)
            bench = TritonBenchmark(
                seed=20_000 + index,
                repetitions=repetitions,
                graph_batch_size=batch_size,
            )
            baseline = bench.evaluate(workload, FIXED)
            if not baseline.usable:
                raise RuntimeError(f"training baseline failed: {baseline.detail}")
            curated = curated_configs()
            remaining = [
                config
                for config in candidate_universe(workload)
                if config not in curated
            ]
            rng = random.Random(101 + index)
            configs = curated + rng.sample(remaining, 16)
            rng.shuffle(configs)
            case = {
                "shape": shape,
                "seed": 20_000 + index,
                "baseline": json_safe(baseline),
                "telemetry_before": telemetry(),
                "trials": [],
            }
            data["cases"].append(case)
            for trial_index, config in enumerate(configs):
                trial_start = perf_counter()
                measured = bench.evaluate(workload, config)
                row = {
                    "config": asdict(config),
                    "measurement": json_safe(measured),
                    "reward": latency_reward(measured, baseline.latency_us),
                    "wall_s": perf_counter() - trial_start,
                }
                case["trials"].append(row)
                log.write(json.dumps({"shape": shape, **row}, allow_nan=False) + "\n")
                log.flush()
                write_json(directory / "dataset.json", data)
                if (trial_index + 1) % 8 == 0:
                    print(
                        f"training {index + 1}/{len(TRAIN_SHAPES)} {shape}: {trial_index + 1}/32",
                        flush=True,
                    )
            case["telemetry_after"] = telemetry()
    data["wall_s"] = perf_counter() - started
    data["complete"] = True
    write_json(directory / "dataset.json", data)
    return data


def case_arrays(case: dict) -> tuple[np.ndarray, np.ndarray]:
    workload = Workload(KernelKind.MATMUL, tuple(case["shape"]))
    x = np.array(
        [
            context_features(workload, ScheduleConfig(**trial["config"]))
            for trial in case["trials"]
        ]
    )
    y = np.array([trial["reward"] for trial in case["trials"]])
    return x, y


def fit_cases(cases: list[dict], alpha: float, ridge: float) -> LinearUCB:
    if not cases:
        raise ValueError("training cases required")
    dimension = len(case_arrays(cases[0])[0][0])
    model = LinearUCB(dimension, alpha, ridge)
    for case in cases:
        x, y = case_arrays(case)
        for features, reward in zip(x, y):
            model.update(features, float(reward))
    return model


def fit(dataset_path: Path, model_path: Path) -> dict:
    data = json.loads(dataset_path.read_text())
    if not data.get("complete"):
        raise ValueError("training collection is incomplete")
    validate_split(data["training_shapes"], data["test_shapes"])
    if [case["shape"] for case in data["cases"]] != data["training_shapes"]:
        raise ValueError("dataset shapes do not match the declared split")
    validation = []
    for ridge in (0.01, 0.1, 1.0):
        for alpha in (0.0, 0.05, 0.15, 0.5):
            fold_rewards = []
            for index, held_out in enumerate(data["cases"]):
                training = [case for j, case in enumerate(data["cases"]) if j != index]
                model = fit_cases(training, alpha, ridge)
                x, rewards = case_arrays(held_out)
                available = np.ones(len(x), dtype=bool)
                best = -6.0
                fold_rng = np.random.default_rng(700 + index)
                for _ in range(min(16, len(x))):
                    action = model.select(x, available, fold_rng)
                    available[action] = False
                    model.update(x[action], float(rewards[action]))
                    best = max(best, float(rewards[action]))
                fold_rewards.append(best)
            validation.append(
                {
                    "ridge": ridge,
                    "alpha": alpha,
                    "mean_best_log_speedup": float(np.mean(fold_rewards)),
                    "fold_best_log_speedups": fold_rewards,
                }
            )
    chosen = max(validation, key=lambda row: row["mean_best_log_speedup"])
    model = fit_cases(data["cases"], chosen["alpha"], chosen["ridge"])
    metadata = {
        key: data[key]
        for key in (
            "training_shapes",
            "test_shapes",
            "device_key",
            "timing",
            "wall_s",
            "correctness_reference",
        )
    }
    metadata.update(
        {
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "feature_version": FEATURE_VERSION,
            "validation": validation,
            "validation_scope": "leave-one-training-shape-out, measured 32-candidate pool, budget16",
            "selected_hyperparameters": {
                "alpha": chosen["alpha"],
                "ridge": chosen["ridge"],
            },
            "training_trials": model.observations,
            "reference_evaluations": len(data["cases"]),
        }
    )
    model.save(model_path, metadata)
    print(
        f"saved {model_path}: {model.observations} observations, alpha={model.alpha}, ridge={model.ridge}",
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect training measurements and fit a contextual bandit."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    collection = sub.add_parser("collect")
    collection.add_argument("--output-dir", type=Path, required=True)
    collection.add_argument("--repetitions", type=int, default=7)
    collection.add_argument("--batch-size", type=int, default=256)
    training = sub.add_parser("fit")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output_dir, args.repetitions, args.batch_size)
    else:
        fit(args.dataset, args.model)


if __name__ == "__main__":
    main()
