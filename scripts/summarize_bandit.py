import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from rl_gpu_autotuner.experiments import usable_record, write_json


def export_suite(directory: Path, output: Path, training_digest: str) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["bandit_metadata"]["dataset_sha256"] != training_digest:
        raise ValueError(f"training dataset does not match model in {directory}")
    expected = {
        (tuple(workload["shape"]), seed)
        for workload in manifest["shapes"]
        for seed in manifest["seeds"]
    }
    cases, trial_logs = [], []
    for path in sorted(directory.glob("*/result.json")):
        case = json.loads(path.read_text())
        if "summary" not in case or len(case["confirmation"]) != manifest["rounds"]:
            raise ValueError(f"incomplete case: {path}")
        cases.append(case)
        logs = [
            json.loads(line)
            for line in (path.parent / "trials.jsonl").read_text().splitlines()
        ]
        if {row["method"] for row in logs} != set(case["search"]):
            raise ValueError(f"trial log methods differ from case history: {path}")
        for method, search in case["search"].items():
            records = [row for row in logs if row["method"] == method]
            if [row["step"] for row in records] != list(
                range(1, manifest["budget"] + 1)
            ) or [row["trial"] for row in records] != search["history"]:
                raise ValueError(
                    f"trial log differs from case history: {path} {method}"
                )
        for row in logs:
            trial_logs.append(
                {
                    "shape": case["workload"]["shape"],
                    "seed": case["seed"],
                    **row,
                }
            )
    observed = {(tuple(c["workload"]["shape"]), c["seed"]) for c in cases}
    if observed != expected or len(cases) != len(expected):
        raise ValueError(f"missing or duplicate cases in {directory}")
    by_shape = defaultdict(list)
    for case in cases:
        by_shape[tuple(case["workload"]["shape"])].append(case)
        for search in case["search"].values():
            if len(search["history"]) != manifest["budget"]:
                raise ValueError("incomplete search budget")
    shapes = []
    for shape, group in by_shape.items():
        latencies = {}
        for method in group[0]["summary"]:
            if not all(c["summary"][method]["all_valid"] for c in group):
                raise ValueError(f"invalid confirmation for {shape} {method}")
            values = [c["summary"][method]["median_us"] for c in group]
            latencies[method] = {
                "median_us": median(values),
                "seed_min_us": min(values),
                "seed_max_us": max(values),
            }
        reductions = {}
        for baseline in ("pytorch", "fixed", "random", "curated", "bandit_first"):
            values = [
                median(
                    100
                    * (
                        1
                        - row["measurements"]["bandit"]["latency_us"]
                        / row["measurements"][baseline]["latency_us"]
                    )
                    for row in case["confirmation"]
                )
                for case in group
            ]
            reductions[baseline] = {
                "median_percent": median(values),
                "seed_min_percent": min(values),
                "seed_max_percent": max(values),
                "seed_values_percent": values,
            }
        shapes.append(
            {"shape": shape, "latencies": latencies, "bandit_reduction": reductions}
        )
    timings = {}
    for method in ("random", "curated", "bandit"):
        elapsed = [row["elapsed_s"] for row in trial_logs if row["method"] == method]
        decisions = [c["decision_seconds"][method] for c in cases]
        timings[method] = {
            "trial_wall_s": sum(elapsed),
            "median_trial_wall_s": median(elapsed),
            "decision_wall_s": sum(decisions),
            "median_decision_ms_per_trial": 1000
            * median(decisions)
            / manifest["budget"],
        }
    confirmations = [
        m for c in cases for r in c["confirmation"] for m in r["measurements"].values()
    ]
    search = [r["trial"]["measurement"] for r in trial_logs]
    references = [c["common_reference"] for c in cases]
    metrics = {
        "suite": directory.name,
        "cases": len(cases),
        "search_evaluations": len(search),
        "confirmation_evaluations": len(confirmations),
        "reference_evaluations": len(references),
        "invalid_measurements": sum(
            not usable_record(r) for r in search + confirmations + references
        ),
        "total_wall_s": sum(c["total_wall_s"] for c in cases),
        "setup_wall_s": sum(c["setup_wall_s"] for c in cases),
        "search_wall_s": sum(c["search_wall_s"] for c in cases),
        "method_costs": timings,
        "shapes": shapes,
    }
    write_json(
        output / f"{directory.name}.json",
        {"manifest": manifest, "cases": cases, "trial_logs": trial_logs},
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export complete bandit measurements and paired summaries."
    )
    parser.add_argument("--suite", type=Path, action="append", required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.training.read_text())
    if not data.get("complete"):
        raise ValueError("incomplete training dataset")
    digest = hashlib.sha256(args.training.read_bytes()).hexdigest()
    metrics = [
        export_suite(directory, args.output_dir, digest) for directory in args.suite
    ]
    write_json(args.output_dir / "training.json", data)
    write_json(args.output_dir / "metrics.json", metrics)
    for suite in metrics:
        print(
            suite["suite"],
            "cases:",
            suite["cases"],
            "invalid:",
            suite["invalid_measurements"],
        )
        for row in suite["shapes"]:
            print(
                row["shape"],
                {m: round(v["median_us"], 3) for m, v in row["latencies"].items()},
                "vs shortlist:",
                round(row["bandit_reduction"]["curated"]["median_percent"], 2),
            )


if __name__ == "__main__":
    main()
