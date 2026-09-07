import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rl_gpu_autotuner import experiments
from rl_gpu_autotuner.benchmark import Measurement
from rl_gpu_autotuner.domain import KernelKind, Workload


class ExperimentTests(unittest.TestCase):
    def test_strict_json_keeps_failure_detail_and_replaces_nonfinite_numbers(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            experiments.write_json(
                path,
                {"failed": Measurement(math.inf, valid=False, detail="resource limit")},
            )
            result = json.loads(
                path.read_text(), parse_constant=lambda value: self.fail(value)
            )
            self.assertIsNone(result["failed"]["latency_us"])
            self.assertEqual(result["failed"]["detail"], "resource limit")

    def test_curated_search_stays_in_shared_candidate_universe(self):
        workload = Workload(KernelKind.MATMUL, (1024, 1024, 1024))
        curated = experiments.curated_configs()
        universe = experiments.candidate_universe(workload)
        self.assertEqual(len(curated), 16)
        self.assertEqual(len(set(curated)), 16)
        self.assertTrue(all(c.is_legal() and c in universe for c in curated))
        self.assertTrue(any(c.block_n == 256 for c in curated))
        self.assertTrue(any(c.group_size_m == 1 for c in curated))

    def test_summary_uses_independent_confirmation_and_paired_speedups(self):
        case = {
            "confirmation": [
                {
                    "measurements": {
                        "pytorch": {"latency_us": 10, "valid": True},
                        "random": {"latency_us": 5, "valid": True},
                    }
                },
                {
                    "measurements": {
                        "pytorch": {"latency_us": 20, "valid": True},
                        "random": {"latency_us": 10, "valid": True},
                    }
                },
            ]
        }
        summary = experiments.summarize_case(case)
        self.assertEqual(summary["random"]["median_us"], 7.5)
        self.assertEqual(summary["random"]["paired_speedup_vs_pytorch"], 2)
        case["confirmation"][0]["measurements"]["random"]["valid"] = False
        self.assertFalse(experiments.summarize_case(case)["random"]["all_valid"])
