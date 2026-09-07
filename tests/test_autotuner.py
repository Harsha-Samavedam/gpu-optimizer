import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rl_gpu_autotuner.benchmark import SimulatedBenchmark
from rl_gpu_autotuner.domain import KernelKind, ScheduleConfig, Workload
from rl_gpu_autotuner.environment import TuningEnvironment
from rl_gpu_autotuner.policies import exhaustive_search, random_search
from rl_gpu_autotuner.search_space import configs_for
from rl_gpu_autotuner.triton_backend import triton_preflight


class AutotunerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = Workload(KernelKind.MATMUL, (512, 512, 512))
        self.candidates = configs_for(self.workload)
        self.benchmark = SimulatedBenchmark(seed=13)

    def test_workload_dimension_validation(self) -> None:
        with self.assertRaises(ValueError):
            Workload(KernelKind.MATMUL, (1, 2))

    def test_search_space_is_legal(self) -> None:
        self.assertTrue(self.candidates)
        self.assertTrue(all(config.is_legal() for config in self.candidates))
        self.assertFalse(ScheduleConfig(16, 16, 16, 16, 1).is_legal())

    def test_exhaustive_is_no_worse_than_same_budget_random(self) -> None:
        random_result = random_search(self.benchmark, self.workload, self.candidates, budget=12, seed=4)
        oracle = exhaustive_search(self.benchmark, self.workload, self.candidates)
        self.assertLessEqual(oracle.measurement.latency_us, random_result.measurement.latency_us)

    def test_environment_enforces_budget(self) -> None:
        env = TuningEnvironment(self.benchmark, self.workload, self.candidates, budget=2)
        _, _, done, _ = env.step(0)
        self.assertFalse(done)
        _, _, done, _ = env.step(1)
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            env.step(2)

    def test_triton_preflight_is_safe_without_optional_dependencies(self) -> None:
        result = triton_preflight()
        self.assertIn("available", result)

    def test_cli_writes_simulated_results(self) -> None:
        # Keep the migration-time result format testable without CUDA/Triton.
        from unittest.mock import patch
        from rl_gpu_autotuner import cli

        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "results.json"
            with patch(
                "sys.argv",
                [
                    "rl-gpu-autotune",
                    "compare",
                    "--shape",
                    "64,64,64",
                    "--budget",
                    "2",
                    "--results-json",
                    str(result_path),
                ],
            ):
                cli.main()
            self.assertIn('"random"', result_path.read_text())


if __name__ == "__main__":
    unittest.main()
