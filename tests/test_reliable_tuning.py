import math
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock

from rl_gpu_autotuner import triton_backend
from rl_gpu_autotuner.benchmark import Benchmark, Measurement
from rl_gpu_autotuner.domain import KernelKind, ScheduleConfig, Workload
from rl_gpu_autotuner.environment import TuningEnvironment
from rl_gpu_autotuner.policies import random_search


class SequenceBenchmark(Benchmark):
    def __init__(self, measurements):
        self.measurements = iter(measurements)

    def evaluate(self, workload, config):
        return next(self.measurements)


class ReliableTuningTests(unittest.TestCase):
    def setUp(self):
        self.workload = Workload(KernelKind.MATMUL, (128, 256, 64))
        self.configs = [ScheduleConfig(32, 32, 32, 4, 2), ScheduleConfig(64, 64, 32, 4, 3)]

    def test_graph_timer_reports_per_call_units_and_replays_without_python_launches(self):
        torch = MagicMock()
        torch.cuda.device.return_value = nullcontext()
        torch.cuda.stream.return_value = nullcontext()
        torch.cuda.graph.return_value = nullcontext()
        torch.cuda.Event.return_value.elapsed_time.return_value = 2.0
        launch = MagicMock()
        samples = triton_backend.time_cuda_callable(
            launch, torch, "cuda:0", warmup=2, repetitions=3, graph_batch_size=100
        )
        self.assertEqual(samples, (20.0, 20.0, 20.0))
        # Warm-up and capture invoke Python; measurement only replays the graph.
        self.assertEqual(launch.call_count, 102)
        self.assertGreaterEqual(torch.cuda.CUDAGraph.return_value.replay.call_count, 3)

    def test_timer_rejects_bad_settings_before_touching_runtime(self):
        for kwargs in ({"graph_batch_size": 0}, {"repetitions": 2}, {"timing_method": "bad"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                triton_backend.time_cuda_callable(MagicMock(), MagicMock(), "cuda", **kwargs)

    def test_search_filters_failed_and_nonfinite_results_and_keeps_full_history(self):
        benchmark = SequenceBenchmark([Measurement(0.1, valid=False), Measurement(5.0)])
        result = random_search(benchmark, self.workload, self.configs, 2)
        self.assertEqual(result.measurement.latency_us, 5.0)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.evaluations, 2)
        with self.assertRaisesRegex(RuntimeError, "valid"):
            random_search(SequenceBenchmark([Measurement(math.nan), Measurement(math.inf)]),
                          self.workload, self.configs, 2)

    def test_environment_has_context_history_masks_and_scale_invariant_reward(self):
        for scale in (1.0, 100.0):
            env = TuningEnvironment(SequenceBenchmark([Measurement(8 * scale)]),
                                    self.workload, self.configs, 2, baseline_latency_us=10 * scale)
            observation, reward, done, info = env.step(0)
            self.assertEqual(observation["shape"], self.workload.shape)
            self.assertEqual(observation["action_mask"], (False, True))
            self.assertEqual(len(observation["history"]), 1)
            self.assertEqual(observation["best_config"], self.configs[0])
            self.assertAlmostEqual(reward, math.log(10 / 8))
            self.assertFalse(done)
            with self.assertRaises(ValueError):
                env.step(0)
            self.assertEqual(env.evaluations, 1)

    def test_invalid_measurement_spends_budget_without_improving_incumbent(self):
        env = TuningEnvironment(SequenceBenchmark([Measurement(0.1, valid=False), Measurement(math.nan)]),
                                self.workload, self.configs, 2, baseline_latency_us=10)
        for action in (0, 1):
            _, reward, done, info = env.step(action)
            self.assertLess(reward, 0)
            self.assertEqual(env.best_latency, 10)
        self.assertTrue(done)

    def test_environment_rejects_empty_or_invalid_budget_and_baseline(self):
        for budget, candidates, baseline in ((0, self.configs, 10), (1, [], 10), (1, self.configs, math.nan)):
            with self.assertRaises(ValueError):
                TuningEnvironment(SequenceBenchmark([]), self.workload, candidates, budget,
                                  baseline_latency_us=baseline)

    def test_schedule_checks_and_grouping_parameter(self):
        self.assertFalse(ScheduleConfig(17, 32, 32, 4, 2).is_legal())
        self.assertTrue(ScheduleConfig(128, 256, 64, 8, 3, group_size_m=4).is_legal())
        self.assertFalse(ScheduleConfig(32, 32, 32, 4, 2, group_size_m=0).is_legal())


if __name__ == "__main__":
    unittest.main()
