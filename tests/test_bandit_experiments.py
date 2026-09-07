import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from rl_gpu_autotuner import experiments, train_bandit
from rl_gpu_autotuner.bandit import LinearUCB, context_features
from rl_gpu_autotuner.benchmark import Measurement
from rl_gpu_autotuner.domain import KernelKind, Workload


class FakeBenchmark:
    calls = 0

    def __init__(self, **kwargs):
        pass

    def evaluate(self, workload, config):
        FakeBenchmark.calls += 1
        return Measurement(5 + config.block_m / 64)


class BanditExperimentTests(unittest.TestCase):
    def test_training_split_excludes_all_test_shapes(self):
        train_bandit.validate_split(train_bandit.TRAIN_SHAPES, train_bandit.TEST_SHAPES)
        with self.assertRaises(ValueError):
            train_bandit.validate_split([(64, 64, 64)], [(64, 64, 64)])

    def test_bandit_has_same_budget_unique_actions_and_no_prior_mutation(self):
        workload = Workload(KernelKind.MATMUL, (64, 96, 48))
        prior = LinearUCB(len(context_features(workload, experiments.FIXED)))
        old_inverse = prior.inverse.copy()
        FakeBenchmark.calls = 0
        with TemporaryDirectory() as directory, patch.object(experiments, 'TritonBenchmark', FakeBenchmark), patch.object(experiments, 'benchmark_torch_matmul', return_value=Measurement(10)), patch.object(experiments, 'telemetry', return_value={}):
            case = experiments.run_case(workload, 2, Path(directory)/'case', budget=3, rounds=3,
                                        repetitions=3, batch_size=2, bandit_model=prior)
        for method in ('random', 'curated', 'bandit'):
            self.assertEqual(case['search'][method]['evaluations'], 3)
            configs = [tuple(t['config'].values()) for t in case['search'][method]['history']]
            self.assertEqual(len(set(configs)), 3)
        self.assertEqual(case['confirmation_budget']['measurements_per_round'], 5)
        self.assertEqual(FakeBenchmark.calls, 1 + 9 + 3*4)
        self.assertEqual(prior.observations, 0)
        np.testing.assert_array_equal(prior.inverse, old_inverse)
        self.assertEqual(case['bandit_updates'], 3)
        self.assertEqual(len(case['budget_curve']['bandit']), 3)
