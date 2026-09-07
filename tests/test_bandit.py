import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from rl_gpu_autotuner.bandit import LinearUCB, context_features, latency_reward
from rl_gpu_autotuner.benchmark import Measurement
from rl_gpu_autotuner.domain import KernelKind, ScheduleConfig, Workload


class BanditTests(unittest.TestCase):
    def test_online_update_matches_batch_ridge_solution(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(40, 5))
        y = rng.normal(size=40)
        model = LinearUCB(5, ridge=0.5)
        for features, reward in zip(x, y):
            model.update(features, reward)
        expected_a = 0.5 * np.eye(5) + x.T @ x
        np.testing.assert_allclose(model.inverse, np.linalg.inv(expected_a), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(model.weights, np.linalg.solve(expected_a, x.T @ y), rtol=1e-10)

    def test_context_changes_the_preferred_action(self):
        model = LinearUCB(2, alpha=0)
        for _ in range(20):
            model.update(np.array([1., 0.]), 1)
            model.update(np.array([0., 1.]), -1)
        rng = np.random.default_rng(4)
        self.assertEqual(model.select(np.eye(2), np.ones(2, dtype=bool), rng), 0)
        self.assertEqual(model.select(np.eye(2)[::-1], np.ones(2, dtype=bool), rng), 1)

    def test_exploration_and_masking(self):
        model = LinearUCB(2, alpha=1)
        model.update(np.array([1., 0.]), 0)
        rng = np.random.default_rng(7)
        self.assertEqual(model.select(np.eye(2), np.array([True, True]), rng), 1)
        self.assertEqual(model.select(np.eye(2), np.array([True, False]), rng), 0)
        with self.assertRaises(ValueError):
            model.select(np.eye(2), np.zeros(2, dtype=bool), rng)

    def test_snapshot_roundtrip_and_episode_isolation(self):
        model = LinearUCB(2, alpha=.2)
        model.update(np.array([1., 2.]), .4)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            model.save(path, {'training_shapes': [[128, 128, 128]]})
            loaded, metadata = LinearUCB.load(path)
            np.testing.assert_allclose(loaded.weights, model.weights)
            self.assertEqual(metadata['training_shapes'], [[128, 128, 128]])
            copied = loaded.copy()
            copied.update(np.array([1., 0.]), 3)
            np.testing.assert_allclose(loaded.weights, model.weights)
            self.assertFalse(np.allclose(copied.weights, loaded.weights))

    def test_reward_is_scale_invariant_and_failures_are_penalized(self):
        self.assertAlmostEqual(latency_reward(Measurement(8), 10), math.log(10/8))
        self.assertAlmostEqual(latency_reward(Measurement(800), 1000), math.log(10/8))
        for measurement in (Measurement(.01, valid=False), Measurement(math.inf), Measurement(math.nan), Measurement(0)):
            self.assertEqual(latency_reward(measurement, 10), -6)
        with self.assertRaises(ValueError):
            latency_reward(Measurement(1), 0)

    def test_features_cover_workload_config_and_tail_dimensions(self):
        config = ScheduleConfig(64, 128, 32, 4, 3)
        aligned = context_features(Workload(KernelKind.MATMUL, (1024, 1024, 1024)), config)
        irregular = context_features(Workload(KernelKind.MATMUL, (1000, 769, 513)), config)
        changed = context_features(Workload(KernelKind.MATMUL, (1024, 1024, 1024)), ScheduleConfig(32, 64, 64, 4, 3))
        self.assertTrue(np.isfinite(aligned).all())
        self.assertEqual(aligned.shape, irregular.shape)
        self.assertFalse(np.allclose(aligned, irregular))
        self.assertFalse(np.allclose(aligned, changed))
        with self.assertRaises(ValueError):
            context_features(Workload(KernelKind.MATMUL, (1, 1, 1)), ScheduleConfig(17, 32, 32, 4, 2))

    def test_rejects_bad_dimensions_and_nonfinite_updates(self):
        for kwargs in ({'dimension': 0}, {'dimension': 2, 'alpha': -1}, {'dimension': 2, 'ridge': 0}):
            with self.assertRaises(ValueError):
                LinearUCB(**kwargs)
        model = LinearUCB(2)
        for x, y in ((np.array([1.]), 1), (np.array([1., math.nan]), 1), (np.ones(2), math.inf)):
            with self.assertRaises(ValueError):
                model.update(x, y)
        self.assertEqual(model.observations, 0)
