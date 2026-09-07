import unittest
from unittest.mock import MagicMock

from rl_gpu_autotuner import triton_backend


class ReferenceTests(unittest.TestCase):
    def test_fp32_reference_disables_tf32_and_restores_setting(self):
        torch = MagicMock()
        torch.backends.cuda.matmul.allow_tf32 = True
        a, b = MagicMock(), MagicMock()

        def matmul(left, right):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertIs(left, a.float.return_value)
            self.assertIs(right, b.float.return_value)
            return MagicMock()

        torch.matmul.side_effect = matmul
        triton_backend.matmul_reference(a, b, torch)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        torch.matmul.side_effect = RuntimeError("failure")
        with self.assertRaises(RuntimeError):
            triton_backend.matmul_reference(a, b, torch)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
