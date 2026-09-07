# RL-GPU-Autotuner

GPU kernel autotuning with Triton. The project benchmarks FP16 matrix multiplication and compares fixed configurations, random search, and a hand-picked shortlist under a fixed trial budget.

It includes a simulator for CPU-only development and a tuning environment for future RL experiments. No learned policy has been trained yet.

## Setup

Requires Python 3.10 or newer. GPU benchmarks also need Linux, a supported NVIDIA GPU and driver, CUDA-enabled PyTorch, Triton, a C compiler, and Python development headers.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For GPU support, install PyTorch using the [official installation instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install triton numpy
python -m rl_gpu_autotuner.cli preflight
```

Preflight checks runtime availability. If Triton compilation reports a missing `Python.h`, install the development headers matching your Python version.

## Usage

Run a simulated comparison:

```bash
python -m rl_gpu_autotuner.cli compare --shape 1024,1024,1024 --budget 16
```

Run on the GPU:

```bash
python -m rl_gpu_autotuner.cli compare \
  --backend triton --shape 1024,1024,1024 --budget 16 \
  --results-json results/comparison.json
```

Shapes are specified as `M,N,K` for `A[M,K] @ B[K,N]`. Add `--oracle` to evaluate the full search space.

Run the repeated experiment suite or check the timing method:

```bash
python -m rl_gpu_autotuner.experiments --output-dir results/suite
python -m rl_gpu_autotuner.experiments --output-dir results/timing --diagnostic-only
```

The suite defaults to four shapes, three search seeds, 16 trials per search, and five confirmation rounds. Choose a new output directory for each run. Results include all trials, selected configurations, timing samples, and GPU telemetry.

## Benchmarks

Both Triton and PyTorch use CUDA Graph batches to reduce Python launch overhead. Timings measure repeated operations on hot buffers, with compilation and correctness checks outside the timed region. Selected configurations are remeasured separately in randomized order.

Results from an RTX 3070 with PyTorch 2.14 and Triton 3.8, in microseconds:

| Shape (M,N,K) | PyTorch | Fixed Triton | Random (16 trials) | Shortlist (16 trials) |
|---|---:|---:|---:|---:|
| 512,512,512 | 11.50 | 11.71 | 11.79 | 9.86 |
| 1024,1024,1024 | 87.99 | 71.92 | 71.99 | 67.74 |
| 512,2048,1024 | 131.41 | 72.72 | 72.59 | 68.66 |
| 1000,769,513 | 56.79 | 55.54 | 55.45 | 55.53 |

Values are medians of per-seed confirmation medians. These are throughput measurements on a GPU also running a desktop; small differences need further validation. See the [experiment report](outputs/reliable_gpu_experiments_2026-09-07.md) for the protocol and limitations.

## Development

```bash
python -m unittest discover -s tests -v
```

Next steps are held-out shape evaluation, a cost-model or contextual-bandit baseline, and tuning budgets based on elapsed time.
