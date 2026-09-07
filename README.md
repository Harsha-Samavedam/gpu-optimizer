# RL-GPU-Autotuner

GPU kernel autotuning with Triton. The project benchmarks FP16 matrix multiplication and compares fixed configurations, random search, a hand-picked shortlist, and a learned contextual bandit under a fixed trial budget.

It includes a CPU simulator and a budgeted tuning environment. The bandit is trained on real GPU measurements; a full sequential RL policy has not been trained.

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

Train and evaluate the contextual bandit:

```bash
OPENBLAS_NUM_THREADS=1 python -m rl_gpu_autotuner.train_bandit collect \
  --output-dir results/training
OPENBLAS_NUM_THREADS=1 python -m rl_gpu_autotuner.train_bandit fit \
  --dataset results/training/dataset.json --model models/bandit.json
OPENBLAS_NUM_THREADS=1 python -m rl_gpu_autotuner.experiments \
  --output-dir results/bandit --bandit-model models/bandit.json
```

The saved [RTX 3070 model](models/linucb_fp16_rtx3070.json) uses 320 measurements from ten training shapes. Training-only cross-validation selects the ridge penalty and exploration bonus. Evaluation rejects overlapping training shapes and a different GPU/runtime. Each evaluation starts from the saved model, updates only from its own trials, and also confirms the initial recommendation separately. The runner records the extra fixed reference measurement and tuning time.

## Benchmarks

Both Triton and PyTorch use CUDA Graph batches to reduce Python launch overhead. Timings measure repeated operations on hot buffers, with compilation and correctness checks outside the timed region. Selected configurations are remeasured separately in randomized order.

Fresh RTX 3070 results with PyTorch 2.14 and Triton 3.8, in microseconds. Each search uses 16 trials:

| Shape (M,N,K) | PyTorch | Fixed | Random | Shortlist | Bandit |
|---|---:|---:|---:|---:|---:|
| 512×512×512 | 12.918 | 13.072 | 12.670 | 11.002 | 10.740 |
| 1024×1024×1024 | 86.322 | 69.774 | 72.438 | 66.664 | 66.682 |
| 512×2048×1024 | 127.808 | 70.376 | 70.420 | 66.590 | 66.676 |
| 1000×769×513 | 56.832 | 56.302 | 56.478 | 56.468 | 44.764 |
| 640×960×768 | 35.170 | 35.378 | 35.926 | 34.302 | 35.614 |
| 896×1408×640 | 67.334 | 54.128 | 55.560 | 54.078 | 54.202 |
| 1023×1025×767 | 84.066 | 99.840 | 101.098 | 93.394 | 96.694 |
| 2048×256×1024 | 46.246 | 38.346 | 38.504 | 38.342 | 39.000 |

Values are medians of three per-seed confirmation medians. The bandit improves the original irregular case substantially, but loses to the shortlist on three of four additional shapes. It does not consistently beat PyTorch. Small differences need further validation on a quiet GPU.

The [full report](outputs/contextual_bandit_2026-09-07.md) includes paired comparisons, first-choice results, costs, correctness details, and the decision about full RL. [Complete measurements](outputs/bandit_results/README.md) and the trained model are included in version control. The [earlier timing study](outputs/reliable_gpu_experiments_2026-09-07.md) is retained for reference.

## Development

```bash
python -m unittest discover -s tests -v
```

Next steps are broader training coverage, tuning budgets based on elapsed time, and a stop action when further measurements are unlikely to pay off.
