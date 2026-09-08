# Contextual-Bandit GPU Kernel Autotuner

A small Triton autotuning project for FP16 matrix multiplication. Given a matrix shape and a 16-trial measurement budget, the tuner chooses tile sizes, warp counts, pipeline stages, and program-group settings to benchmark.

The learned method is a linear contextual bandit with online updates. It is not a full sequential RL policy.

## What is included

- FP16 Triton matmul kernel with configurable schedules.
- CUDA Graph timing, FP32-reference correctness checks, and repeated confirmation runs.
- Fixed, random-search, shortlist, and contextual-bandit baselines.
- A saved RTX 3070 bandit model trained on 320 measurements from ten shapes.
- CPU tests for search budgets, feature construction, online ridge updates, timing, and result handling.

## Setup

Requires Python 3.10+. GPU runs require Linux, a supported NVIDIA GPU, CUDA-enabled PyTorch, and Triton.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install triton numpy
python -m rl_gpu_autotuner.cli preflight
```

## Run

```bash
# CPU simulator
python -m rl_gpu_autotuner.cli compare --shape 1024,1024,1024 --budget 16

# GPU comparison
python -m rl_gpu_autotuner.cli compare \
  --backend triton --shape 1024,1024,1024 --budget 16 \
  --results-json results/comparison.json

# Train and evaluate the bandit
OPENBLAS_NUM_THREADS=1 python -m rl_gpu_autotuner.train_bandit collect --output-dir results/training
OPENBLAS_NUM_THREADS=1 python -m rl_gpu_autotuner.train_bandit fit \
  --dataset results/training/dataset.json --model models/bandit.json
OPENBLAS_NUM_THREADS=1 python -m rl_gpu_autotuner.experiments \
  --output-dir results/bandit --bandit-model models/bandit.json
```

## Results

On an RTX 3070 with PyTorch 2.14 and Triton 3.8, each method received 16 search trials. Latencies are microseconds; values are medians of repeated, independently confirmed measurements. The percentages use paired measurements, so they do not always match ratios of displayed medians.

| Shape (M,N,K) | PyTorch | Random | Bandit | Bandit vs. PyTorch | Bandit vs. random |
|---|---:|---:|---:|---:|---:|
| 512x512x512 | 12.918 | 12.670 | 10.740 | 16.80% faster | 14.41% faster |
| 1024x1024x1024 | 86.322 | 72.438 | 66.682 | 22.78% faster | 6.13% faster |
| 512x2048x1024 | 127.808 | 70.420 | 66.676 | 47.87% faster | 5.60% faster |
| 1000x769x513 | 56.832 | 56.478 | 44.764 | 21.17% faster | 20.75% faster |
| 640x960x768 | 35.170 | 35.926 | 35.614 | 1.09% slower | 1.31% faster |
| 896x1408x640 | 67.334 | 55.560 | 54.202 | 19.51% faster | 10.62% faster |
| 1023x1025x767 | 84.066 | 101.098 | 96.694 | 15.06% slower | 3.71% faster |
| 2048x256x1024 | 46.246 | 38.504 | 39.000 | 15.76% faster | 1.25% slower |

The bandit beat PyTorch on six of eight shapes and random search on seven of eight. Results are specific to this GPU, runtime, kernel, candidate space, and hot-buffer timing setup. Full methodology, limitations, and raw exports are in [the results report](outputs/contextual_bandit_2026-09-07.md).

## Tests

```bash
python -m unittest discover -s tests -v
```
