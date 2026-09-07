# RL-GPU-Autotuner

A learning-focused project for **budget-aware, shape-aware GPU kernel autotuning**. The end goal is to use reinforcement learning to choose efficient Triton kernel launch/schedule configurations under an expensive measurement budget.

The repository works without a GPU today. It starts with a deterministic simulator that behaves like a noisy GPU benchmark; that lets us build and test the search/RL plumbing before replacing it with real Triton timings.

## Project path

1. **Now — simulator and baselines:** define valid schedule configurations, benchmark them, and compare random search with exhaustive search.
2. **GPU available — real measurements:** add Triton kernels and a `TritonBenchmark` backend; preserve the same public interface.
3. **RL:** train a contextual-bandit policy first, then a sequential policy (e.g. PPO) for dependent choices such as tile size → warps → pipeline stages.
4. **Evaluation:** use equal tuning budgets, correctness checks, latency distributions, and held-out tensor shapes.

## Quick start (no GPU required)

```bash
python -m pip install -e .
python -m rl_gpu_autotuner.cli compare --kernel matmul --shape 1024,1024,1024 --budget 12
python -m unittest discover -s tests -v
```

`compare` uses the simulator unless you explicitly add a real backend. The reported time is a synthetic latency in microseconds, so it is useful for tests and algorithm development—not performance claims.

## GPU-machine setup and first real comparison

The first hardware milestone is **FP16 matrix multiplication on an NVIDIA CUDA GPU**. On the GPU machine, install a CUDA-compatible PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/), install Triton, then install this project:

```bash
python -m pip install triton
python -m pip install -e .
python -m rl_gpu_autotuner.cli preflight
```

`preflight` must report `"available": true` and identify the GPU before benchmarking. Then run a budget-matched first comparison:

```bash
python -m rl_gpu_autotuner.cli compare \
  --backend triton \
  --kernel matmul \
  --shape 1024,1024,1024 \
  --dtype fp16 \
  --budget 16 \
  --results-json results/first_gpu_run.json
```

This measures a PyTorch/cuBLAS reference, a fixed Triton configuration, and random search under the requested budget. Add `--oracle` only after the basic run works: it compiles and evaluates every candidate and can take a long time.

## Architecture

```text
Workload(shape, dtype) + ScheduleConfig
              │
              ▼
       Benchmark backend
       ├── SimulatedBenchmark (now)
       └── TritonBenchmark    (future)
              │
              ▼
          Search policy
       ├── random baseline
       ├── exhaustive oracle
       └── RL environment/policy
```

## Definition of success for v0.1

- The search space rejects configurations that violate basic resource constraints.
- Searches obey a fixed benchmark budget.
- The simulator is deterministic for a seed but returns controlled measurement noise.
- A future hardware backend can be added without changing policies or result logging.

## Next coding milestone

The FP16 Triton matmul kernel and `TritonBenchmark.evaluate` are implemented. Each evaluation compiles outside the timed section, validates against PyTorch, warms up, records repeated CUDA-event timings, and returns median latency. Do not train RL until fixed/random/exhaustive baselines are working on real measurements.
