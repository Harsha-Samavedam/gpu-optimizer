# RL-GPU-Autotuner

A learning-focused project for **budget-aware, shape-aware GPU kernel autotuning**. The end goal is to use reinforcement learning to choose efficient Triton kernel launch/schedule configurations under an expensive measurement budget.

The repository supports both a CPU-only simulator and a real FP16 Triton matmul backend. GPU measurements use CUDA Graph batches to reduce host launch overhead, with full trial logs and independent confirmation of selected configurations.

## Project path

1. **Implemented — simulator and baselines:** define valid schedule configurations, benchmark them, and compare random search with exhaustive search.
2. **Now — real measurements:** validate timing and compare fixed, random, and curated schedules across shapes and seeds on an RTX 3070.
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
       └── TritonBenchmark    (CUDA Graph batch timing)
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

The FP16 Triton matmul kernel and `TritonBenchmark.evaluate` are implemented. Each evaluation compiles outside timing, validates against PyTorch, warms up, and measures repeated CUDA Graph replays. Each graph contains 512 launches by default; elapsed time is divided by the batch size. The result is the median of per-batch means, measuring hot-buffer throughput rather than single-request latency. PyTorch uses the same timer. Use `--timing-method events` only to diagnose the legacy host-gap-sensitive method.

The next milestone is a larger, held-out shape dataset and a cost-model/contextual-bandit baseline. The old exhaustive result below used the legacy timer and is not a corrected performance oracle.

## Repeated experiment suite

The completed 2026-09-07 study covers 12 cases, 384 search trials, and 240
confirmation measurements. See the [experiment report](outputs/reliable_gpu_experiments_2026-09-07.md)
for the timing diagnostic, measured comparisons, MDP changes, and limitations.
For 1024-square FP16 matmul, the curated winner measured about 23% lower latency
than PyTorch and 6% lower than fixed Triton under the new hot-buffer graph protocol.
This is not a direct optimization delta from the legacy 8% result.

After the local environment setup below, run:

```bash
python -m rl_gpu_autotuner.experiments --output-dir results/my_suite
python -m rl_gpu_autotuner.experiments --output-dir results/my_timer_check --diagnostic-only
```

Output directories must be new. Defaults are four shapes, three search seeds,
16 trials each for random and curated search, and five confirmation rounds.
Both searches use the same candidate universe; the curated baseline prioritizes
16 predefined configurations, including larger tiles and alternate `group_size_m`.
Search measurements are interleaved and every trial is logged, including failures.
Winners are frozen before confirmation on new inputs, with randomized method order.
Confirmation costs are separate from tuning budgets. JSON records include raw batch
samples, selected configurations, failure details, timings, runtime versions, and GPU telemetry.

## Tuning environment

Observations now contain shape/dtype/kernel, supplied device features, candidate
configurations, full trial history, incumbent latency/configuration, budget, and
an action mask excluding already measured configurations. Invalid or nonfinite
measurements spend a trial and receive a penalty without changing the incumbent.

With an independently measured fixed baseline, positive rewards are
`log(previous_best / new_best)`, making improvements comparable across latency
scales. Baseline measurement cost must be accounted for separately and equally
across policies. Without a baseline, positive reward is terminal-only
`1 / (1 + best_latency_us)`; a deliberately slow first trial cannot inflate it.
Use the baseline-normalized mode for cross-shape learning. This is still an
environment scaffold, not a trained policy or a Gymnasium adapter.

## Local GPU setup (verified 2026-09-07)

This machine has an NVIDIA GeForce RTX 3070 (8 GB, compute capability 8.6),
driver 595.71.05, Ubuntu 26.04, an i7-11700KF, and approximately 30 GiB RAM.
The project-local `.venv` contains Python 3.14.4, PyTorch 2.14.0+cu130,
Triton 3.8.0, and NumPy. GPU preflight passes when run with GPU device access;
the Codex filesystem sandbox hides the NVIDIA devices.

Triton's launcher also needs Python development headers. Because system package
installation required interactive sudo authentication, the matching Ubuntu
`libpython3.14-dev=3.14.4-1` package was downloaded and extracted into
`.venv/python-dev`. Use these local headers when compiling:

```bash
source .venv/bin/activate
export CPATH="$PWD/.venv/python-dev/usr/include/python3.14:$PWD/.venv/python-dev/usr/include"
python -m rl_gpu_autotuner.cli preflight
python -m rl_gpu_autotuner.cli compare --backend triton --kernel matmul \
  --shape 1024,1024,1024 --dtype fp16 --budget 16 \
  --results-json results/first_gpu_run.json
```

The first real run completed with valid fixed and selected random configurations.
Its median latencies were 91.136 us for PyTorch, 84.928 us for fixed Triton,
and 86.016 us for the best of 16 random candidates (seed 0). Raw samples are in
`results/first_gpu_run.json`; installed versions are in `results/gpu_environment.txt`.
The subsequent exhaustive run completed all 768 evaluations and selected a valid
configuration with blocks `(64, 128, 32)`, 4 warps, and 3 stages at 82.944 us.
That run measured PyTorch at 90.112 us, fixed Triton at 84.992 us, and random
search at 86.016 us; its samples are in `results/first_gpu_oracle.json`.
These files and `.venv` are local, ignored artifacts.

These legacy results are retained as historical smoke tests; the earlier 8%
comparison must not be treated as a validated optimization gain. Consult the
repeated CUDA Graph experiments for the current comparison. The GPU still runs
the desktop, so clocks, temperature, contention, and run-to-run variability remain
part of the interpretation. Neither the driver nor GPU clock settings were changed.
