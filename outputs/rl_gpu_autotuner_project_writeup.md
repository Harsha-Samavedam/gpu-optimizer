# RL-Guided GPU Kernel Autotuning

## Project write-up

### Abstract

This project investigates whether reinforcement learning can reduce the cost of GPU kernel autotuning. GPU kernels frequently expose several valid execution schedules—for example, different tile sizes, warp counts, and pipeline depths. The fastest schedule depends on the operation, tensor shape, data type, and target GPU. Exhaustively compiling and benchmarking every candidate is expensive, while simple heuristics may leave performance on the table.

The project builds a budget-aware tuning system for Triton kernels. Given a workload and a limited number of hardware measurements, the tuner selects which legal kernel configuration to benchmark next. Its performance will be compared against fixed heuristics, random search, exhaustive search where practical, and a non-RL search baseline. The goal is not to beat vendor libraries universally; it is to build a reproducible study of when learned policies can find good schedules efficiently.

### Motivation

Modern machine-learning programs spend much of their time in a relatively small set of GPU operations: matrix multiplication, reductions, softmax, normalization, and fused transformer kernels. These operations implement the same mathematics regardless of schedule, but their execution time can differ substantially depending on how work is mapped to the GPU.

For a matrix multiplication, a kernel must decide how much of the output matrix one GPU program computes, how the reduction dimension is chunked, and how much parallel and pipelined execution it uses. A configuration that is efficient for a large square FP16 matrix on one GPU may be poor for a small, rectangular, or differently typed workload on another.

Autotuning solves this by generating several kernel variants, measuring them on hardware, and retaining a fast one. However, real measurements are not free: candidates must be compiled or fetched from cache, validated for correctness, warmed up, and timed repeatedly. This makes the problem well suited to budgeted decision-making.

### Background: GPU execution and Triton

A GPU consists of many streaming multiprocessors (SMs). A kernel launch creates a grid of independent work units. Each unit is assigned to an SM, where it executes using groups of 32 threads called warps. Global GPU memory is large but relatively slow; registers and shared memory on an SM are fast but limited. Performance depends on balancing memory traffic, reuse, register/shared-memory pressure, and available parallel work.

Triton is a Python-based language and compiler for high-performance GPU kernels. Rather than writing from the perspective of an individual CUDA thread, a Triton programmer expresses a blocked algorithm: one program instance computes a block, or tile, of tensor work. Triton then maps that blocked work to threads and warps and performs GPU-specific code generation. Triton's programming model is designed to make blocked tensor algorithms practical while enabling compiler optimizations such as coalescing, vectorization, and tensor-core-aware instruction selection. [1]

Triton is therefore a useful target for this project. It exposes performance-relevant choices such as tile sizes, `num_warps`, and `num_stages`, but it avoids requiring hand-written GPU assembly. Triton's standard `autotune` mechanism already benchmarks user-provided configurations; this project examines whether a learned policy can choose trials more efficiently under a constrained measurement budget. [2]

### Scope

The first real kernel target is FP16 matrix multiplication, now implemented and benchmarked on an RTX 3070. The project will later extend to a row-wise reduction, softmax, and one fused operation such as bias-plus-activation or RMSNorm if time allows.

The initial action space for matmul is deliberately small and interpretable:

| Parameter | Example values | Role |
| --- | --- | --- |
| `BLOCK_M` | 16, 32, 64, 128 | Output rows computed by one Triton program |
| `BLOCK_N` | 16, 32, 64, 128 | Output columns computed by one Triton program |
| `BLOCK_K` | 16, 32, 64 | Reduction chunk per loop iteration |
| `num_warps` | 1, 2, 4, 8 | Hardware parallelism allocated to a program |
| `num_stages` | 1–4 | Memory/compute pipeline depth |

These parameters control important tradeoffs. Larger output tiles can increase reuse of values loaded from global memory, but they can consume more registers and reduce the number of active programs that fit on an SM. More warps or pipeline stages may hide memory latency, but can also increase resource pressure. The tuner does not change the numerical result or the GPU instruction set; it selects a legal schedule for the same algorithm.

### System design

```text
Workload + target-GPU features
            |
            v
      Policy / search strategy
            |
            v
  Candidate Triton schedule configuration
            |
            v
Compile/cache → correctness validation → warm-up → timing
            |
            v
     Measurement record and reward
            |
            +-------- feeds the next policy decision
```

The system is organized around a benchmark-backend interface. It makes search policies independent of the measurement source:

- `SimulatedBenchmark` provides a deterministic, noisy synthetic latency surface for development without GPU hardware.
- `TritonBenchmark` compiles, validates, and measures real Triton kernels using CUDA Graph batches on an NVIDIA GPU.
- Search policies consume the same workload, candidate, and measurement interfaces regardless of backend.

This separation makes the project testable before hardware access and prevents the RL/search logic from being tied to a single compiler implementation.

### Reinforcement-learning formulation

The final version is a finite-horizon, budgeted Markov decision process.

**Episode.** Tune one kernel workload on one target GPU with a fixed number of measurements, such as 16 or 32.

**State.** The implemented observation includes kernel kind, shape (`M`, `N`, `K`), dtype, supplied GPU features, remaining trial budget, current best latency/configuration, full prior configuration/measurement history, candidate configurations, and an action mask. The contextual bandit now uses 67 numeric workload/configuration features; a Gymnasium adapter remains future work.

**Action.** Choose the next legal configuration to benchmark. Invalid choices will be removed from the candidate set in advance rather than made an interesting part of the RL task.

**Transition.** The backend evaluates the configuration, updates benchmark history and best observed latency, and spends one budget unit.

**Reward.** The primary reward is improvement in the incumbent latency:

```text
reward_t = log(best_latency_before / best_latency_after)
```

The incumbent starts from an independently measured fixed baseline; log improvement is dimensionless and sums to log speedup when the discount factor is one. The baseline cost must be reported equally across policies. Invalid measurements spend budget and incur a penalty. Repeated actions are masked. Without an external baseline, the scaffold uses terminal-only reward `1 / (1 + best_latency_us)` so a slow initial action cannot inflate return; use fixed-baseline mode for cross-workload learning.

**Terminal condition.** The episode ends when the measurement budget is exhausted.

The implemented contextual bandit ranks complete schedules using workload, tile, and estimated resource features. It learns clipped log speedup against a separately measured fixed reference and updates after each of its own trials. Its parameters are specific to the training GPU/runtime. Both the first recommendation and the result after 16 trials are confirmed independently; this is not a trained full sequential RL policy.

### Why RL—and why not assume it wins?

The reward is a noisy, expensive, non-differentiable hardware measurement. A policy must decide whether to explore configurations it knows little about or refine variants near a promising one. With many related workloads, a learned policy may amortize experience: it can carry lessons from previously tuned shapes to a new shape and make a better initial choice.

However, RL is not presumed to be the universal winner. If the candidate space is small and tuning happens offline, exhaustive search is simpler and more reliable. For low-dimensional black-box tuning, Bayesian optimization can be highly competitive. TVM MetaSchedule, for example, uses a cost model and evolutionary search to explore transformations such as tiling, vectorization, and thread binding, reducing the number of real measurements required. [3]

The project is therefore an evaluation, not a marketing claim. RL is justified only if it finds comparable or better schedules using fewer measurements, generalizes to held-out workloads, or improves cold-start schedule selection.

### Experimental plan

Each method will receive the same candidate space, correctness checks, target GPU, and measurement budget.

1. **Fixed heuristic.** A reasonable default schedule for each kernel.
2. **Random search.** A simple and essential budget-matched baseline.
3. **Exhaustive search.** An oracle for small configuration spaces; not necessarily affordable for all real experiments.
4. **Non-RL model-based search.** Bayesian optimization or a lightweight learned performance model plus evolutionary search.
5. **Contextual bandit.** A learned one-shot schedule recommender.
6. **Sequential RL policy.** A history-aware policy that chooses the next configuration to benchmark.

Core metrics will be best latency found at each budget, speedup over the fixed baseline, number of trials needed to reach a fraction of the oracle result, tuning overhead, correctness pass rate, and performance on held-out tensor shapes. All reported GPU results will use repeated timings and a robust statistic such as median latency.

### Current implementation status

The simulator and real FP16 GPU backend are implemented. The project currently provides:

- workload and schedule-configuration data models;
- a constrained, legal candidate search space;
- a deterministic simulator with controlled measurement noise;
- random-search and exhaustive-search policies;
- a budget-enforcing RL-compatible environment;
- regression tests covering legality, graph timing units, budget behavior, invalid measurements, reward incentives, and strict result serialization;
- a repeated experiment runner with equal random/curated/bandit trial budgets, complete trial logs, frozen winners, shuffled confirmation order, and GPU telemetry;
- a trained LinearUCB model, training-only hyperparameter selection, and disjoint training/evaluation shapes.

The next milestone is broader training coverage and tuning under elapsed-time budgets. The early single-launch timing result is historical only: current experiments use medians of CUDA Graph batch means on repeated hot buffers. This reduces host launch gaps but does not measure cold-cache or end-to-end request latency. See the README and the dated experiment reports for measured results and limitations.

### Relationship to existing work

This project is not the first use of machine learning or RL in compiler optimization. ML-guided compiler optimization has been deployed in LLVM, and GPU-specific research has used RL to learn compiler heuristics. [4] RL has also been applied at the much lower SASS assembly-scheduling layer, where a policy mutates instruction schedules after compilation. [5]

The project differs in scope and purpose. It focuses on an understandable, Triton-level schedule-selection problem rather than generating arbitrary source code or manipulating assembly. Its contribution is a reproducible, learning-oriented comparison of budget-aware policies and conventional search methods on real GPU measurements.

### Usefulness

The project is useful in three ways:

1. **Practical systems relevance.** Kernel tuning matters for efficient ML training and inference, especially when standard libraries cannot provide a customized fused implementation.
2. **Research and engineering practice.** It teaches correct GPU benchmarking, workload characterization, compiler scheduling, and experimental baselines—not only RL training.
3. **Portfolio value.** It demonstrates an unusual but coherent combination of RL, computer architecture, GPU programming, compilers, and performance engineering.

A successful outcome is not limited to showing an RL speedup. It may demonstrate that RL needs fewer trials on related workloads, identify where classical search remains superior, or reveal which hardware/workload features predict a schedule's performance. A negative result with careful benchmarking and fair baselines is still informative.

### Limitations and honest claims

Results will be specific to the selected GPU, driver, Triton version, workload suite, dtypes, and configuration space. Simulated results are for software development only and will not be presented as GPU-performance evidence. Vendor libraries such as cuBLAS encode extensive architecture-specific engineering, so the project does not claim to replace them generally.

The intended claim is narrower and defensible:

> This project studies whether a learned, shape- and hardware-aware policy can allocate a limited GPU benchmarking budget more effectively than standard search baselines when tuning Triton kernel schedules.

### References

1. [Triton: Programming Model](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
2. [Triton: `autotune` API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
3. [Apache TVM: MetaSchedule Search-Based Auto-Tuning](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
4. [Generating GPU Compiler Heuristics Using Reinforcement Learning](https://arxiv.org/abs/2111.12055)
5. [CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning](https://arxiv.org/abs/2501.08071)
