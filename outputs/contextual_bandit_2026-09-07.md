# Contextual-bandit results

## Setup

- GPU: RTX 3070 (compute capability 8.6)
- Runtime: PyTorch 2.14.0+cu130 and Triton 3.8.0
- Kernel: FP16 matrix multiplication
- Search budget: 16 unique configurations per method
- Bandit: linear contextual model trained on 320 GPU measurements from ten shapes

The bandit uses matrix-shape and schedule features to rank configurations, then updates from its own measurements. It is an adaptive contextual bandit, not a full RL policy.

Each result below uses three search seeds. Winners were remeasured with fresh inputs in randomized order. Timings use CUDA Graph batches on hot buffers; compilation, allocation, and correctness checks are outside the timed loop.

## Results

Latency is in microseconds. Percentage comparisons use paired timing rounds.

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

The bandit beat PyTorch on six of eight shapes and random search on seven of eight. Its largest gain was 47.87% versus PyTorch on 512x2048x1024. It was not uniformly better: PyTorch won on two cases, including a 15.06% advantage on 1023x1025x767.

## Limitations

- The results apply only to this Triton matmul kernel, configuration space, RTX 3070, and software stack.
- GPU clocks were not locked, so small differences should be treated cautiously.
- The evaluation measures steady-state kernel throughput, not end-to-end application latency.
- A manually curated shortlist is retained in the raw exports, but PyTorch and equal-budget random search are the primary baselines here.

Raw trial logs, selected schedules, timing samples, and training metadata are in [bandit_results](bandit_results/). All 28 CPU tests passed for this revision.
