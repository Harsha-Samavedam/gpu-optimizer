# Contextual bandit experiments — 2026-09-07

The trained bandit finds a strong improvement on one original shape, but does not consistently beat the shortlist on fresh shapes. Both suites are complete: eight shapes, three seeds each, and zero invalid measurements. Full RL was not trained; the evidence points first to training coverage and better validation.

## Implementation

The tuner now has a trained shared-parameter LinearUCB model. A 67-element feature vector combines log workload dimensions and aspect ratios, schedule parameters, quadratic action terms, workload/action interactions, tail utilization, grid size, and rough resource estimates. These estimates are features, not measured occupancy or a compiler resource guarantee. The model is specific to one GPU/runtime.

Each action chooses a complete configuration from the same 777-candidate universe used by random search. The predictor estimates clipped log speedup against an independently measured fixed kernel. Invalid results receive −6 and spend a trial. Ridge regression is updated with the Sherman–Morrison formula; scoring optionally adds an uncertainty bonus. This follows the linear contextual-bandit approach described by [Li et al.](https://arxiv.org/abs/1003.0146) and [Chu et al.](https://proceedings.mlr.press/v15/chu11a.html), using a shared workload/configuration feature model.

The saved model is copied at the start of every evaluation case. Updates use only that bandit's own feedback; random and shortlist measurements are never supplied to it. Tried configurations are masked. Winners are frozen before independent confirmation. This is an adaptive contextual-bandit tuner, not a trained full sequential RL policy.

## Training and validation

- RTX 3070, compute capability 8.6; PyTorch 2.14.0+cu130, Triton 3.8.0.
- Ten training shapes: 256×256×256, 384×384×384, 768×768×768, 1536×1536×512, 256×1024×512, 384×1536×768, 768×384×1024, 1536×512×768, 767×513×385, and 1151×779×651.
- 32 configurations per shape: the 16-config shortlist plus 16 uniform non-shortlist samples, measured in shuffled order. All 320 training trials and ten fixed references passed correctness.
- Seven timed graph replays of 256 operations per candidate. Collection took 135.87 seconds with the existing compilation cache. This is additional offline cost, not free knowledge.
- Leave-one-training-shape-out validation selected ridge=0.01 and alpha=0 from twelve combinations. Multiple combinations tied; the data do not establish that zero exploration is uniquely best. The selected run is greedy prediction with online updates.
- Validation only replays each held-out training shape's 32 measured candidates at budget 16. It cannot certify the best choice in the full 777-candidate universe.
- An additional training-only ablation compared a frozen ranking with online updates. Both found the best measured candidate in all ten 32-candidate pools by trial 16. This limited replay does not establish an advantage for sequential planning.
- The original four evaluation shapes are excluded from model fitting and hyperparameter selection, but were studied previously when developing the project and shortlist. They are not an untouched research test set. Four additional shapes were recorded before their GPU evaluation, with the model and hyperparameters frozen.

## Correctness reference

The first collection attempt stopped at the ninth shape because directly comparing two FP16 implementations produced a false failure near the tolerance boundary. At one inspected element, Triton and the FP32 reference rounded to FP16 both gave −0.4465332, while default PyTorch FP16 matmul gave −0.4321289. Each passed the existing tolerance against the FP32 reference separately.

Both implementations now validate against FP32 matmul with TF32 disabled, rounded back to FP16. The previous TF32 setting is restored even if reference calculation fails. Tolerances remain rtol=atol=0.01. Reference work is outside timing; PyTorch's timed reduced-precision settings were not changed. [PyTorch documents the numerical effects of reduced-precision accumulation](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

The 256 completed candidate measurements from the interrupted first collection were excluded, and all ten training shapes were recollected with the corrected reference. Those discarded candidate measurements consumed another 107.19 seconds of logged trial wall time; complete failed-run wall time was not recorded.

## Evaluation protocol

Every shape uses three search seeds, 16 trials each for random, shortlist, and bandit, plus one separately recorded common fixed-reference measurement. The bandit therefore costs 16 search evaluations plus its calibration reference when deployed alone. Training cost is reported separately.

Each case has five confirmation rounds with fresh inputs and shuffled method order. Six methods are measured: PyTorch, fixed Triton, random winner, shortlist winner, bandit winner, and the bandit's first recommendation. The latter is a one-search-trial result, confirmed independently; it shares the separately charged calibration reference. No confirmation result changes a winner.

Each timing uses ten warm-up launches, three warm graph replays, and 15 timed replays of 512 operations. Values describe repeated hot-buffer throughput; allocation, compilation, validation, and capture are outside kernel timing. They are not cold-cache or end-to-end request latencies. GPU clocks were not locked and the desktop remained active. Trial wall times include compilation-cache effects and interleaved reuse, so this is an equal-trial comparison, not a controlled equal-wall-time contest.

Latency tables use the median of three per-seed confirmation medians. Percentage reductions use paired within-round ratios, then the median within each seed and across seeds. Seed ranges are descriptive, not confidence intervals. Small differences should not be treated as established improvements.

## Results

### Original shapes, fresh measurements

All latency values are in microseconds. “First” is the independently confirmed first recommendation; all other search columns use 16 trials.

| Shape (M,N,K) | PyTorch | Fixed | Random | Shortlist | Bandit | First | Bandit reduction vs shortlist |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512×512×512 | 12.918 | 13.072 | 12.670 | 11.002 | 10.740 | 13.082 | 2.24% |
| 1024×1024×1024 | 86.322 | 69.774 | 72.438 | 66.664 | 66.682 | 66.594 | −0.08% |
| 512×2048×1024 | 127.808 | 70.376 | 70.420 | 66.590 | 66.676 | 67.746 | −0.11% |
| 1000×769×513 | 56.832 | 56.302 | 56.478 | 56.468 | 44.764 | 55.006 | 20.30% |

Negative reductions mean the bandit was slower. Percentages use paired ratios rather than ratios of the displayed aggregate medians, so they need not equal arithmetic on those columns.

| Shape | Bandit reduction vs PyTorch | vs fixed | vs random | vs shortlist, seed range |
|---|---:|---:|---:|---:|
| 512×512×512 | 16.80% | 17.93% | 14.41% | 2.07%–2.99% |
| 1024×1024×1024 | 22.78% | 4.76% | 6.13% | −0.23%–0.30% |
| 512×2048×1024 | 47.87% | 5.78% | 5.60% | −0.32%–0.39% |
| 1000×769×513 | 21.17% | 20.20% | 20.75% | 19.54%–20.72% |

The bandit beat the shortlist in all three seeds for the small square and original irregular shape. The other two comparisons are effectively ties. The approximately 20% irregular-shape improvement is the strongest finding; the small-square improvement is modest and deserves further quiet-GPU validation.

### Additional shapes

| Shape (M,N,K) | PyTorch | Fixed | Random | Shortlist | Bandit | First | Bandit reduction vs shortlist |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640×960×768 | 35.170 | 35.378 | 35.926 | 34.302 | 35.614 | 35.622 | -3.76% |
| 896×1408×640 | 67.334 | 54.128 | 55.560 | 54.078 | 54.202 | 56.674 | -0.02% |
| 1023×1025×767 | 84.066 | 99.840 | 101.098 | 93.394 | 96.694 | 97.102 | -3.42% |
| 2048×256×1024 | 46.246 | 38.346 | 38.504 | 38.342 | 39.000 | 45.010 | -1.69% |

| Shape | Bandit reduction vs PyTorch | vs fixed | vs random | vs shortlist, seed range |
|---|---:|---:|---:|---:|
| 640×960×768 | -1.09% | -0.42% | 1.31% | -3.89%–-3.61% |
| 896×1408×640 | 19.51% | 0.08% | 10.62% | -0.15%–4.29% |
| 1023×1025×767 | -15.06% | 3.35% | 3.71% | -4.41%–-3.16% |
| 2048×256×1024 | 15.76% | -1.60% | -1.25% | -1.79%–-1.65% |

The bandit lost to the shortlist in all three seeds on 640×960×768, 1023×1025×767, and 2048×256×1024. The median comparison on 896×1408×640 is a tie, with one seed favoring the bandit. Absolute timing changed substantially in that middle seed, reinforcing the need for paired comparisons.

PyTorch was faster than the bandit on 640×960×768 and 1023×1025×767. The latter loss is about 15%, so a deployment should retain a library fallback. The successful original irregular-shape result does not generalize to every irregular workload.

## Kernel and policy findings

For 512×512×512, the bandit selected 32×64×32 tiles in two seeds and 64×32×32 in the third, with four warps and three stages. The shortlist's winner was 32×64×64. Smaller K chunks and the alternate output-tile orientation were available to the bandit but absent from the shortlist. Smaller output tiles can provide more grid parallelism; these measurements do not isolate occupancy or any single parameter's causal effect.

For 1000×769×513, all three bandit winners used 64×128×16 tiles, four warps, and two or three stages. The shortlist retained 64×64×32. The bandit reached the faster four-warp variant at trial 10 in the first seed; it had tried the same tile dimensions with eight warps earlier. Changing output width, K chunk, and warp count can alter reuse, tail work, and resource pressure; compiler profiling would be needed to separate those effects. This is configuration discovery within the existing kernel implementation, not a newly written kernel algorithm.

The first recommendation is not a universal replacement for tuning. On the original irregular and small-square shapes, the full bandit result is about 19% and 18% lower in latency than its first recommendation. On the 1024-square shape, the first recommendation already matches the shortlist and full bandit within noise.

“Irregular” alone is not enough to choose a K chunk. In the first 1023×1025×767 case, the shortlist selected K=64 with four stages, while the bandit selected K=32 with three stages. Both K chunks pad 767 to 768, but K=64 halves the reduction-loop count. By comparison, K=513 has much more padding with larger chunks. These arithmetic differences motivate broader remainder coverage in training; the measurements changed other parameters too, so this is a mechanism to investigate rather than an isolated causal result.

These gains also expose a limitation of the comparison: a fixed shortlist cannot find configurations it does not contain. Adding the discovered winners to a future shortlist may narrow the gap. That updated baseline should be tested on new shapes, rather than claimed as an independent comparison on the same observed results.

## Cost

The original suite contains 576 search evaluations, 360 confirmations, and 12 calibration references, with zero invalid measurements. It took 795.93 seconds of recorded case wall time. Across its twelve cases, logged search-trial wall time totaled 101.87 seconds for bandit, 194.95 for shortlist, and 304.74 for random. These totals reflect selected kernel speeds, resource use, and compilation-cache reuse; they are not isolated cold-start tuner comparisons.

Bandit scoring plus updating took a median 0.57 ms per trial with `OPENBLAS_NUM_THREADS=1`. Use the tuner to choose and cache a configuration, not on every matmul call. On the original irregular shape, the measured search-plus-setup cost would need roughly 0.8 million repeated calls to repay itself versus fixed Triton using the observed per-call saving. That rough estimate excludes offline training and confirmation costs and assumes steady timing.

The additional suite took 879.60 seconds. Together, the two suites contain 1,152 search evaluations, 720 confirmations, and 24 calibration references. Including successful training, there are 2,226 GPU evaluations: 2,106 Triton and 120 PyTorch, all passing correctness. The discarded first collection is archived separately and excluded from these counts. Total recorded case wall time for the two evaluation suites was 1,675.53 seconds (27.93 minutes), plus offline collection and fitting.

## Full RL decision and next steps

No full RL policy was trained in this run. The bandit has a substantial repeatable gain on one original shape, modest gains or ties on others, and generalization failures on additional shapes. A larger sequential policy is not yet supported by the available evidence: frozen ranking and online adaptation both exhaust the useful signal in the small 32-candidate training replay at budget 16, while most of the 777 actions remain unmeasured for each training shape. Training RL on that same replay would not resolve the missing hardware outcomes.

A full RL policy could eventually optimize terminal best latency, measurement cost, stopping, and remeasurement decisions. Before that, the most useful changes are:

1. Expand training shapes and measured candidate coverage, especially tile-grid boundaries and different irregular remainders. Validate at budgets 1, 4, 8, and 16 over a broader candidate pool; the current validation ties are a weak basis for choosing exploration.
2. Compare a richer nonlinear cost model with this linear predictor. Include measured compiler resource information and GPU-specific grid-wave features where available. The current resource features are coarse estimates.
3. Add a charged confirmation/remeasurement action and a stop action under an elapsed-time budget. Protect the incumbent against noisy minima and retain a measured fixed/PyTorch fallback when tuning does not improve it.
4. Use focused paired kernel ablations before rewriting kernels or expanding the action space. No new full-space oracle was measured here, so remaining headroom is unknown.

## Verification

All 28 CPU regression tests pass. They cover ridge-update math, exploration and masking, context-dependent rankings, persistence, episode isolation, reward handling, equal budgets, and the FP32 reference. Independent review also matched the full feature model's online update against batch ridge regression within 1e-9 and checked the exported original-shape table values. Export integrity checks reject a truncated trial log and a mismatched training dataset. Ruff lint, formatting, and Git whitespace checks pass.

Combined CPU/GPU Python line coverage is 89%. Triton JIT device-body execution is not represented as ordinary Python line coverage. Real-GPU search and confirmation measurements provide the kernel correctness checks.

## Reproduction and saved artifacts

The README has collection, fitting, and evaluation commands. The saved model is [models/linucb_fp16_rtx3070.json](../models/linucb_fp16_rtx3070.json). Training metadata records the exact dataset SHA256, hardware/runtime, reference protocol, and cross-validation scores.

The additional suite used:

```bash
OPENBLAS_NUM_THREADS=1 python -m rl_gpu_autotuner.experiments \
  --output-dir results/bandit_unseen_v1 \
  --bandit-model models/linucb_fp16_rtx3070.json \
  --shapes '640,960,768;896,1408,640;1023,1025,767;2048,256,1024'
```

Choose unused output directories when rerunning. This machine also set `CPATH` to its locally extracted Python development headers; a normal installation of matching development headers does not need that override.

The exporter verifies that each trial log matches its case history, all expected cases are complete, and the attached training data matches the model's dataset hash. Run:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/summarize_bandit.py \
  --suite results/bandit_suite_v1 --suite results/bandit_unseen_v1 \
  --training results/bandit_training_v2/dataset.json \
  --output-dir outputs/bandit_results
```

Raw working files remain in ignored `results/` directories. Complete successful training and evaluation exports, including candidate measurements, confirmation samples, selected configurations, costs, and telemetry, are also saved under `outputs/bandit_results/` for version control. The older [timing study](reliable_gpu_experiments_2026-09-07.md) remains historical; use the freshly measured baselines for this comparison.
