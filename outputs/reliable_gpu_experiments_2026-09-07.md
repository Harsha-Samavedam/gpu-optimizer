# Repeated GPU experiments — 2026-09-07

## Conclusion

The timer is now shared between Triton and PyTorch and defaults to CUDA Graph batches. The earlier approximately 8% result is historical, not a valid before/after optimization baseline. With the new method and frozen-winner confirmation, the curated 16-trial baseline has lower latency than PyTorch on these four tested shapes, but gains vary sharply by shape. Additional improvement over the existing fixed Triton kernel is substantially smaller.

## Protocol

- NVIDIA RTX 3070, compute capability 8.6, driver 595.71.05; Python 3.14.4, PyTorch 2.14.0+cu130, Triton 3.8.0.
- FP16 inputs/output, FP32 accumulation in Triton; correctness against PyTorch with rtol=atol=0.01. PyTorch reduced-precision-reduction flag is recorded in the manifest.
- Four shapes × three search seeds; each method pays for 16 candidate evaluations. The common universe has 777 configurations. Random samples without replacement; curated search uses a predefined shortlist informed by the earlier 1024-square experiment and Triton tile families. This is a heuristic baseline, not RL.
- Searches are interleaved in shuffled order. Every candidate, failure status, raw samples, and wall time is written to JSONL. Winners are frozen before confirmation.
- Five separate confirmation rounds per case, with new inputs relative to search and shuffled method order. Four methods are measured per round. No winner is replaced based on confirmation.
- Ten warm-up launches, three warm-up graph replays, then 15 timed replays; 512 identical operations per graph. Each sample is elapsed graph time divided by 512, and each evaluation reports the median batch mean.
- The measured scope is hot-buffer steady-state throughput. Allocation, compilation, validation, and graph capture are outside timing. This is not single-request, cold-cache, or end-to-end application latency.
- GPU clock/power settings were not changed; the desktop remained active. Telemetry is logged before and after every confirmation round.
- The v1 runner used overlapping confirmation input seeds: seven distinct input datasets per shape across 15 separate timing rounds. These are not 15 independent input datasets. The runner now derives disjoint seeds within a suite.

## Confirmed results

Latency columns are medians of three per-seed medians; each per-seed value uses five confirmation rounds. Percentage reductions use within-round paired ratios, aggregated within seed and then across seeds. Seed ranges below are descriptive ranges, not confidence intervals.

| Shape (M,N,K) | PyTorch µs | Fixed µs | Random µs | Curated µs | Curated reduction vs PyTorch | Curated reduction vs fixed |
|---|---:|---:|---:|---:|---:|---:|
| 512×512×512 | 11.502 | 11.714 | 11.790 | 9.864 | 14.60% | 15.82% |
| 512×2048×1024 | 131.412 | 72.724 | 72.590 | 68.664 | 47.75% | 5.70% |
| 1000×769×513 | 56.792 | 55.540 | 55.448 | 55.532 | 2.12% | -0.04% |
| 1024×1024×1024 | 87.988 | 71.920 | 71.986 | 67.736 | 22.85% | 5.89% |

| Shape | Curated reduction vs PyTorch across search seeds |
|---|---:|
| 512×512×512 | 14.25%–14.91% |
| 512×2048×1024 | 47.24%–47.92% |
| 1000×769×513 | 2.05%–2.14% |
| 1024×1024×1024 | 22.44%–22.98% |

All 384 search trials and 180 Triton confirmation evaluations passed correctness (564 total). The suite also contains 60 timed PyTorch references. No new exhaustive oracle was run; the old 768-candidate result used a different timer and cannot establish regret for this suite.

## Timing diagnostic

Three shuffled diagnostic rounds compared single-launch events with graph batches. Values below are medians of the three evaluation medians. Diagnostic method comparisons always ran PyTorch then fixed then old winner, so use the main shuffled confirmation suite for performance conclusions.

| Timer | Batch size | PyTorch µs | Fixed µs | Previous winner µs |
|---|---:|---:|---:|---:|
| events | 1 | 81.760 | 75.776 | 73.728 |
| cuda_graph | 256 | 87.408 | 69.972 | 65.596 |
| cuda_graph | 512 | 87.420 | 69.652 | 66.520 |
| cuda_graph | 1024 | 87.471 | 70.556 | 67.022 |

The graph medians are broadly consistent across batch sizes, but individual rounds still drift. For example, the first 1024-square suite case had curated confirmation medians from 57.80 to 66.24 µs. Large batches reduce Python gaps; they do not eliminate thermal, power, cache, or desktop contention effects. Results near 2% should be treated as marginal until repeated in a quieter controlled session.

## Tuning findings

| Shape | Curated winner family | Interpretation |
|---|---|---|
| 512×512×512 | blocks 32×64×64, 4 warps, 3 stages, group 8 | Same choice in all three searches; a smaller output tile beats the fixed 64×64 tile. |
| 1024×1024×1024 | blocks 64×128×32, 4 warps, 3 stages; groups 8/4/1 | The output tile family is consistent; these data do not establish a reliable winner among group sizes. |
| 512×2048×1024 | blocks 64×128×32, 4 warps, 3 stages | Similar family to the square workload, with a much larger PyTorch-relative advantage. |
| 1000×769×513 | fixed blocks 64×64×32, 4 warps, 3 stages | Curated search retains the default; extra tuning gives no confirmed benefit. |

No curated winner used the new 256-wide/tall tiles. Bigger tiles did not automatically help this suite. The recorded candidate table supports shape-dependent priors; it does not isolate the causal effect of each parameter. Group size is now exposed as an action parameter, but its small differences need focused paired ablations.

For the original 1024-square workload, the new comparison is approximately 23% lower latency than PyTorch and approximately 6% lower than fixed Triton. Most of the apparent change from the earlier 8% claim is a change in measurement, not evidence of a 15-percentage-point kernel optimization.

## MDP changes implemented

- Observations include workload context, supplied device features, candidate configurations, incumbent, full measurement history, remaining budget, and an action mask.
- Already tried configurations are masked, and the episode ends when the budget or candidate set is exhausted.
- Invalid/nonfinite timings spend a trial, receive a penalty, and cannot improve the incumbent. Search and summary code also reject nonpositive/nonfinite latencies.
- With a fixed independent baseline, reward is log(previous_best / new_best). With discount 1, the positive improvement terms telescope to log speedup; failure penalties are additional.
- Without a baseline, positive reward is terminal-only 1/(1+best_latency_us). Regression tests prevent a policy from gaining reward by deliberately choosing a slow first trial. For cross-shape learning, use an independently measured baseline and account for its cost equally.
- Larger tiles, K=128, and alternate group sizes can be represented. Static legality now requires power-of-two block dimensions; device resource limits are still checked by compilation/execution, not a complete predictive resource model.

## Next changes, ranked by evidence

1. **Shape-aware starting prior / contextual bandit.** Learn a ranking of complete configurations using log dimensions, aspect ratios, tile remainders, estimated tile count, dtype, and hardware features. The observed shape-dependent winners justify this before adding sequential PPO.
2. **Elapsed-time budget and stop action.** Keep trial count as a reference but expose remaining seconds, measured trial costs, and incumbent improvement. A stop action could retain the default on shapes with little benefit. Wall time includes compilation, validation, capture, and measurement; it is not pure compile time.
3. **Explicit remeasurement action with a cost.** Current masking prevents duplicate search trials. A future noise-aware policy should be allowed to spend budget to confirm an uncertain winner, using sample uncertainty and improvement margins rather than rewarding every noisy minimum.
4. **More held-out shapes and a cost-model baseline.** The current four shapes are exploratory, not a predeclared train/test split. Compare learned rankings and a classical model-based policy under the same candidate and time budgets before RL claims.
5. **Focused kernel work.** Profile the slower regimes, test padding/tail-aware schedules for irregular dimensions, and only then consider fusion such as bias+activation. A fused kernel must be compared to the same fused workload; this suite does not establish any fusion gain.

Observed candidate wall times: min 0.099 s, median 0.816 s, max 38.207 s. This wide range directly motivates cost-aware decisions.

## Verification and artifacts

- 18 CPU regression tests pass; Ruff lint and formatting pass; a separate real-GPU integration case exercises the final reviewed runner, including disjoint confirmation seeds.
- Combined CPU/GPU Python line coverage: 86%. Triton JIT device-body execution is not captured as ordinary Python line coverage.
- Python review found no remaining high-priority correctness issue after reward and summary fixes.
- Raw study: `results/reliable_suite_v1/manifest.json`, `*/trials.jsonl`, `*/result.json`, `summary.json`, and `analysis.json`.
- Timer diagnostic: `results/timing_diagnostic_v1/timing_diagnostic.json`.
- Final integration check: `results/final_integration_check/` (different small shape/budget; excluded from the performance table).
- Raw artifacts are locally ignored by Git; this report and the README preserve the conclusions.

## Sources

The graph replay approach follows the same principle as [Triton do_bench_cudagraph](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench_cudagraph.html) and [PyTorch CUDA Graph semantics](https://docs.pytorch.org/docs/main/notes/cuda.html). Candidate tile and grouping families are informed by the [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). Performance numbers above are from this repository’s local measurements, not those sources.
