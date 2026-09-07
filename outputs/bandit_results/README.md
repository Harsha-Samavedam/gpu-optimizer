# Bandit measurements

Complete local GPU measurements from 2026-09-07. See the [report](../contextual_bandit_2026-09-07.md) for interpretation and reproduction commands.

| File | Contents |
|---|---|
| [metrics.json](metrics.json) | Per-shape latency summaries, paired reductions, seed ranges, counts, and costs for both suites. |
| [bandit_suite_v1.json](bandit_suite_v1.json) | Original four shapes: all 12 cases, search histories, trial wall times, confirmation samples, configurations, and telemetry. |
| [bandit_unseen_v1.json](bandit_unseen_v1.json) | Four additional shapes: the same complete records for 12 more cases. |
| [training.json](training.json) | All 320 training trials and ten calibration references. Its SHA256 matches the saved model metadata. |
| [training_ablation.json](training_ablation.json) | Training-only comparison of frozen ranking and online updates over the measured 32-candidate pools. |
| [unseen_plan.json](unseen_plan.json) | Additional shapes and evaluation settings recorded before their GPU run. |
| [discarded_training.json](discarded_training.json) | Interrupted first collection, explicitly excluded from training and reported performance. |

The [trained model](../../models/linucb_fp16_rtx3070.json) contains coefficients, inverse covariance, training provenance, and cross-validation results. These exports are tracked by Git. Working JSONL logs also remain locally under the ignored `results/` directory.
