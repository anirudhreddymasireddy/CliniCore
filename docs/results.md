# Results and honest limits

Every number below is measured on data the model never saw while training. Figures marked *synthetic* only show that the pipeline works; they are not evidence of clinical performance.

## Stage 1: NHAMCS, train 2018–2021, test 2022 (11,607 adult ED visits, 17 groups)
| Metric | Value |
|---|---|
| Accuracy (top guess) | 59.0 % (most-common-group baseline 17 %) |
| Right group in top 3 | 82.8 % |
| Calibration error (ECE) | 0.022 |
| 90 % plausible set | contains the true group 89.1 % of the time, 3.9 groups on average |

**The confidence cut-off is a policy choice** (2022):

| Cut-off | Answered automatically | Correct when answered |
|---|---|---|
| 0.5 | 64.6 % | 71.0 % |
| 0.7 | 38.6 % | 78.6 % |
| **0.8 (used)** | **23.1 %** | **83.5 %** |
| 0.9 | 9.0 % | 89.1 % |

Validation (held-out 2018–2021 visits) and the 2022 test year agree within about 1 point, so the cut-off transfers across years.

**Per group (2022)**: strongest are abdominal (80 %), chest pain (75 %), gynecologic/pregnancy (75 %), mental health (73 %) and injury (73 %). Weakest are sepsis/fever/other infection (18 %), metabolic (28 %), other symptoms (28 %), heart rhythm/failure/BP (31 %) and viral/COVID/flu (32 %). Triage information alone cannot separate these. The model is unsure about them, so they go to a doctor, which is the intended behaviour.

## Stage 3 sepsis: PhysioNet 2019, train hospital A (16,268 + 4,068 validation), test all 20,000 hospital-B patients
| Model | AUPRC (95 % CI) | AUROC | Utility* | Septic patients alerted | False-alert h / 24 h |
|---|---|---|---|---|---|
| **LightGBM** | 0.069 (0.063–0.077) | 0.800 | 0.306 (0.285–0.331) | 62 % | 1.24 |
| GRU-D | 0.045 (0.040–0.051) | 0.737 | 0.110 | 73 % | 3.94 |
| GRU-D + LightGBM | 0.075 (0.067–0.085) | 0.801 | 0.301 | 60 % | 1.27 |

\*Re-implementation of the PhysioNet utility (0 = never alert, 1 = ideal). Chance-level AUPRC is about 0.018 (1.8 % of hours are labelled). The decision rule, fixed in advance, was to keep the simplest model unless another wins on both AUPRC and utility with 95 % confidence. GRU-D was still improving at 10 epochs.

## Stage 3, all four risks on real patients
Run `python pipeline/physionet_icu.py`. Per-risk AUROC/AUPRC on hospital B, with the current-tier, qSOFA and NEWS2 baselines, are written to `results/stage3_multirisk_metrics.csv` (and `CAREGUARD_RESULTS.ST3_RISK_METRICS`).

## Synthetic only (pipeline checks)
Multi-risk heads on a shifted synthetic hospital: AUROC 0.88–0.91. Automatic-retraining demo: Stage 1 accuracy 0.66 → 0.89 over 3 promoted retrains, and the Stage 3 gate correctly rejected candidates that lost sensitivity.

## Limits
- No out-of-hospital validation of Stage 1. NHAMCS is a US survey and German EDs may differ.
- The criticality score is our own composite and not validated. The ROX, SCAI and KDIGO proxies are limited by missing data (see data.md).
- Learning from false alarms can make a model too quiet. This is mitigated by the safety gate and the "treatment prevented the event" exclusion, not eliminated.
- No authentication: roles are chosen in the UI. Not for clinical use.
