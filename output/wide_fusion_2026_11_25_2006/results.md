# wide_fusion: CN vs MCI

Mean over 10 seeds, 95% bootstrap CI (B = 2000).

| Model | Feature set | Test AUC [95% CI] | Test Acc [95% CI] | Test Sen [95% CI] | Test Spec [95% CI] |
|---|---|---|---|---|---|
| wide_cf_w050 | fusion | 0.738 [0.663, 0.812] | 0.648 [0.589, 0.709] | 0.741 [0.633, 0.843] | 0.607 [0.533, 0.681] |
| kd_meanprob | fusion | 0.679 [0.597, 0.763] | — | — | — |
| kd_meanreg | fusion | 0.688 [0.602, 0.773] | — | — | — |
| kdwide_cls | fusion | 0.683 [0.610, 0.757] | — | — | — |
| kdwide_reg | fusion | 0.676 [0.594, 0.759] | — | — | — |
| rdwide_cls | fusion | 0.675 [0.591, 0.754] | — | — | — |
| rdwide_reg | fusion | 0.722 [0.640, 0.798] | — | — | — |

## Paired contrasts

| Contrast | ΔAUC [95% CI] | Seeds > 0 | Visits |
|---|---|---|---|
| wide_cf_w050 − cogfusion_w050 | +0.013 [-0.013, +0.042] | 7/10 | 176 |
| kdwide_cls − kd_meanprob | +0.004 [-0.031, +0.039] | 5/10 | 176 |
| kdwide_reg − kd_meanreg | -0.012 [-0.043, +0.021] | 2/10 | 176 |

## Reference cells

| Reference | Path | Test AUC [95% CI] |
|---|---|---|
| tabpfn_meanprob | output/baseline_2026_11_24_1207/tabpfn_meanprob/fusion | 0.701 [0.619, 0.781] |
| cogfusion_w050 | output/cognitive_fusion_2026_11_24_1305/cogfusion_w050/fusion | 0.725 [0.640, 0.807] |

## MMSE rules

| Rule | AUC [95% CI] | Sens [95% CI] | Spec [95% CI] | Visits |
|---|---|---|---|---|
| mmse_cutoff (MCI if MMSE <= 28) | 0.693 [0.619, 0.762] | 0.755 [0.650, 0.857] | 0.631 [0.546, 0.713] | 175 |
| mmse_cutoff_lt28 (MCI if MMSE < 28) | 0.666 [0.580, 0.751] | 0.528 [0.375, 0.680] | 0.803 [0.730, 0.875] | 175 |
