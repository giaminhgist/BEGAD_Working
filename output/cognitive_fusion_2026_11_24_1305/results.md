# cognitive_fusion: CN vs MCI

Mean over 10 seeds, 95% bootstrap CI (B = 2000).

| Model | Feature set | Test AUC [95% CI] | Test Acc [95% CI] | Test Sen [95% CI] | Test Spec [95% CI] |
|---|---|---|---|---|---|
| cogfusion_w000 | fusion | 0.708 [0.626, 0.787] | 0.618 [0.556, 0.677] | 0.687 [0.574, 0.795] | 0.588 [0.516, 0.657] |
| cogfusion_w025 | fusion | 0.720 [0.638, 0.800] | 0.641 [0.575, 0.703] | 0.652 [0.525, 0.773] | 0.636 [0.560, 0.706] |
| cogfusion_w050 | fusion | 0.725 [0.640, 0.807] | 0.669 [0.602, 0.732] | 0.661 [0.530, 0.788] | 0.672 [0.596, 0.743] |
| cogfusion_w075 | fusion | 0.723 [0.638, 0.807] | 0.660 [0.597, 0.720] | 0.657 [0.535, 0.782] | 0.661 [0.590, 0.732] |
| cogfusion_w100 | fusion | 0.718 [0.633, 0.802] | 0.667 [0.603, 0.728] | 0.630 [0.511, 0.754] | 0.684 [0.612, 0.757] |

## Paired contrasts

| Contrast | ΔAUC [95% CI] | Seeds > 0 | Visits |
|---|---|---|---|
| cogfusion_w000 − tabpfn_meanprob | +0.006 [+0.002, +0.011] | 9/10 | 176 |
| cogfusion_w025 − tabpfn_meanprob | +0.018 [+0.004, +0.035] | 10/10 | 176 |
| cogfusion_w050 − tabpfn_meanprob | +0.023 [-0.006, +0.054] | 10/10 | 176 |
| cogfusion_w075 − tabpfn_meanprob | +0.022 [-0.020, +0.068] | 9/10 | 176 |
| cogfusion_w100 − tabpfn_meanprob | +0.017 [-0.036, +0.075] | 8/10 | 176 |
