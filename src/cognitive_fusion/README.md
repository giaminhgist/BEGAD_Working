# cognitive_fusion

This family fuses two scores for CN vs MCI: the baseline TabPFN mean-probability score p̄, and the mean r̄ of six
per-task TabPFN regressors of a clinical composite C. The regressors predict C from the eye-tracking features only.
The two scores are combined on the z scale:

    s_w = (1 − w) · z(logit p̄) + w · z(r̄)

How the parts are built:

- **p̄** is read from the per-task `tabpfn` cells of a finished baseline output directory and is not refit. A
  start-up check confirms that it equals the baseline `tabpfn_meanprob/fusion` cell.
- **C** (`composite.py`) is the mean of the sign-aligned z-scores of five clinical scores: MMSE, verbal fluency,
  word-list total, naming and delayed recall.
  - The z-scores are standardised on the training rows of each fit, and higher C means more impaired.
  - Visits with fewer than 3 available scores are left out of the regressor's training rows.
  - The clinical scores are training targets only, never inputs.
- **The regressors** (`regress.py`) use the same nested folds as the classifiers.
- **The z references** (`combine.py`) come from the inner out-of-fold rows of each (seed, outer fold) only. The
  decision threshold is the Youden threshold on the fused inner out-of-fold scores.
- **The contrasts** (`contrast.py`) are paired Test AUC differences against the reference cell, with a shared-draw
  bootstrap CI. No multiplicity adjustment is applied.

Settings are in `config.yaml`: `baseline_out`, the source cells, the reference cells, the score signs,
`min_scores`, the regressor `n_estimators`, `w_grid`, `clip` and `primary`.

Run (from the repository root):

    PYTHONPATH=src .venv/bin/python -m cognitive_fusion.run --baseline-out output/smoke_YYYY_MM_DD_HHMM/baseline --smoke
    PYTHONPATH=src .venv/bin/python -m cognitive_fusion.run --baseline-out output/baseline_YYYY_MM_DD_HHMM

A smoke run writes next to its baseline, in `output/smoke_YYYY_MM_DD_HHMM/cognitive_fusion`. A full run needs a
completed, non-smoke baseline output.

Arms: `cogfusion_w000`, `_w025`, `_w050` (primary), `_w075` and `_w100`, where w is the weight of the regressor
score.

Output: `output/cognitive_fusion_YYYY_MM_DD_HHMM/<arm>/fusion/` contains `predictions.csv`, `inner_oof.csv`,
`selected_params.json` and `metrics.json`. The top level holds:

- `regression/task<t>/` and `regression/composite_params.json`;
- `contrasts.json`, `diagnostics.json`;
- `results.md`, `run_info.json`.

## Adding or removing an arm, or the family

- **Add or remove an arm.** Edit `w_grid` in `config.yaml`. The arm name is derived from w, for example 0.3 gives
  `cogfusion_w030`. `primary` must name an arm in `w_grid`.
- **Change the reference cells.** Edit `reference` and `early_fusion`. If a reference cell is not in the baseline
  output, the identity check and the contrasts are skipped.
- **Remove the family.** Delete `src/cognitive_fusion/` and `output/cognitive_fusion_*`. Nothing in `src/common` or
  `src/baseline` imports it.

Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q src/cognitive_fusion/tests`.
