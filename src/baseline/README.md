# baseline

This family classifies CN vs MCI from eye-tracking features. It runs five machine-learning models (`lr_l2`,
`linear_svm`, `rf`, `fnn`, `tabpfn`) on seven feature sets (`task1`, `task3`, `task4`, `task5`, `task7`, `task9`,
`fusion`). It also builds a TabPFN mean-probability late fusion (`tabpfn_meanprob`) and applies a fixed clinical
rule (`mmse_cutoff`: MCI if MMSE ≤ 28). Every model uses the shared nested cross-validation, folds, seeds and
bootstrap draws from `src/common`.

- `config.yaml`: the enabled models, their fixed parameters and tuning grids (listed in tie-break order), and the
  feature sets.
- `models/`: one file per model. Each file registers itself with `@register("<name>")`.
- `run.py`: the entry point.

Run (from the repository root):

    PYTHONPATH=src .venv/bin/python -m baseline.run --smoke   # code check: seed 0, reduced grid, B = 200 -> output/smoke_YYYY_MM_DD_HHMM/baseline
    PYTHONPATH=src .venv/bin/python -m baseline.run           # full run: 10 seeds, B = 2000 -> output/baseline_YYYY_MM_DD_HHMM

`--models` and `--feature-sets` restrict a run to a subset. `--out DIR` reuses a directory, and finished jobs are
resumed unless `--no-resume` is given.

Output: `output/baseline_YYYY_MM_DD_HHMM/<model>/<feature_set>/` contains `predictions.csv`, `inner_oof.csv`,
`cv_results.csv`, `selected_params.json` and `metrics.json`. `mmse_cutoff` writes to `mmse_cutoff/clinical/` and
has no `inner_oof.csv`. `results.md` and `run_info.json` are at the top level.

## Adding or removing a model

**Add a fitted model.** Create `models/<name>.py` with a class that subclasses `EstimatorModel` and is decorated with
`@register("<name>")`.

- Set `pipeline`: `scaled` (median impute and standardise), `native` (the model handles NaN itself) or `tabpfn`.
- Set `score`: `predict_proba` or `decision_function`.
- Implement `build(params, fixed, ctx)`, which returns the bare estimator. Use `ctx.seed` as its random state.

Then add an entry to `config.yaml` with `name`, `fixed` and `grid`. `smoke_fixed` is optional. An empty grid means
no tuning.

**Add a derived model** (built from other cells or from data, not fitted). Subclass `DerivedModel`:

- Set `feature_set` (the output sub-folder).
- Implement `requires(model_cfg)`, which returns the cells it reads.
- Implement `run(ctx)`, which returns `predictions`, `inner_oof` and `selected`.
- Set `has_validation = False` for a fixed rule without model selection.

Then add its config entry.

**Remove a model.** Delete its entry from `config.yaml`, or comment it out, and the model is no longer run. Also
delete its file in `models/` to remove it for good. `tabpfn_meanprob` needs the `tabpfn` cells of tasks 1, 3, 4,
5, 7 and 9, and `cognitive_fusion` reads the same cells.

Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q src/baseline/tests`. The slow tests fit TabPFN; deselect them
with `-m "not slow"`.
