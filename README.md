# CN vs MCI classification from eye-tracking features

This repository contains the code, the fold definitions and the results for a CN vs MCI classification study.
The study separates cognitively normal (CN) visits from visits with mild cognitive impairment (MCI) using
eye-tracking features recorded during six tasks of the BEGAD cohort (tasks 1, 3, 4, 5, 7 and 9). It compares:

- **baseline**: five machine-learning models on each single task and on all tasks together, a late fusion of the
  per-task TabPFN probabilities, and a fixed clinical MMSE cutoff rule;
- **cognitive_fusion**: the late-fusion TabPFN score combined with a TabPFN regression of a clinical composite.
  The regression is trained on clinical test scores and predicted from the eye-tracking features only;
- **wide_fusion**: the same fusion of a classifier and a regressor of the clinical composite, but with one TabPFN per
  block of tasks (King-Devick tasks 1, 3, 4, 5 and reading tasks 7, 9) that sees the per-task features side by side,
  instead of one TabPFN per task.

Every model uses the same cohort, the same cross-validation folds, the same seeds and the same bootstrap draws,
so all rows of the result tables can be compared directly.

## Contents

- [Cohort](#cohort)
- [Features](#features)
- [Repository layout](#repository-layout)
- [Evaluation](#evaluation)
- [Models](#models)
- [cognitive_fusion method](#cognitive_fusion-method)
- [wide_fusion method](#wide_fusion-method)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Output format](#output-format)
- [Results](#results)
- [Limitations](#limitations)
- [Reference](#reference)

## Cohort

| | Visits | Subjects |
|---|---:|---:|
| CN | 122 | 100 |
| MCI | 54 | 44 |
| **Total** | **176** | **141** |

- The unit of analysis is the **visit**: 141 first visits and 35 follow-up visits. 106 subjects have one visit and
  35 subjects have two.
- 3 subjects have a different diagnosis at their two visits. This is why the subject counts per group add up to
  144 and not 141.
- Labels: MCI = 1, CN = 0. Visits with other diagnoses in the feature files (for example AD) are not part of this
  cohort.
- The clinical scores come from `dataset/original_dataset/COGDATA_20260930.csv` and are joined on the visit id.
  One MCI visit has no MMSE value and no value for the other four scores used here (verbal fluency, word-list
  learning total, naming, delayed recall).

## Features

The feature definitions, formulas, units and task applicability are in
[`dataset/docs/features.md`](dataset/docs/features.md): 69 features per task recording. The per-task feature
tables are `dataset/features/<task>/features.csv` for tasks 1, 3, 4, 5, 7 and 9.

- `gaze_quadrant_prop_bottom_left` is dropped from every task and from fusion. Its gaze bounding box is pooled over
  all recordings, including held-out visits, so it would carry information from outside the training folds.
- A feature that does not apply to a task is blank for every recording of that task, and it is removed for that
  task. For example, some features apply only to tasks 1, 3, 4 and 5, and others only to tasks 7 and 9.

| Feature set | Tasks | Features |
|---|---|---:|
| `task1`, `task3`, `task4`, `task5` | one task each | 63 |
| `task7`, `task9` | one task each | 60 |
| `fusion` (early fusion) | all six side by side, columns prefixed `t<task>__` | 372 |

In addition, every fitted pipeline drops columns that have at most one distinct non-missing value on its training
rows.

## Repository layout

```
README.md                     this file
requirements.txt              exact package versions (checked at start-up)
pytest.ini
dataset/
  original_dataset/           source recordings and clinical data (read-only)
  processed_dataset/          per-task processed recordings (read-only)
  features/<task>/features.csv
  docs/features.md            feature dictionary
  fold/                       shared folds and bootstrap draws (see dataset/fold/README.md)
    seeds.yaml                the 10 seeds
    seed_<seed>.csv           outer and inner fold of every visit, one file per seed
    bootstrap_draws.npz       B = 2000 stratified subject-level resamples
src/
  common/                     code shared by all families (data, folds, nested CV, metrics, bootstrap, report)
  baseline/                   baseline family: config.yaml, run.py, models/ (one file per model)
  cognitive_fusion/           cognitive_fusion family: config.yaml, run.py, composite.py, regress.py, combine.py, contrast.py
  wide_fusion/                wide_fusion family: config.yaml, run.py, features.py, nested.py, combine.py, derived.py, mmse.py
output/
  baseline_2026_11_24_1207/          final baseline results
  cognitive_fusion_2026_11_24_1305/  final cognitive_fusion results (built from the baseline output above)
  wide_fusion_2026_11_25_2006/       final wide_fusion results (reads the two output directories above)
```

The families import only `src/common` and never import each other. `cognitive_fusion` reads the saved
per-task TabPFN outputs of a finished baseline output directory and does not refit them. `wide_fusion` reads the saved
per-task outputs of the baseline and cognitive_fusion directories (for its comparison cells and the identity checks)
and fits its own block models.

## Evaluation

**Nested cross-validation.** Every (model, feature set, seed) uses an outer 4-fold × inner 4-fold
`StratifiedGroupKFold`. The splits are stratified by label and grouped by subject (`ParticipantID`), so no subject
is ever in both the training and the test part of a split, at either level. The folds are generated once and
stored in `dataset/fold/seed_<seed>.csv`, and all models read the same files.

**Seeds.** 0, 1, 7, 13, 42, 123, 777, 1234, 2024, 3407. Each seed sets both the fold split and the random state of
the models.

**Model selection.** Inside each outer training set, every grid configuration is evaluated by 4-fold inner
cross-validation. The configuration with the highest AUC over the pooled inner out-of-fold (OOF) predictions is
selected, and ties go to the first configuration in grid order. The selected configuration is refit on the full
outer training set and predicts the outer test fold.

**Decision threshold.** The threshold is the Youden-index threshold of the inner OOF scores of the selected
configuration (inner OOF only, never test rows). It is applied to that outer fold's test scores.

**Metrics.** Test AUC, accuracy (Acc), sensitivity (Sen, MCI recall) and specificity (Spec, CN recall). For each
seed, the outer-test predictions of the four outer folds are pooled, so each visit is a test row exactly once per
seed. The metric is computed on the pooled rows and then averaged over the 10 seeds.

**95% confidence intervals.** These come from a stratified subject-level bootstrap with B = 2000:

- Subjects are drawn with replacement within two strata, subjects ever diagnosed MCI and CN-only subjects. Each
  drawn subject contributes all of its visits.
- For each replicate, the pooled per-seed metric is recomputed and averaged over the seeds. The CI is the 2.5th and
  97.5th percentiles of these replicate values.
- The draws are generated once (`dataset/fold/bootstrap_draws.npz`) and shared by every model. This makes paired
  comparisons between models possible.
- The models are not refit inside the bootstrap.

## Models

All settings are in `src/baseline/config.yaml`. Grid values are listed in tie-break order.

| Model | Preprocessing | Fixed settings | Tuning grid |
|---|---|---|---|
| `lr_l2`: L2 logistic regression | median impute, standardise | lbfgs, max_iter 5000 | C ∈ {0.001, 0.01, 0.1, 1, 10, 100}; class_weight ∈ {none, balanced} |
| `linear_svm`: linear SVM (score = signed distance) | median impute, standardise | squared hinge, max_iter 20000 | C ∈ {0.0001, 0.001, 0.01, 0.1, 1, 10}; class_weight ∈ {none, balanced} |
| `rf`: random forest | none (native missing-value handling) | 500 trees, gini, bootstrap | min_samples_leaf ∈ {5, 3, 1}; max_features ∈ {sqrt, 0.33}; class_weight ∈ {none, balanced_subsample} |
| `fnn`: feed-forward network (scikit-learn MLP) | median impute, standardise | ReLU, Adam, lr 0.001, max_iter 1000, no early stopping | alpha ∈ {10, 1, 0.1, 0.001}; hidden layers ∈ {(32), (64, 32)}; class_weight ∈ {none, balanced} (balanced = sample weight n / (2 n_c)) |
| `tabpfn`: frozen TabPFN v3.5 classifier | none (missing values passed to TabPFN) | pretrained checkpoint used as is, no fine-tuning, `n_estimators = 2`, CPU | none |
| `tabpfn_meanprob`: TabPFN late fusion | – | mean of the six per-task `tabpfn` MCI probabilities, with no learned weights | none |
| `mmse_cutoff`: clinical rule | – | MCI if MMSE ≤ 28, else CN | none |

Notes:

- The five fitted models are run on every feature set: `task1`, `task3`, `task4`, `task5`, `task7`, `task9` and
  `fusion`.
- `tabpfn` has no grid. Its inner cross-validation is used only for the decision threshold.
- **`tabpfn_meanprob`** is built from the saved `tabpfn/task<t>` outputs.
  - Its test score for a visit is the mean of the six per-task outer-test probabilities, and its inner OOF score is
    the mean of the six per-task inner OOF probabilities. The inner folds are the same for all tasks.
  - Its threshold is the Youden threshold of the averaged inner OOF scores.
- **`mmse_cutoff`** is a fixed rule with no training and no model selection: MCI if MMSE ≤ 28 (cutoff from Atiya et
  al., 2026; see [Reference](#reference)).
  - Its score is the rule prediction (1 if MMSE ≤ 28, else 0), so AUC, Acc, Sen and Spec all come from the fixed
    cutoff, with the same seeds and bootstrap draws as the other models.
  - The one visit without an MMSE is excluded from this cell only, which leaves n = 175 visits (140 subjects;
    53 MCI, 122 CN). All other models use all 176 visits.

## cognitive_fusion method

The fused score for a visit is

    s_w = (1 − w) · z(logit p̄) + w · z(r̄)

- **p̄** is the TabPFN mean probability: the score of `tabpfn_meanprob`, read from the baseline output. A start-up
  check confirms that it is identical to the baseline cell. Probabilities are clipped to [1e−6, 1 − 1e−6] before the
  logit.
- **r̄** is the mean of six per-task TabPFN regressors (frozen TabPFN v3.5, `n_estimators = 2`). Each regressor
  predicts the clinical composite **C** from the eye-tracking features of one task, using the same columns as the
  classifier. No clinical score or age is an input. An input check rejects any column that carries a clinical-data
  column name or a name matching age, CDR or MMSE, and any column almost perfectly correlated (|r| > 0.999) with
  age or a clinical score.
- **C** is defined from five clinical scores: MMSE, verbal fluency, word-list learning total, naming, and delayed
  recall (`src/cognitive_fusion/composite.py`).
  - On the training rows of each fit, each score k is standardised with its training mean and SD (all diagnoses
    together), and its sign is set so that a higher C means more impaired.
  - C is the mean of the signed z-scores the visit has. Visits with fewer than 3 available scores are left out of
    the regressor's training rows.
  - The clinical scores are used only as training targets and never as inputs. Test visits are scored from their
    eye-tracking features alone.
  - The regressors follow the same nested folds as the classifiers: inner fits give inner OOF predictions, and a
    refit on the outer training set predicts the test fold.
- **z(·) references** come from the inner OOF rows only, for each (seed, outer fold).
  - Test rows use the mean and SD over all inner OOF rows of that outer fold.
  - The inner OOF rows of inner fold i use the other three inner folds.
  - No test visit enters a reference.
- **Threshold**: the Youden threshold of the fused inner OOF scores, as for every other model.
- **Arms** (`w_grid` in `src/cognitive_fusion/config.yaml`): w = 0, 0.25, **0.5 (primary)**, 0.75, 1. The arms
  are named `cogfusion_w000`, `cogfusion_w025`, `cogfusion_w050`, `cogfusion_w075` and `cogfusion_w100`. With
  w = 1 only the regressor score is used.
- **Paired contrasts**: every arm is compared with `tabpfn_meanprob` on identical (seed, visit) rows, and the
  primary arm is also compared with `tabpfn` on `fusion`.
  - ΔAUC is the mean over seeds of the per-seed AUC difference.
  - The 95% CI is the percentile interval of the same-draw bootstrap differences.
  - No multiplicity adjustment is applied.

## wide_fusion method

The earlier families fit one TabPFN per task and average the outputs. wide_fusion keeps the features of the tasks of a
block **side by side** in one row per visit, so one TabPFN sees all tasks of the block at once
(`src/wide_fusion/README.md`):

- **KD block**: tasks 1, 3, 4, 5, with 63 columns per task, 4 × 63 = 252 columns.
- **Reading block**: tasks 7, 9, with 60 columns per task, 2 × 60 = 120 columns.

Within every fit (each inner fit and each outer refit), each task's columns are z-scaled with the training visits of
that fit only. A column with SD 0 or fewer than 2 values is kept as NaN. TabPFN v3.5 uses up to 768 features per
estimator, so it sees every column; every fit checks this and the run stops if TabPFN would subsample features.

Two members are fitted with the nested folds above (frozen TabPFN v3.5, `n_estimators = 2`, one model per block):

- `wide_cls_e2`, a classifier of the diagnosis. p̄ is the mean of the KD and reading probabilities.
- `wide_reg_C_e2`, a regressor of the clinical composite C, defined as in cognitive_fusion and recomputed on the
  training rows of every fit. r̄ is the mean of the KD and reading predictions. The clinical scores are training
  targets only.

The fused cell **`wide_cf_w050`** is

    s = 0.5 · z(logit p̄) + 0.5 · z(r̄)

with the z references from the inner OOF rows of each (seed, outer fold) and the Youden threshold on the fused inner
OOF scores, as in cognitive_fusion. Descriptive cells score a single block or member as is: `kdwide_cls`,
`kdwide_reg`, `rdwide_cls`, `rdwide_reg`, and, built from the saved per-task outputs of the KD tasks, `kd_meanprob`
and `kd_meanreg`. Paired contrasts use the same definition as in cognitive_fusion.

## Installation

The results were produced with Python 3.14.4 on CPU (no GPU is used).

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` lists exact versions (numpy, pandas, scipy, scikit-learn, torch, tabpfn, python-dotenv, pyyaml,
pytest, threadpoolctl). The entry points refuse to start if the installed versions differ. `--skip-pin-check`
disables this check.

**TabPFN token.** TabPFN needs an access token. Provide it as the environment variable `TABPFN_TOKEN`, or put a line
`TABPFN_TOKEN=<your token>` in a file `.env` at the repository root. `.env` is ignored by version control, the code
reads it automatically and never prints it. Also leave `TABPFN_MODEL_CACHE_SIZE` unset or set it to 0.

## How to run

Run all commands from the repository root.

```bash
# smoke test (code check only, not a result: seed 0, first/last grid values, B = 200)
PYTHONPATH=src .venv/bin/python -m baseline.run --smoke
PYTHONPATH=src .venv/bin/python -m cognitive_fusion.run --baseline-out output/smoke_YYYY_MM_DD_HHMM/baseline --smoke
PYTHONPATH=src .venv/bin/python -m wide_fusion.run --smoke

# full run (10 seeds, B = 2000); cognitive_fusion needs the finished baseline output directory
PYTHONPATH=src .venv/bin/python -m baseline.run
PYTHONPATH=src .venv/bin/python -m cognitive_fusion.run --baseline-out output/baseline_YYYY_MM_DD_HHMM
PYTHONPATH=src .venv/bin/python -m wide_fusion.run    # reads the two output directories named in src/wide_fusion/config.yaml

# rebuild results.md from the saved metrics.json / contrasts.json (no refitting)
PYTHONPATH=src .venv/bin/python -m common.report --out output/baseline_2026_11_24_1207 output/cognitive_fusion_2026_11_24_1305

# tests (the slow tests fit TabPFN; deselect them with -m "not slow")
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m pytest -q -m "not slow"

# regenerate the fold files and the bootstrap draws (already provided; draws are not overwritten without --force)
PYTHONPATH=src .venv/bin/python -m common.make_folds
PYTHONPATH=src .venv/bin/python -m common.bootstrap
```

Options for `baseline.run`:

- `--models` and `--feature-sets` restrict a run to a subset.
- `--seeds` sets the seeds.
- `--out DIR` sets the output directory.
- `--workers`, `--tabpfn-workers` and `--tabpfn-threads` control parallelism.
- `--n-boot` sets the number of bootstrap replicates.
- `--no-resume` forces recomputation. By default, a run resumes from finished jobs in the same `--out`.

On a 16-core CPU, the full baseline run took about 58 minutes, the cognitive_fusion run about 10 minutes and the
wide_fusion run about 13 minutes.
Smoke runs write to `output/smoke_YYYY_MM_DD_HHMM/`, which is ignored by version control.

## Output format

Each output directory is named `<family>_YYYY_MM_DD_HHMM` and contains one sub-folder per cell,
`<model>/<feature_set>/`. For `mmse_cutoff` the feature set is `clinical`, and for the cognitive_fusion arms it is
`fusion`.

| File | Content |
|---|---|
| `predictions.csv` | outer-test rows of every seed: `seed, outer_fold, RealTag, ParticipantID, IsFollowup, y_true, y_score, threshold, y_hat` |
| `inner_oof.csv` | inner OOF scores of the selected configuration: `seed, outer_fold, inner_fold, RealTag, ParticipantID, IsFollowup, y_true, y_score`. Not written for `mmse_cutoff` |
| `cv_results.csv` | fitted models only: for every grid configuration and outer fold, the inner cross-validation score used for selection |
| `selected_params.json` | fitted models: for every seed and outer fold, the selected configuration, the threshold and the dropped degenerate columns. Derived cells: the rule used (for `mmse_cutoff` also the visit without MMSE) |
| `metrics.json` | the summary (mean and 95% CI of Test AUC, Acc, Sen and Spec), per-seed and per-fold metrics, the spread over seeds (SD, min, max), confusion counts, n and bootstrap settings. It also stores the inner-CV AUC |

The top level of each output directory holds `results.md` (the table, built from the `metrics.json` files) and
`run_info.json` (command, status, seeds, package versions and timing). cognitive_fusion also writes:

- `regression/task<t>/`: the per-task regressor predictions. In `inner_oof.csv`, `y_true` is the training-row C,
  and in `predictions.csv` it is blank.
- `regression/composite_params.json`: the training means and SDs of the five scores for every fit.
- `contrasts.json`: the paired differences, including per-seed values.
- `diagnostics.json`: the inner OOF Spearman ρ between r̄ and C, and fit times.

wide_fusion writes one folder per cell, `<cell>/fusion/`, and also:

- `members/<member>/`: the member predictions, the per-block scores, fit logs and, for the regressor, the training
  means and SDs of C.
- `contrasts.json`, `diagnostics.json`.
- `mmse_rules.json`: the AUC, Sen and Spec of the MMSE rules (MMSE ≤ 28 and MMSE < 28) with their CIs.

## Results

All numbers are copied from `output/baseline_2026_11_24_1207/`, `output/cognitive_fusion_2026_11_24_1305/` and
`output/wide_fusion_2026_11_25_2006/` (`results.md`, `metrics.json`, `contrasts.json`, `diagnostics.json`,
`mmse_rules.json`; the MMSE AUC is taken from the baseline output). Each value is the mean over 10 seeds, with the
95% bootstrap CI in brackets (B = 2000). n = 176 visits unless stated.

### Key rows

| Model | Feature set | Test AUC [95% CI] | Test Acc [95% CI] | Test Sen [95% CI] | Test Spec [95% CI] |
|---|---|---|---|---|---|
| lr_l2 | fusion | 0.646 [0.588, 0.712] | 0.616 [0.558, 0.677] | 0.648 [0.539, 0.755] | 0.602 [0.531, 0.677] |
| linear_svm | fusion | 0.658 [0.593, 0.731] | 0.616 [0.558, 0.676] | 0.650 [0.545, 0.758] | 0.601 [0.529, 0.675] |
| rf | fusion | 0.664 [0.584, 0.744] | 0.595 [0.539, 0.653] | 0.635 [0.527, 0.740] | 0.578 [0.507, 0.653] |
| fnn | fusion | 0.689 [0.623, 0.756] | 0.620 [0.569, 0.674] | 0.670 [0.590, 0.746] | 0.598 [0.532, 0.666] |
| tabpfn | task7 | 0.694 [0.614, 0.772] | 0.618 [0.562, 0.674] | 0.730 [0.631, 0.828] | 0.568 [0.498, 0.636] |
| tabpfn | fusion | 0.692 [0.618, 0.767] | 0.617 [0.559, 0.676] | 0.685 [0.588, 0.782] | 0.587 [0.513, 0.661] |
| **tabpfn_meanprob** | fusion | **0.701 [0.619, 0.781]** | 0.621 [0.561, 0.680] | 0.713 [0.602, 0.823] | 0.580 [0.506, 0.650] |
| **cogfusion_w050** (primary) | fusion | **0.725 [0.640, 0.807]** | 0.669 [0.602, 0.732] | 0.661 [0.530, 0.788] | 0.672 [0.596, 0.743] |
| **wide_cf_w050** | fusion | **0.738 [0.663, 0.812]** | 0.648 [0.589, 0.709] | 0.741 [0.633, 0.843] | 0.607 [0.533, 0.681] |
| mmse_cutoff (n = 175) | clinical | 0.693 [0.619, 0.762] | 0.669 [0.599, 0.738] | 0.755 [0.650, 0.857] | 0.631 [0.546, 0.713] |

### Baseline: all cells

Source: [`output/baseline_2026_11_24_1207/results.md`](output/baseline_2026_11_24_1207/results.md).

| Model | Feature set | Test AUC [95% CI] | Test Acc [95% CI] | Test Sen [95% CI] | Test Spec [95% CI] |
|---|---|---|---|---|---|
| lr_l2 | task1 | 0.589 [0.521, 0.656] | 0.620 [0.565, 0.675] | 0.469 [0.353, 0.583] | 0.687 [0.623, 0.746] |
| lr_l2 | task3 | 0.591 [0.525, 0.656] | 0.543 [0.482, 0.602] | 0.644 [0.554, 0.738] | 0.498 [0.419, 0.575] |
| lr_l2 | task4 | 0.640 [0.568, 0.714] | 0.611 [0.561, 0.663] | 0.598 [0.502, 0.696] | 0.617 [0.558, 0.678] |
| lr_l2 | task5 | 0.604 [0.550, 0.660] | 0.634 [0.575, 0.694] | 0.680 [0.572, 0.786] | 0.613 [0.547, 0.684] |
| lr_l2 | task7 | 0.631 [0.559, 0.703] | 0.586 [0.532, 0.642] | 0.628 [0.518, 0.732] | 0.568 [0.502, 0.634] |
| lr_l2 | task9 | 0.605 [0.532, 0.672] | 0.605 [0.547, 0.660] | 0.602 [0.493, 0.705] | 0.606 [0.538, 0.675] |
| lr_l2 | fusion | 0.646 [0.588, 0.712] | 0.616 [0.558, 0.677] | 0.648 [0.539, 0.755] | 0.602 [0.531, 0.677] |
| linear_svm | task1 | 0.602 [0.526, 0.675] | 0.643 [0.588, 0.696] | 0.437 [0.321, 0.552] | 0.734 [0.675, 0.788] |
| linear_svm | task3 | 0.609 [0.529, 0.688] | 0.537 [0.477, 0.596] | 0.641 [0.548, 0.734] | 0.491 [0.414, 0.566] |
| linear_svm | task4 | 0.651 [0.580, 0.724] | 0.620 [0.570, 0.670] | 0.602 [0.498, 0.702] | 0.628 [0.572, 0.686] |
| linear_svm | task5 | 0.643 [0.573, 0.715] | 0.611 [0.553, 0.670] | 0.674 [0.567, 0.781] | 0.583 [0.516, 0.653] |
| linear_svm | task7 | 0.640 [0.564, 0.717] | 0.589 [0.537, 0.639] | 0.641 [0.539, 0.737] | 0.566 [0.503, 0.629] |
| linear_svm | task9 | 0.621 [0.547, 0.692] | 0.610 [0.553, 0.668] | 0.609 [0.498, 0.715] | 0.611 [0.545, 0.680] |
| linear_svm | fusion | 0.658 [0.593, 0.731] | 0.616 [0.558, 0.676] | 0.650 [0.545, 0.758] | 0.601 [0.529, 0.675] |
| rf | task1 | 0.592 [0.498, 0.678] | 0.587 [0.532, 0.641] | 0.507 [0.391, 0.617] | 0.623 [0.563, 0.682] |
| rf | task3 | 0.570 [0.489, 0.651] | 0.519 [0.461, 0.578] | 0.635 [0.535, 0.733] | 0.468 [0.392, 0.544] |
| rf | task4 | 0.618 [0.536, 0.702] | 0.582 [0.530, 0.635] | 0.507 [0.409, 0.610] | 0.616 [0.551, 0.678] |
| rf | task5 | 0.625 [0.543, 0.712] | 0.588 [0.527, 0.650] | 0.641 [0.527, 0.748] | 0.565 [0.494, 0.637] |
| rf | task7 | 0.656 [0.584, 0.734] | 0.578 [0.526, 0.634] | 0.643 [0.557, 0.733] | 0.550 [0.479, 0.619] |
| rf | task9 | 0.653 [0.572, 0.733] | 0.613 [0.556, 0.665] | 0.548 [0.446, 0.646] | 0.641 [0.576, 0.705] |
| rf | fusion | 0.664 [0.584, 0.744] | 0.595 [0.539, 0.653] | 0.635 [0.527, 0.740] | 0.578 [0.507, 0.653] |
| fnn | task1 | 0.589 [0.511, 0.663] | 0.567 [0.516, 0.617] | 0.528 [0.429, 0.621] | 0.584 [0.527, 0.643] |
| fnn | task3 | 0.547 [0.477, 0.619] | 0.524 [0.470, 0.574] | 0.570 [0.489, 0.655] | 0.503 [0.434, 0.568] |
| fnn | task4 | 0.662 [0.588, 0.733] | 0.637 [0.586, 0.687] | 0.556 [0.453, 0.659] | 0.673 [0.621, 0.731] |
| fnn | task5 | 0.635 [0.566, 0.704] | 0.598 [0.546, 0.654] | 0.657 [0.571, 0.740] | 0.572 [0.505, 0.642] |
| fnn | task7 | 0.677 [0.605, 0.748] | 0.615 [0.561, 0.670] | 0.674 [0.576, 0.768] | 0.589 [0.522, 0.656] |
| fnn | task9 | 0.597 [0.520, 0.674] | 0.592 [0.536, 0.643] | 0.509 [0.413, 0.602] | 0.629 [0.563, 0.691] |
| fnn | fusion | 0.689 [0.623, 0.756] | 0.620 [0.569, 0.674] | 0.670 [0.590, 0.746] | 0.598 [0.532, 0.666] |
| tabpfn | task1 | 0.635 [0.549, 0.717] | 0.599 [0.546, 0.653] | 0.580 [0.471, 0.684] | 0.607 [0.542, 0.668] |
| tabpfn | task3 | 0.585 [0.506, 0.667] | 0.524 [0.469, 0.577] | 0.667 [0.584, 0.753] | 0.461 [0.386, 0.529] |
| tabpfn | task4 | 0.651 [0.563, 0.732] | 0.598 [0.546, 0.650] | 0.570 [0.460, 0.670] | 0.611 [0.549, 0.673] |
| tabpfn | task5 | 0.658 [0.578, 0.741] | 0.599 [0.542, 0.661] | 0.652 [0.546, 0.755] | 0.576 [0.505, 0.648] |
| tabpfn | task7 | 0.694 [0.614, 0.772] | 0.618 [0.562, 0.674] | 0.730 [0.631, 0.828] | 0.568 [0.498, 0.636] |
| tabpfn | task9 | 0.624 [0.536, 0.707] | 0.573 [0.517, 0.628] | 0.591 [0.490, 0.689] | 0.566 [0.501, 0.633] |
| tabpfn | fusion | 0.692 [0.618, 0.767] | 0.617 [0.559, 0.676] | 0.685 [0.588, 0.782] | 0.587 [0.513, 0.661] |
| tabpfn_meanprob | fusion | 0.701 [0.619, 0.781] | 0.621 [0.561, 0.680] | 0.713 [0.602, 0.823] | 0.580 [0.506, 0.650] |
| mmse_cutoff | clinical | 0.693 [0.619, 0.762] | 0.669 [0.599, 0.738] | 0.755 [0.650, 0.857] | 0.631 [0.546, 0.713] |

`mmse_cutoff` uses 175 visits (140 subjects; 53 MCI, 122 CN): the visit without an MMSE is excluded from this cell
only. With the cutoff MMSE < 28 the AUC is
0.666 [0.580, 0.751] (Sen 0.528 [0.375, 0.680], Spec 0.803 [0.730, 0.875]).

### cognitive_fusion arms

Source: [`output/cognitive_fusion_2026_11_24_1305/results.md`](output/cognitive_fusion_2026_11_24_1305/results.md).

| Arm | w | Test AUC [95% CI] | Test Acc [95% CI] | Test Sen [95% CI] | Test Spec [95% CI] |
|---|---:|---|---|---|---|
| cogfusion_w000 | 0 | 0.708 [0.626, 0.787] | 0.618 [0.556, 0.677] | 0.687 [0.574, 0.795] | 0.588 [0.516, 0.657] |
| cogfusion_w025 | 0.25 | 0.720 [0.638, 0.800] | 0.641 [0.575, 0.703] | 0.652 [0.525, 0.773] | 0.636 [0.560, 0.706] |
| **cogfusion_w050** (primary) | 0.5 | 0.725 [0.640, 0.807] | 0.669 [0.602, 0.732] | 0.661 [0.530, 0.788] | 0.672 [0.596, 0.743] |
| cogfusion_w075 | 0.75 | 0.723 [0.638, 0.807] | 0.660 [0.597, 0.720] | 0.657 [0.535, 0.782] | 0.661 [0.590, 0.732] |
| cogfusion_w100 | 1 | 0.718 [0.633, 0.802] | 0.667 [0.603, 0.728] | 0.630 [0.511, 0.754] | 0.684 [0.612, 0.757] |

### Paired Test AUC differences

These are the paired differences in Test AUC on identical (seed, visit) rows and shared bootstrap draws (B = 2000,
`contrasts.json`). ΔAUC is the mean over seeds of (AUC of the arm − AUC of the reference). "Seeds > 0" counts the
seeds with a positive per-seed difference.

| Arm | Reference | ΔAUC | 95% CI | Seeds > 0 |
|---|---|---:|---|---:|
| cogfusion_w000 | tabpfn_meanprob / fusion | +0.006 | [+0.002, +0.011] | 9 / 10 |
| cogfusion_w025 | tabpfn_meanprob / fusion | +0.018 | [+0.004, +0.035] | 10 / 10 |
| **cogfusion_w050** | tabpfn_meanprob / fusion | +0.023 | [−0.006, +0.054] | 10 / 10 |
| cogfusion_w075 | tabpfn_meanprob / fusion | +0.022 | [−0.020, +0.068] | 9 / 10 |
| cogfusion_w100 | tabpfn_meanprob / fusion | +0.017 | [−0.036, +0.075] | 8 / 10 |
| **cogfusion_w050** | tabpfn / fusion | +0.032 | [−0.011, +0.077] | 9 / 10 |

How to read these numbers:

- **No multiplicity adjustment is applied across the five arms; the CIs are descriptive only.**
- For the primary arm (w = 0.5), both CIs include 0.
- Only the w = 0 and w = 0.25 arms have CIs that exclude 0 against `tabpfn_meanprob`.
- The w = 0 arm contains no regressor information. Its score is a monotone transform of p̄ within each outer fold.
  Its small difference from `tabpfn_meanprob` comes from the per-fold z standardisation, which changes how scores
  from different outer folds rank against each other once the test rows of a seed are pooled.

Regressor diagnostic (`diagnostics.json`): the inner OOF Spearman ρ between r̄ and the training-row composite C is
0.384 on average over the 10 seeds (per-seed range 0.346 to 0.402).

### wide_fusion cells

Source: [`output/wide_fusion_2026_11_25_2006/results.md`](output/wide_fusion_2026_11_25_2006/results.md).
Acc, Sen and Spec are reported for the fused cell only.

| Cell | Test AUC [95% CI] | Test Acc [95% CI] | Test Sen [95% CI] | Test Spec [95% CI] |
|---|---|---|---|---|
| **wide_cf_w050** | **0.738 [0.663, 0.812]** | 0.648 [0.589, 0.709] | 0.741 [0.633, 0.843] | 0.607 [0.533, 0.681] |
| kdwide_cls | 0.683 [0.610, 0.757] | – | – | – |
| kdwide_reg | 0.676 [0.594, 0.759] | – | – | – |
| rdwide_cls | 0.675 [0.591, 0.754] | – | – | – |
| rdwide_reg | 0.722 [0.640, 0.798] | – | – | – |
| kd_meanprob | 0.679 [0.597, 0.763] | – | – | – |
| kd_meanreg | 0.688 [0.602, 0.773] | – | – | – |

Paired Test AUC differences (`contrasts.json`, same definition as above):

| Cell | Reference | ΔAUC | 95% CI | Seeds > 0 |
|---|---|---:|---|---:|
| **wide_cf_w050** | cogfusion_w050 | +0.013 | [−0.013, +0.042] | 7 / 10 |
| kdwide_cls | kd_meanprob | +0.004 | [−0.031, +0.039] | 5 / 10 |
| kdwide_reg | kd_meanreg | −0.012 | [−0.043, +0.021] | 2 / 10 |

The CI of wide_cf_w050 against cogfusion_w050 includes 0.

## Limitations

- **Small sample, single cohort.** There are 176 visits from 141 subjects (54 MCI visits), and the CIs are wide. No
  external validation cohort is available, so the results describe this cohort only.
- **Visit-level evaluation.** 35 subjects contribute two visits, and 3 of them change diagnosis between visits.
  Folds and bootstrap are grouped by subject, but the metrics weight subjects with two visits more than subjects
  with one.
- **What the CI covers.** The bootstrap resamples subjects among the saved test predictions. It does not refit the
  models, so the CIs do not include the variability of model training. The spread over seeds is reported separately
  in `metrics.json`.
- **Seeds.** Each seed sets both the fold split and the model random state, so the variation between seeds mixes
  these two sources.
- **MMSE cell.** The MMSE rule is evaluated on 175 visits instead of 176.
- **MMSE as a comparator.** MMSE is part of the clinical assessment of the cohort. The MMSE rule is therefore not an
  independent comparator for the diagnostic labels and should be read as a clinical reference point.
- **Use of clinical scores in cognitive_fusion.** In the same way, the cognitive_fusion regressors are trained on
  clinical scores that include MMSE. These scores are used as training targets only and never as test-time inputs.
- **Multiple comparisons.** Five cognitive_fusion arms and many baseline cells are reported, and no multiplicity
  adjustment is applied.

## Reference

Atiya S, Aschebrook-Kilfoy B, Konda S; ADNI. Comparative methods for determining MoCA, MMSE, and CDR cut-offs in
Alzheimer's disease: insights from the ADNI study. *Alzheimers Dement (Amst)*. 2026;18(1):e70291.
doi:[10.1002/dad2.70291](https://doi.org/10.1002/dad2.70291)
