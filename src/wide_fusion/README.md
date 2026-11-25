# wide_fusion

This family scores CN vs MCI from the eye-tracking features of one visit with **one model per block** instead of one
model per task. A visit has up to six recordings (cards): four King-Devick cards (tasks 1, 3, 4, 5) and two reading
passages (tasks 7, 9). The earlier families fit one TabPFN per card and average the six outputs. Here the features of
the cards of a block are kept **side by side** in one row, so a single TabPFN sees them all at once:

- **KD block**: tasks 1, 3, 4, 5 with 63 columns per card, 4 × 63 = 252 columns.
- **Reading block**: tasks 7, 9 with 60 columns per card, 2 × 60 = 120 columns.

One row per visit, one TabPFN per block. Nothing is averaged across cards inside the model input.

Two members use this layout, each with `n_estimators` 2:

- `wide_cls_e2`: a classifier of the diagnosis. p̄ is the mean of the two block probabilities (KD, reading).
- `wide_reg_C_e2`: a regressor of the clinical composite C (the one of `cognitive_fusion`: mean of the sign-aligned
  z-scores of MMSE, verbal fluency, word-list total, naming and delayed recall; recomputed on the training rows of
  every fit; visits with fewer than 3 scores are left out of the regressor's training rows). r̄ is the mean of the two
  block regressions. The clinical scores are training targets only, never inputs.

The two scores are fused on the z scale, as in `cognitive_fusion`:

    s = 0.5 · z(logit p̄) + 0.5 · z(r̄)

The z references come from the inner out-of-fold rows of each (seed, outer fold) only; the decision threshold is the
Youden threshold on the fused inner out-of-fold scores.

Inside every fit (each inner fit and each outer refit), the per-card, per-column z-scaling (ddof 1) uses that fit's
training visits only. A card-column with SD 0, a non-finite SD or fewer than 2 values is NaN for every visit; it is kept
as a NaN column and never counted as 0. TabPFN may drop all-constant or all-NaN columns itself; that is logged.
**Feature subsampling is a stop.** Every fit is run with TabPFN's subsampling warnings turned into errors, and the
per-estimator feature count is checked against the fitted width. If an estimator would see fewer columns than it is
given, the run halts and nothing is changed to get around it. The counts are logged in `run_info.json`
(`tabpfn_feature_check`).

## Run

Settings are in `config.yaml`: tasks and blocks, the target C, the two members, the saved runs, the cells, `w`,
`clip`, the reference runs and the contrasts.

From the repository root:

    PYTHONPATH=src .venv/bin/python -m wide_fusion.run --smoke
    PYTHONPATH=src .venv/bin/python -m wide_fusion.run

A smoke run (seed 0, 200 bootstrap draws) writes `output/smoke_YYYY_MM_DD_HHMM/wide_fusion`; its numbers are a code
check, not results. The full run uses every seed of `dataset/fold/seeds.yaml` and 2000 draws, and needs the two finished
full runs named in `config.yaml` (it stops if one is missing, not completed, or a smoke run). Jobs are (member, seed,
outer fold) and run on `--tabpfn-workers` × `--tabpfn-threads` (default 4 × 4). Each job writes a partial result with a
fingerprint (settings, code, library versions, inputs); a rerun resumes from matching partials and stops if one does not
match, so use a new `--out` after a code or settings change. Expected time on 16 cores: about 15 minutes.

## Output

`output/wide_fusion_YYYY_MM_DD_HHMM/`:

- `<cell>/fusion/` with `predictions.csv`, `inner_oof.csv`, `selected_params.json` and `metrics.json`, for every cell above;
- `members/<member>/` with the member's `predictions.csv`, `inner_oof.csv`, the per-block scores (`rows_predictions.csv`,
  `rows_inner_oof.csv`), `fit_log.csv`, `wide_fit_log.json` and, for the regressor, `target_params.json`;
- `contrasts.json`, `mmse_rules.json`, `diagnostics.json`;

`metrics.json` also keeps a validation AUC in `summary.val_auc`; it is not shown in `results.md`.

