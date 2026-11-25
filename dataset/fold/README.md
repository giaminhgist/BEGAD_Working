# Cross-validation folds

One file per seed, `seed_{seed}.csv`, over the 176-visit CN/MCI cohort (141 subjects). Every model and every
feature set uses the same files. `seeds.yaml` lists the seeds.

Columns:
- `RealTag`: visit id (join key to the features and to the clinical data).
- `ParticipantID`: subject id; the grouping variable (no subject spans training and test).
- `label`: 1 = MCI, 0 = CN.
- `outer_fold`: 0..3, the outer test fold of this row.
- `inner_fold_o0` .. `inner_fold_o3`: for outer fold k, the inner validation fold (0..3) of the row within the
  training set of outer fold k; -1 if the row is in the test set of outer fold k (so `inner_fold_ok == -1` iff
  `outer_fold == k`).

Generation: `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)`, stratified by `label`, grouped by
`ParticipantID`; the inner splits use the same class and the same seed on the outer training rows. Rows are ordered
by RealTag.

`bootstrap_draws.npz` holds the shared bootstrap draws: B = 2000 stratified subject-level resamples (subjects
drawn with replacement within the ever-MCI and the CN-only strata). Arrays: `subjects` (sorted ParticipantID),
`stratum` (1 = ever-MCI), `counts` (B x n_subjects draw multiplicities), `seed`, `B`.

Regenerate (repository root):

    PYTHONPATH=src .venv/bin/python -m common.make_folds
    PYTHONPATH=src .venv/bin/python -m common.bootstrap
