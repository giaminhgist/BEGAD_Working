"""Generate the outer/inner StratifiedGroupKFold fold files dataset/fold/seed_{seed}.csv and seeds.yaml.

Run (repository root):  PYTHONPATH=src .venv/bin/python -m common.make_folds [--seeds 0 1 ...]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .config import fold_dir, load_data_config, load_seeds
from .data import load_cohort

DEFAULT_SEEDS = [0, 1, 7, 13, 42, 123, 777, 1234, 2024, 3407]


def make_folds(cohort: pd.DataFrame, seed: int, outer: int = 4, inner: int = 4) -> pd.DataFrame:
    df = cohort[["RealTag", "ParticipantID", "label"]].reset_index(drop=True).copy()
    y, g = df["label"].to_numpy(), df["ParticipantID"].to_numpy()
    df["outer_fold"] = -1
    for o in range(outer):
        df[f"inner_fold_o{o}"] = -1
    outer_cv = StratifiedGroupKFold(n_splits=outer, shuffle=True, random_state=seed)
    for o, (tr, te) in enumerate(outer_cv.split(np.zeros(len(df)), y, g)):
        df.loc[te, "outer_fold"] = o
        inner_cv = StratifiedGroupKFold(n_splits=inner, shuffle=True, random_state=seed)
        for i, (_, va) in enumerate(inner_cv.split(np.zeros(len(tr)), y[tr], g[tr])):
            df.loc[tr[va], f"inner_fold_o{o}"] = i
    return df


def load_folds(seed: int) -> pd.DataFrame:
    return pd.read_csv(fold_dir() / f"seed_{seed}.csv")


README = """# Cross-validation folds

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
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_data_config()
    out = fold_dir()
    out.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else (load_seeds() if (out / "seeds.yaml").exists() else DEFAULT_SEEDS)
    cohort = load_cohort(cfg)
    for s in seeds:
        df = make_folds(cohort, s, cfg["cv"]["outer_splits"], cfg["cv"]["inner_splits"])
        df.to_csv(out / f"seed_{s}.csv", index=False)
        print(f"seed {s}: outer sizes {df['outer_fold'].value_counts().sort_index().tolist()}")
    (out / "seeds.yaml").write_text(f"seeds: {[int(s) for s in seeds]}\n", encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
