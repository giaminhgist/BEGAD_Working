"""Helpers for scores built from the saved outputs of other cells (late fusion): merging per-task frames, the
leakage check on predictions vs inner OOF rows, and the Youden-threshold step that turns scores into a cell."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import youden_threshold

TASKS = [1, 3, 4, 5, 7, 9]
KEYS_P = ["seed", "outer_fold", "RealTag"]
KEYS_O = ["seed", "outer_fold", "inner_fold", "RealTag"]
EXTRA = ["ParticipantID", "IsFollowup", "y_true"]
PRED_COLS = ["seed", "outer_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score", "threshold", "y_hat"]
OOF_COLS = ["seed", "outer_fold", "inner_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score"]


def mean_prob(P) -> np.ndarray:
    """Row-wise mean of the per-task probabilities; every entry must be finite."""
    P = np.asarray(P, dtype=np.float64)
    if not np.isfinite(P).all():
        raise AssertionError("a member score is missing or not finite")
    return P.sum(axis=1) / P.shape[1]


def members(frames: dict[int, pd.DataFrame], keys: list[str], col: str = "y_score"):
    """Merge per-task frames on `keys` (labels must agree). Returns (base frame, matrix with one column per task).

    Row order is that of first appearance over the tasks. Every row must be present for every task.
    """
    tasks = sorted(frames)
    base = pd.concat([frames[t][keys + EXTRA] for t in tasks], ignore_index=True).drop_duplicates(keys, ignore_index=True)
    keyed = pd.MultiIndex.from_frame(base[keys])
    P = np.full((len(base), len(tasks)), np.nan)
    for j, t in enumerate(tasks):
        f = frames[t]
        if f[keys].duplicated().any():
            raise AssertionError(f"task {t}: duplicated rows")
        s = f.set_index(keys)
        P[:, j] = s[col].reindex(keyed).to_numpy()
        y = s["y_true"].reindex(keyed).to_numpy()
        ok = ~np.isnan(y)
        if not np.array_equal(y[ok], base["y_true"].to_numpy()[ok]):
            raise AssertionError(f"task {t}: labels differ between tasks")
    if not np.isfinite(P).all():
        raise AssertionError("a member score is missing for some row")
    return base, P


def check_no_leakage(pred: pd.DataFrame, oof: pd.DataFrame) -> None:
    """No outer-test visit or subject among the inner-OOF rows of the same (seed, outer fold); every (seed, visit)
    is predicted once; no visit appears twice in an inner OOF."""
    if pred.duplicated(["seed", "RealTag"]).any():
        raise AssertionError("a (seed, visit) is predicted in more than one outer fold")
    for (s, o), te in pred.groupby(["seed", "outer_fold"]):
        tr = oof[(oof.seed == s) & (oof.outer_fold == o)]
        if set(te["RealTag"]) & set(tr["RealTag"]) or set(te["ParticipantID"]) & set(tr["ParticipantID"]):
            raise AssertionError(f"seed {s} outer fold {o}: inner-OOF rows overlap the outer-test rows/subjects")
        if tr.duplicated(["RealTag"]).any():
            raise AssertionError(f"seed {s} outer fold {o}: a visit appears twice in the inner OOF")


def finish_cell(pred: pd.DataFrame, oof: pd.DataFrame):
    """(predictions, inner_oof) in the cell layout from frames that carry y_score: the Youden threshold of each
    (seed, outer fold) is computed on its pooled inner-OOF scores and applied to that fold's test scores."""
    check_no_leakage(pred, oof)
    thr = {(s, o): youden_threshold(q["y_true"].to_numpy(), q["y_score"].to_numpy())
           for (s, o), q in oof.groupby(["seed", "outer_fold"])}
    pred = pred.copy()
    pred["threshold"] = [thr[(s, o)] for s, o in zip(pred["seed"], pred["outer_fold"])]
    pred["y_hat"] = (pred["y_score"] >= pred["threshold"]).astype(int)
    return pred[PRED_COLS].reset_index(drop=True), oof[OOF_COLS].reset_index(drop=True)


def load_task_cells(run_dir, model: str, tasks: list[int], seeds: list[int]):
    """Per-task predictions and inner OOF of `model` from `run_dir/<model>/task{t}/`, restricted to `seeds`."""
    from pathlib import Path
    pred, oof = {}, {}
    for t in tasks:
        cell = Path(run_dir) / model / f"task{t}"
        for name, store in (("predictions", pred), ("inner_oof", oof)):
            f = cell / f"{name}.csv"
            if not f.exists():
                raise FileNotFoundError(f"{model}/task{t}/{name}.csv not found in {Path(run_dir).name}")
            d = pd.read_csv(f)
            d = d[d.seed.isin(seeds)].reset_index(drop=True)
            if set(d.seed.unique()) != set(seeds):
                raise RuntimeError(f"{model}/task{t}: seeds {sorted(d.seed.unique())} differ from the requested {seeds}")
            store[t] = d
    return pred, oof
