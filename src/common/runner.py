"""Nested CV for one job = (model, feature set, seed): inner-CV config selection, Youden threshold on the inner OOF,
outer refit and test prediction.

Explicit loops (not GridSearchCV) so that the selected configuration's inner OOF predictions are saved.
"""
from __future__ import annotations
import gc
import importlib
import json
import random
import time
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from .config import estimator_fixed, expand_grid
from .data import load_analysis_data, merge_folds
from .make_folds import load_folds
from .metrics import auc_score, youden_threshold
from .pipelines import BuildContext, describe_features, wrap_pipeline


@dataclass
class Job:
    """Picklable description of one unit of work (one seed of one cell)."""
    family: str
    registry: str                 # module that exposes REGISTRY, e.g. "baseline.models"
    model: str
    model_cfg: dict
    feature_set: str
    seed: int
    smoke: bool = False
    n_threads: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model}/{self.feature_set}/seed_{self.seed}"

    def spec(self):
        return importlib.import_module(self.registry).REGISTRY[self.model]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import sys
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def _score(model, X, how: str) -> np.ndarray:
    if how == "predict_proba":
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.decision_function(X), dtype=float)


def fit_model(model, X, y):
    """Fit one estimator; return (model, seconds, number of ConvergenceWarnings)."""
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X, y)
    n_conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    return model, time.perf_counter() - t0, n_conv


def select_config(scores: list[float]) -> int:
    """Index of the config with the highest inner AUC; ties and NaN -> grid order (first wins)."""
    best, best_v = 0, None
    for i, v in enumerate(scores):
        if np.isnan(v):
            continue
        if best_v is None or v > best_v:
            best, best_v = i, v
    return best


def _build(spec, params, fixed, ctx):
    return wrap_pipeline(spec.build(dict(params), fixed, ctx), spec.pipeline)


def run_job(job: Job) -> dict:
    """All outer folds of one (model, feature set, seed). Returns plain-Python / pandas results."""
    t_start = time.perf_counter()
    seed_everything(job.seed)
    spec, mcfg = job.spec(), job.model_cfg
    X_df, y, groups, cohort = load_analysis_data(job.feature_set)
    folds = merge_folds(cohort, load_folds(job.seed))
    X = X_df.to_numpy(dtype=float)
    columns = np.array(X_df.columns)
    ids = cohort[["RealTag", "ParticipantID", "IsFollowup"]]
    grid = expand_grid(mcfg, job.smoke)
    fixed = estimator_fixed(mcfg, job.smoke)
    ctx = BuildContext(seed=job.seed, smoke=job.smoke, n_threads=job.n_threads)
    pred_rows, oof_rows, cv_rows, selected = [], [], [], {}
    outer = folds["outer_fold"].to_numpy()

    for o in range(4):
        tr_idx, te_idx = np.flatnonzero(outer != o), np.flatnonzero(outer == o)
        inner = folds[f"inner_fold_o{o}"].to_numpy()
        oofs, cfg_rows = [], []
        for ci, params in enumerate(grid):
            oof = np.full(len(y), np.nan)
            fold_scores, times, nconv = [], [], 0
            for i in range(4):
                itr, iva = tr_idx[inner[tr_idx] != i], tr_idx[inner[tr_idx] == i]
                m, secs, nc = fit_model(_build(spec, params, fixed, ctx), X[itr], y[itr])
                oof[iva] = _score(m, X[iva], spec.score)
                fold_scores.append(auc_score(y[iva], oof[iva]))
                times.append(secs)
                nconv += nc
                del m
                gc.collect()
            cfg_rows.append({"seed": job.seed, "outer_fold": o, "config_idx": ci, "params": json.dumps(params),
                             "inner_score": auc_score(y[tr_idx], oof[tr_idx]),
                             "inner_fold_scores": json.dumps(fold_scores), "fit_seconds": json.dumps(times),
                             "n_convergence_warnings": nconv})
            oofs.append(oof)
        bi = select_config([r["inner_score"] for r in cfg_rows])
        cv_rows += cfg_rows
        params, oof = grid[bi], oofs[bi]
        thr = youden_threshold(y[tr_idx], oof[tr_idx])           # inner OOF only
        sel = {"params": params, "config_idx": bi, "inner_score": cfg_rows[bi]["inner_score"], "threshold": thr}
        for r in tr_idx:
            oof_rows.append({"seed": job.seed, "outer_fold": o, "inner_fold": int(inner[r]), **ids.iloc[r].to_dict(),
                             "y_true": float(y[r]), "y_score": float(oof[r])})

        m, secs, nc = fit_model(_build(spec, params, fixed, ctx), X[tr_idx], y[tr_idx])
        te_pred = _score(m, X[te_idx], spec.score)
        sel.update({"refit_seconds": secs, "n_convergence_warnings": nc})
        sel.update(describe_features(X[tr_idx], columns))
        if getattr(m, "checkpoint_", None):
            sel["checkpoint"] = m.checkpoint_
        del m
        gc.collect()
        selected[o] = sel
        for r, p in zip(te_idx, te_pred):
            pred_rows.append({"seed": job.seed, "outer_fold": o, **ids.iloc[r].to_dict(), "y_true": float(y[r]),
                              "y_score": float(p), "threshold": thr, "y_hat": int(p >= thr)})

    return {"job": job.key, "predictions": pd.DataFrame(pred_rows), "inner_oof": pd.DataFrame(oof_rows),
            "cv_results": pd.DataFrame(cv_rows), "selected": selected,
            "seconds": time.perf_counter() - t_start, "n_features": int(X.shape[1])}
