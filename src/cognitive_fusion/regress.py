"""Regressor member: per-task nested-CV regression of the composite C from the eye features (no clinical score and
no age as input).

`run_regression_job` is a pure function of arrays (X, scores, labels, folds); the model comes from `make_model(seed)`, so
tests can use a stub. Every fit records the visits it was fitted on (`fit_log`), which `audit_fit_log` checks.
"""
from __future__ import annotations
import gc
import re
import time
import numpy as np
import pandas as pd
from .composite import SCORE_COLS, composite, composite_params

FORBIDDEN_NAME = re.compile(r"Ikä|age|CDR|MMSE", re.IGNORECASE)


def audit_inputs(cols, clf_cols, X: np.ndarray, age: np.ndarray, S: np.ndarray, cogdata_cols) -> None:
    """Regressor columns = classifier columns (same order); no COGDATA/age/CDR/MMSE column name; no column with
    |Pearson r| > 0.999 with age or with any clinical score over the cohort."""
    if list(cols) != list(clf_cols):
        raise AssertionError("regressor X columns differ from the classifier columns")
    bad = [c for c in cols if c in set(cogdata_cols) or FORBIDDEN_NAME.search(c)]
    if bad:
        raise AssertionError(f"forbidden input columns: {bad}")
    targets = {"age": np.asarray(age, float), **{k: S[:, i] for i, k in enumerate(SCORE_COLS)}}
    for j, c in enumerate(cols):
        for name, v in targets.items():
            ok = ~np.isnan(X[:, j]) & ~np.isnan(v)
            if ok.sum() > 2 and np.std(X[ok, j]) > 0 and np.std(v[ok]) > 0 and abs(np.corrcoef(X[ok, j], v[ok])[0, 1]) > 0.999:
                raise AssertionError(f"column {c} has |r| > 0.999 with {name}")


def run_regression_job(task: int, seed: int, X: np.ndarray, cols, S: np.ndarray, y: np.ndarray, ids: pd.DataFrame,
                       folds: pd.DataFrame, make_model, sign: np.ndarray, min_scores: int = 3,
                       outer_folds=range(4)) -> dict:
    """All outer folds of one (task, seed). X: eye features only; S: clinical scores (training targets only, never inputs);
    y: labels (only the start-up sign assertion on training rows reads them); ids: RealTag, ParticipantID, IsFollowup."""
    t_start = time.perf_counter()
    pred_rows, oof_rows, params_log, selected, fit_log = [], [], {}, {}, []
    outer = folds["outer_fold"].to_numpy()
    tags = ids["RealTag"].to_numpy()
    for o in outer_folds:
        tr_idx, te_idx = np.flatnonzero(outer != o), np.flatnonzero(outer == o)
        inner = folds[f"inner_fold_o{o}"].to_numpy()
        # composite parameters of the refit (outer-train rows) are also the scale of the saved training-row C of this fold
        p_refit = composite_params(S[tr_idx], y[tr_idx])
        C_tr, nA_tr = composite(S[tr_idx], p_refit, sign, min_scores)
        c_of = dict(zip(tr_idx.tolist(), C_tr.tolist()))
        params_log[o] = {"refit": {**_pj(p_refit), "n_excluded": int((nA_tr < min_scores).sum()),
                                   "excluded": tags[tr_idx][nA_tr < min_scores].tolist()}, "inner": {}}
        sel = {"fit_seconds": [], "predict_seconds": []}
        for i in range(4):
            itr, iva = tr_idx[inner[tr_idx] != i], tr_idx[inner[tr_idx] == i]
            p_i = composite_params(S[itr], y[itr])
            C, nA = composite(S[itr], p_i, sign, min_scores)
            keep = nA >= min_scores
            params_log[o]["inner"][i] = {**_pj(p_i), "n_excluded": int((~keep).sum()), "excluded": tags[itr][~keep].tolist()}
            m = make_model(seed)
            t0 = time.perf_counter()
            m.fit(X[itr][keep], C[keep])
            sel["fit_seconds"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            pr = np.asarray(m.predict(X[iva]), dtype=float)
            sel["predict_seconds"].append(time.perf_counter() - t0)
            fit_log.append({"seed": seed, "outer_fold": o, "fit": f"inner{i}", "task": task, "train": tags[itr][keep].tolist(),
                            "eval": tags[iva].tolist()})
            for r, p in zip(iva, pr):
                oof_rows.append({"seed": seed, "outer_fold": o, "inner_fold": int(i), **ids.iloc[r].to_dict(),
                                 "y_true": c_of[int(r)], "y_pred": float(p)})
            del m
            gc.collect()
        keep = ~np.isnan(C_tr)
        m = make_model(seed)
        t0 = time.perf_counter()
        m.fit(X[tr_idx][keep], C_tr[keep])
        sel["refit_seconds"] = time.perf_counter() - t0
        sup = getattr(m, "support_", None)
        sel["dropped_columns"] = [] if sup is None else [c for j, c in enumerate(cols) if j not in set(np.asarray(sup).tolist())]
        t0 = time.perf_counter()
        pr = np.asarray(m.predict(X[te_idx]), dtype=float)
        sel["refit_predict_seconds"] = time.perf_counter() - t0
        fit_log.append({"seed": seed, "outer_fold": o, "fit": "refit", "task": task, "train": tags[tr_idx][keep].tolist(),
                        "eval": tags[te_idx].tolist()})
        for r, p in zip(te_idx, pr):
            pred_rows.append({"seed": seed, "outer_fold": o, **ids.iloc[r].to_dict(), "y_true": np.nan, "y_pred": float(p)})
        del m
        gc.collect()
        selected[o] = sel
    return {"predictions": pd.DataFrame(pred_rows), "inner_oof": pd.DataFrame(oof_rows), "composite_params": params_log,
            "selected": selected, "fit_log": fit_log, "seconds": time.perf_counter() - t_start}


def _pj(p: dict) -> dict:
    return {"mu": p["mu"].tolist(), "sd": p["sd"].tolist()}


def audit_fit_log(fit_log: list, ids: pd.DataFrame, folds: pd.DataFrame) -> None:
    """For every fit, the visits it was fitted on are training rows of that fit and are disjoint (visit and subject)
    from its evaluation rows; inner fits stay inside the outer-training rows."""
    pid = dict(zip(ids["RealTag"], ids["ParticipantID"]))
    outer = dict(zip(ids["RealTag"], folds["outer_fold"].to_numpy()))
    for f in fit_log:
        tr, ev, o = set(f["train"]), set(f["eval"]), f["outer_fold"]
        assert not (tr & ev), f"{f['fit']}: train/eval visit overlap"
        assert not ({pid[t] for t in tr} & {pid[t] for t in ev}), f"{f['fit']}: train/eval subject overlap"
        assert all(outer[t] != o for t in tr), f"{f['fit']}: a fold-{o} test visit was fitted on"
        if f["fit"] != "refit":
            i = int(f["fit"][5:])
            inner = dict(zip(ids["RealTag"], folds[f"inner_fold_o{o}"].to_numpy()))
            assert all(inner[t] != i for t in tr) and all(inner[t] == i for t in ev), f"{f['fit']}: wrong inner split"
