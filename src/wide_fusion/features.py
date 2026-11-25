"""Wide card layout: the per-card features of one visit side by side, one row per visit.

Every function takes the *training rows* of one fit explicitly. The per-card z parameters come from those rows only and
never see a query row or a label. All statistics ignore NaN; a card-column whose SD is zero, non-finite or based on
fewer than 2 values is *degenerate*: its z is NaN for every visit (it is kept as a NaN column, never counted as 0).

Shapes: `Xs` is a list with one (n_visits, p) array per card (same visit order, same p columns); Z is (T, n, p).
"""
from __future__ import annotations
import numpy as np


def card_params(Xs, rows) -> dict:
    """Per card: nan-mean, nan-SD (ddof = 1) of every column over `rows`, and degenerate[t, j] (boolean).
    degenerate = fewer than 2 non-NaN values, or SD zero / non-finite."""
    rows = np.asarray(rows)
    mu, sd, deg = [], [], []
    for X in Xs:
        A = np.asarray(X, float)[rows]
        n = (~np.isnan(A)).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(n > 0, np.nansum(A, axis=0) / np.maximum(n, 1), np.nan)
            dev = np.where(np.isnan(A), 0.0, A - np.where(np.isnan(m), 0.0, m))
            s = np.sqrt((dev ** 2).sum(axis=0) / np.maximum(n - 1, 1))
        bad = (n < 2) | ~np.isfinite(s) | ~(s > 0)
        mu.append(m)
        sd.append(np.where(bad, np.nan, s))
        deg.append(bad)
    return {"mu": np.stack(mu), "sd": np.stack(sd), "degenerate": np.stack(deg)}


def z_cards(Xs, rows, params) -> np.ndarray:
    """(T, len(rows), p) per-card z of the given rows; NaN wherever the card-column is degenerate or the value is NaN."""
    rows = np.asarray(rows)
    out = []
    for t, X in enumerate(Xs):
        Z = (np.asarray(X, float)[rows] - params["mu"][t]) / params["sd"][t]
        Z[:, params["degenerate"][t]] = np.nan
        out.append(Z)
    return np.stack(out)


def _check_rows(train_rows, query_rows) -> tuple:
    tr, q = np.asarray(train_rows), np.asarray(query_rows)
    if len(np.intersect1d(tr, q)) or len(np.unique(q)) != len(q) or len(np.unique(tr)) != len(tr):
        raise AssertionError("training and query visits overlap, or a visit is repeated")
    return tr, q


def wide_features(Xs, cols, card_ids, train_rows, query_rows) -> dict:
    """Design of one fit: the per-card z blocks side by side (T * p columns, names prefixed by card). Degenerate
    card-columns are NaN. The per-card z is close to a no-op for TabPFN (it preprocesses each column itself)."""
    tr, q = _check_rows(train_rows, query_rows)
    cols = list(cols)
    par = card_params(Xs, tr)
    Ztr, Zq = z_cards(Xs, tr, par), z_cards(Xs, q, par)
    cat = lambda Z: np.hstack([Z[t] for t in range(Z.shape[0])])  # noqa: E731
    names = [f"t{c}__{col}" for c in card_ids for col in cols]
    log = {"degenerate": par["degenerate"].astype(int).tolist(), "n_degenerate": int(par["degenerate"].sum()),
           "n_columns_in": len(names), "n_features": len(names), "excluded": []}
    return {"Xtr": cat(Ztr), "Xq": cat(Zq), "names": names, "log": log}
