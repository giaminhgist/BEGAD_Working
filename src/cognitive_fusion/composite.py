"""Clinical composite C (regression target), fitted on the training rows of each fit only."""
from __future__ import annotations
import numpy as np

SCORE_COLS = ["mmse", "verbal", "word", "naming", "delayed"]   # cohort columns: MMSE, verbal fluency, word list, naming, delayed recall


def sign_vector(sign_cfg: dict) -> np.ndarray:
    """Config `sign` (score name -> +1/-1) as a vector in SCORE_COLS order; -1 everywhere makes higher C = more impaired."""
    return np.array([sign_cfg[c] for c in SCORE_COLS], dtype=float)


def composite_params(S: np.ndarray, y: np.ndarray | None, check_sign: bool = True) -> dict:
    """mu_k, sd_k (ddof = 1; rows with a non-NaN score k, all diagnoses) of the training set S.

    Sign assertion: in every score the training-set CN mean exceeds the MCI mean; otherwise stop.
    """
    S = np.asarray(S, dtype=float)
    mu, sd = np.empty(S.shape[1]), np.empty(S.shape[1])
    for k in range(S.shape[1]):
        v = S[:, k]
        ok = ~np.isnan(v)
        mu[k], sd[k] = v[ok].mean(), v[ok].std(ddof=1)
        if check_sign and y is not None:
            yy = np.asarray(y)[ok]
            if not v[ok][yy == 0].mean() > v[ok][yy == 1].mean():
                raise RuntimeError(f"score {SCORE_COLS[k]}: training-set CN mean is not greater than the MCI mean; "
                                   "stop and report (a sign is never flipped silently)")
    if not (np.isfinite(sd).all() and (sd > 0).all()):
        raise RuntimeError("composite: a score has a zero or non-finite SD on the training set")
    return {"mu": mu, "sd": sd}


def composite(S: np.ndarray, params: dict, sign: np.ndarray, min_scores: int = 3):
    """C = (1/|A|) sum_{k in A} sign_k z_k over the row's non-NaN scores A (sign -1 everywhere: C = -mean(z)).

    Also returns |A| per row. Rows with |A| < min_scores get C = NaN (left out of the regressor's training rows only).
    """
    S = np.asarray(S, dtype=float)
    z = (S - params["mu"]) / params["sd"] * sign
    A = ~np.isnan(S)
    nA = A.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(A, z, 0.0).sum(axis=1) / nA
    return np.where(nA >= min_scores, C, np.nan), nA
