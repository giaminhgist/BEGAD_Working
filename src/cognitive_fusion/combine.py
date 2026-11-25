"""Fusion of the classifier member and the regressor member on the z scale, plus the Spearman diagnostics.

- classifier member p: mean of the per-task TabPFN MCI probabilities (read from the baseline output);
- regressor member r: mean of the per-task TabPFN regressions of the clinical composite;
- l = logit(clip(p)); score s_w = (1 - w) z(l) + w z(r), with z fitted per (seed, outer fold) on inner-OOF rows only:
  test rows use the mean/SD over all inner-OOF rows of the fold, the inner-OOF rows of inner fold i use the other
  inner folds (leave own fold out).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common.latefusion import EXTRA, KEYS_O, KEYS_P, TASKS, finish_cell, mean_prob, members  # noqa: F401
from common.metrics import auc_score


def logit(p, clip: float = 1e-6):
    q = np.clip(np.asarray(p, dtype=np.float64), clip, 1 - clip)
    return np.log(q / (1 - q))


def rbar(reg_frames: dict, base: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Regressor score r = plain mean over the per-task regressors, aligned to the rows of `base`; all must be finite."""
    keyed = pd.MultiIndex.from_frame(base[keys])
    cols = []
    for t in TASKS:
        f = reg_frames[t]
        if f[keys].duplicated().any():
            raise AssertionError(f"regression task {t}: duplicated rows")
        cols.append(f.set_index(keys)["y_pred"].reindex(keyed).to_numpy())
    V = np.stack(cols, axis=1)
    if not np.isfinite(V).all():
        raise AssertionError("a regressor prediction is missing or not finite")
    return np.ascontiguousarray(V).sum(axis=1) / V.shape[1]


def mean_sd(v: np.ndarray) -> tuple[float, float]:
    m, s = float(np.mean(v)), float(np.std(v, ddof=1))
    if not (np.isfinite(s) and s > 0):
        raise ValueError("zero or non-finite SD in the z-scale reference")
    return m, s


class ZRef:
    """Mean/SD of l and r over a reference set; remembers the visits and inner folds it was built from."""

    def __init__(self, l, r, tags, folds):
        self.m_l, self.s_l = mean_sd(l)
        self.m_r, self.s_r = mean_sd(r)
        self.tags, self.inner_folds = tuple(tags), tuple(int(f) for f in folds)


def combine_fold(l_oof, r_oof, inner_fold, tags_oof, l_te, r_te, ws, ref_log=None):
    """Scores s_w for every w: {w: (inner-OOF scores, test scores)} of one (seed, outer fold)."""
    l_oof, r_oof, l_te, r_te = (np.asarray(a, float) for a in (l_oof, r_oof, l_te, r_te))
    inner_fold, tags_oof = np.asarray(inner_fold), np.asarray(tags_oof)
    zl_o, zr_o = np.empty_like(l_oof), np.empty_like(r_oof)
    for i in sorted(set(inner_fold.tolist())):
        m = inner_fold == i
        ref = ZRef(l_oof[~m], r_oof[~m], tags_oof[~m], inner_fold[~m])
        zl_o[m], zr_o[m] = (l_oof[m] - ref.m_l) / ref.s_l, (r_oof[m] - ref.m_r) / ref.s_r
        if ref_log is not None:
            ref_log.append({"target": f"inner{i}", "ref": ref, "target_tags": tuple(tags_oof[m])})
    ref = ZRef(l_oof, r_oof, tags_oof, inner_fold)
    zl_t, zr_t = (l_te - ref.m_l) / ref.s_l, (r_te - ref.m_r) / ref.s_r
    if ref_log is not None:
        ref_log.append({"target": "test", "ref": ref, "target_tags": None})
    return {w: ((1 - w) * zl_o + w * zr_o, (1 - w) * zl_t + w * zr_t) for w in ws}


def audit_references(ref_log: list, train_tags, test_tags) -> None:
    """z references contain only outer-training visits, never the target inner fold or any test visit."""
    train, test = set(train_tags), set(test_tags)
    for r in ref_log:
        ref, tags = r["ref"], set(r["ref"].tags)
        assert tags <= train and not (tags & test), "reference contains a non-training visit"
        if r["target"] == "test":
            assert tags == train, "test reference must be all outer-training visits"
        else:
            assert int(r["target"][5:]) not in set(ref.inner_folds), "reference contains rows of the target inner fold"
            assert not (tags & set(r["target_tags"])), "reference contains target rows"


def fusion_arms(pred_t: dict, oof_t: dict, reg_pred: dict, reg_oof: dict, ws, clip: float = 1e-6):
    """Cells {w: (predictions, inner_oof)} for every w, and the classifier member's scores for the identity check."""
    bt, Pt = members(pred_t, KEYS_P)
    bo, Po = members(oof_t, KEYS_O)
    p_te, p_oo = mean_prob(Pt), mean_prob(Po)
    r_te, r_oo = rbar(reg_pred, bt, KEYS_P), rbar(reg_oof, bo, KEYS_O)
    l_te, l_oo = logit(p_te, clip), logit(p_oo, clip)
    S_te = {w: np.full(len(bt), np.nan) for w in ws}
    S_oo = {w: np.full(len(bo), np.nan) for w in ws}
    g_oo_all = bo.groupby(["seed", "outer_fold"]).indices
    for (s, o), gi_te in bt.groupby(["seed", "outer_fold"]).indices.items():
        gi_oo = g_oo_all[(s, o)]
        log: list = []
        out = combine_fold(l_oo[gi_oo], r_oo[gi_oo], bo["inner_fold"].to_numpy()[gi_oo],
                           bo["RealTag"].to_numpy()[gi_oo], l_te[gi_te], r_te[gi_te], ws, log)
        audit_references(log, bo["RealTag"].to_numpy()[gi_oo], bt["RealTag"].to_numpy()[gi_te])
        for w in ws:
            S_oo[w][gi_oo], S_te[w][gi_te] = out[w]
    cells = {w: finish_cell(bt.assign(y_score=S_te[w]), bo.assign(y_score=S_oo[w])) for w in ws}
    return cells, {"p_te": (bt, p_te), "p_oo": (bo, p_oo), "r_oo": r_oo}


def arm_name(w: float) -> str:
    return f"cogfusion_w{int(round(w * 100)):03d}"


def rho(x, y) -> float:
    """Spearman rho on the pairs where both values are finite; NaN with fewer than 3 pairs or a constant side."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x[ok], y[ok])[0])


def seed_mean(values) -> float:
    """Mean over the seeds with a defined value; NaN if none is defined."""
    v = np.asarray(list(values), dtype=float)
    ok = ~np.isnan(v)
    return float(v[ok].mean()) if ok.any() else float("nan")


def inner_spearman(reg_oof_first: pd.DataFrame, bo: pd.DataFrame, r_oo: np.ndarray) -> dict:
    """Inner-OOF Spearman rho(r, C) per seed (training-row C) and its mean over seeds."""
    c = reg_oof_first.set_index(KEYS_O)["y_true"]
    cc = c.reindex(pd.MultiIndex.from_frame(bo[KEYS_O])).to_numpy()
    out = {}
    for s in sorted(bo["seed"].unique()):
        m = bo["seed"].to_numpy() == s
        out[int(s)] = rho(r_oo[m], cc[m])
    return {"per_seed": out, "mean": seed_mean(out.values())}
