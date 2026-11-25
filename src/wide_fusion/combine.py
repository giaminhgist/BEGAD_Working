"""Fusion of a classifier score (p̄) and a regressor score (r̄) on the z scale (same arithmetic as
cognitive_fusion/combine.py, copied so that this family imports no other family):

    s = (1 - w) z(logit clip(p̄)) + w z(r̄),  w = 0.5

z is fitted per (seed, outer fold) on inner-OOF rows only: test rows use mean/SD over all inner-OOF rows of the fold;
the inner-OOF rows of inner fold i use the other inner folds (own fold left out). The Youden threshold is computed on
the fused inner-OOF scores (common.latefusion.finish_cell)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.latefusion import KEYS_O, KEYS_P, finish_cell


def logit(p, clip: float = 1e-6):
    q = np.clip(np.asarray(p, dtype=np.float64), clip, 1 - clip)
    return np.log(q / (1 - q))


def mean_sd(v) -> tuple[float, float]:
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

    def as_dict(self) -> dict:
        return {"m_l": self.m_l, "s_l": self.s_l, "m_r": self.m_r, "s_r": self.s_r, "n_rows": len(self.tags)}


def combine_fold(l_oof, r_oof, inner_fold, tags_oof, l_te, r_te, w, ref_log=None):
    """(inner-OOF fused scores, test fused scores) of one (seed, outer fold)."""
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
    return (1 - w) * zl_o + w * zr_o, (1 - w) * zl_t + w * zr_t


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


def _align(base: pd.DataFrame, other: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """other's y_score on base's rows; same row set and the same labels are required."""
    if other[keys].duplicated().any() or base[keys].duplicated().any():
        raise AssertionError("duplicated member rows")
    if len(other) != len(base):
        raise AssertionError("members cover different rows")
    o = other.set_index(keys).reindex(pd.MultiIndex.from_frame(base[keys]))
    if o["y_score"].isna().any() or not np.isfinite(o["y_score"].to_numpy(float)).all():
        raise AssertionError("members cover different rows")
    if not np.array_equal(o["y_true"].to_numpy(float), base["y_true"].to_numpy(float)):
        raise AssertionError("members disagree on labels")
    if not np.array_equal(o["ParticipantID"].astype(str).to_numpy(), base["ParticipantID"].astype(str).to_numpy()):
        raise AssertionError("members disagree on ParticipantID")
    return o["y_score"].to_numpy(float)


def fuse(cls: tuple, reg: tuple, w: float, clip: float):
    """cls / reg = (test frame, inner-OOF frame) with y_score = p̄ / r̄. Returns ((predictions, inner_oof), selected)."""
    # rows in key order, so the z references do not depend on the row order of the member files
    bt = cls[0].sort_values(KEYS_P, kind="mergesort").reset_index(drop=True)
    bo = cls[1].sort_values(KEYS_O, kind="mergesort").reset_index(drop=True)
    r_te, r_oo = _align(bt, reg[0], KEYS_P), _align(bo, reg[1], KEYS_O)
    l_te, l_oo = logit(bt["y_score"], clip), logit(bo["y_score"], clip)
    S_te, S_oo = np.full(len(bt), np.nan), np.full(len(bo), np.nan)
    g_oo = bo.groupby(["seed", "outer_fold"]).indices
    sel: dict = {}
    for (s, o), gi_te in bt.groupby(["seed", "outer_fold"]).indices.items():
        gi = g_oo[(s, o)]
        log: list = []
        S_oo[gi], S_te[gi_te] = combine_fold(l_oo[gi], r_oo[gi], bo["inner_fold"].to_numpy()[gi], bo["RealTag"].to_numpy()[gi],
                                             l_te[gi_te], r_te[gi_te], w, log)
        audit_references(log, bo["RealTag"].to_numpy()[gi], bt["RealTag"].to_numpy()[gi_te])
        sel.setdefault(str(int(s)), {})[str(int(o))] = {
            "z_ref_test": log[-1]["ref"].as_dict(),
            "z_ref_inner": {r["target"]: r["ref"].as_dict() for r in log[:-1]},
            "n_clipped_rows": int(((bo["y_score"].to_numpy()[gi] < clip) | (bo["y_score"].to_numpy()[gi] > 1 - clip)).sum()
                                  + ((bt["y_score"].to_numpy()[gi_te] < clip) | (bt["y_score"].to_numpy()[gi_te] > 1 - clip)).sum())}
    pred, oof = finish_cell(bt.assign(y_score=S_te), bo.assign(y_score=S_oo))
    for s, d in sel.items():
        for o, v in d.items():
            thr = pred.loc[(pred.seed == int(s)) & (pred.outer_fold == int(o)), "threshold"].unique()
            assert len(thr) == 1
            v["threshold"] = float(thr[0])
    return (pred, oof), sel


def single_cell(frames: tuple):
    """A member alone as a cell: score = p̄ or r̄ as is; Youden threshold on its inner OOF."""
    return finish_cell(frames[0].copy(), frames[1].copy())


def identity_check(pred: pd.DataFrame, oof: pd.DataFrame, ref_cell, seeds) -> None:
    """The classifier member equals the baseline tabpfn_meanprob cell exactly (test and inner-OOF scores)."""
    from pathlib import Path
    for frame, name, keys in ((pred, "predictions.csv", KEYS_P), (oof, "inner_oof.csv", KEYS_O)):
        r = pd.read_csv(Path(ref_cell) / name)
        r = r[r.seed.isin(seeds)]
        if len(r) != len(frame):
            raise AssertionError(f"{name}: row count differs from the reference cell")
        rr = r.set_index(keys)["y_score"].reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
        d = float(np.max(np.abs(rr - frame["y_score"].to_numpy())))
        if not d == 0.0:
            raise AssertionError(f"classifier member differs from the reference cell {name} (max |d| = {d})")
