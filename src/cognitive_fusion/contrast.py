"""Paired contrast of one cell against a reference cell on identical (seed, visit) rows.

Delta AUC = mean over seeds of (AUC_arm - AUC_ref). The CI is the percentile interval of the same-draw difference of
the subject-level bootstrap thetas. No p-value is computed. No multiplicity adjustment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.bootstrap import bootstrap_thetas, percentile_ci
from common.evaluate import ci_metric_fn
from common.metrics import auc_score


def _groups(pred: pd.DataFrame, seeds: list[int], subj_index: dict) -> list[list[dict]]:
    out = []
    for s in seeds:
        p = pred[pred.seed == s]
        out.append([{"sidx": p["ParticipantID"].map(subj_index).to_numpy(), "y": p["y_true"].to_numpy(),
                     "score": p["y_score"].to_numpy()}])
    return out


def paired_delta(pred_cc: pd.DataFrame, pred_ref: pd.DataFrame, seeds: list[int], draws: dict) -> dict:
    """Delta AUC = mean over seeds of (AUC_cc - AUC_ref); CI from the same-draw difference of bootstrap thetas
    (float, percentiles)."""
    subj_index = {s: i for i, s in enumerate(draws["subjects"])}
    per_seed = [auc_score(pred_cc[pred_cc.seed == s]["y_true"], pred_cc[pred_cc.seed == s]["y_score"])
                - auc_score(pred_ref[pred_ref.seed == s]["y_true"], pred_ref[pred_ref.seed == s]["y_score"])
                for s in seeds]
    th_cc = bootstrap_thetas(_groups(pred_cc, seeds, subj_index), draws["counts"], ci_metric_fn, ["auc"])["auc"]
    th_ref = bootstrap_thetas(_groups(pred_ref, seeds, subj_index), draws["counts"], ci_metric_fn, ["auc"])["auc"]
    d = th_cc - th_ref
    d = d[np.isfinite(d)]
    return {"delta": float(np.mean(per_seed)), "ci": percentile_ci(d), "B": len(d)}


def paired_report(pred: pd.DataFrame, ref: pd.DataFrame, draws: dict, seeds: list[int] | None = None) -> dict:
    """paired_delta plus the per-seed deltas; the two frames must hold the same (seed, RealTag, y_true, ParticipantID)."""
    seeds = seeds if seeds is not None else sorted(int(s) for s in pred["seed"].unique())
    a = pred[pred.seed.isin(seeds)].sort_values(["seed", "RealTag"]).reset_index(drop=True)
    b = ref[ref.seed.isin(seeds)].sort_values(["seed", "RealTag"]).reset_index(drop=True)
    if not (a[["seed", "RealTag"]].equals(b[["seed", "RealTag"]]) and np.array_equal(a["y_true"], b["y_true"])):
        raise AssertionError("paired rows differ (seed, RealTag, y_true)")
    if not np.array_equal(a["ParticipantID"].astype(str).to_numpy(), b["ParticipantID"].astype(str).to_numpy()):
        raise AssertionError("paired rows differ in ParticipantID")
    r = paired_delta(a, b, seeds, draws)
    per_seed = {s: auc_score(a[a.seed == s]["y_true"], a[a.seed == s]["y_score"])
                - auc_score(b[b.seed == s]["y_true"], b[b.seed == s]["y_score"]) for s in seeds}
    return {"delta": r["delta"], "lo": r["ci"]["lo"], "hi": r["ci"]["hi"], "B": r["B"],
            "per_seed": per_seed, "n_pos": int(sum(v > 0 for v in per_seed.values())), "n_seeds": len(seeds)}
