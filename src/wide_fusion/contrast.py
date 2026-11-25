"""Paired contrast of one cell against a reference on identical (seed, RealTag, outer_fold) rows (same arithmetic as
cognitive_fusion/contrast.py, with an outer-fold check; copied so that this family imports no other family).

Delta AUC = mean over seeds of (AUC_cell - AUC_ref). CI = percentile interval of the same-draw difference of the
subject-level bootstrap thetas (shared draws). No p-value is computed or stored."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.bootstrap import bootstrap_thetas, percentile_ci
from common.evaluate import ci_metric_fn
from common.metrics import auc_score


def _groups(pred: pd.DataFrame, seeds, subj_index: dict) -> list:
    out = []
    for s in seeds:
        p = pred[pred.seed == s]
        out.append([{"sidx": p["ParticipantID"].map(subj_index).to_numpy(), "y": p["y_true"].to_numpy(),
                     "score": p["y_score"].to_numpy()}])
    return out


def paired_report(pred: pd.DataFrame, ref: pd.DataFrame, draws: dict, seeds) -> dict:
    seeds = [int(s) for s in seeds]
    a = pred[pred.seed.isin(seeds)].sort_values(["seed", "RealTag"]).reset_index(drop=True)
    b = ref[ref.seed.isin(seeds)].sort_values(["seed", "RealTag"]).reset_index(drop=True)
    if not (len(a) == len(b) and a[["seed", "RealTag"]].equals(b[["seed", "RealTag"]])
            and np.array_equal(a["y_true"].to_numpy(float), b["y_true"].to_numpy(float))):
        raise AssertionError("paired rows differ (seed, RealTag, y_true)")
    if not np.array_equal(a["outer_fold"].to_numpy(int), b["outer_fold"].to_numpy(int)):
        raise AssertionError("paired rows differ in outer_fold (different splits)")
    if not np.array_equal(a["ParticipantID"].astype(str).to_numpy(), b["ParticipantID"].astype(str).to_numpy()):
        raise AssertionError("paired rows differ in ParticipantID")
    if sorted(set(a.seed)) != sorted(seeds):
        raise AssertionError("a requested seed is missing")
    subj = {s: i for i, s in enumerate(draws["subjects"])}
    per_seed = {s: auc_score(a[a.seed == s]["y_true"], a[a.seed == s]["y_score"])
                - auc_score(b[b.seed == s]["y_true"], b[b.seed == s]["y_score"]) for s in seeds}
    th_a = bootstrap_thetas(_groups(a, seeds, subj), draws["counts"], ci_metric_fn, ["auc"])["auc"]
    th_b = bootstrap_thetas(_groups(b, seeds, subj), draws["counts"], ci_metric_fn, ["auc"])["auc"]
    d = th_a - th_b
    d = d[np.isfinite(d)]
    ci = percentile_ci(d)
    return {"delta": float(np.mean(list(per_seed.values()))), "lo": ci["lo"], "hi": ci["hi"], "B": int(len(d)),
            "per_seed": per_seed, "n_seeds_positive": int(sum(v > 0 for v in per_seed.values())),
            "n_seeds": len(seeds), "n_visits": int((a.seed == seeds[0]).sum())}
