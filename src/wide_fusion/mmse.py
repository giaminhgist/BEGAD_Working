"""MMSE rule rows: the rules "MCI if MMSE <= 28" and "MCI if MMSE < 28" scored as a binary prediction.

AUC = ROC AUC of the rule prediction = (sensitivity + specificity) / 2. Nothing is fitted. The rule rows are
the outer-test rows of the finished baseline `mmse_cutoff` cell (visits without an MMSE are not in it); the second
rule uses the same rows and the MMSE of the cohort. Aggregation is the one of the other cells: per seed the pooled
outer-test rows give the metric, the metric is averaged over the seeds, and the 95% CI is the percentile interval of the
same statistic over the shared stratified subject-level bootstrap draws.

Replicating each subject's visits m_j times equals weighting each row by m_j, so tp/fn/tn/fp are weighted sums (a
test checks this against `common.bootstrap.bootstrap_thetas`)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.bootstrap import percentile_ci
from common.config import repo_path
from common.data import load_cohort
from common.metrics import auc_score

NAMES = ["auc01", "sens", "spec"]
RULES = {"mmse_cutoff": 28.0, "mmse_cutoff_lt28": 27.0}   # MCI iff MMSE <= value (MMSE < 28 is MMSE <= 27: scores are integers)


def point_metrics(y, yhat) -> dict:
    """AUC (rank form on y_hat, ties 1/2), sensitivity, specificity of one pooled set of rows."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    pos, neg = y == 1, y != 1
    sens = float((yhat[pos] == 1).mean()) if pos.any() else float("nan")
    spec = float((yhat[neg] == 0).mean()) if neg.any() else float("nan")
    return {"auc01": auc_score(y, yhat), "sens": sens, "spec": spec}


def boot_thetas(pred: pd.DataFrame, draws: dict) -> dict:
    """theta*_b = mean over seeds of the pooled-seed metric on the replicated rows (vectorised, B x n weights)."""
    subj_index = {s: i for i, s in enumerate(draws["subjects"])}
    counts = draws["counts"].astype(float)                       # B x n_subjects
    acc = {n: [] for n in NAMES}
    for s in sorted(pred["seed"].unique()):
        p = pred[pred.seed == s]
        W = counts[:, p["ParticipantID"].map(subj_index).to_numpy()]   # B x rows
        y, h = p["y_true"].to_numpy() == 1, p["y_hat"].to_numpy() == 1
        with np.errstate(invalid="ignore", divide="ignore"):
            n1, n0 = W @ y, W @ ~y
            sens, spec = (W @ (y & h)) / n1, (W @ (~y & ~h)) / n0
        acc["sens"].append(sens)
        acc["spec"].append(spec)
        acc["auc01"].append((sens + spec) / 2)
    return {n: np.mean(acc[n], axis=0) for n in NAMES}            # NaN anywhere -> NaN replicate (as in the official code)


def score_rule(pred: pd.DataFrame, draws: dict) -> dict:
    seeds = sorted(pred["seed"].unique())
    per_seed = {int(s): point_metrics(pred.loc[pred.seed == s, "y_true"], pred.loc[pred.seed == s, "y_hat"]) for s in seeds}
    th = boot_thetas(pred, draws)
    out = {"n_seeds": len(seeds), "n_visits": int((pred.seed == seeds[0]).sum()), "per_seed": per_seed}
    for n in NAMES:
        ci = percentile_ci(th[n])
        out[n] = {"mean": float(np.mean([per_seed[s][n] for s in per_seed])), "lo": ci["lo"], "hi": ci["hi"]}
    return out


def rule_frames(saved: pd.DataFrame, mmse_by_tag: pd.Series) -> dict:
    """{rule name: frame with y_hat} on the rows of the saved mmse_cutoff cell. The saved score must be the rule
    prediction (1 if MMSE <= 28, else 0) of the cohort and the saved decision must be the same; the other rule is
    computed from the same MMSE."""
    mmse = mmse_by_tag.reindex(saved["RealTag"]).to_numpy(float)
    if not np.array_equal(saved["y_score"].to_numpy(float), (mmse <= RULES["mmse_cutoff"]).astype(float)):
        raise AssertionError("mmse_cutoff scores differ from the rule prediction (MMSE <= 28) of the cohort")
    out = {}
    for name, cut in RULES.items():
        d = saved.copy()
        d["y_hat"] = (mmse <= cut).astype(int)
        out[name] = d
    if not np.array_equal(out["mmse_cutoff"]["y_hat"].to_numpy(), saved["y_hat"].to_numpy()):
        raise AssertionError("mmse_cutoff decisions differ from the rule MMSE <= 28")
    return out


def mmse_rows(baseline_run, seeds, draws: dict) -> dict:
    """Rows of both rules from the finished baseline run and the cohort MMSE."""
    f = repo_path(baseline_run) / "mmse_cutoff" / "clinical" / "predictions.csv"
    saved = pd.read_csv(f)
    saved = saved[saved.seed.isin(seeds)].reset_index(drop=True)
    cohort = load_cohort()
    frames = rule_frames(saved, cohort.set_index("RealTag")["mmse"])
    return {name: score_rule(d, draws) for name, d in frames.items()}
