"""Classification metrics: rank-based AUC (cheap enough for bootstrap replicates), Youden threshold, Acc/Sen/Spec."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

NAN = float("nan")


def auc_score(y, s) -> float:
    """ROC AUC (Mann-Whitney form, ties count 1/2). NaN if only one class is present."""
    y = np.asarray(y)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return NAN
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden_threshold(y, score) -> float:
    """Threshold of maximal Youden J over finite thresholds of roc_curve(drop_intermediate=False); ties -> highest."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        raise ValueError("Youden threshold needs both classes in the inner OOF scores")
    fpr, tpr, thr = roc_curve(y, score, drop_intermediate=False)
    ok = np.isfinite(thr)
    j = (tpr - fpr)[ok]
    return float(thr[ok][int(np.argmax(j))])


def threshold_metrics(y, yhat) -> dict:
    """Accuracy, sensitivity (MCI recall), specificity (CN recall) and the confusion counts at a fixed rule."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    n = tp + fn + tn + fp
    sens = tp / (tp + fn) if tp + fn else NAN
    spec = tn / (tn + fp) if tn + fp else NAN
    return {"acc": (tp + tn) / n if n else NAN, "sens": sens, "spec": spec,
            "bal_acc": (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else NAN,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def cls_metrics(y, score, yhat=None) -> dict:
    out = {"auc": auc_score(y, score)}
    if yhat is not None:
        out.update(threshold_metrics(y, yhat))
    return out
