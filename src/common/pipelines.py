"""Pipeline building blocks shared by the model families: `DropDegenerate`, the balanced-weight wrapper and
`wrap_pipeline`, which puts a bare estimator behind the preprocessing named by its pipeline type."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data


class DropDegenerate(SelectorMixin, BaseEstimator):
    """Drop columns with <= 1 unique non-NaN value (fitted on the training rows only)."""

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=np.float64, reset=True)
        n_unique = np.array([len(np.unique(X[~np.isnan(X[:, j]), j])) for j in range(X.shape[1])])
        self.support_ = n_unique > 1
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


@dataclass
class BuildContext:
    seed: int
    smoke: bool = False
    n_threads: int = 1


class BalancedSampleWeight(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """Fit `estimator` with sample_weight = n / (2 n_c), computed on the rows it is fitted on."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        w_c = dict(zip(classes, len(y) / (len(classes) * counts)))
        w = np.array([w_c[v] for v in y])
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=w)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def wrap_pipeline(final, ptype: str):
    """`scaled`: DropDegenerate -> median impute -> StandardScaler -> model; `native`: DropDegenerate -> model
    (the model handles NaN itself); `tabpfn`: the estimator is used as is (it drops degenerate columns itself)."""
    if ptype == "tabpfn":
        return final
    steps = [("drop", DropDegenerate())]
    if ptype == "scaled":
        steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    elif ptype != "native":
        raise ValueError(f"unknown pipeline type {ptype!r}")
    steps.append(("model", final))
    return Pipeline(steps)


def describe_features(X: np.ndarray, columns) -> dict:
    """Columns dropped as degenerate in the final fit (same deterministic step for every model)."""
    kept = DropDegenerate().fit(X).get_support()
    return {"n_features_after_drop": int(kept.sum()), "dropped_columns": np.asarray(columns)[~kept].tolist()}
