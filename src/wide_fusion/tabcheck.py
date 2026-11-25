"""Feature-subsampling guard: every TabPFN estimator must see all the columns it is given.

`fit_checked` turns the TabPFN UserWarnings about feature subsampling / feature coverage into errors, and reads the
per-estimator feature counts of the fitted model (`ensemble_preprocessor_.subsample_feature_indices`: None = all
columns). If any estimator sees fewer columns than the model received, it raises: the run stops and the settings
(n_estimators, columns) are NOT changed. A model without that attribute (a test stub) returns None."""
from __future__ import annotations
import warnings

import numpy as np

class SubsamplingStop(RuntimeError):
    """TabPFN feature subsampling: a STOP (the run halts at once; nothing is changed to get around it)."""


SUBSAMPLE_WARNING = r".*(covers at most|eature subsampling|never sampled|max_features_per_estimator|cover all features).*"


def estimator_features(model, n_estimators: int | None = None, n_offered: int | None = None, names=None) -> dict | None:
    """Per fit: offered width, columns dropped as degenerate (by the wrapper's DropDegenerate rule, allowed), fitted
    width, per-estimator counts. Subsampling (a per-estimator count != fitted width) is a STOP; so is
    offered - dropped != fitted width."""
    m = getattr(model, "model_", None)
    ep = getattr(m, "ensemble_preprocessor_", None)
    if ep is None:
        return None
    n_total = int(ep.feature_schema.num_columns)
    per = [n_total if i is None else int(len(i)) for i in ep.subsample_feature_indices]
    budget = sorted({int(c.preprocess_config.max_features_per_estimator) for c in ep.configs})
    offered = n_total if n_offered is None else int(n_offered)
    support = getattr(model, "support_", None)
    if support is None:
        dropped_idx = []
    else:
        dropped_idx = sorted(set(range(offered)) - {int(i) for i in support})
    dropped = [str(names[i]) if names is not None and i < len(names) else int(i) for i in dropped_idx]
    info = {"n_columns_offered": offered, "n_degenerate_dropped": len(dropped_idx), "degenerate_dropped": dropped,
            "n_features_in_model": n_total, "per_estimator_features": per, "max_features_per_estimator": budget,
            "n_estimators": len(per)}
    fi = getattr(model, "fit_info_", {}) or {}
    if "n_features_in_model" in fi and int(fi["n_features_in_model"]) != n_total:
        raise AssertionError(f"TabPFN preprocessor sees {n_total} columns, wrapper reports {fi['n_features_in_model']}")
    if n_estimators is not None and len(per) != n_estimators:
        raise AssertionError(f"{len(per)} estimators, expected {n_estimators}")
    if offered - len(dropped_idx) != n_total:
        raise AssertionError(f"offered {offered} - dropped {len(dropped_idx)} != fitted width {n_total}")
    if any(c != n_total for c in per):
        raise SubsamplingStop(f"TabPFN feature subsampling: per-estimator feature counts {per} of {n_total} fitted columns "
                           f"(budget {budget}). STOP: do not change n_estimators or columns; report.")
    return info


def fit_checked(model, X, y, n_estimators: int | None = None, names=None) -> dict | None:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=SUBSAMPLE_WARNING, category=UserWarning)
        model.fit(X, y)
    return estimator_features(model, n_estimators, n_offered=np.shape(X)[1], names=names)
