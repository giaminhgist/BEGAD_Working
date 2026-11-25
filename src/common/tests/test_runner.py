import numpy as np
import pytest
from common import runner
from common.data import load_analysis_data, load_cohort, merge_folds
from common.make_folds import load_folds
from common.metrics import auc_score, youden_threshold
from common.runner import Job, run_job, select_config

CFG = {"name": "stub_lr", "fixed": {"max_iter": 2000}, "grid": {"C": [0.01, 1.0]}}


def make_job(seed=0, fs="task1"):
    return Job(family="baseline", registry="common.tests.stub_models", model="stub_lr", model_cfg=CFG,
               feature_set=fs, seed=seed)


@pytest.fixture(scope="module")
def result():
    return run_job(make_job())


def test_select_config_ties_and_nan():
    assert select_config([0.5, 0.5, 0.4]) == 0
    assert select_config([np.nan, 0.3, 0.3]) == 1
    assert select_config([0.2, 0.9, 0.9]) == 1


def test_each_visit_predicted_once_per_seed(result):
    p = result["predictions"]
    assert len(p) == 176 and p["RealTag"].is_unique
    assert set(p["outer_fold"]) == {0, 1, 2, 3}
    assert list(p.columns) == ["seed", "outer_fold", "RealTag", "ParticipantID", "IsFollowup", "y_true", "y_score",
                               "threshold", "y_hat"]


def test_inner_oof_is_training_rows_only_and_subject_disjoint(result):
    p, o = result["predictions"], result["inner_oof"]
    assert len(o) == 4 * 176 - 176 and not o["y_score"].isna().any()
    for f, te in p.groupby("outer_fold"):
        tr = o[o.outer_fold == f]
        assert not (set(te["RealTag"]) & set(tr["RealTag"]))
        assert not (set(te["ParticipantID"]) & set(tr["ParticipantID"]))
        assert tr["RealTag"].is_unique


def test_threshold_is_youden_of_inner_oof(result):
    p, o = result["predictions"], result["inner_oof"]
    for f, te in p.groupby("outer_fold"):
        tr = o[o.outer_fold == f]
        thr = youden_threshold(tr["y_true"], tr["y_score"])
        assert (te["threshold"] == thr).all()
        assert (te["y_hat"] == (te["y_score"] >= thr).astype(int)).all()


def test_selected_config_has_best_inner_auc(result):
    cv = result["cv_results"]
    for f, sel in result["selected"].items():
        rows = cv[cv.outer_fold == f]
        assert sel["inner_score"] == pytest.approx(rows["inner_score"].max())


def test_outer_test_labels_do_not_influence_test_scores(monkeypatch, result):
    """Flip the labels of the outer-test rows of fold 0: their own scores must not change (no test-label leakage)."""
    X, y, g, c = load_analysis_data("task1")
    folds = merge_folds(c, load_folds(0))
    te = folds["outer_fold"].to_numpy() == 0
    y2 = y.copy()
    y2[te] = 1 - y2[te]
    monkeypatch.setattr(runner, "load_analysis_data", lambda fs: (X, y2, g, c))
    other = run_job(make_job())
    a, b = result["predictions"], other["predictions"]
    fold0 = a["outer_fold"] == 0
    assert np.array_equal(a.loc[fold0, "y_score"], b.loc[fold0, "y_score"])
    assert np.array_equal(a.loc[fold0, "threshold"], b.loc[fold0, "threshold"])
    assert not np.array_equal(a.loc[fold0, "y_true"], b.loc[fold0, "y_true"])


def test_outer_test_features_do_not_influence_inner_oof_or_threshold(monkeypatch, result):
    X, y, g, c = load_analysis_data("task1")
    folds = merge_folds(c, load_folds(0))
    te = folds["outer_fold"].to_numpy() == 2
    X2 = X.copy()
    X2.iloc[np.flatnonzero(te), :] = 1e6
    monkeypatch.setattr(runner, "load_analysis_data", lambda fs: (X2, y, g, c))
    other = run_job(make_job())
    a, b = result["inner_oof"], other["inner_oof"]
    m = a["outer_fold"] == 2
    assert np.array_equal(a.loc[m, "y_score"], b.loc[m, "y_score"])
    assert np.array_equal(result["predictions"].query("outer_fold == 2")["threshold"],
                          other["predictions"].query("outer_fold == 2")["threshold"])
