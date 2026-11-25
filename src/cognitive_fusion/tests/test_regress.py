import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from cognitive_fusion.composite import SCORE_COLS, sign_vector
from cognitive_fusion.regress import audit_fit_log, audit_inputs, run_regression_job
from common.data import load_analysis_data, merge_folds
from common.make_folds import load_folds

SIGN = sign_vector({c: -1 for c in SCORE_COLS})


def stub(seed):
    return make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=10.0))


@pytest.fixture(scope="module")
def setup():
    X_df, y, g, cohort = load_analysis_data("task1")
    folds = merge_folds(cohort, load_folds(0))
    ids = cohort[["RealTag", "ParticipantID", "IsFollowup"]]
    return X_df.to_numpy(float), list(X_df.columns), cohort[SCORE_COLS].to_numpy(float), y, ids, folds


@pytest.fixture(scope="module")
def result(setup):
    X, cols, S, y, ids, folds = setup
    return run_regression_job(1, 0, X, cols, S, y, ids, folds, stub, SIGN)


def test_fit_log_audit_passes_and_covers_every_fit(setup, result):
    X, cols, S, y, ids, folds = setup
    audit_fit_log(result["fit_log"], ids, folds)
    assert len(result["fit_log"]) == 4 * 5
    assert len(result["predictions"]) == 176 and result["predictions"]["y_true"].isna().all()
    assert len(result["inner_oof"]) == 3 * 176 and result["inner_oof"]["y_true"].notna().sum() >= 3 * 176 - 3


def test_fit_log_audit_detects_test_visit_in_training(setup, result):
    X, cols, S, y, ids, folds = setup
    bad = [dict(f) for f in result["fit_log"]]
    refit = next(f for f in bad if f["fit"] == "refit" and f["outer_fold"] == 0)
    refit["train"] = refit["train"] + [refit["eval"][0]]
    with pytest.raises(AssertionError):
        audit_fit_log(bad, ids, folds)


def test_test_scores_and_labels_do_not_change_test_predictions(setup, result):
    """Replace the clinical scores and labels of the outer-test rows of fold 1: their predictions must not move."""
    X, cols, S, y, ids, folds = setup
    te = folds["outer_fold"].to_numpy() == 1
    S2, y2 = S.copy(), y.copy()
    S2[te] = 0.0
    y2[te] = 1 - y2[te]
    other = run_regression_job(1, 0, X, cols, S2, y2, ids, folds, stub, SIGN)
    a, b = result["predictions"], other["predictions"]
    m = a["outer_fold"] == 1
    assert np.allclose(a.loc[m, "y_pred"], b.loc[m, "y_pred"], atol=1e-12)


def test_inner_validation_scores_do_not_change_inner_oof_of_that_fold(setup, result):
    X, cols, S, y, ids, folds = setup
    inner = folds["inner_fold_o0"].to_numpy()
    va = inner == 2
    S2 = S.copy()
    S2[va] = np.nan_to_num(S2[va]) * 0 + 1.0
    other = run_regression_job(1, 0, X, cols, S2, y, ids, folds, stub, SIGN, outer_folds=[0])
    a = result["inner_oof"].query("outer_fold == 0 and inner_fold == 2").set_index("RealTag")["y_pred"]
    b = other["inner_oof"].query("outer_fold == 0 and inner_fold == 2").set_index("RealTag")["y_pred"]
    assert np.allclose(a.loc[b.index], b, atol=1e-12)


def test_composite_of_training_rows_uses_outer_training_parameters(setup, result):
    X, cols, S, y, ids, folds = setup
    cp = result["composite_params"][0]["refit"]
    tr = folds["outer_fold"].to_numpy() != 0
    assert np.allclose(cp["mu"], np.nanmean(S[tr], axis=0))
    assert np.allclose(cp["sd"], np.nanstd(S[tr], axis=0, ddof=1))
    assert cp["excluded"] == ["BEGMCI-010"] or cp["excluded"] == []


def test_input_audit_rejects_clinical_columns(setup):
    X, cols, S, y, ids, folds = setup
    age = np.arange(len(X), dtype=float)
    audit_inputs(cols, cols, X, age, S, ["Ikä", "MMSE"])
    with pytest.raises(AssertionError):
        audit_inputs(cols, cols[::-1], X, age, S, ["Ikä"])
    X2 = np.c_[X, S[:, 0] * 2 + 1]
    with pytest.raises(AssertionError):
        audit_inputs(cols + ["leaky"], cols + ["leaky"], X2, age, S, ["Ikä"])
    with pytest.raises(AssertionError):
        audit_inputs(cols + ["MMSE_total"], cols + ["MMSE_total"], np.c_[X, np.zeros(len(X))], age, S, [])
